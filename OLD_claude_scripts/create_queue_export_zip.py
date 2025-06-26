#!/usr/bin/env python3
"""
Script to create a ZIP file with all the queue-based export system files
Run this script to generate dnne_queue_export_system.zip
"""

import zipfile
import os
from pathlib import Path

# Define all files and their contents
files_to_create = {
    # Main export system files
    "export_system/graph_exporter.py": '''#!/usr/bin/env python3
"""
DNNE Queue-Based Export System
Converts node graphs to reactive Python scripts using async queues
"""

from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import logging

class ExportableNode:
    """Base class for nodes that can be exported to code"""
    
    @classmethod
    def get_template_name(cls) -> str:
        """Return the template file name for this node type"""
        raise NotImplementedError
    
    @classmethod
    def prepare_template_vars(cls, node_id: str, node_data: Dict, 
                            connections: Dict) -> Dict[str, Any]:
        """Prepare variables for template substitution"""
        raise NotImplementedError
    
    @classmethod
    def get_imports(cls) -> List[str]:
        """Return list of import statements needed by this node"""
        return []


class GraphExporter:
    """Main export system that converts graphs to queue-based Python scripts"""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"
        self.node_registry = {}  # Maps node types to exportable classes
        self.logger = logging.getLogger(__name__)
        
    def register_node(self, node_type: str, node_class: type):
        """Register an exportable node type"""
        self.node_registry[node_type] = node_class
        self.logger.info(f"Registered node type: {node_type}")
    
    def export_workflow(self, workflow: Dict, output_path: Optional[Path] = None) -> str:
        """Convert workflow JSON to queue-based Python script"""
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])
        metadata = workflow.get("metadata", {})
        
        # Collect all imports
        imports = {
            "#!/usr/bin/env python3",
            "import asyncio",
            "import time",
            "import logging",
            "from typing import Dict, Any, List, Optional",
            "from abc import ABC, abstractmethod",
            "from asyncio import Queue",
            "",
            "# Configure logging",
            "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')",
        }
        
        # Generate node implementations
        node_implementations = []
        node_instances = []
        
        for node in nodes:
            node_id = str(node["id"])
            node_type = node["class_type"]
            
            if node_type in self.node_registry:
                node_class = self.node_registry[node_type]
                
                # Get template and prepare variables
                template_name = node_class.get_template_name()
                template_vars = node_class.prepare_template_vars(
                    node_id, node, self._get_node_connections(node_id, links, nodes)
                )
                
                # Load and process template
                template_content = self._load_template(template_name)
                node_code = self._process_template(template_content, template_vars)
                node_implementations.append(node_code)
                
                # Create instance
                class_name = template_vars.get("CLASS_NAME", node_type + "Node")
                instance_name = f"{node_id}_node"
                node_instances.append(f'{instance_name} = {class_name}_{node_id}("{node_id}")')
                
                # Add imports
                imports.update(node_class.get_imports())
            else:
                self.logger.warning(f"Unknown node type: {node_type}")
                # Generate placeholder
                placeholder_code = self._generate_placeholder_node(node_id, node_type)
                node_implementations.append(placeholder_code)
                node_instances.append(f'{node_id}_node = PlaceholderNode_{node_id}("{node_id}")')
        
        # Load base framework template
        base_framework = self._load_template("base/queue_framework.py")
        
        # Generate connections
        connections = self._generate_connections(links, nodes)
        
        # Assemble final script
        script = self._assemble_script(
            sorted(list(imports)),
            base_framework,
            node_implementations,
            node_instances,
            connections,
            metadata
        )
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(script)
            self.logger.info(f"Exported script to: {output_path}")
        
        return script
    
    def _load_template(self, template_name: str) -> str:
        """Load template file content"""
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return template_path.read_text()
    
    def _process_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Process template by replacing variables"""
        # Replace template variables
        for key, value in variables.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        # Remove template_vars declaration section
        lines = template.split('\\n')
        processed_lines = []
        skip_template_vars = False
        
        for line in lines:
            if line.strip().startswith('template_vars = {'):
                skip_template_vars = True
                continue
            elif skip_template_vars and line.strip() == '}':
                skip_template_vars = False
                continue
            elif not skip_template_vars:
                processed_lines.append(line)
        
        return '\\n'.join(processed_lines)
    
    def _get_node_connections(self, node_id: str, links: List, nodes: List) -> Dict:
        """Get incoming and outgoing connections for a node"""
        connections = {
            "inputs": {},
            "outputs": {}
        }
        
        for link in links:
            # Link format: [link_id, from_node, from_slot, to_node, to_slot]
            if len(link) >= 5:
                from_node = str(link[1])
                to_node = str(link[3])
                
                if to_node == node_id:
                    # Incoming connection
                    connections["inputs"][link[4]] = {
                        "from_node": from_node,
                        "from_slot": link[2]
                    }
                elif from_node == node_id:
                    # Outgoing connection
                    if link[2] not in connections["outputs"]:
                        connections["outputs"][link[2]] = []
                    connections["outputs"][link[2]].append({
                        "to_node": to_node,
                        "to_slot": link[4]
                    })
        
        return connections
    
    def _generate_connections(self, links: List, nodes: List) -> List[str]:
        """Generate connection tuples for wire_nodes"""
        connections = []
        
        # Map node types to their output names
        node_outputs = {}
        for node in nodes:
            node_id = str(node["id"])
            node_type = node["class_type"]
            
            # Default output names based on node type
            if node_type == "MNISTDataset":
                node_outputs[node_id] = ["batch_data", "batch_labels"]
            elif node_type == "LinearLayer":
                node_outputs[node_id] = ["output_tensor"]
            elif node_type == "CameraSensor":
                node_outputs[node_id] = ["image", "timestamp"]
            elif node_type == "VisionNetwork":
                node_outputs[node_id] = ["vision_features"]
            # Add more as needed
            else:
                node_outputs[node_id] = [f"output_{i}" for i in range(3)]
        
        for link in links:
            if len(link) >= 5:
                from_node = str(link[1])
                from_slot = link[2]
                to_node = str(link[3])
                to_slot = link[4]
                
                # Get actual output name
                outputs = node_outputs.get(from_node, [])
                output_name = outputs[from_slot] if from_slot < len(outputs) else f"output_{from_slot}"
                
                # Input names are more standardized
                input_name = self._get_input_name_for_slot(nodes, to_node, to_slot)
                
                connections.append(
                    f'("{from_node}", "{output_name}", "{to_node}", "{input_name}")'
                )
        
        return connections
    
    def _get_input_name_for_slot(self, nodes: List, node_id: str, slot: int) -> str:
        """Get input name for a given slot"""
        # Find node type
        for node in nodes:
            if str(node["id"]) == node_id:
                node_type = node["class_type"]
                
                # Map based on node type
                if node_type == "LinearLayer":
                    return "input_tensor"
                elif node_type == "VisionNetwork":
                    return "camera_data"
                # Add more mappings as needed
                
        return f"input_{slot}"
    
    def _generate_placeholder_node(self, node_id: str, node_type: str) -> str:
        """Generate placeholder for unknown node types"""
        return f\'\'\'
class PlaceholderNode_{node_id}(QueueNode):
    """Placeholder for {node_type} node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input_0"])
        self.setup_outputs(["output_0"])
        self.logger.warning(f"Using placeholder for unknown node type: {node_type}")
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        self.logger.info(f"Placeholder compute for {node_type}")
        return {{"output_0": inputs.get("input_0", None)}}
\'\'\'
    
    def _assemble_script(self, imports: List[str], base_framework: str,
                        node_implementations: List[str], node_instances: List[str],
                        connections: List[str], metadata: Dict) -> str:
        """Assemble the complete script"""
        script_parts = []
        
        # Header
        script_parts.extend([
            imports[0],  # Shebang
            '"""',
            "Generated by DNNE Queue-Based Export System",
            f"Metadata: {json.dumps(metadata, indent=2) if metadata else 'None'}",
            '"""',
            "",
            "# Imports",
        ])
        
        # Add imports (skip shebang)
        script_parts.extend(imports[1:])
        
        # Add framework
        script_parts.extend([
            "",
            "# =" * 40,
            "# Queue Framework",
            "# =" * 40,
            base_framework,
            "",
            "# =" * 40,
            "# Node Implementations",
            "# =" * 40,
        ])
        
        # Add node implementations
        for impl in node_implementations:
            script_parts.append(impl)
            script_parts.append("")
        
        # Add main function
        script_parts.extend([
            "# =" * 40,
            "# Main Execution",
            "# =" * 40,
            "",
            "async def main():",
            '    """Main execution function"""',
            '    print("🚀 Starting DNNE Queue-Based Execution")',
            '    print("=" * 60)',
            "",
            "    # Create nodes",
        ])
        
        # Add node instances
        for instance in node_instances:
            script_parts.append(f"    {instance}")
        
        script_parts.extend([
            "",
            "    # Create runner",
            "    runner = GraphRunner()",
            "",
            "    # Add nodes to runner",
        ])
        
        # Add nodes to runner
        for instance in node_instances:
            node_var = instance.split(" = ")[0].strip()
            script_parts.append(f"    runner.add_node({node_var})")
        
        script_parts.extend([
            "",
            "    # Wire connections",
            "    connections = [",
        ])
        
        # Add connections
        for conn in connections:
            script_parts.append(f"        {conn},")
        
        script_parts.extend([
            "    ]",
            "    runner.wire_nodes(connections)",
            "",
            "    # Run the graph",
            "    try:",
            "        # Run indefinitely (Ctrl+C to stop)",
            "        await runner.run()",
            "        # Or run for specific duration:",
            "        # await runner.run(duration=10.0)  # Run for 10 seconds",
            "    except KeyboardInterrupt:",
            "        print('\\\\n🛑 Stopped by user')",
            "",
            "    # Show final statistics",
            "    print('\\\\n📊 Final Statistics:')",
            "    stats = runner.get_stats()",
            "    for node_id, node_stats in stats.items():",
            "        print(f'  {node_id}: {node_stats[\\"compute_count\\"]} computations, '",
            "              f'avg time: {node_stats[\\"last_compute_time\\"]:.3f}s')",
            "",
            "",
            "if __name__ == '__main__':",
            "    asyncio.run(main())",
        ])
        
        return "\\n".join(script_parts)
''',

    # Node exporters
    "export_system/node_exporters/__init__.py": '''"""
Node exporter classes that handle code generation using queue-based templates
"""

from .ml_nodes import (
    MNISTDatasetExporter,
    LinearLayerExporter,
    LossExporter,
    OptimizerExporter,
    DisplayExporter,
    register_ml_exporters
)

from .robotics_nodes import (
    CameraSensorExporter,
    IMUSensorExporter,
    VisionNetworkExporter,
    SoundNetworkExporter,
    DecisionNetworkExporter,
    RobotControllerExporter,
    IsaacGymEnvExporter,
    register_robotics_exporters
)

# Register all exporters
def register_all_exporters(exporter):
    """Register all node exporters with the graph exporter"""
    register_ml_exporters(exporter)
    register_robotics_exporters(exporter)
    
    # Log registration summary
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Registered {len(exporter.node_registry)} node types for export")

# Export all classes for direct access
__all__ = [
    # ML nodes
    'MNISTDatasetExporter',
    'LinearLayerExporter', 
    'LossExporter',
    'OptimizerExporter',
    'DisplayExporter',
    # Robotics nodes
    'CameraSensorExporter',
    'IMUSensorExporter',
    'VisionNetworkExporter',
    'SoundNetworkExporter',
    'DecisionNetworkExporter',
    'RobotControllerExporter',
    'IsaacGymEnvExporter',
    # Registration functions
    'register_all_exporters',
    'register_ml_exporters',
    'register_robotics_exporters'
]
''',

    "export_system/node_exporters/ml_nodes.py": '''#!/usr/bin/env python3
"""
Exporters for ML nodes using queue-based templates
"""

from ..graph_exporter import ExportableNode

class MNISTDatasetExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/mnist_dataset_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "MNISTDatasetNode",
            "DATA_PATH": params.get("data_path", "./data"),
            "TRAIN": params.get("train", True),
            "DOWNLOAD": params.get("download", True),
            "BATCH_SIZE": params.get("batch_size", 32),
            "EMIT_RATE": params.get("emit_rate", 10.0)  # Batches per second
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from torch.utils.data import DataLoader",
            "from torchvision import datasets, transforms",
        ]


class LinearLayerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/linear_layer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "LinearLayerNode",
            "INPUT_SIZE": params.get("input_size", 784),  # Default for MNIST
            "OUTPUT_SIZE": params.get("output_size", 128),
            "ACTIVATION": params.get("activation", "relu"),
            "DROPOUT": params.get("dropout", 0.0),
            "BIAS": params.get("bias", True)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        ]


class LossExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/loss_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "LossNode",
            "LOSS_TYPE": params.get("loss_type", "cross_entropy")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]


class OptimizerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/optimizer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        # Extract model nodes from connections
        model_nodes = []
        # TODO: Parse from connections to find upstream model nodes
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "OptimizerNode",
            "OPTIMIZER_TYPE": params.get("optimizer", "adam"),
            "LEARNING_RATE": params.get("learning_rate", 0.001),
            "MODEL_NODES": str(model_nodes)  # Will be a list
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.optim as optim",
        ]


class DisplayExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/display_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "DisplayNode",
            "DISPLAY_TYPE": params.get("display_type", "tensor_stats"),
            "LOG_INTERVAL": params.get("log_interval", 10)  # Log every N inputs
        }
    
    @classmethod
    def get_imports(cls):
        return []


# Registration function
def register_ml_exporters(exporter):
    """Register all ML node exporters"""
    exporter.register_node("MNISTDataset", MNISTDatasetExporter)
    exporter.register_node("LinearLayer", LinearLayerExporter)
    exporter.register_node("Loss", LossExporter)
    exporter.register_node("Optimizer", OptimizerExporter)
    exporter.register_node("Display", DisplayExporter)
    
    # Aliases for compatibility
    exporter.register_node("Linear", LinearLayerExporter)
    exporter.register_node("CrossEntropyLoss", LossExporter)
    exporter.register_node("BatchSampler", MNISTDatasetExporter)  # Combined with dataset
''',

    "export_system/node_exporters/robotics_nodes.py": '''#!/usr/bin/env python3
"""
Exporters for robotics nodes using queue-based templates
"""

from ..graph_exporter import ExportableNode

class CameraSensorExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/camera_sensor_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        # Parse resolution
        resolution = params.get("resolution", "640x480")
        width, height = map(int, resolution.split('x'))
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CameraSensorNode",
            "FPS": params.get("fps", 30.0),
            "WIDTH": width,
            "HEIGHT": height,
            "USE_REAL_CAMERA": params.get("use_real_camera", False),
            "CAMERA_INDEX": params.get("camera_index", 0)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import numpy as np",
            "import torch",
        ]


class IMUSensorExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/imu_sensor_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IMUSensorNode",
            "SAMPLE_RATE": params.get("sample_rate", 100.0),
            "ADD_NOISE": params.get("add_noise", True)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import numpy as np",
            "import torch",
        ]


class VisionNetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/vision_network_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "VisionNetworkNode",
            "MODEL_TYPE": params.get("model", "resnet18"),
            "PRETRAINED": params.get("pretrained", True),
            "OUTPUT_DIM": params.get("output_dim", 512),
            "DEVICE": params.get("device", "cuda")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
            "import torchvision.transforms as transforms",
        ]


class SoundNetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/sound_network_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SoundNetworkNode",
            "MODEL_TYPE": params.get("model", "wav2vec"),
            "OUTPUT_DIM": params.get("output_dim", 256),
            "DEVICE": params.get("device", "cuda")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]


class DecisionNetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/decision_network_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        # Count input connections to determine input dimension
        num_inputs = len(connections.get("inputs", {}))
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "DecisionNetworkNode",
            "NUM_INPUTS": num_inputs,
            "ACTION_DIM": params.get("action_dim", 6),
            "HIDDEN_SIZE": params.get("hidden_size", 256),
            "DEVICE": params.get("device", "cuda")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]


class RobotControllerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/robot_controller_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        # Parse joint limits
        joint_limits = params.get("joint_limits", [-3.14, 3.14])
        if isinstance(joint_limits, str):
            joint_limits = eval(joint_limits)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "RobotControllerNode",
            "JOINT_LIMITS_MIN": joint_limits[0],
            "JOINT_LIMITS_MAX": joint_limits[1],
            "CONTROL_TYPE": params.get("control_type", "position"),
            "NUM_JOINTS": params.get("num_joints", 7)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import numpy as np",
            "import torch",
        ]


class IsaacGymEnvExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/isaac_gym_env_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IsaacGymEnvNode",
            "ENV_NAME": params.get("env_name", "Cartpole"),
            "NUM_ENVS": params.get("num_envs", 1),
            "DEVICE": params.get("device", "cuda"),
            "HEADLESS": params.get("headless", False)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import numpy as np",
            "# from isaacgym import gymapi  # Uncomment when Isaac Gym is available",
        ]


# Registration function
def register_robotics_exporters(exporter):
    """Register all robotics node exporters"""
    # Sensors
    exporter.register_node("CameraSensor", CameraSensorExporter)
    exporter.register_node("IMUSensor", IMUSensorExporter)
    exporter.register_node("AudioSensor", CameraSensorExporter)  # Reuse camera template
    
    # Processing networks
    exporter.register_node("VisionNetwork", VisionNetworkExporter)
    exporter.register_node("SoundNetwork", SoundNetworkExporter)
    exporter.register_node("DecisionNetwork", DecisionNetworkExporter)
    
    # Control
    exporter.register_node("RobotController", RobotControllerExporter)
    
    # Isaac Gym
    exporter.register_node("IsaacGymEnv", IsaacGymEnvExporter)
''',

    # Test script
    "export_system/test_export.py": '''#!/usr/bin/env python3
"""
Test script for the queue-based export system
"""

from graph_exporter import GraphExporter
from node_exporters import register_all_exporters
import json
import time

def test_mnist_export():
    """Test exporting a simple MNIST training workflow"""
    print("=" * 60)
    print("Test 1: MNIST Training Workflow")
    print("=" * 60)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    # Test workflow - MNIST with two linear layers
    mnist_workflow = {
        "nodes": [
            {
                "id": "1",
                "class_type": "MNISTDataset",
                "inputs": {
                    "data_path": "./data",
                    "train": True,
                    "download": True,
                    "batch_size": 64,
                    "emit_rate": 5.0  # 5 batches per second
                }
            },
            {
                "id": "2", 
                "class_type": "LinearLayer",
                "inputs": {
                    "input_size": 784,
                    "output_size": 256,
                    "activation": "relu",
                    "dropout": 0.2
                }
            },
            {
                "id": "3",
                "class_type": "LinearLayer", 
                "inputs": {
                    "input_size": 256,
                    "output_size": 10,
                    "activation": "none"
                }
            },
            {
                "id": "4",
                "class_type": "Loss",
                "inputs": {
                    "loss_type": "cross_entropy"
                }
            }
        ],
        "links": [
            [1, "1", 0, "2", 0],  # MNIST data -> Linear1
            [2, "2", 0, "3", 0],  # Linear1 -> Linear2
            [3, "3", 0, "4", 0],  # Linear2 -> Loss (predictions)
            [4, "1", 1, "4", 1],  # MNIST labels -> Loss (labels)
        ],
        "metadata": {
            "name": "MNIST Classifier",
            "description": "Simple two-layer neural network for MNIST",
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Export
    script = exporter.export_workflow(mnist_workflow)
    
    # Save to file
    output_file = "test_mnist_queue.py"
    with open(output_file, "w") as f:
        f.write(script)
    
    print(f"✅ Exported MNIST workflow to: {output_file}")
    print(f"   Total lines: {len(script.splitlines())}")
    

def test_robotics_export():
    """Test exporting a robotics workflow with vision and sound"""
    print("\\n" + "=" * 60)
    print("Test 2: Vision + Sound Robotics Workflow")
    print("=" * 60)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    # Robotics workflow
    robotics_workflow = {
        "nodes": [
            {
                "id": "camera",
                "class_type": "CameraSensor",
                "inputs": {
                    "fps": 30.0,
                    "resolution": "640x480",
                    "use_real_camera": False
                }
            },
            {
                "id": "microphone", 
                "class_type": "AudioSensor",
                "inputs": {
                    "sample_rate": 16.0,
                    "channels": 2
                }
            },
            {
                "id": "vision_net",
                "class_type": "VisionNetwork",
                "inputs": {
                    "model": "resnet18",
                    "pretrained": True,
                    "device": "cuda"
                }
            },
            {
                "id": "sound_net",
                "class_type": "SoundNetwork", 
                "inputs": {
                    "model": "wav2vec",
                    "device": "cuda"
                }
            },
            {
                "id": "decision_net",
                "class_type": "DecisionNetwork",
                "inputs": {
                    "action_dim": 6,
                    "hidden_size": 512,
                    "device": "cuda"
                }
            },
            {
                "id": "robot_ctrl",
                "class_type": "RobotController",
                "inputs": {
                    "num_joints": 7,
                    "joint_limits": [-3.14, 3.14],
                    "control_type": "position"
                }
            }
        ],
        "links": [
            [1, "camera", 0, "vision_net", 0],      # Camera -> Vision
            [2, "microphone", 0, "sound_net", 0],   # Audio -> Sound
            [3, "vision_net", 0, "decision_net", 0], # Vision -> Decision
            [4, "sound_net", 0, "decision_net", 1],  # Sound -> Decision  
            [5, "decision_net", 0, "robot_ctrl", 0]  # Decision -> Robot
        ],
        "metadata": {
            "name": "Multimodal Robot Control",
            "description": "Vision and sound processing for robot decision making",
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Export
    script = exporter.export_workflow(robotics_workflow)
    
    # Save to file
    output_file = "test_robotics_queue.py"
    with open(output_file, "w") as f:
        f.write(script)
    
    print(f"✅ Exported robotics workflow to: {output_file}")
    print(f"   Total lines: {len(script.splitlines())}")
    

def test_simple_linear():
    """Test a minimal workflow"""
    print("\\n" + "=" * 60)
    print("Test 3: Simple Linear Transform")
    print("=" * 60)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    # Minimal workflow
    simple_workflow = {
        "nodes": [
            {
                "id": "data",
                "class_type": "MNISTDataset",
                "inputs": {
                    "batch_size": 32,
                    "emit_rate": 10.0
                }
            },
            {
                "id": "linear",
                "class_type": "LinearLayer",
                "inputs": {
                    "input_size": 784,
                    "output_size": 10
                }
            },
            {
                "id": "display",
                "class_type": "Display",
                "inputs": {
                    "display_type": "tensor_stats",
                    "log_interval": 5
                }
            }
        ],
        "links": [
            [1, "data", 0, "linear", 0],     # Data -> Linear
            [2, "linear", 0, "display", 0]   # Linear -> Display
        ],
        "metadata": {
            "name": "Simple Test",
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Export
    script = exporter.export_workflow(simple_workflow)
    
    # Save to file
    output_file = "test_simple_queue.py"
    with open(output_file, "w") as f:
        f.write(script)
    
    print(f"✅ Exported simple workflow to: {output_file}")
    print(f"   Total lines: {len(script.splitlines())}")


if __name__ == "__main__":
    print("🚀 DNNE Queue-Based Export System Tests")
    print("=" * 60)
    
    # Run all tests
    test_mnist_export()
    test_robotics_export()
    test_simple_linear()
    
    print("\\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\\nGenerated scripts:")
    print("  - test_mnist_queue.py    : MNIST training with async queues")
    print("  - test_robotics_queue.py : Vision+Sound robotics control")
    print("  - test_simple_queue.py   : Simple linear transform demo")
    print("\\nRun any script with: python <script_name>.py")
    print("Press Ctrl+C to stop the async execution.")
''',

    # Template files - Base framework
    "export_system/templates/base/queue_framework.py": '''# Queue-Based Node Framework
class QueueNode(ABC):
    """Base class for all queue-based nodes"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.input_queues: Dict[str, Queue] = {}
        self.output_subscribers: Dict[str, List[Queue]] = {}
        self.required_inputs: List[str] = []
        self.output_names: List[str] = []
        self.running = False
        self.compute_count = 0
        self.last_compute_time = 0.0
        self.logger = logging.getLogger(f"Node.{node_id}")
    
    def setup_inputs(self, required: List[str], queue_size: int = 100):
        """Setup input queues"""
        self.required_inputs = required
        for input_name in required:
            self.input_queues[input_name] = Queue(maxsize=queue_size)
    
    def setup_outputs(self, outputs: List[str]):
        """Setup output specifications"""
        self.output_names = outputs
        for output_name in outputs:
            self.output_subscribers[output_name] = []
    
    async def send_output(self, output_name: str, value: Any):
        """Send output to all subscribers"""
        if output_name in self.output_subscribers:
            for queue in self.output_subscribers[output_name]:
                await queue.put(value)
    
    @abstractmethod
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Override this to implement node logic"""
        pass
    
    async def run(self):
        """Main execution loop"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            while self.running:
                # Gather all required inputs
                inputs = {}
                for input_name in self.required_inputs:
                    value = await self.input_queues[input_name].get()
                    inputs[input_name] = value
                
                # Execute compute
                start_time = time.time()
                outputs = await self.compute(**inputs)
                self.last_compute_time = time.time() - start_time
                self.compute_count += 1
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                    
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False


class SensorNode(QueueNode):
    """Base class for sensor nodes that generate data at fixed rates"""
    
    def __init__(self, node_id: str, update_rate: float):
        super().__init__(node_id)
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
    
    async def run(self):
        """Sensor run loop with fixed rate"""
        self.running = True
        self.logger.info(f"Starting sensor {self.node_id} at {self.update_rate}Hz")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Execute compute
                outputs = await self.compute()
                self.compute_count += 1
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                
                # Sleep to maintain rate
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
                self.last_compute_time = time.time() - start_time
                
        except asyncio.CancelledError:
            self.logger.info(f"Sensor {self.node_id} cancelled")
            raise
        finally:
            self.running = False


class GraphRunner:
    """Manages and runs the complete node graph"""
    
    def __init__(self):
        self.nodes: Dict[str, QueueNode] = {}
        self.tasks: List[asyncio.Task] = []
        self.logger = logging.getLogger("GraphRunner")
    
    def add_node(self, node: QueueNode):
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        self.logger.info(f"Added node: {node.node_id}")
    
    def wire_nodes(self, connections: List[tuple]):
        """Wire nodes together: (from_id, output, to_id, input)"""
        for from_id, output_name, to_id, input_name in connections:
            from_node = self.nodes[from_id]
            to_node = self.nodes[to_id]
            
            # Subscribe to_node's input queue to from_node's output
            from_node.output_subscribers[output_name].append(
                to_node.input_queues[input_name]
            )
            self.logger.info(f"Connected {from_id}.{output_name} -> {to_id}.{input_name}")
    
    async def run(self, duration: Optional[float] = None):
        """Run all nodes"""
        self.logger.info("Starting graph execution")
        
        # Start all nodes
        for node in self.nodes.values():
            task = asyncio.create_task(node.run())
            self.tasks.append(task)
        
        try:
            if duration:
                await asyncio.sleep(duration)
                self.logger.info(f"Stopping after {duration}s")
            else:
                # Run until cancelled
                await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.logger.info("All nodes stopped")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics"""
        return {
            node_id: {
                "compute_count": node.compute_count,
                "last_compute_time": node.last_compute_time,
                "running": node.running
            }
            for node_id, node in self.nodes.items()
        }
''',

    # ML node templates
    "export_system/templates/nodes/mnist_dataset_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "mnist_1",
    "CLASS_NAME": "MNISTDatasetNode",
    "DATA_PATH": "./data",
    "TRAIN": True,
    "DOWNLOAD": True,
    "BATCH_SIZE": 32,
    "EMIT_RATE": 10.0
}

class {CLASS_NAME}_{NODE_ID}(SensorNode):
    """MNIST dataset loader that emits batches at fixed rate"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, update_rate={EMIT_RATE})
        self.setup_outputs(["batch_data", "batch_labels"])
        
        # Setup dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.dataset = datasets.MNIST(
            root="{DATA_PATH}",
            train={TRAIN},
            download={DOWNLOAD},
            transform=transform
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size={BATCH_SIZE},
            shuffle=True,
            num_workers=0
        )
        
        self.data_iter = iter(self.dataloader)
        self.epoch = 0
    
    async def compute(self) -> Dict[str, Any]:
        try:
            images, labels = next(self.data_iter)
        except StopIteration:
            # Reset iterator at end of epoch
            self.epoch += 1
            self.data_iter = iter(self.dataloader)
            images, labels = next(self.data_iter)
            self.logger.info(f"Starting epoch {{self.epoch}}")
        
        return {{
            "batch_data": images,
            "batch_labels": labels
        }}
''',

    "export_system/templates/nodes/linear_layer_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "linear_1",
    "CLASS_NAME": "LinearLayerNode",
    "INPUT_SIZE": 784,
    "OUTPUT_SIZE": 128,
    "ACTIVATION": "relu",
    "DROPOUT": 0.0,
    "BIAS": True
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Linear layer with activation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input_tensor"])
        self.setup_outputs(["output_tensor"])
        
        # Create layer
        self.linear = nn.Linear({INPUT_SIZE}, {OUTPUT_SIZE}, bias={BIAS})
        self.dropout = nn.Dropout({DROPOUT}) if {DROPOUT} > 0 else None
        self.activation = "{ACTIVATION}"
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = self.linear.to(self.device)
        
    async def compute(self, input_tensor) -> Dict[str, Any]:
        # Ensure input is on correct device
        x = input_tensor.to(self.device)
        
        # Flatten if needed (for MNIST)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass
        x = self.linear(x)
        
        # Activation
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        elif self.activation == "sigmoid":
            x = torch.sigmoid(x)
        
        # Dropout if training
        if self.dropout is not None:
            x = self.dropout(x)
        
        return {{"output_tensor": x}}
''',

    "export_system/templates/nodes/loss_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "loss_1",
    "CLASS_NAME": "LossNode",
    "LOSS_TYPE": "cross_entropy"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Loss computation node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["predictions", "labels"])
        self.setup_outputs(["loss", "accuracy"])
        
        # Setup loss function
        self.loss_type = "{LOSS_TYPE}"
        if self.loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()
    
    async def compute(self, predictions, labels) -> Dict[str, Any]:
        # Compute loss
        loss = self.criterion(predictions, labels)
        
        # Compute accuracy for classification
        accuracy = 0.0
        if self.loss_type == "cross_entropy":
            _, predicted = torch.max(predictions, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total if total > 0 else 0.0
        
        self.logger.info(f"Loss: {{loss.item():.4f}}, Accuracy: {{accuracy:.2%}}")
        
        return {{
            "loss": loss,
            "accuracy": accuracy
        }}
''',

    "export_system/templates/nodes/display_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "display_1",
    "CLASS_NAME": "DisplayNode",
    "DISPLAY_TYPE": "tensor_stats",
    "LOG_INTERVAL": 10
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Display/logging node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input_0"])
        self.setup_outputs([])  # No outputs
        
        self.display_type = "{DISPLAY_TYPE}"
        self.display_count = 0
        self.log_interval = {LOG_INTERVAL}
    
    async def compute(self, input_0) -> Dict[str, Any]:
        self.display_count += 1
        
        # Only log at intervals
        if self.display_count % self.log_interval == 0:
            if self.display_type == "tensor_stats" and hasattr(input_0, 'shape'):
                self.logger.info(f"[{{self.display_count}}] Tensor shape: {{input_0.shape}}, "
                              f"min: {{input_0.min().item():.4f}}, "
                              f"max: {{input_0.max().item():.4f}}, "
                              f"mean: {{input_0.mean().item():.4f}}")
            elif self.display_type == "value":
                self.logger.info(f"[{{self.display_count}}] Value: {{input_0}}")
            else:
                self.logger.info(f"[{{self.display_count}}] {{type(input_0)}}")
        
        return {{}}  # No outputs
''',

    # Robotics node templates
    "export_system/templates/nodes/camera_sensor_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "camera_1",
    "CLASS_NAME": "CameraSensorNode",
    "FPS": 30.0,
    "WIDTH": 640,
    "HEIGHT": 480,
    "USE_REAL_CAMERA": False,
    "CAMERA_INDEX": 0
}

class {CLASS_NAME}_{NODE_ID}(SensorNode):
    """Camera sensor that generates image data at fixed FPS"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, update_rate={FPS})
        self.setup_outputs(["image", "timestamp"])
        
        self.width = {WIDTH}
        self.height = {HEIGHT}
        self.channels = 3
        self.frame_count = 0
        self.use_real_camera = {USE_REAL_CAMERA}
        
        if self.use_real_camera:
            import cv2
            self.camera = cv2.VideoCapture({CAMERA_INDEX})
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, {FPS})
        
    async def compute(self) -> Dict[str, Any]:
        if self.use_real_camera:
            # Read from real camera
            ret, frame = self.camera.read()
            if ret:
                # Convert BGR to RGB and normalize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            else:
                # Fallback to random if camera fails
                image = torch.rand(self.channels, self.height, self.width)
        else:
            # Simulated camera data for testing
            image = torch.rand(self.channels, self.height, self.width)
        
        timestamp = time.time()
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            self.logger.info(f"Captured frame {{self.frame_count}}")
        
        return {{
            "image": image,
            "timestamp": timestamp
        }}
    
    def __del__(self):
        """Cleanup camera resources"""
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
''',

    "export_system/templates/nodes/vision_network_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "vision_1",
    "CLASS_NAME": "VisionNetworkNode",
    "MODEL_TYPE": "resnet18",
    "PRETRAINED": True,
    "OUTPUT_DIM": 512,
    "DEVICE": "cuda"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Vision processing network"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["camera_data"])
        self.setup_outputs(["vision_features"])
        
        # Setup device
        self.device = torch.device("{DEVICE}" if torch.cuda.is_available() else "cpu")
        
        # Load vision model
        if "{MODEL_TYPE}" == "resnet18":
            from torchvision.models import resnet18
            self.model = resnet18(pretrained={PRETRAINED})
            # Remove final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif "{MODEL_TYPE}" == "mobilenet":
            from torchvision.models import mobilenet_v2
            self.model = mobilenet_v2(pretrained={PRETRAINED})
            self.model.classifier = nn.Identity()
        else:
            # Custom model placeholder
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, {OUTPUT_DIM})
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to eval mode
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    async def compute(self, camera_data) -> Dict[str, Any]:
        # Move to device
        image = camera_data.to(self.device)
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Normalize
        image = self.transform(image)
        
        # Forward pass
        with torch.no_grad():
            features = self.model(image)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        return {{"vision_features": features}}
''',

    "export_system/templates/nodes/decision_network_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "decision_1",
    "CLASS_NAME": "DecisionNetworkNode",
    "NUM_INPUTS": 2,
    "ACTION_DIM": 6,
    "HIDDEN_SIZE": 256,
    "DEVICE": "cuda"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Decision network that waits for all inputs before processing"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # This node waits for both vision and sound features
        self.setup_inputs(required=["vision_features", "sound_features"])
        self.setup_outputs(["action", "confidence"])
        
        # Setup device
        self.device = torch.device("{DEVICE}" if torch.cuda.is_available() else "cpu")
        
        # Build decision network
        # Assumes vision_features and sound_features are flattened vectors
        self.fusion_layer = nn.Linear(512 + 256, {HIDDEN_SIZE})  # Adjust based on actual sizes
        self.hidden_layer = nn.Linear({HIDDEN_SIZE}, {HIDDEN_SIZE})
        self.action_layer = nn.Linear({HIDDEN_SIZE}, {ACTION_DIM})
        self.confidence_layer = nn.Linear({HIDDEN_SIZE}, 1)
        
        # Move to device
        self.fusion_layer = self.fusion_layer.to(self.device)
        self.hidden_layer = self.hidden_layer.to(self.device)
        self.action_layer = self.action_layer.to(self.device)
        self.confidence_layer = self.confidence_layer.to(self.device)
        
    async def compute(self, vision_features, sound_features) -> Dict[str, Any]:
        # This method only executes when BOTH inputs are available
        # The queue framework handles the synchronization automatically
        
        # Move inputs to device
        vision = vision_features.to(self.device)
        sound = sound_features.to(self.device)
        
        # Ensure batch dimension
        if vision.dim() == 1:
            vision = vision.unsqueeze(0)
        if sound.dim() == 1:
            sound = sound.unsqueeze(0)
        
        # Fuse multimodal features
        fused = torch.cat([vision, sound], dim=1)
        
        # Forward pass
        x = F.relu(self.fusion_layer(fused))
        x = F.relu(self.hidden_layer(x))
        
        # Generate action and confidence
        action = self.action_layer(x)
        confidence = torch.sigmoid(self.confidence_layer(x))
        
        self.logger.info(f"Decision made with confidence: {{confidence.mean().item():.2f}}")
        
        return {{
            "action": action,
            "confidence": confidence
        }}
''',

    # Add more templates as needed...
}

def create_zip_file():
    """Create a zip file with all the queue export system files"""
    zip_filename = "dnne_queue_export_system.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, content in files_to_create.items():
            # Add file to zip
            zipf.writestr(file_path, content)
            print(f"Added: {file_path}")
    
    print(f"\n✅ Created {zip_filename}")
    print(f"Total files: {len(files_to_create)}")
    print("\nTo install:")
    print(f"1. Copy {zip_filename} to your DNNE-UI project root")
    print(f"2. Run: unzip {zip_filename}")
    print("3. Test with: python export_system/test_export.py")

if __name__ == "__main__":
    create_zip_file()
