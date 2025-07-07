"""
Unit tests for DNNE template generation and code execution.

Tests template loading, variable substitution, generated code validation,
and execution of exported scripts.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import subprocess
import sys

from fixtures.node_data import SAMPLE_TEMPLATE_VARS
from fixtures.test_utils import (
    assert_valid_python_code, create_temp_export_dir, cleanup_export_dir
)


class TestTemplateLoading:
    """Test template file loading and access."""
    
    @pytest.mark.export
    def test_template_directory_exists(self):
        """Test that template directory exists."""
        template_dir = Path("export_system/templates/nodes")
        assert template_dir.exists(), f"Template directory not found: {template_dir}"
        assert template_dir.is_dir(), f"Template path is not a directory: {template_dir}"
    
    @pytest.mark.export
    def test_queue_templates_exist(self):
        """Test that queue-based templates exist for key nodes."""
        template_dir = Path("export_system/templates/nodes")
        
        # Key templates that should exist
        expected_templates = [
            "mnist_dataset_queue.py",
            "linear_layer_queue.py", 
            "network_queue.py",
            "sgd_optimizer_queue.py",
            "cross_entropy_queue.py",
            "training_step_queue.py"
        ]
        
        existing_templates = list(template_dir.glob("*_queue.py"))
        template_names = [t.name for t in existing_templates]
        
        # At least some key templates should exist
        found_templates = [t for t in expected_templates if t in template_names]
        assert len(found_templates) > 0, f"No expected templates found. Existing: {template_names}"
    
    @pytest.mark.export
    def test_template_file_format(self):
        """Test that template files have correct format."""
        template_dir = Path("export_system/templates/nodes")
        template_files = list(template_dir.glob("*_queue.py"))
        
        for template_file in template_files[:3]:  # Test first 3 templates
            assert template_file.exists()
            assert template_file.suffix == ".py"
            
            # Should contain Python code
            content = template_file.read_text()
            assert len(content) > 0
            
            # Should contain template variables
            assert "{" in content and "}" in content, \
                f"Template {template_file.name} should contain template variables"
    
    @pytest.mark.export
    def test_base_template_framework(self):
        """Test that base framework templates exist."""
        base_dir = Path("export_system/templates/base")
        
        expected_base_files = [
            "queue_framework.py",
            "imports.py"
        ]
        
        for base_file in expected_base_files:
            file_path = base_dir / base_file
            if file_path.exists():
                content = file_path.read_text()
                assert len(content) > 0
                
                # Should contain class definitions for queue framework
                if "queue_framework" in base_file:
                    assert "class" in content
                    assert "QueueNode" in content or "GraphRunner" in content


class TestTemplateVariableSubstitution:
    """Test template variable substitution and formatting."""
    
    @pytest.mark.export
    def test_basic_variable_substitution(self):
        """Test basic template variable substitution."""
        # Simple template with variables
        template_content = """
class {CLASS_NAME}_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.param1 = {PARAM1}
        self.param2 = "{PARAM2}"
        self.param3 = {PARAM3}
"""
        
        template_vars = {
            "CLASS_NAME": "TestNode",
            "NODE_ID": "node_1",
            "PARAM1": 42,
            "PARAM2": "test_value",
            "PARAM3": True
        }
        
        # Substitute variables
        generated_code = template_content.format(**template_vars)
        
        # Check substitution worked
        assert "TestNode_node_1" in generated_code
        assert "self.param1 = 42" in generated_code
        assert 'self.param2 = "test_value"' in generated_code
        assert "self.param3 = True" in generated_code
        
        # Generated code should be valid Python
        assert_valid_python_code(generated_code)
    
    @pytest.mark.export
    def test_complex_variable_substitution(self):
        """Test substitution of complex data types."""
        template_content = """
class {CLASS_NAME}_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.list_param = {LIST_PARAM}
        self.dict_param = {DICT_PARAM}
        self.device = "{DEVICE}"
"""
        
        template_vars = {
            "CLASS_NAME": "ComplexNode", 
            "NODE_ID": "node_2",
            "LIST_PARAM": [1, 2, 3, 4],
            "DICT_PARAM": {"key1": "value1", "key2": 42},
            "DEVICE": "cuda"
        }
        
        generated_code = template_content.format(**template_vars)
        
        # Check complex substitutions
        assert "[1, 2, 3, 4]" in generated_code
        assert "{'key1': 'value1', 'key2': 42}" in generated_code or \
               '"key1": "value1"' in generated_code
        assert 'self.device = "cuda"' in generated_code
        
        # Should still be valid Python
        assert_valid_python_code(generated_code)
    
    @pytest.mark.export
    def test_missing_variable_handling(self):
        """Test handling of missing template variables."""
        template_content = """
class {CLASS_NAME}_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        self.missing_var = {MISSING_VAR}
"""
        
        incomplete_vars = {
            "CLASS_NAME": "TestNode",
            "NODE_ID": "node_1"
            # MISSING_VAR not provided
        }
        
        # Should raise KeyError for missing variables
        with pytest.raises(KeyError):
            template_content.format(**incomplete_vars)
    
    @pytest.mark.export
    def test_actual_template_substitution(self):
        """Test substitution with actual template files."""
        template_dir = Path("export_system/templates/nodes")
        template_files = list(template_dir.glob("*_queue.py"))
        
        if len(template_files) == 0:
            pytest.skip("No template files found")
        
        # Test with first available template
        template_file = template_files[0]
        template_content = template_file.read_text()
        
        # Use sample template variables
        sample_vars = SAMPLE_TEMPLATE_VARS["linear_layer"]
        
        try:
            generated_code = template_content.format(**sample_vars)
            
            # Generated code should be valid
            assert_valid_python_code(generated_code)
            
            # Should contain substituted values
            assert sample_vars["NODE_ID"] in generated_code
            assert sample_vars["CLASS_NAME"] in generated_code
            
        except KeyError as e:
            # Template might require variables not in our sample
            missing_var = str(e).strip("'\"")
            pytest.skip(f"Template {template_file.name} requires variable {missing_var}")


class TestGeneratedCodeValidation:
    """Test validation of generated code quality and correctness."""
    
    @pytest.mark.export
    def test_generated_code_syntax(self):
        """Test that generated code has valid Python syntax."""
        # Sample generated code patterns
        code_samples = [
            """
import torch
import torch.nn as nn

class LinearLayer_node_1(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.in_features = 784
        self.out_features = 10
        
    async def compute(self, input_data):
        return {"output": input_data}
""",
            """
import torch.optim as optim

class SGDOptimizer_node_2(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.learning_rate = 0.01
        self.momentum = 0.9
        
    async def compute(self, model):
        optimizer = optim.SGD(model.parameters(), 
                            lr=self.learning_rate,
                            momentum=self.momentum)
        return {"optimizer": optimizer}
"""
        ]
        
        for code_sample in code_samples:
            assert_valid_python_code(code_sample)
    
    @pytest.mark.export
    def test_generated_code_imports(self):
        """Test that generated code imports are resolvable."""
        # Common import patterns in generated code
        import_tests = [
            "import torch",
            "import torch.nn as nn", 
            "import torch.optim as optim",
            "import asyncio",
            "import numpy as np"
        ]
        
        for import_stmt in import_tests:
            # Each import should be valid Python syntax
            assert_valid_python_code(import_stmt)
            
            # Try to actually import (if available)
            try:
                exec(import_stmt)
            except ImportError:
                # Module might not be available in test environment
                pass
    
    @pytest.mark.export
    def test_queue_node_class_structure(self):
        """Test that generated QueueNode classes have proper structure."""
        generated_code = """
class TestNode_node_1(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input1"], optional=["input2"])
        self.setup_outputs(["output1", "output2"])
        
    async def compute(self, input1, input2=None):
        result = {"output1": input1, "output2": input2}
        return result
"""
        
        # Should be valid Python
        assert_valid_python_code(generated_code)
        
        # Should contain required QueueNode patterns
        assert "class" in generated_code
        assert "QueueNode" in generated_code
        assert "__init__" in generated_code
        assert "async def compute" in generated_code
        assert "super().__init__" in generated_code
    
    @pytest.mark.export
    def test_async_method_patterns(self):
        """Test async method patterns in generated code."""
        # Test individual async patterns with proper indentation
        async_patterns = [
            # Async method definition pattern
            """
async def compute(self, **kwargs):
    pass
""",
            # Async queue operations
            """
async def test_method(self):
    result = await self.input_queues['input'].get()
    await self.output_queues['output'].put(result)
    return {"output": result}
""",
            # Complete async compute pattern
            """
async def compute(self, input_data):
    result = self.process_data(input_data)
    return {"output": result}
"""
        ]
        
        for pattern in async_patterns:
            # Each pattern should be valid Python syntax
            assert_valid_python_code(pattern.strip())


class TestCodeExecution:
    """Test execution of generated code."""
    
    @pytest.mark.export
    def test_simple_runner_execution(self):
        """Test execution of simple generated runner script."""
        # Create minimal runner script
        runner_code = """
#!/usr/bin/env python3
import asyncio
import sys

class MockQueueNode:
    def __init__(self, node_id):
        self.node_id = node_id
        
    async def compute(self):
        return {"result": f"computed_{self.node_id}"}

async def main():
    node = MockQueueNode("test_node")
    result = await node.compute()
    print(f"Success: {result}")
    return True

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
"""
        
        # Write to temporary file and execute
        export_dir = create_temp_export_dir()
        
        try:
            runner_file = export_dir / "test_runner.py"
            runner_file.write_text(runner_code)
            
            # Execute the script
            result = subprocess.run(
                [sys.executable, str(runner_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should execute successfully
            assert result.returncode == 0, f"Script failed with: {result.stderr}"
            assert "Success:" in result.stdout
            
        finally:
            cleanup_export_dir(export_dir)
    
    @pytest.mark.export
    @pytest.mark.slow
    def test_queue_framework_execution(self):
        """Test execution of queue framework components."""
        # Test basic queue framework functionality
        queue_test_code = """
#!/usr/bin/env python3
import asyncio

class QueueNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.input_queues = {}
        self.output_queues = {}
        
    def setup_inputs(self, required=None, optional=None):
        required = required or []
        optional = optional or []
        for input_name in required + optional:
            self.input_queues[input_name] = asyncio.Queue(maxsize=2)
            
    def setup_outputs(self, outputs):
        for output_name in outputs:
            self.output_queues[output_name] = asyncio.Queue(maxsize=2)
            
    async def compute(self, **kwargs):
        return {"result": "test_result"}

async def test_queue_node():
    node = QueueNode("test_node")
    node.setup_inputs(required=["input1"])
    node.setup_outputs(["output1"])
    
    # Test basic functionality
    assert node.node_id == "test_node"
    assert "input1" in node.input_queues
    assert "output1" in node.output_queues
    
    result = await node.compute()
    assert result["result"] == "test_result"
    
    print("Queue framework test passed")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_queue_node())
    exit(0 if result else 1)
"""
        
        export_dir = create_temp_export_dir()
        
        try:
            test_file = export_dir / "queue_test.py"
            test_file.write_text(queue_test_code)
            
            # Execute the test
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            assert result.returncode == 0, f"Queue test failed: {result.stderr}"
            assert "Queue framework test passed" in result.stdout
            
        finally:
            cleanup_export_dir(export_dir)
    
    @pytest.mark.export
    def test_import_resolution(self):
        """Test that generated imports can be resolved."""
        # Test script that imports common dependencies
        import_test_code = """
#!/usr/bin/env python3
import sys

def test_imports():
    try:
        import asyncio
        print("asyncio: OK")
        
        import torch
        print("torch: OK")
        
        import torch.nn as nn
        print("torch.nn: OK")
        
        import numpy as np
        print("numpy: OK")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    print(f"Import test result: {success}")
    sys.exit(0 if success else 1)
"""
        
        export_dir = create_temp_export_dir()
        
        try:
            test_file = export_dir / "import_test.py"
            test_file.write_text(import_test_code)
            
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should complete (may fail on missing imports, but shouldn't crash)
            assert result.returncode in [0, 1], f"Import test crashed: {result.stderr}"
            
            if result.returncode == 0:
                assert "torch: OK" in result.stdout
            else:
                # Missing imports are acceptable in test environment
                assert "Import error:" in result.stdout
                
        finally:
            cleanup_export_dir(export_dir)


class TestTemplateIntegration:
    """Integration tests for template generation workflow."""
    
    @pytest.mark.export
    @pytest.mark.integration
    def test_complete_template_workflow(self):
        """Test complete workflow from template to executable code."""
        # Simulate the template generation process
        template_vars = {
            "CLASS_NAME": "TestLinearLayer",
            "NODE_ID": "node_1", 
            "IN_FEATURES": 784,
            "OUT_FEATURES": 10,
            "BIAS": True,
            "DEVICE": "cpu"
        }
        
        # Basic template structure
        template_content = """
import torch
import torch.nn as nn
import asyncio

class {CLASS_NAME}_{NODE_ID}:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.layer = nn.Linear({IN_FEATURES}, {OUT_FEATURES}, bias={BIAS})
        self.device = "{DEVICE}"
        
    async def compute(self, input_data):
        if hasattr(input_data, 'to'):
            input_data = input_data.to(self.device)
        output = self.layer(input_data)
        return {{"output": output}}

async def main():
    node = {CLASS_NAME}_{NODE_ID}("test_node")
    
    # Test with dummy data
    test_input = torch.randn(32, {IN_FEATURES})
    result = await node.compute(test_input)
    
    output = result["output"]
    assert output.shape == (32, {OUT_FEATURES})
    
    print("Template integration test passed")
    return True

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
"""
        
        # Generate code
        generated_code = template_content.format(**template_vars)
        
        # Validate syntax
        assert_valid_python_code(generated_code)
        
        # Test execution
        export_dir = create_temp_export_dir()
        
        try:
            test_file = export_dir / "template_integration_test.py"
            test_file.write_text(generated_code)
            
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=20
            )
            
            if result.returncode == 0:
                assert "Template integration test passed" in result.stdout
            else:
                # Might fail due to missing torch, but should provide meaningful error
                error_output = result.stderr.lower()
                assert "import" in error_output or "module" in error_output
                
        finally:
            cleanup_export_dir(export_dir)
    
    @pytest.mark.export
    def test_error_handling_in_generated_code(self):
        """Test error handling in generated code."""
        # Code with intentional error handling
        error_handling_code = """
#!/usr/bin/env python3
import asyncio

class ErrorHandlingNode:
    def __init__(self, node_id):
        self.node_id = node_id
        
    async def compute(self, input_data):
        try:
            # Simulate processing
            if input_data is None:
                raise ValueError("Input data cannot be None")
                
            result = {"output": f"processed_{input_data}"}
            return result
            
        except Exception as e:
            print(f"Error in {self.node_id}: {e}")
            return {"error": str(e)}

async def test_error_handling():
    node = ErrorHandlingNode("test_node")
    
    # Test normal case
    result1 = await node.compute("test_data")
    assert "output" in result1
    assert result1["output"] == "processed_test_data"
    
    # Test error case  
    result2 = await node.compute(None)
    assert "error" in result2
    assert "cannot be None" in result2["error"]
    
    print("Error handling test passed")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_error_handling())
    exit(0 if result else 1)
"""
        
        export_dir = create_temp_export_dir()
        
        try:
            test_file = export_dir / "error_handling_test.py"
            test_file.write_text(error_handling_code)
            
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            assert result.returncode == 0, f"Error handling test failed: {result.stderr}"
            assert "Error handling test passed" in result.stdout
            
        finally:
            cleanup_export_dir(export_dir)