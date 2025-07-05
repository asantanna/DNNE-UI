# DNNE Development TODO

## Overview
This file is a permanent record of development tasks created and completed during the DNNE project. It serves as a log of the development process, tracking what is currently being worked on and what has been completed.

For larger features and architectural considerations not currently being worked on, see TASKS.md.

---

## OPEN TO-DO ITEMS

### High Priority
1. **Test SYNC-based synchronized training loop**: Verify that the new SYNC type and training synchronization fixes the tensor gradient computation error:
   - Test the updated MNIST workflow with TrainingStep.ready → GetBatch.trigger connection
   - Confirm no "tensor modified by inplace operation" errors
   - Validate performance maintains reasonable training speed

2. **Training Validation**: Complete full MNIST training runs to validate 90%+ accuracy achievement:
   - Run full 5-epoch training cycles  
   - Compare against standard PyTorch MNIST benchmarks
   - Document performance characteristics

### Medium Priority
1. **ConvNet Support**: Add convolutional layer nodes for image processing:
   - Conv2D, MaxPool2D, BatchNorm2D nodes
   - Proper shape propagation through conv layers
   - CNN-based MNIST classifier example

2. **Advanced Optimizers**: Extend beyond SGD:
   - Adam, AdamW, RMSprop optimizer nodes
   - Learning rate scheduling nodes
   - Optimizer parameter tuning guidance

3. **Data Pipeline Enhancement**:
   - Custom dataset loading nodes
   - Data augmentation nodes (rotation, scaling, noise)
   - Validation dataset splitting

### Low Priority
1. **Visualization Improvements**:
   - Real-time loss/accuracy plotting nodes
   - Network architecture visualization
   - Training progress dashboards

2. **Model Export/Import**:
   - Save/load trained model weights
   - ONNX export capabilities
   - Model versioning system

### System Maintenance
1. **Export System Health**:
   - Monitor for any regression in slot corruption fix
   - Ensure new nodes follow queue-based template patterns
   - Improve error messages in export failures

2. **Code Quality**:
   - Add inline documentation to complex template generation logic
   - Expand automated tests for export system
   - Consider consolidating similar node patterns

---

## COMPLETED TO-DO ITEMS

### Recently Completed (January 2025)
- ~~**Refactor Export System to Generate Modular Package Structure**: Transform from monolithic 719-line runner.py to well-organized Python package with separate modules for each node~~
- ~~**Create backup of current graph_exporter.py before refactoring~~
- ~~**Add new methods to GraphExporter**: _create_package_structure, _export_framework, _export_node_to_file, _generate_node_init, _generate_minimal_runner~~
- ~~**Refactor export_workflow() method to use new modular approach~~
- ~~**Update _process_template() to handle per-node imports~~
- ~~**Test the new export system with MNIST Test workflow~~
- ~~**Fix missing imports in generated node files**: Added automatic detection for Dict, Any, asyncio, time imports~~
- ~~**Fix relative imports to absolute imports**: Changed from ..framework.base to framework.base~~
- ~~**Fix node architecture deadlock issues**: Resolved queue-based execution hanging~~
- ~~**Test modular MNIST export successfully**: Verified training runs with loss/accuracy output~~

### Major Achievements (June 2025)
- ~~**Slot Mapping Crisis Resolution**: Fixed critical ComfyUI slot corruption issue with JSON-based workaround~~
- ~~**Network Node Implementation**: Successfully implemented Network node pattern consolidating LinearLayers~~
- ~~**Training Progress System**: Added EpochTracker, enhanced GetBatch and CrossEntropyLoss for real-time stats~~
- ~~**MNIST Pipeline Complete**: Full working MNIST classification with proper training loop~~
- ~~**Device Compatibility**: Fixed GPU/CPU tensor device mismatch issues~~
- ~~**Export Registration**: Completed node exporter registration for all ML nodes~~

### Technical Fixes (June 2025)
- ~~Template parameter substitution bugs fixed~~
- ~~Workflow title feature implemented~~
- ~~Widget value reading from UI inputs fixed~~
- ~~PyTorch dependencies and imports resolved~~
- ~~Connection wiring between complex node patterns~~
- ~~Model parameter collection for optimizer nodes~~
- ~~Variable naming issues (node_10 vs 10_node format)~~
- ~~Bias parameter processing in LinearLayer templates~~
- ~~Abstract method implementations in optimizer templates~~
- ~~**SYNC Data Type Implementation**: Created SYNC type for node synchronization and training coordination~~
- ~~**Training Loop Synchronization**: Implemented TrainingStep.ready → GetBatch.trigger synchronization to fix tensor gradient computation errors~~
- ~~**Queue-Based SYNC Signals**: Modified templates to send/receive SYNC signals for proper training sequence control~~
- ~~**UI Node Updates**: Added SYNC inputs/outputs to TrainingStep and GetBatch node definitions~~
- ~~**Export System Updates**: Updated node exporters to handle new SYNC-based inputs and outputs~~