# DNNE Future Tasks and Features

## Overview
This file contains larger features, system improvements, and architectural considerations for the DNNE project that are not currently being actively worked on. Use this file to select high-level features for future implementation.

## Priority Features

### High Priority
1. **Performance Optimization**: Current MNIST training takes ~3+ minutes per epoch (1875 batches at 10Hz). Consider:
   - Increasing batch processing rate for training scenarios
   - Adding configurable update rates per node type
   - Implementing batch size optimization recommendations

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

3. **GetBatch Rate Limiting Widget**:
   - Add optional rate limiting widget to GetBatch node
   - Default setting: "off" (no rate limiting)
   - Allow users to set specific Hz for robotics applications
   - Widget should be clearly labeled as "for robotics timing only"

## System Maintenance

### Export System Health
- **Slot Mapping**: Monitor for any regression in slot corruption fix
- **Template Updates**: Ensure new nodes follow queue-based template patterns
- **Error Handling**: Improve error messages in export failures

### Code Quality
- **Documentation**: Add inline documentation to complex template generation logic
- **Testing**: Expand automated tests for export system
- **Refactoring**: Consider consolidating similar node patterns

## Architecture Notes for Future Development

### Current System Strengths
1. **Queue-Based Architecture**: Solid foundation for real-time robotics applications
2. **Template System**: Flexible and extensible code generation
3. **Node Pattern**: Proven pattern for implementing new node types
4. **Connection Resolution**: Robust handling of complex node interconnections

### Technical Debt Areas
1. **Performance**: Training loop timing optimization needed
2. **Error Handling**: More graceful handling of export edge cases
3. **UI Feedback**: Better progress indication during long exports
4. **Documentation**: Need more comprehensive template development guide

### Next Developer Guidance
When continuing this project:
1. Follow existing queue-based template patterns in `templates/nodes/`
2. Test all new nodes with actual export and execution
3. Monitor ComfyUI slot mapping for any regressions
4. Maintain device compatibility (CPU/GPU) in all tensor operations
5. Use the MNIST Test workflow as integration test for major changes

## Long-term Vision
- Complete ML/robotics visual programming environment
- Production-ready code generation for NVIDIA Isaac Gym
- Seamless integration with major ML frameworks
- Real-time training and inference capabilities
- Visual debugging and profiling tools