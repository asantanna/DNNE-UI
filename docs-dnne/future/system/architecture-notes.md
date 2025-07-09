# Architecture Notes and Technical Debt

## Priority
Ongoing

## Description
Important architectural considerations and technical debt areas for future DNNE development.

## Current System Strengths
1. **Queue-Based Architecture**: Solid foundation for real-time robotics applications
2. **Template System**: Flexible and extensible code generation
3. **Node Pattern**: Proven pattern for implementing new node types
4. **Connection Resolution**: Robust handling of complex node interconnections

## Technical Debt Areas
1. **Performance**: Training loop timing optimization needed
2. **Error Handling**: More graceful handling of export edge cases
3. **UI Feedback**: Better progress indication during long exports
4. **Documentation**: Need more comprehensive template development guide

## Development Guidelines
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

## Success Metrics
- Maintain clean architecture as system grows
- Keep export times reasonable
- Ensure generated code is readable and debuggable
- Support diverse use cases without compromising core design