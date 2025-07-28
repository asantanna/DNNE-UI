# DNNE Experiments

This directory contains research, experiments, and solutions for various DNNE challenges and innovations.

## Available Experiments

### 1. [Yield Tests](yield_tests/) - Thread-Safe Yielding for Concurrent Execution
**Problem**: How to allow synchronous code (like RL training) to yield control in DNNE's async architecture  
**Solution**: Thread-safe yielding mechanism using executors and queue-based communication  
**Status**: ✅ Implemented and tested  
**Key Files**: 
- `yield_tests/README.md` - Full documentation
- `yield_tests/solution/` - Working implementation
- `yield_tests/USAGE_EXAMPLE.md` - How to use in your nodes

## Adding New Experiments

When conducting research or experiments:

1. Create a new directory: `experiments/your_experiment_name/`
2. Include:
   - `README.md` - Problem statement and overview
   - Test programs that reproduce the issue
   - Solution implementations
   - Documentation of findings
3. Update this file with a summary

## Guidelines

- **Reproducible**: Include minimal test cases
- **Well-documented**: Explain the problem and solution clearly
- **Practical**: Provide usage examples
- **Organized**: Follow the structure used in existing experiments

---

*These experiments represent the ongoing evolution of DNNE's architecture and capabilities.*