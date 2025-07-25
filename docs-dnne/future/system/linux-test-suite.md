# Linux-Side Test Suite

**Priority**: Low  
**Status**: Future consideration

## Description

Create a test suite that runs on the Linux/WSL2 side to test exported code functionality that cannot be tested from Windows.

## Motivation

Some DNNE features only make sense to test in their actual runtime environment:
- Adaptive yielding with event loop manipulation
- Isaac Gym integration
- CUDA/GPU operations
- Linux-specific path handling
- Performance benchmarks

## Implementation Notes

Possible approach:
1. Windows test suite command: `dnne-test linux-export`
2. Exports test cases to `dnne-test-suite-linux/`
3. Instructs user to run from Linux: `./dnne-test-linux`
4. Linux-side runner executes tests and reports results

## Dependencies

- Would need to handle cross-platform file access
- Requires Linux/WSL2 environment with proper setup
- May need special handling for GPU tests

## Estimated Effort

Medium - Would need to design cross-platform test coordination system

## Notes

Currently not needed as:
- Manual testing is sufficient for most features
- Exported workflows serve as implicit tests
- Added complexity may not be worth the benefit

Revisit if we accumulate many Linux-specific features that need automated testing.