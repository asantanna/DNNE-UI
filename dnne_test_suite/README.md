# DNNE Test Suite

Centralized test organization for the DNNE (Drag and Drop Neural Network Environment) project.

**Note**: As of July 2025, the test suite has been consolidated into this single location. The previous `tests-dnne/` directory has been removed as it was a complete duplicate of `dnne_test_suite/core/`. All test infrastructure is now contained within `dnne_test_suite/`.

## Structure

```
dnne_test_suite/
├── README.md                          # This file
├── core/                              # Core pytest-based tests
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   ├── fixtures/                      # Test utilities and data
│   ├── run_tests.sh                   # Main test runner
│   └── check_dependencies.py         # Dependency checker
├── specialized/                       # Specialized test scripts
│   ├── cartpole_ppo_comprehensive.py # RL comprehensive tests
│   ├── checkpoint_system.py          # Checkpoint system tests
│   ├── mnist_inference_complete.py   # Inference mode tests
│   └── export_system/                # Export system tests
├── profiling/                         # Performance tests
├── archived/                          # Historical tests
├── utilities/                         # Test utilities
│   ├── programmatic_export.py        # Export utility
│   └── validate_connections.py       # Connection validator
├── scripts/                           # Test command infrastructure
│   ├── commands.sh                    # Main test command implementations
│   ├── dnne_test                      # Wrapper: main test suite
│   ├── dnne_test-unit                 # Wrapper: unit tests only
│   ├── dnne_test-ml                   # Wrapper: ML tests only
│   └── ... (all test commands)
└── outputs/                           # Test outputs and logs
    ├── coverage/                      # Coverage reports (htmlcov/)
    └── logs/                          # Test execution logs
```

## Running Tests

The main entry point is the `dnne_test` command in the project root, which provides access to all test categories:

```bash
# Show all available test commands
dnne_test help

# Run quick unit tests
dnne_test quick

# Run full test suite
dnne_test full

# Run specific test categories
dnne_test ml          # ML node tests only
dnne_test robotics    # Robotics/Isaac Gym tests
dnne_test export      # Export system tests
dnne_test rl          # RL comprehensive tests
```

## Test Categories

### Core Tests (`core/`)
- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: End-to-end workflow tests

### Specialized Tests (`specialized/`)
- **Cartpole PPO Comprehensive**: Full reinforcement learning pipeline tests
- **Checkpoint System**: Model checkpoint save/load functionality
- **MNIST Inference Complete**: Complete training + inference validation
- **Export System**: Code generation and export functionality

### Profiling Tests (`profiling/`)
- **Performance Comparison**: DNNE vs PyTorch direct benchmarks
- **Async Overhead**: Queue-based execution performance analysis
- **PPO Agent Context**: Comparison of different implementation approaches

### Archived Tests (`archived/`)
Historical test implementations kept for reference. These are not run by the main test suite but may contain useful patterns or debugging approaches.

## Dependencies

Test dependencies are automatically installed when running tests via `dnne_test`. Manual installation:

```bash
pip install pytest pytest-timeout pytest-asyncio pytest-cov
```

## Configuration

Environment variables:
- `DNNE_TEST_DATA_PATH`: Custom data directory (default: ./data)
- `DNNE_TEST_DOWNLOAD`: Enable/disable downloads (default: true)

## Migration Notes

This directory consolidates tests from multiple locations:
- `tests-dnne/` → `core/`
- `claude_scripts/test_*.py` → `specialized/`
- `export_system/test_*.py` → `specialized/export_system/`
- `claude_scripts/profiling/` → `profiling/`
- `claude_scripts/archive/` → `archived/`

The original `tests-unit/` directory containing ComfyUI legacy tests remains untouched.