# DNNE Future Features

This directory contains ideas and plans for future DNNE features. Each feature has its own file for easy tracking and development.

## How to Use This Directory

1. **Browse by category** to find features in your area of interest
2. **Add new ideas** by creating a new `.md` file in the appropriate category
3. **Update priority** as features become more/less important
4. **Convert to issues** when ready to implement

## Feature Categories

### 🚀 Performance
Optimizations to make DNNE faster and more efficient.

- **[Training Optimization](performance/training-optimization.md)** - *High Priority* - Speed up the current 3+ min/epoch training time

### 🧠 ML Features
Machine learning capabilities and nodes.

- **[ConvNet Support](ml-features/convnet-support.md)** - *Medium Priority* - Add Conv2D, MaxPool2D, BatchNorm2D nodes
- **[Advanced Optimizers](ml-features/advanced-optimizers.md)** - *Medium Priority* - Adam, AdamW, RMSprop, and LR schedulers
- **[Data Pipeline Enhancement](ml-features/data-pipeline.md)** - *Medium Priority* - Custom datasets, augmentation, validation splitting
- **[Model Export/Import](ml-features/model-export-import.md)** - *Low Priority* - Save/load models, ONNX export

### 🤖 Robotics
Robotics-specific features and improvements.

- **[PPO Decomposition](robotics/ppo-decomposition.md)** - *Low Priority* - Split PPO into 4 nodes for visibility
- **[Rate Limiting](robotics/rate-limiting.md)** - *Low Priority* - Add Hz control to GetBatch for robotics

### 🔧 System
Core system improvements and infrastructure.

- **[Export Improvements](system/export-improvements.md)** - *Medium Priority* - Better error handling and monitoring
- **[UI Feedback](system/ui-feedback.md)** - *Low Priority* - Progress bars and better UX

### 📊 Visualization
Data visualization and monitoring features.

- **[Training Dashboards](visualization/training-dashboards.md)** - *Low Priority* - Real-time training visualization

## Priority Guide

- **High**: Critical for usability or blocking other work
- **Medium**: Important features that would significantly improve DNNE
- **Low**: Nice-to-have features or experimental ideas

## Adding New Features

Create a new file using this template:

```markdown
# Feature Name

## Priority
High/Medium/Low

## Description
Brief description of the feature

## Motivation
Why this feature is needed

## Implementation Notes
Technical considerations

## Dependencies
What needs to be done first

## Estimated Effort
Small/Medium/Large

## Success Metrics
How we'll know it's working
```

## Feature Lifecycle

1. **Idea**: Initial concept documented here
2. **Design**: Detailed technical design added
3. **GitHub Issue**: Convert to issue when ready
4. **Implementation**: Active development
5. **Testing**: Validation and refinement
6. **Release**: Feature complete and documented

## Notes for Developers

When implementing features from this directory:
- Review the full feature file first
- Check dependencies and prerequisites
- Consider the priority and effort estimates
- Update or remove the file when complete
- Add proper documentation to main docs

## Vision

These features represent the evolution of DNNE toward a complete visual programming environment for ML and robotics, with production-ready code generation and seamless integration with modern frameworks.