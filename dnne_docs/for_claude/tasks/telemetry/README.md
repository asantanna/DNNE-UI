# Telemetry System Refactoring

## Overview

This directory contains task tracking and implementation details for the DNNE telemetry system refactoring project.

## Goals

Streamline the telemetry system by:
- Eliminating redundant systems (MetricsLogger)
- Reducing telemetry data volume by 50%
- Making all intervals configurable
- Focusing on essential, actionable metrics

## Key Changes

1. **Single System**: Remove MetricsLogger, use only TelemetryClient
2. **Simplified Metrics**: Focus on essential metrics, remove redundant statistics
3. **Configurable**: All intervals and levels configurable via `--override`
4. **No File I/O**: Exported code does UDP only, no file writing

## Documents

- **[TASKS.md](TASKS.md)** - Phase-based task tracking
- **[implementation.md](implementation.md)** - Detailed implementation guide
- **[telemetry_policy.md](../../architecture/telemetry_policy.md)** - Architectural policy and guidelines

## Quick Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Remove old MetricsLogger system | 🔴 Not Started |
| Phase 2 | Update UI nodes & exporters | 🔴 Not Started |
| Phase 3 | Update node templates | 🔴 Not Started |
| Phase 4 | Testing & documentation | 🔴 Not Started |

## Key Metrics to Track

### Performance/Timing (BalancerNode)
- `frequency_current` - Current execution rate
- `latency_avg` - Average processing time
- `violation_count` - Number of violations

### Training Progress (EpochTracker)
- `epoch` - Current epoch number
- `loss_mean` - Average loss for epoch
- `accuracy_mean` - Average accuracy for epoch

### Simulation Progress (SimulationTracker)
- `episodes` - Total episode count
- `timesteps` - Total environment steps
- `loss_mean` - Average training loss (primary metric)

## Command-Line Configuration

All telemetry configurable at runtime:

```bash
# Enable telemetry with levels
python runner.py \
    --enable-telemetry balancer_10,sim_tracker_42 \
    --override balancer_10:telemetry_level=essential \
    --override balancer_10:report_interval=200 \
    --override sim_tracker_42:telemetry_interval=50_episodes
```

## Simplified Interval Format

New unified format: `{value}_{unit}` or time suffixes
- `100_steps` - Every 100 timesteps
- `10_episodes` - Every 10 episodes  
- `30s` - Every 30 seconds
- `5m` - Every 5 minutes

---
*Part of DNNE telemetry system refactoring - September 2025*