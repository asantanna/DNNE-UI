# DNNE Telemetry Policy & Organization

## Executive Summary

The current telemetry system has evolved organically and now contains significant redundancy and confusion between multiple systems (TelemetryClient, MetricsLogger, and node-specific reporting). This document proposes a clear policy and implementation plan to streamline telemetry collection.

## Current State Analysis

### Telemetry-Reporting Nodes

Currently, only three nodes report telemetry:

1. **BalancerNode** (`balancer_node_queue.tpl`)
   - Reports: throughput, latency, frequency, queue depths, violations
   - Dual reporting: Both TelemetryClient AND MetricsLogger
   - Purpose: Performance monitoring and timing violation detection

2. **EpochTracker** (`epoch_tracker_queue.tpl`)
   - Reports: Training loss/accuracy statistics, epoch completion metrics
   - Window-based aggregation (time or batch-based)
   - Purpose: ML training progress monitoring

3. **SimulationTracker** (`simulation_tracker_queue.tpl`)
   - Reports: Episode rewards, lengths, success rates, loss statistics
   - Configurable reporting modes (time/steps/episodes)
   - Purpose: RL/robotics training progress monitoring

### Infrastructure Components

1. **TelemetryClient** (`framework/telemetry.py`)
   - Fire-and-forget UDP to agent
   - Rate-limited violations
   - Remote monitoring focus

2. **MetricsLogger** (`framework/metrics_logger.py`)
   - Local file logging
   - Singleton pattern
   - Violation tracking with severity

3. **Agent/Server Pipeline**
   - UDP → WebSocket → File storage
   - Violation aggregation
   - Remote workflow monitoring

### Identified Problems

1. **Dual Reporting**: BalancerNode reports to both TelemetryClient and MetricsLogger
2. **Inconsistent Metrics**: Different nodes report different granularities
3. **Excessive Detail**: EpochTracker reports 15+ metrics per window (mean, min, max, std, p25, p50, p75 for both loss and accuracy)
4. **Unclear Ownership**: No clear decision on when to use which system
5. **Redundant Infrastructure**: Two parallel systems doing similar things

## Proposed Telemetry Policy

### Core Principles

1. **Minimal Overhead**: Telemetry should never impact node performance (<0.1% CPU)
2. **Essential Information Only**: Report only metrics needed for decision-making
3. **Single Source of Truth**: Each metric should have one authoritative source
4. **Clear Purpose**: Every metric must have a defined consumer and use case

### What to Track

#### 1. Performance Monitoring (BalancerNode)
**Purpose**: Detect timing violations and performance bottlenecks

**Essential Metrics**:
- `frequency_current`: Current execution rate (Hz)
- `latency_avg`: Average processing time (ms) 
- `violation_count`: Number of violations in window

**Optional Metrics** (enabled via config):
- `queue_depth_*`: Input/output queue sizes
- `frequency_avg`: Moving average frequency

**Remove**:
- Duplicate reporting to MetricsLogger
- Per-violation telemetry (use summaries only)
- Redundant throughput metrics

#### 2. Training Progress (EpochTracker)
**Purpose**: Monitor ML model training convergence

**Essential Metrics** (per epoch):
- `epoch_number`: Current epoch
- `loss_mean`: Average loss for epoch
- `accuracy_mean`: Average accuracy for epoch
- `batches_completed`: Number of batches

**Optional Metrics** (via config):
- `loss_std`: Standard deviation (for stability monitoring)
- `accuracy_trend`: Improvement rate

**Remove**:
- Percentile statistics (p25, p50, p75)
- Window-based reporting during epoch
- Min/max values (use mean+std instead)

#### 3. Simulation Progress (SimulationTracker)  
**Purpose**: Monitor simulator-based training (physics learning, biologically plausible networks, etc.)

**Essential Metrics** (per reporting interval):
- `episodes_completed`: Total episode count
- `loss_mean`: Average training loss (primary metric for non-RL training)
- `timesteps_total`: Total environment steps

**Optional Metrics** (via config):
- `episode_length_mean`: Average episode duration
- `success_rate`: Task-specific success metric (if applicable)
- `loss_std`: Loss standard deviation for stability monitoring

**Remove**:
- Reward tracking (PPO_Agent handles its own logging)
- Per-step reporting
- Detailed loss statistics (percentiles)
- Redundant sample counts

### When to Report

All reporting intervals are configurable via `--override` flags:

1. **BalancerNode**: Every N executions (default: 100, configurable via `--override {node_id}:report_interval=200`)
2. **EpochTracker**: Only at epoch completion (no mid-epoch reporting)
3. **SimulationTracker**: Based on configured mode (configurable via `--override {node_id}:telemetry_interval=VALUE`):
   - Episodes: Every N episodes (default: 10)
   - Time: Every N seconds (default: 30s)
   - Steps: Every N timesteps (default: 10000)

### Which System to Use

**Use TelemetryClient for all telemetry**:
- All remote monitoring (agent/server pipeline)
- Production deployments
- Cross-node coordination metrics
- Local logging (if needed) via TelemetryClient options

**Delete MetricsLogger immediately**:
- Not currently used
- Adds unnecessary complexity
- All functionality available in TelemetryClient

## Implementation Plan

### Single-Shot Implementation (No gradual rollout)

1. **Delete MetricsLogger**
   - Remove `framework/metrics_logger.py`
   - Remove all MetricsLogger imports and calls
   - Update Global class to remove MetricsLogger references

2. **Enhance TelemetryClient**
   ```python
   # Enhanced API
   telemetry.report_metric(node_id, metric_name, value, aggregate=True)
   telemetry.report_violation(node_id, type, expected, actual)
   telemetry.start_window(node_id, window_type)
   telemetry.end_window(node_id, stats_dict)
   ```

3. **Update All Node Templates**
   - **BalancerNode**: Remove MetricsLogger, add configurable intervals
   - **EpochTracker**: Remove window reporting, simplify metrics
   - **SimulationTracker**: Focus on loss tracking, remove reward metrics

4. **Configuration via --override**
   All telemetry settings configurable at runtime:
   ```bash
   # Examples:
   --override balancer_10:report_interval=200
   --override balancer_10:telemetry_level=extended
   --override sim_tracker_42:telemetry_interval=50_episodes
   ```

## No Backward Compatibility Needed

- Break existing telemetry if necessary
- Clean implementation without legacy support
- Direct migration to new system

## Success Metrics

1. **Performance**: <0.1% CPU overhead with telemetry enabled
2. **Data Volume**: 50% reduction in telemetry data size
3. **Clarity**: Single API, single system, clear guidelines
4. **Adoption**: All nodes using unified telemetry within 6 months

## Summary

The proposed policy focuses on:
- **Fewer, more meaningful metrics** instead of exhaustive statistics
- **Single unified system** instead of dual TelemetryClient/MetricsLogger
- **Clear per-node responsibilities** with defined essential vs optional metrics
- **Configurable granularity** to support both production and debug scenarios

This will result in a cleaner, more maintainable telemetry system that provides actionable insights without overwhelming users or systems with unnecessary data.