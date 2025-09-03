# Telemetry Refactoring Implementation Guide

## Overview

This document provides specific implementation details for the telemetry refactoring outlined in `telemetry_policy.md`.

## Simplified Telemetry Interval Format

The new telemetry interval format unifies all reporting modes into a single string format:

### Format: `{value}_{unit}`

**Examples**:
- `100_steps` - Report every 100 timesteps
- `10_episodes` - Report every 10 episodes
- `30s` - Report every 30 seconds
- `5m` - Report every 5 minutes
- `1h` - Report every hour

### Parsing Logic:

```python
def parse_interval(interval_str: str) -> tuple[float, str]:
    """
    Parse interval string into (value, unit) tuple.
    
    Examples:
        "100_steps" -> (100, "steps")
        "10_episodes" -> (10, "episodes")  
        "30s" -> (30, "seconds")
        "5m" -> (300, "seconds")
    """
    # Handle underscore format (steps/episodes)
    if '_' in interval_str:
        parts = interval_str.split('_')
        return (float(parts[0]), parts[1])
    
    # Handle time suffixes
    if interval_str.endswith('s'):
        return (float(interval_str[:-1]), "seconds")
    elif interval_str.endswith('m'):
        return (float(interval_str[:-1]) * 60, "seconds")
    elif interval_str.endswith('h'):
        return (float(interval_str[:-1]) * 3600, "seconds")
    
    # Default to steps if no unit specified
    return (float(interval_str), "steps")
```

### Benefits:
- Single widget instead of separate mode/interval widgets
- Self-documenting format
- Easy command-line override: `--override sim_42:telemetry_interval=50_episodes`
- No need for separate telemetry_mode widget

## Code Changes by Component

### 1. Enhanced TelemetryClient (`framework/telemetry.py`)

**IMPORTANT**: TelemetryClient does NO file logging. All telemetry is sent via UDP to the agent client, which forwards to DNNE server for file storage. The exported runner.py code never writes telemetry files.

```python
# New features to add:

class TelemetryClient:
    def __init__(self, enabled=False, host="localhost", port=9999,
                 violation_rate_limit=10):
        # ... existing code ...
        # NO file logging in TelemetryClient - purely UDP fire-and-forget
    
    def report_metric(self, node_id: str, metric_name: str, value: float, 
                     aggregate: bool = True):  # NEW: Aggregation hint
        """
        Unified metric reporting with aggregation hint.
        
        Args:
            aggregate: If True, this metric should be aggregated over windows
        """
        # Implementation combines report_custom with aggregation metadata
    
    def start_window(self, node_id: str, window_type: str):
        """Mark the start of a telemetry window for aggregation."""
        self.window_start_time = time.time()
        self.window_metrics = defaultdict(list)
    
    def end_window(self, node_id: str, stats: Dict[str, float]):
        """
        End window and report aggregated statistics.
        More efficient than multiple individual reports.
        """
        packet = {
            "type": "window_stats",
            "node_id": node_id,
            "duration": time.time() - self.window_start_time,
            "stats": stats
        }
        self._send_json(packet)
```

### 2. Simplified BalancerNode

**Current Issues**:
- Reports to both TelemetryClient and MetricsLogger
- Sends metrics every execution AND every 100 executions
- Reports 8+ different metrics

**Proposed Changes**:

```python
# In balancer_node_queue.tpl

class BalancerNode_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        # ... existing setup ...
        
        # Simplified configuration
        self.telemetry_level = g.get_node_config(
            self.node_id, 'telemetry_level', 'essential'
        )  # 'off', 'essential', 'extended', 'debug'
        
        # Remove MetricsLogger completely
        # self._register_with_global()  # DELETE THIS
    
    async def compute(self, input):
        # ... existing measurement code ...
        
        # Report only at intervals (not every execution)
        if self.execution_count % 100 == 0:
            self._report_telemetry_window()
        
        # Remove per-execution telemetry
        # DELETE: telemetry.report_throughput(...)
        # DELETE: telemetry.report_latency(...)
        
        return {"output": input}
    
    def _report_telemetry_window(self):
        """Report aggregated metrics for the window."""
        if not self.telemetry_enabled:
            return
        
        stats = {}
        
        # Essential metrics (always included)
        if self.telemetry_level in ['essential', 'extended', 'debug']:
            stats['frequency_current'] = self.current_frequency
            stats['latency_avg'] = self.average_latency
            stats['violation_count'] = len(self.violations)
        
        # Extended metrics (optional)
        if self.telemetry_level in ['extended', 'debug']:
            stats['frequency_avg'] = self.average_frequency
            stats['frequency_std'] = self._calculate_std(self.frequency_window)
            for name, queue in self.input_queues.items():
                stats[f'queue_input_{name}'] = queue.qsize()
        
        # Debug metrics (verbose)
        if self.telemetry_level == 'debug':
            stats['frequency_min'] = min(self.frequency_window) if self.frequency_window else 0
            stats['frequency_max'] = max(self.frequency_window) if self.frequency_window else 0
            stats['execution_count'] = self.execution_count
        
        # Single efficient call
        telemetry.end_window(self.node_id, stats)
        
        # Clear violation buffer
        self.violations = []
```

### 3. Streamlined EpochTracker

**Current Issues**:
- Reports 23 metrics per window (7 loss + 7 accuracy + metadata)
- Window-based reporting during epoch (not needed)
- Excessive statistical detail

**Proposed Changes**:

```python
# In epoch_tracker_queue.tpl

class EpochTrackerNode_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        # ... existing setup ...
        
        # Remove window-based telemetry config
        # DELETE: self.telemetry_batch_window
        # DELETE: self.telemetry_time_window
        # DELETE: telemetry buffers
        
        self.telemetry_level = g.get_node_config(
            self.node_id, 'telemetry_level', 'essential'
        )
    
    async def compute(self, epoch_stats, loss, accuracy):
        # ... existing tracking ...
        
        # Remove window-based reporting
        # DELETE: All window telemetry code
        
        if epoch_stats.get("completed", False):
            # Report only at epoch completion
            self._report_epoch_telemetry(epoch_num)
        
        return {"control_metrics": control_metrics}
    
    def _report_epoch_telemetry(self, epoch_num):
        """Report epoch completion metrics."""
        if not self.telemetry_enabled:
            return
        
        stats = {}
        
        # Essential metrics
        if self.telemetry_level in ['essential', 'extended', 'debug']:
            stats['epoch'] = float(epoch_num + 1)
            stats['loss_mean'] = sum(self.epoch_losses) / len(self.epoch_losses)
            stats['accuracy_mean'] = sum(self.epoch_accuracies) / len(self.epoch_accuracies)
            stats['batches'] = float(len(self.epoch_losses))
        
        # Extended metrics
        if self.telemetry_level in ['extended', 'debug']:
            # Calculate trend (improvement from previous epoch)
            if hasattr(self, 'prev_loss_mean'):
                stats['loss_delta'] = stats['loss_mean'] - self.prev_loss_mean
                stats['accuracy_delta'] = stats['accuracy_mean'] - self.prev_accuracy_mean
            self.prev_loss_mean = stats['loss_mean']
            self.prev_accuracy_mean = stats['accuracy_mean']
            
            # Standard deviation for stability monitoring
            stats['loss_std'] = self._calculate_std(self.epoch_losses)
            stats['accuracy_std'] = self._calculate_std(self.epoch_accuracies)
        
        # Debug metrics
        if self.telemetry_level == 'debug':
            stats['loss_min'] = min(self.epoch_losses)
            stats['loss_max'] = max(self.epoch_losses)
            stats['accuracy_min'] = min(self.epoch_accuracies)
            stats['accuracy_max'] = max(self.epoch_accuracies)
        
        # Single call for all metrics
        telemetry.report_metric(self.node_id, 'epoch_complete', 1.0, aggregate=False)
        for key, value in stats.items():
            telemetry.report_metric(self.node_id, f'epoch_{key}', value, aggregate=False)
```

### 4. Focused SimulationTracker (Simulator-specific, not RL-specific)

**Current Issues**:
- Reports 20+ metrics per window
- Complex multi-mode configuration
- Redundant statistics (percentiles, etc.)
- Incorrectly focused on RL rewards instead of general loss tracking

**Proposed Changes**:

```python
# In simulation_tracker_queue.tpl

class SimulationTracker_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        # ... existing setup ...
        
        # Simplified interval configuration
        self.report_interval = self._parse_report_interval()
        self.telemetry_level = g.get_node_config(
            self.node_id, 'telemetry_level', 'essential'
        )
    
    def _parse_report_interval(self):
        """Parse interval string like '10_episodes', '30s', '10000_steps'."""
        interval_str = g.get_node_config(
            self.node_id, 'telemetry_interval', '10_episodes'
        )
        # Parse and return (value, unit) tuple
    
    def _should_report(self):
        """Simplified reporting check."""
        value, unit = self.report_interval
        
        if unit == 'episodes':
            return self.episode_count % value == 0
        elif unit == 'seconds':
            return time.time() - self.last_report_time >= value
        elif unit == 'steps':
            return self.timestep_count - self.last_report_step >= value
    
    def _report_telemetry(self):
        """Report simulation metrics."""
        if not self.telemetry_enabled:
            return
        
        stats = {}
        
        # Essential metrics (loss is primary for physics/bio-plausible learning)
        if self.telemetry_level in ['essential', 'extended', 'debug']:
            stats['episodes'] = float(self.episode_count)
            stats['timesteps'] = float(self.timestep_count)
            
            # Loss is the primary metric for non-RL training
            if self.losses:
                recent_losses = self.losses[-1000:]  # Last 1000 samples
                stats['loss_mean'] = sum(recent_losses) / len(recent_losses)
        
        # Extended metrics
        if self.telemetry_level in ['extended', 'debug']:
            # Loss statistics for stability monitoring
            if self.losses and len(self.losses) > 1:
                stats['loss_std'] = self._calculate_std(recent_losses)
            
            recent_lengths = self.episode_lengths[-self.window_size:]
            if recent_lengths:
                stats['episode_length_mean'] = sum(recent_lengths) / len(recent_lengths)
            
            # Success rate if applicable (task-specific)
            recent_successes = self.episode_successes[-self.window_size:]
            if recent_successes:
                stats['success_rate'] = sum(recent_successes) / len(recent_successes)
        
        # Debug metrics
        if self.telemetry_level == 'debug':
            if self.losses:
                stats['loss_min'] = min(recent_losses)
                stats['loss_max'] = max(recent_losses)
            stats['episodes_since_improvement'] = (
                self.episode_count - self.last_improvement_episode
            )
        
        # Report all at once
        for key, value in stats.items():
            telemetry.report_metric(self.node_id, f'sim_{key}', value)
        
        # Update markers
        self.last_report_time = time.time()
        self.last_report_step = self.timestep_count
```

## Implementation Strategy (No Backwards Compatibility)

### Step 1: Delete MetricsLogger

```bash
# Remove the unused MetricsLogger system entirely
rm export_system/templates/framework/metrics_logger.py
```

Remove all references from:
- `balancer_node_queue.tpl` - Delete `_register_with_global()` and all MetricsLogger calls
- `framework/globals.py` - Remove any MetricsLogger imports or references

### Step 2: Update UI Nodes (*_visnode.py)

#### BalancerNode (`custom_nodes/balancer_visnode.py`)
- Remove: MetricsLogger-related widgets
- Add: `report_interval` widget (INT, default: 100, min: 1)
- Keep: `telemetry_level` widget if exists, or add (COMBO: ["off", "essential", "extended", "debug"])

#### EpochTracker (`custom_nodes/epoch_tracker_visnode.py`)
- Remove: `telemetry_batch_window`, `telemetry_time_window` widgets
- Remove: `telemetry_stats` widget (replaced by telemetry_level)
- Add/Keep: `telemetry_level` widget

#### SimulationTracker (`custom_nodes/simulation_tracker_visnode.py`)
- Update: `telemetry_mode` widget - remove if exists (mode now inferred from interval format)
- Update: `telemetry_interval` widget to accept simplified format (STRING)
- Remove: `telemetry_stats` widget (replaced by telemetry_level)
- Add: `telemetry_level` widget

### Step 3: Update Node Exporters

#### ml_nodes.py
- Remove window-based telemetry parameter extraction for EpochTracker
- Simplify to just pass telemetry_level

#### robotics_nodes.py  
- Update SimulationTracker to use simplified interval format
- Remove reward-related parameter extraction
- Focus on loss tracking parameters

#### utility_nodes.py
- Update BalancerNode export to include report_interval
- Remove MetricsLogger configuration extraction

### Step 4: Update Node Templates

No legacy support - directly modify templates:
1. `balancer_node_queue.tpl` - Remove dual reporting, add configurable intervals
2. `epoch_tracker_queue.tpl` - Remove window reporting, simplify to epoch-only
3. `simulation_tracker_queue.tpl` - Focus on loss, remove reward tracking

### Step 5: Make All Values Configurable

```python
# In each node template, replace hardcoded values:

# BEFORE:
if self.execution_count % 100 == 0:  # Hardcoded
    self._report_telemetry()

# AFTER: 
self.report_interval = g.get_node_config(
    self.node_id, 'report_interval', 100  # Default 100, configurable
)
if self.execution_count % self.report_interval == 0:
    self._report_telemetry()
```

## Testing Plan

### Unit Tests

1. **TelemetryClient enhancements**
   - Test window aggregation
   - Test local logging option
   - Test backward compatibility

2. **Node telemetry**
   - Test each telemetry level (off/essential/extended/debug)
   - Verify metric counts and names
   - Test interval calculations

### Integration Tests

1. **End-to-end workflow**
   - Export workflow with mixed telemetry configs
   - Run and verify telemetry output
   - Check file sizes and performance

2. **Migration test**
   - Load workflow with old telemetry config
   - Verify automatic migration
   - Check deprecation warnings

### Performance Tests

1. **Overhead measurement**
   - Run workflow with telemetry off
   - Run with essential telemetry
   - Run with debug telemetry
   - Compare execution times (<0.1% overhead target)

2. **Data volume**
   - Measure telemetry data size per hour
   - Compare old vs new system
   - Verify 50% reduction target

## Telemetry Data Flow

**Clear separation of responsibilities**:

1. **runner.py (exported code)**:
   - Sends UDP packets via TelemetryClient
   - NO file I/O
   - Fire-and-forget pattern
   - Zero blocking

2. **Agent Client (dnne_agent_client.py)**:
   - Receives UDP packets on port 9999
   - Aggregates violations
   - Forwards via WebSocket to DNNE server
   - Could add local file logging here in future (not now)

3. **DNNE Server (server.py)**:
   - Receives telemetry via WebSocket
   - Writes to timestamped directories
   - Manages file rotation and cleanup
   - Only place where files are written

## Rollout Timeline

### Single-Shot Implementation
Complete all changes in one pass:

1. **Delete MetricsLogger** (`framework/metrics_logger.py`)
2. **Enhance TelemetryClient** with window aggregation features
3. **Update all UI nodes** (*_visnode.py files)
4. **Update all node exporters** (ml_nodes.py, robotics_nodes.py, utility_nodes.py)
5. **Update all three node templates**:
   - BalancerNode: Remove MetricsLogger, add configurable intervals
   - EpochTracker: Remove window reporting, epoch-only metrics
   - SimulationTracker: Focus on loss, remove rewards
6. **Test end-to-end** with example workflows
7. **Update documentation**

## Configuration Examples

**CLARIFICATION**: The `telemetry_config` dictionaries shown below are for documentation purposes only - they illustrate how telemetry is configured via command-line `--override` flags. The actual configuration is done at runtime through the Global class using `--override` arguments.

### Example 1: Production Simulator Training

```bash
# Minimal telemetry for production (command-line configuration)
python runner.py \
    --enable-telemetry balancer_10,simulation_tracker_42 \
    --override balancer_10:telemetry_level=essential \
    --override simulation_tracker_42:telemetry_level=essential \
    --override simulation_tracker_42:telemetry_interval=100_episodes
```

This is conceptually equivalent to:
```python
# FOR ILLUSTRATION ONLY - not actual code
telemetry_config = {
    "balancer_10": {
        "telemetry_enabled": True,
        "telemetry_level": "essential"
    },
    "simulation_tracker_42": {
        "telemetry_enabled": True,
        "telemetry_level": "essential",
        "telemetry_interval": "100_episodes"
    }
}
```

### Example 2: Debug ML Training

```bash
# Verbose telemetry for debugging
python runner.py \
    --enable-telemetry balancer_10,epoch_tracker_67 \
    --override balancer_10:telemetry_level=debug \
    --override epoch_tracker_67:telemetry_level=extended
```

### Example 3: Performance Profiling

```bash
# Focus on timing metrics with consistent intervals
python runner.py \
    --enable-telemetry balancer_10,balancer_11 \
    --override balancer_10:telemetry_level=extended \
    --override balancer_10:report_interval=200 \
    --override balancer_11:telemetry_level=extended \
    --override balancer_11:report_interval=200
```

## Success Criteria

1. **Code Reduction**: 40% fewer lines in telemetry-related code
2. **Performance**: <0.1% overhead with essential telemetry
3. **Data Volume**: 50% reduction in telemetry file sizes
4. **Clarity**: Single API, clear documentation
5. **Migration**: Zero breaking changes for existing workflows