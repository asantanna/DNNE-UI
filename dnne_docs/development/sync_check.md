# Network/Optimizer Sync Checking

## Overview
Automatic detection of synchronization violations between neural networks and optimizers during training. Prevents silent corruption of training by ensuring lockstep execution.

## Problem Solved
In dataflow architectures, networks can run ahead of optimizers, processing multiple forward passes before gradients are applied. This creates "runaway execution" bugs that silently corrupt training - the opposite of deadlocks but equally destructive.

Example: Shadow_Train workflow generated 1.2M+ events when DataStreamer ran unchecked, causing the network to process data without synchronized gradient updates.

## How It Works

### Execution Counting
- Network and Optimizer both maintain `execution_count` starting at 0
- Network increments when receiving input data
- Optimizer increments after completing gradient step
- Before processing, Network checks: `network.execution_count == optimizer.get_execution_count()`

### Sync Violation Detection
When counts mismatch, training halts with clear error:
```
🔥 SYNC VIOLATION DETECTED! 🔥
Network execution count: 1
Optimizer execution count: 0
Network is running ahead of optimizer - this will corrupt training!
```

### Connection Discovery
- Network finds its optimizer by following the `model` output connection
- FAIL-FAST if no optimizer connected (except inference mode)
- Supports multiple network/optimizer pairs in same workflow

## Configuration

### Default Behavior
- **Enabled by default** - negligible overhead (integer comparison)
- Shows status on startup:
  - `Network(6) sync checking enabled with Optimizer(15)`
  - `Network(6) sync checking disabled by user`

### Command Line Control
- `--disable-sync-check` - Disable checking (not recommended)
- Useful for experimental workflows or debugging

## Implementation Details

### Files Modified
- `network_exporter.py` - Finds connected optimizer via model output
- `network_queue.tpl` - Implements sync checking before forward pass
- `sgd_optimizer_queue.tpl` - Tracks execution count after gradient step
- `globals.py` - Stores disable_sync_check flag with FAIL-FAST validation
- `arg_parser.tpl` - Adds command-line argument

### FAIL-FAST Principles
- Unknown parameters to `Global.initialize()` trigger immediate error
- Missing optimizer connection fails export (not runtime)
- Clear error messages with actionable fixes

## Benefits
- **Early Detection** - Catches sync issues immediately, not after hours of corrupted training
- **Zero Cost When Correct** - No performance impact for properly synchronized workflows
- **Clear Diagnostics** - Shows exact execution counts where desync occurred
- **Backward Compatible** - Can be disabled for legacy workflows

## Testing
Verified with Shadow_Train workflow:
- Detected sync violation when workflow was broken
- Ran correctly when workflow was fixed
- Properly disabled with `--disable-sync-check` flag