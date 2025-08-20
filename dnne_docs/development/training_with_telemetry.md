# Training with Telemetry

Quick guide for monitoring ML training progress via telemetry.

## Enable Training Telemetry

Training telemetry reports statistics (mean, min, max, std dev, percentiles) for loss and accuracy.

### Time-Based Windows (Recommended)
```bash
# Report every 30 seconds
--enable-telemetry 67 --override 67:telemetry_time_window=30

# Report every 5 minutes
--enable-telemetry 67 --override 67:telemetry_time_window=300
```

### Batch-Based Windows
```bash
# Report every 100 batches
--enable-telemetry 67 --override 67:telemetry_batch_window=100

# Report every 500 batches
--enable-telemetry 67 --override 67:telemetry_batch_window=500
```

## Example Commands

### MNIST Training (2 epochs, updates every minute)
```bash
python runner.py --epochs 2 --enable-telemetry 67 --override 67:telemetry_time_window=60 --debug
```

### Franka Coop (3 networks, each reporting every 2 minutes)
```bash
python runner.py --enable-telemetry 104,141,178 \
  --override 104:telemetry_time_window=120 \
  --override 141:telemetry_time_window=120 \
  --override 178:telemetry_time_window=120
```

## Telemetry Output

Statistics reported per window:
- `train_loss_mean`, `train_loss_min`, `train_loss_max`, `train_loss_std`
- `train_loss_p25`, `train_loss_p50` (median), `train_loss_p75`
- `train_acc_mean`, `train_acc_min`, `train_acc_max`, `train_acc_std`
- `train_acc_p25`, `train_acc_p50` (median), `train_acc_p75`
- `train_window_size` (number of batches in window)
- `train_total_batches` (cumulative count)
- `train_current_epoch`

Per-epoch summaries:
- `epoch_completed`, `epoch_avg_loss`, `epoch_avg_accuracy`, `epoch_total_batches`

## Requirements

Telemetry requires remote deployment:
1. **DNNE Server** running (Windows): `python main.py`
2. **Agent Client** running (WSL/Linux): `python dnne_agent/dnne_agent_client.py`
3. **Deploy from UI**: Use Export button with runtime args

Files written to: `remote_clients/{hostname}/{workflow}/telemetry/telem_{timestamp}/node_{id}.dat`

## Performance Notes

- **Zero overhead when disabled**: No buffers allocated, no statistics computed
- **Minimal overhead when enabled**: Only statistical aggregation, no per-batch I/O
- **Time windows preferred**: More intuitive than batch counts
- **Window precedence**: If both specified, time-based takes priority

## Finding Node IDs

EpochTracker nodes handle training telemetry. Find their IDs in the exported runner.py:
```python
# Look for lines like:
epoch_tracker_node_67 = EpochTrackerNode_67("67")
```

Or check the workflow JSON for nodes with class_type "EpochTracker".