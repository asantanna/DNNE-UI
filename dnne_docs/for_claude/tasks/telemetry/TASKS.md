# Telemetry System Tasks

## Current Status
✅ **COMPLETE** - Telemetry system fully refactored and operational

### Important Documents/Files
- Policy document: `dnne_docs/architecture/telemetry_policy.md`
- Implementation guide: `dnne_docs/for_claude/tasks/telemetry/implementation.md`
- Current telemetry framework: `export_system/templates/framework/telemetry.py`
- Node templates to update:
  - `export_system/templates/nodes/balancer_node_queue.tpl`
  - `export_system/templates/nodes/epoch_tracker_queue.tpl`
  - `export_system/templates/nodes/simulation_tracker_queue.tpl`

## Quick Reference

### Files Impacted
- Templates: `balancer_node_queue.tpl`, `epoch_tracker_queue.tpl`, `simulation_tracker_queue.tpl`
- Framework: `telemetry.py` (enhanced), ~~`metrics_logger.py`~~ (deleted)
- UI Nodes: All tracker nodes updated with `telemetry_level` widget
- Exporters: Updated to pass telemetry configuration

### Completed Work (See HISTORY.md for details)
- ✅ Merged TelemetryClient and MetricsLogger into single system
- ✅ Implemented configurable intervals with runtime overrides
- ✅ Simplified metrics to essential-only by default
- ✅ Removed all reward tracking (biologically plausible loss-only)
- ✅ Fixed --enable-telemetry flag to set telemetry_level
- ✅ Created 28 unit tests (all passing)

## Key Implementation Notes

### Simplified Telemetry Interval Format
- Format: `{value}_{unit}` or time suffixes
- Examples: `100_steps`, `10_episodes`, `30s`, `5m`, `1h`
- Single widget replaces mode/interval combination

### Configuration Strategy
- All intervals configurable via `--override`
- Example: `--override balancer_10:report_interval=200`
- No hardcoded values - use defaults with override capability

### Data Flow (No File I/O in runner.py)
1. **runner.py**: UDP packets only (fire-and-forget)
2. **Agent Client**: Aggregation and forwarding
3. **DNNE Server**: Only place where files are written

### Expected Outcomes
- 50% reduction in telemetry data volume
- 40% fewer lines of telemetry code
- Single unified system (no MetricsLogger)
- All metrics meaningful and actionable

## Active TODOs

### Documentation
- [ ] Update `dnne_docs/architecture/telemetry.md` with new API
- [ ] Add examples of `--override` configurations  
- [ ] Document simplified interval format
- [ ] Create migration notes for existing workflows

### Testing
- [ ] Measure actual performance overhead (<0.1% target)
- [ ] Verify data volume reduction (50% target)

---
*Last Updated: 2025-09-03*