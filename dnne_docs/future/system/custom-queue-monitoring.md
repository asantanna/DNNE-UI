# Custom Queue Monitoring

**Priority**: Medium

## Description
Several nodes bypass standard queue monitoring by directly manipulating queues with `self.input_queues[X].get()`, making them invisible to deadlock analysis. We need an automatic solution to track these operations.

## Problem Nodes
The following nodes use custom queue manipulation that escapes monitoring:

1. **GetBatchNode** - Directly calls `await self.input_queues["dataloader"].get()` and similar for schema/trigger
2. **IsaacGymSimNode** - Creates tasks with `asyncio.create_task(self.input_queues["action"].get())`
3. **DataStreamerNode** - Direct queue access for sync/reset inputs
4. **SGDOptimizerNode** - Has virtual connection handling (partially fixed)
5. **Nodes using input_waiter** - BarrierNode, Eat_NNode, NetworkNode, ConcatNode (may need review)

## Current Workaround
Manually adding logging calls around custom queue operations in templates (Solution B from deadlock analysis investigation).

## Proposed Solutions

### Option 1: Helper Methods (Recommended)
Add `get_input_monitored()` method to base_nodes.py that includes logging, then update templates to use it.

### Option 2: MonitoredQueue Wrapper
Replace Queue instances with a MonitoredQueue class that automatically logs all operations.

### Option 3: Eliminate Custom Queue Operations
Refactor nodes to use standard queue patterns, eliminating the need for custom manipulation.

## Motivation
- Deadlock analysis currently misses these operations, leading to incorrect root cause identification
- GetBatchNode appears to "never receive input" even when working correctly
- Makes debugging complex workflows harder

## Implementation Notes
- Must maintain backward compatibility
- Logging should have minimal performance impact when disabled
- Consider making this opt-in initially with `--debug-deadlock` flag

## Dependencies
- Requires updates to deadlock_utils.py
- May need changes to base_nodes.py
- Template updates for affected nodes

## Estimated Effort
2-3 days for full implementation and testing across all affected nodes