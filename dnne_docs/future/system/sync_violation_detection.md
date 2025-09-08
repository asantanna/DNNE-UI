# Sync Violation Detection System

**Priority**: High  
**Status**: Proposed

## Description
System to detect and prevent "runaway execution" bugs where nodes race ahead of each other instead of maintaining lockstep synchronization. This is the opposite of deadlocks but equally destructive (e.g., Shadow_Train's 1.2M+ event busy loop).

## Detection Strategies

### 1. Generation Tracking
- Each data item tagged with generation number
- Nodes validate sequential processing (can't skip generations)
- Fatal error if gen N+2 arrives before N+1 processed

### 2. Sync Barriers
- Virtual barrier nodes requiring all inputs before proceeding
- Similar to MPI barriers in parallel computing
- Enforces synchronization checkpoints

### 3. Event Sequence Validation
- Define expected patterns (e.g., obs→action→obs→step_complete)
- Runtime validation against actual sequences
- Fatal on deviation

### 4. Queue Depth Monitoring
- Track queue depths across graph
- Alert/fail if queues exceed threshold
- Indicates producer racing ahead of consumer

## Implementation Components

1. **sync_validator.py** - Framework module with validation strategies
2. **Enhanced base_nodes** - Generation tracking infrastructure
3. **Diagnostic tool** - analyze_sync.py for post-mortem analysis
4. **Test workflows** - Intentionally broken workflows for validation

## Benefits
- Early detection (fail-fast)
- Clear error messages
- Optional overhead (debug flag)
- Reusable patterns

## Estimated Effort
2-3 days for full implementation with all strategies