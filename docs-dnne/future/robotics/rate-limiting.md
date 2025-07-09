# GetBatch Rate Limiting for Robotics

## Priority
Low

## Description
Add optional rate limiting widget to GetBatch node for robotics applications that need specific timing.

## Motivation
- Robotics often requires fixed control frequencies
- Training should run as fast as possible
- Need user control over timing behavior
- Current system optimized for speed only

## Implementation Notes
### Widget Design
- Default: "off" (no rate limiting)
- User can set specific Hz (e.g., 10Hz, 30Hz, 100Hz)
- Clear labeling: "Rate Limit (robotics only)"
- Warning when enabled during training

### Implementation Options
1. **Sleep-based**: Simple asyncio.sleep()
2. **Timer-based**: Precise timing with compensation
3. **Queue-based**: Rate limit at queue level

### Usage Pattern
```
BatchSampler → GetBatch(rate_limit=30Hz) → RobotControl
                                              ↓
                                         Fixed 30Hz loop
```

## Technical Considerations
- Don't affect training performance when disabled
- Accurate timing for real-time control
- Handle timing drift
- Work with async architecture

## Dependencies
- Current GetBatch implementation
- Async timing utilities

## Estimated Effort
Small

## Success Metrics
- Precise rate limiting when enabled
- Zero overhead when disabled
- Easy to understand UI
- Reliable timing for robotics