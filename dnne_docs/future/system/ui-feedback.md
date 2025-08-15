# UI Progress Feedback

## Priority
Low

## Description
Improve user interface feedback during long operations, especially exports and training.

## Motivation
- Exports can take time with no feedback
- Users unsure if system is working
- No progress indication
- Professional UX expected

## Implementation Notes
### Export Progress
- Progress bar during export
- Current node being processed
- Estimated time remaining
- Clear completion notification

### Training Feedback
- Live metrics display
- Epoch progress bar
- Batch processing rate
- Time estimates

### Implementation Options
1. **WebSocket updates**: Real-time progress
2. **Progress API**: Structured progress events
3. **UI overlay**: Non-blocking progress display

## Technical Considerations
- Don't slow down operations
- Handle long-running exports
- Clear error states
- Cancelable operations

## Dependencies
- WebSocket infrastructure
- UI framework capabilities
- Progress tracking hooks

## Estimated Effort
Small-Medium

## Success Metrics
- Clear progress indication
- Accurate time estimates
- Improved user confidence
- Professional appearance