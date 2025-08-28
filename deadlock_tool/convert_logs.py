#!/usr/bin/env python3
"""
Convert data_flow.log (JSON Lines) to events.json for deadlock analysis.
"""

import json
from pathlib import Path

def convert_logs_to_events():
    """Convert the data_flow.log to events.json format"""
    log_path = Path('/tmp/dnne_deadlock_data/data_flow.log')
    output_path = Path('/tmp/dnne_deadlock_data/events.json')
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return False
        
    events = []
    
    # Read JSON Lines format
    with open(log_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                
                # Convert to our event format
                event = {
                    'timestamp': entry.get('ts', 0),
                    'node_id': entry.get('node', ''),
                }
                
                # Map event types
                entry_type = entry.get('type', '')
                
                if entry_type == 'QUEUE_GET_WAIT':
                    event['event_type'] = 'QUEUE_GET_WAIT'
                    event['input_name'] = entry.get('queue', 'input')
                    
                elif entry_type == 'QUEUE_GET_SUCCESS':
                    event['event_type'] = 'QUEUE_GET_SUCCESS'
                    event['input_name'] = entry.get('queue', 'input')
                    event['wait_time'] = entry.get('wait_time', 0)
                    
                elif entry_type == 'QUEUE_PUT':
                    event['event_type'] = 'QUEUE_PUT'
                    event['output_name'] = entry.get('queue', 'output')
                    event['data'] = entry.get('data', {})
                    
                elif entry_type == 'QUEUE_PUT_BLOCKED':
                    event['event_type'] = 'QUEUE_PUT_BLOCKED'
                    event['output_name'] = entry.get('queue', 'output')
                    
                elif entry_type == 'NODE_START':
                    event['event_type'] = 'NODE_START'
                    event['class_name'] = entry.get('class_name', '')
                    
                elif entry_type == 'NODE_EXECUTE':
                    event['event_type'] = 'NODE_EXECUTE'
                    
                elif entry_type == 'QUEUE_STATE':
                    event['event_type'] = 'QUEUE_STATE'
                    event['queue_depths'] = entry.get('queue_depths', {})
                    
                else:
                    # Keep original type for unmapped events
                    event['event_type'] = entry_type
                    event['data'] = entry
                    
                events.append(event)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse line: {e}")
                continue
                
    # Save as JSON array
    with open(output_path, 'w') as f:
        json.dump(events, f, indent=2)
        
    print(f"✓ Converted {len(events)} events")
    print(f"✓ Saved to {output_path}")
    
    # Show event type distribution
    event_types = {}
    for event in events:
        et = event.get('event_type', 'unknown')
        event_types[et] = event_types.get(et, 0) + 1
        
    print("\nEvent Type Distribution:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
        
    # Show time range
    if events:
        start_time = min(e.get('timestamp', 0) for e in events)
        end_time = max(e.get('timestamp', 0) for e in events)
        duration = end_time - start_time
        print(f"\nTime Range: {duration:.3f} seconds")
        print(f"  Start: {start_time:.3f}")
        print(f"  End: {end_time:.3f}")
        
    return True

if __name__ == "__main__":
    convert_logs_to_events()