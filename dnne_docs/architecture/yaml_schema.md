# DNNE YAML Schema Specification

## Overview

DNNE extends Isaac Gym environment YAML files with a flexible hierarchical schema system that provides semantic information about observation and action tensors. This allows nodes like Split to use human-readable names instead of numeric indices.

## Schema Structure

The schema system supports 0 to N levels of hierarchy through the `dnne` section in YAML files.

### Basic Structure

```yaml
dnne:
  # Optional: Define hierarchy levels for schema selection
  schema_levels: ["level1", "level2", ...]  
  
  # Required for each level: Options for that level
  level1_options: ["option1", "option2", ...]
  level2_options: ["optionA", "optionB", ...]
  
  # Schema definitions
  nested_schemas:
    # Structure depends on schema_levels
```

## Three Cases

### Case 1: No Schema Levels (Direct Schema)

Used when there's only one schema for the environment (e.g., Cartpole).

```yaml
dnne:
  # No schema_levels key
  nested_schemas:
    numObservations: 4
    numActions: 1
    observationSchema:
      cart_position: [0, 1]      # Index 0
      cart_velocity: [1, 2]      # Index 1
      pole_angle: [2, 3]         # Index 2
      pole_velocity: [3, 4]      # Index 3
    actionSchema:
      force: [0, 1]              # Single action
```

**UI Behavior:**
- No additional widgets created
- Schema directly available

### Case 2: Single Level Schema

Used when schemas vary by one parameter (e.g., difficulty).

```yaml
dnne:
  schema_levels: ["difficulty"]
  difficulty_options: ["easy", "medium", "hard"]
  
  nested_schemas:
    easy:
      numObservations: 10
      numActions: 2
      observationSchema:
        basic_state: [0, 10]
      actionSchema:
        simple_control: [0, 2]
    
    medium:
      numObservations: 15
      numActions: 3
      observationSchema:
        state: [0, 10]
        extra_sensors: [10, 15]
      actionSchema:
        control: [0, 3]
    
    hard:
      numObservations: 20
      numActions: 4
      observationSchema:
        full_state: [0, 20]
      actionSchema:
        advanced_control: [0, 4]
```

**UI Behavior:**
- Creates one dropdown widget: "difficulty"
- User selects difficulty level
- Schema updates based on selection

### Case 3: Multi-Level Schema

Used for complex environments with multiple configuration axes (e.g., FrankaDNNE).

```yaml
dnne:
  schema_levels: ["subtask", "controlType"]
  subtask_options: ["random_target", "block_ball", "pick_place"]
  controlType_options: ["osc", "joint_tor"]
  
  nested_schemas:
    random_target:
      osc:
        numObservations: 23
        numActions: 8
        observationSchema:
          joint_positions: [0, 7]
          joint_velocities: [7, 14]
          target_position: [14, 17]
          eef_position: [17, 20]
          eef_target_delta: [20, 23]
        actionSchema:
          joint_commands: [0, 7]
          gripper_command: [7, 8]
      
      joint_tor:
        numObservations: 25
        numActions: 7
        observationSchema:
          joint_positions: [0, 7]
          joint_velocities: [7, 14]
          joint_torques: [14, 21]
          target_position: [21, 24]
          eef_position: [24, 25]
        actionSchema:
          joint_torques: [0, 7]
    
    block_ball:
      osc:
        numObservations: 30
        numActions: 8
        observationSchema:
          # Different schema for block_ball task
          joint_positions: [0, 7]
          joint_velocities: [7, 14]
          block_position: [14, 17]
          ball_position: [17, 20]
          # ... etc
        actionSchema:
          joint_commands: [0, 7]
          gripper_command: [7, 8]
      
      joint_tor:
        # ... another variant
```

**UI Behavior:**
- Creates two dropdown widgets: "subtask" and "controlType"
- User selects both options
- Schema at path `nested_schemas[subtask][controlType]`

## Widget Generation Rules

1. **Widget Order:**
   - `task` (always first, selects YAML file)
   - Dynamic widgets (from schema_levels, in order)
   - Existing fixed widgets (dt, num_envs, seed, etc.)
   - `schema_display` (always last, read-only multiline text)

2. **Widget Creation:**
   - For each entry in `schema_levels`, create a dropdown widget
   - Widget name = level name (e.g., "subtask")
   - Widget options = from `{level}_options` array
   - Each widget gets a change callback

3. **Index Management:**
   ```python
   # Example for exporter
   num_dynamic = len(schema_levels) if schema_levels else 0
   
   widget_indices = {
       'task': 0,
       'subtask': 1,           # Dynamic
       'controlType': 2,       # Dynamic
       'dt': 3,               # Fixed, offset by num_dynamic
       'num_envs': 4,         # Fixed, offset by num_dynamic
       # ...
   }
   ```

## Callback Chain

1. **Task Change:**
   - Load new YAML file
   - Parse `dnne` section
   - Create/update dynamic widgets
   - Trigger schema resolution

2. **Dynamic Widget Change:**
   - Navigate to new schema path
   - Update schema_display widget
   - Notify downstream nodes (future)

3. **Schema Display Update:**
   - Format observation/action schemas
   - Show element names with ranges
   - Clear display if no schema

## Schema Display Format

The schema_display widget shows:

```
Observations (23 elements):
• joint_positions [0:7] - 7 joint positions
• joint_velocities [7:14] - 7 joint velocities  
• target_position [14:17] - 3D target position
• eef_position [17:20] - 3D end-effector position
• eef_target_delta [20:23] - EEF to target delta

Actions (8 elements):
• joint_commands [0:7] - 7 joint commands
• gripper_command [7:8] - gripper control
```

## Using Schemas in Split Node

The Split node's "by name" mode uses these schemas:

```
# Full range
"joint_positions" → [0:7]

# With slice notation
"joint_positions[2:5]" → [2:5]
"joint_velocities[:3]" → [7:10]
"target_position[0]" → [14:15]

# Multiple splits
"joint_positions[0:4],joint_positions[4:7],target_position"
```

## Migration Guide

To add schemas to existing YAML files:

1. **Identify tensor structure:** Determine what each observation/action element represents

2. **Choose hierarchy level:**
   - No levels: Single fixed schema
   - 1 level: Varies by one parameter
   - 2+ levels: Multiple configuration axes

3. **Add dnne section:**
   ```yaml
   dnne:
     # Add schema_levels if needed
     # Add option arrays for each level
     nested_schemas:
       # Add schema definitions
   ```

4. **Define semantic names:**
   - Use descriptive names
   - Group related elements
   - Document units/meanings in comments

## Best Practices

1. **Naming:**
   - Use snake_case for schema keys
   - Be descriptive (e.g., `joint_positions` not `jp`)
   - Group logically (all joint data together)

2. **Ranges:**
   - Use [start, end] format (end is exclusive)
   - Ensure no gaps or overlaps
   - Document total size

3. **Documentation:**
   - Add comments explaining units
   - Note coordinate frames
   - Describe special values (e.g., -1 for invalid)

4. **Validation:**
   - Ensure option arrays match actual schemas
   - Test all combinations for multi-level schemas
   - Verify total size matches numObservations/numActions

## Implementation Status

- ✅ Basic schema support (single fixed schema)
- ✅ Split node "by name" mode with slice notation
- 🚧 Dynamic widget generation
- 🚧 Multi-level schema navigation
- 🚧 Real-time schema display
- 🚧 Callback system for updates

## Future Extensions

- Schema validation at export time
- Auto-generate schemas from environment code
- Schema inheritance (base + overrides)
- Type information (float32, int64, etc.)
- Unit annotations (meters, radians, etc.)