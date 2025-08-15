# Type System History

*Component: DNNE Type System*

## Session: 2025-08-15 - Dynamic Color Palette System

### Problem
CONFIG link types (PPO_CONFIG_PYDICT, BALANCING_CONFIG_PYDICT) were showing green (#37745C) instead of the intended red color defined in dnneColors.ts.

### Solution
Implemented dynamic color palette substitution system:
1. Updated dark.json to use placeholders ({CONFIG_COLOR}, {DATA_COLOR}, etc.)
2. Created substitutePaletteColors() function in dnneColors.ts
3. Applied substitution when loading palette in coreColorPalettes.ts
4. Removed TYPE_COLOR_MAP duplication - dark.json is now single source of truth

### Color Updates (from user preferences)
- CONFIG_COLOR: Changed from green to purple (dim(ORIGINAL_COLORS.PURPLE, 0.5))
- TRAINING_COLOR: Changed to ORIGINAL_COLORS.BLUE (undimmed)
- CONTROL_COLOR: Adjusted to 40% dimming (was 50%)
- STATS_COLOR: Adjusted to 30% dimming

### Result
- Clean separation between color definitions and type mappings
- CONFIG links now correctly show intended colors
- Easy to update colors in one place (dnneColors.ts)

## Session: 2025-08-14 - Type System Refinement

### Phase 1: Analysis
- Created scripts to analyze all workflow connections
- Discovered 95 unique link patterns across all workflows
- Identified need for refined type system beyond generic TENSOR/MODEL

### Phase 2: Design
- Designed refined type hierarchy with specific types:
  - Data flow: BATCH_IMAGE_TENSOR, LAYER_TENSOR, etc.
  - Models: NETWORK_MODEL, PPO_AGENT_OBJ
  - Stats: EPOCH_TRAINING_STATS, PPO_EVAL_STATS
  - Config: ISAAC_ENV_CONFIG, PPO_CONFIG, BALANCING_CONFIG
- Created wildcard matching system (*TENSOR matches any _TENSOR suffix)

### Phase 3: Frontend Implementation
- Implemented dnneTypeValidation.ts for wildcard type matching
- Created dnneLinkColorService.ts for link color resolution
- Overrode LiteGraph's connectSlots to resolve types at connection time
- Updated color palette with all specific type colors

### Phase 4: Backend Implementation  
- Updated all 19 node definition files with refined types:
  - Dataset nodes: MNIST_DATASET, CIFAR10_DATASET with schemas
  - Training nodes: specific loss and accuracy types
  - Network nodes: NETWORK_MODEL output type
  - RL nodes: PPO_AGENT_OBJ, training/eval stats
  - Robotics nodes: SIM_OBSERVATION_TENSOR, SIM_DONE_TRIGGER

### Implementation Details

#### Type Resolution Algorithm
1. If output type is specific → use output type
2. Else if input type is specific → use input type  
3. Else if both wildcards → use suffix (last component)
4. Special case (*,*) → use "ANY"

#### Color Groupings
- Purple (#B39DDB) - Main data flow tensors
- Yellow (#FFD500) - Statistics and dictionaries
- Cyan (#64B5F6) - Training components
- Green (#6EE7B7) - Configuration objects
- Light Green (#81C784) - Model objects
- Red (#FF6E6E) - Triggers and actions
- Brown (#8B7355) - Schemas
- Orange (#FFA931) - Data sources

### Files Modified
- 19 node definition files in custom_nodes/
- Frontend services in src/services/
- Color palette in src/assets/palettes/dark.json
- GraphView.vue for integration

### Testing Record
- Frontend builds successfully with no TypeScript errors
- All node definitions updated and validated
- Color resolution system integrated and logging enabled