# DNNE Type System Philosophy

## Core Principles

### 1. Outputs are Specific, Inputs are Permissive
- **Outputs declare exactly what they produce** with specific type names (e.g., `BATCH_LABEL_TENSOR`, `SGD_OPTIMIZER_OBJ`)
- **Inputs accept broader types** using wildcards (e.g., `*TENSOR`, `*OPTIMIZER`)
- This allows maximum flexibility while maintaining type safety

### 2. Format Suffixes for Clarity
All types include their underlying format as a suffix:
- `_TENSOR` - PyTorch tensors of various shapes/purposes
- `_OBJ` - Python objects (models, optimizers, agents)
- `_DICT` - Python dictionaries
- `_JSON` - JSON-serializable configuration data
- `_SCHEMA` - JSON schemas describing data structure
- `_TRIGGER` - Synchronization signals
- `_FLOAT` - Scalar floating point values

### 3. Prefix Context for Specificity
Type names include context about their source/purpose:
- Node-specific: `BATCH_LABEL_TENSOR` (from GetBatch node)
- Algorithm-specific: `SGD_OPTIMIZER_OBJ` (SGD optimizer)
- Domain-specific: `SIM_OBSERVATION_TENSOR` (from simulator)

## Type Categories

### Tensor Types
Data tensors flowing through the network:
- `LAYER_TENSOR` - Virtual layer connections
- `BATCH_IMAGE_TENSOR`, `BATCH_LABEL_TENSOR` - Training data
- `NETWORK_OUTPUT_TENSOR` - Model predictions
- `CROSSENTROPY_LOSS_TENSOR` - Loss values
- `SIM_OBSERVATION_TENSOR` - Simulator observations

### Object Types
Complex Python objects:
- `NETWORK_MODEL` - Neural network architectures
- `SGD_OPTIMIZER_OBJ` - Optimizer instances
- `PPO_AGENT_OBJ` - Trained RL agents
- `MNIST_DATASET`, `CIFAR10_DATASET` - Dataset objects
- `SAMPLER_BATCH_DATALOADER` - Data loading objects

### Configuration Types
Static configuration data:
- `BALANCING_CONFIG` - Load balancing settings
- `PPO_CONFIG` - PPO algorithm parameters
- `ISAAC_ENV_CONFIG` - Environment configuration

### Signal Types
Synchronization and control flow:
- `TRAIN_STEP_DONE_TRIGGER` - Training completion signals
- `SIM_DONE_TRIGGER` - Simulation completion
- `DATASTREAMER_DONE_TRIGGER` - Data streaming completion

### Schema Types
Data structure definitions:
- `MNIST_DATASET_SCHEMA` - MNIST data format
- `CIFAR10_DATASET_SCHEMA` - CIFAR-10 data format
- `SAMPLER_BATCH_SCHEMA` - Batch sampling structure

### Statistics Types
Training and evaluation metrics:
- `BATCH_EPOCH_STATS` - Per-batch statistics
- `EPOCH_TRAINING_STATS` - Training metrics
- `PPO_EVAL_STATS` - RL evaluation metrics

## Special Cases

### Wildcards
- `*` - Universal acceptor (BalancingNode passthrough)
- `*TENSOR` - Accepts any tensor type
- `*OPTIMIZER` - Accepts any optimizer

### Passthrough Nodes
Nodes like `BalancingNode` that simply pass data through unchanged use `*` for both input and output to maintain type flexibility.

### Virtual Connections
`LAYER_TENSOR` represents virtual connections between neural network layers that don't exist as actual runtime objects but are conceptual links in the network architecture.

## Connection Rules

The type system enforces these connection rules:
1. **Exact match**: Types must match exactly when no wildcard is present
2. **Wildcard acceptance**: `*TYPE` accepts any subtype of TYPE
3. **Universal wildcard**: `*` accepts any type
4. **Output specificity**: Outputs cannot use wildcards (except passthrough nodes)

## Implementation Notes

### Type Validation
Connection validity is checked using the `validate_node_input` function in `comfy_execution/validation.py`:
- Non-strict mode: Types must have at least one overlap
- Strict mode: Input must be subset of output
- Wildcard (`*`) in input type accepts anything

### Migration Strategy
1. Update node definitions with new specific output types
2. Update node inputs to use wildcard types where appropriate
3. Color scheme updates to follow (grouped by type category)

## Future Considerations

- Type inheritance/hierarchy for related types
- Automatic type inference for new nodes
- Visual indicators in UI for type categories
- Validation warnings for type mismatches