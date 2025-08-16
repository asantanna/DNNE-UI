# Link Pattern Analysis Report

## Summary Statistics

- **Total Workflows**: 7
- **Total Links**: 102
- **Total Nodes**: 70
- **Total Unconnected Inputs**: 350
- **Total Unconnected Outputs**: 29

- **Unique Connection Patterns**: 29
- **Unique Unconnected Input Types**: 5
- **Unique Unconnected Output Types**: 10

## Connected Link Patterns by Category

### Layer Connections

- **LinearLayer.output** (LAYER_TENSOR) → **LinearLayer.input** (*LAYER_TENSOR)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **LinearLayer.output** (LAYER_TENSOR) → **Network.to_output** (*LAYER_TENSOR)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **Network.layers** (LAYER_TENSOR) → **LinearLayer.input** (*LAYER_TENSOR)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async

### Data Flow

- **BatchSampler.dataloader** (SAMPLER_BATCH_DATALOADER) → **GetBatch.dataloader** (*BATCH_DATALOADER)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **BatchSampler.schema** (SAMPLER_BATCH_SCHEMA) → **GetBatch.schema** (*BATCH_SCHEMA)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **CIFAR10Dataset.dataset** (CIFAR10_DATASET) → **BatchSampler.dataset** (*DATASET)
  - Used in: CIFAR10_Test, Yield_Test_Async
- **CIFAR10Dataset.schema** (CIFAR10_DATASET_SCHEMA) → **BatchSampler.schema** (*DATASET_SCHEMA)
  - Used in: CIFAR10_Test, Yield_Test_Async
- **GetBatch.epoch_stats** (BATCH_EPOCH_STATS) → **EpochTracker.epoch_stats** (*EPOCH_STATS)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **GetBatch.images** (BATCH_IMAGE_TENSOR) → **Network.input** (*TENSOR)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **GetBatch.labels** (BATCH_LABEL_TENSOR) → **CrossEntropyLoss.labels** (*LABEL_TENSOR)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **MNISTDataset.dataset** (MNIST_DATASET) → **BatchSampler.dataset** (*DATASET)
  - Used in: MNIST_Test, Yield_Test, Yield_Test_Async
- **MNISTDataset.schema** (MNIST_DATASET_SCHEMA) → **BatchSampler.schema** (*DATASET_SCHEMA)
  - Used in: MNIST_Test, Yield_Test, Yield_Test_Async

### Training Flow

- **CrossEntropyLoss.accuracy** (CROSSENTROPY_ACCURACY_FLOAT) → **EpochTracker.accuracy** (*ACCURACY_FLOAT)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **CrossEntropyLoss.loss** (CROSSENTROPY_LOSS_TENSOR) → **EpochTracker.loss** (*LOSS_TENSOR)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **CrossEntropyLoss.loss** (CROSSENTROPY_LOSS_TENSOR) → **TrainingStep.loss** (*LOSS_TENSOR)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **SGDOptimizer.optimizer** (SGD_OPTIMIZER_OBJ) → **TrainingStep.optimizer** (*OPTIMIZER)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **TrainingStep.ready** (TRAIN_STEP_DONE_TRIGGER) → **BalancerNode.input** (*)
  - Used in: CIFAR10_Test, Yield_Test
- **TrainingStep.trigger** (TRAIN_STEP_TRIGGER) → **BalancerNode.input** (*)
  - Used in: Yield_Test_Async

### Control Flow

- **BalancerNode.output** (*) → **GetBatch.trigger** (*TRIGGER)
  - Used in: CIFAR10_Test, Yield_Test, Yield_Test_Async
- **IsaacGymSim.done** (SIM_DONE_TRIGGER) → **DataStreamer.reset** (*TRIGGER)
  - Used in: Franka_Minimal_Test
- **TrainingStep.ready** (TRAIN_STEP_DONE_TRIGGER) → **GetBatch.trigger** (*TRIGGER)
  - Used in: MNIST_Test

### Config Connections

- **BalancerConfig.config** (BALANCING_CONFIG) → **PPOAgent.balancing_config** (BALANCING_CONFIG)
  - Used in: Cartpole_PPO, Yield_Test
- **IsaacGymEnvs.env** (ISAAC_ENV_CONFIG) → **IsaacGymSim.env_config** (ISAAC_ENV_CONFIG)
  - Used in: Franka_Coop_Nodes, Franka_Minimal_Test
- **IsaacGymEnvs.env** (ISAAC_ENV_CONFIG) → **PPOAgent.env_config** (ISAAC_ENV_CONFIG)
  - Used in: Cartpole_PPO, Yield_Test
- **PPOConfig.config** (PPO_CONFIG) → **PPOAgent.ppo_config** (PPO_CONFIG)
  - Used in: Cartpole_PPO, Yield_Test

### Other

- **DataStreamer.data** (DATASTREAMER_DATA_TENSOR) → **IsaacGymSim.action** (*TENSOR)
  - Used in: Franka_Minimal_Test
- **IsaacGymSim.observation** (SIM_OBSERVATION_TENSOR) → **CustomComputation.input** (*TENSOR)
  - Used in: Franka_Coop_Nodes
- **Network.model** (NETWORK_MODEL) → **SGDOptimizer.model** (NETWORK_MODEL)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **Network.output** (NETWORK_OUTPUT_TENSOR) → **CrossEntropyLoss.predictions** (*TENSOR)
  - Used in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async

## Unconnected Input Slots

These inputs exist but are not connected in any workflow:

- **CrossEntropyLoss.targets** (*TENSOR)
  - Found in: Yield_Test_Async
- **DataStreamer.sync** (*TRIGGER)
  - Found in: Franka_Minimal_Test
- **IsaacGymEnvs.custom_config** (*CONFIG)
  - Found in: Yield_Test
- **IsaacGymSim.action** (*TENSOR)
  - Found in: Franka_Coop_Nodes
- **IsaacGymSim.reset** (*TRIGGER)
  - Found in: Franka_Coop_Nodes, Franka_Minimal_Test

## Unconnected Output Slots

These outputs exist but are not connected in any workflow:

- **CustomComputation.output** (CUSTOMCOMP_OUTPUT_TENSOR)
  - Found in: Franka_Coop_Nodes
- **DataStreamer.done** (DATASTREAMER_DONE_TRIGGER)
  - Found in: Franka_Minimal_Test
- **DataStreamer.metadata** (DATASTREAMER_METADATA)
  - Found in: Franka_Minimal_Test
- **EpochTracker.training_stats** (EPOCH_TRAINING_STATS)
  - Found in: CIFAR10_Test, Yield_Test_Async
- **EpochTracker.training_summary** (EPOCH_TRAINING_SUMMARY)
  - Found in: CIFAR10_Test, MNIST_Test, Yield_Test, Yield_Test_Async
- **IsaacGymSim.done** (SIM_DONE_TRIGGER)
  - Found in: Franka_Coop_Nodes
- **IsaacGymSim.observation** (SIM_OBSERVATION_TENSOR)
  - Found in: Franka_Minimal_Test
- **PPOAgent.agent** (PPO_AGENT_OBJ)
  - Found in: Cartpole_PPO, Yield_Test
- **PPOAgent.eval_stats** (PPO_EVAL_STATS)
  - Found in: Cartpole_PPO, Yield_Test
- **PPOAgent.training_stats** (PPO_TRAINING_STATS)
  - Found in: Cartpole_PPO, Yield_Test

## Type Usage Analysis

### Types Currently in Use

- *
- BALANCING_CONFIG
- DATALOADER
- DATASET
- DICT
- ISAAC_ENV_CONFIG
- MODEL
- OPTIMIZER
- PPO_CONFIG
- SCHEMA
- SYNC
- TENSOR
- TRIGGER

### Suggested Type Refinements

- ***** → *ACCURACY_FLOAT, CROSSENTROPY_ACCURACY_FLOAT, TRAIN_STEP_DONE_TRIGGER, TRAIN_STEP_TRIGGER
- **DATALOADER** → *BATCH_DATALOADER, SAMPLER_BATCH_DATALOADER
- **DATASET** → *DATASET, CIFAR10_DATASET, MNIST_DATASET
- **DICT** → *EPOCH_STATS, BATCH_EPOCH_STATS
- **MODEL** → NETWORK_MODEL
- **OPTIMIZER** → *OPTIMIZER, SGD_OPTIMIZER_OBJ
- **SCHEMA** → *BATCH_SCHEMA, *DATASET_SCHEMA, CIFAR10_DATASET_SCHEMA, MNIST_DATASET_SCHEMA, SAMPLER_BATCH_SCHEMA
- **SYNC** → *, *TRIGGER, TRAIN_STEP_DONE_TRIGGER
- **TENSOR** → *LABEL_TENSOR, *LAYER_TENSOR, *LOSS_TENSOR, *TENSOR, BATCH_IMAGE_TENSOR, BATCH_LABEL_TENSOR, CROSSENTROPY_LOSS_TENSOR, DATASTREAMER_DATA_TENSOR, LAYER_TENSOR, NETWORK_OUTPUT_TENSOR, SIM_OBSERVATION_TENSOR
- **TRIGGER** → *TRIGGER, SIM_DONE_TRIGGER

### Types in Use after Changes (these are grouped by LINK COLOR)

#### Main Data Flow Types (PURPLE)
- BATCH_IMAGE_TENSOR
- CUSTOMCOMP_OUTPUT_TENSOR
- DATASTREAMER_DATA_TENSOR
- LAYER_TENSOR
- NETWORK_OUTPUT_TENSOR
- SIM_OBSERVATION_TENSOR

#### Dictionary/Stats Types (YELLOW)
- BATCH_EPOCH_STATS
- EPOCH_TRAINING_STATS
- EPOCH_TRAINING_SUMMARY
- PPO_EVAL_STATS
- PPO_TRAINING_STATS

## Training Types (CYAN)
- BATCH_LABEL_TENSOR
- CROSSENTROPY_ACCURACY_FLOAT
- CROSSENTROPY_LOSS_TENSOR

#### Configuration Types (GREEN)
- BALANCING_CONFIG
- ISAAC_ENV_CONFIG
- PPO_CONFIG
- DATASTREAMER_METADATA

#### Object Types (LIGHT BLUE)
- CIFAR10_DATASET
- CIFAR10_DATASET_SCHEMA
- MNIST_DATASET
- MNIST_DATASET_SCHEMA
- NETWORK_MODEL
- PPO_AGENT_OBJ
- SAMPLER_BATCH_DATALOADER
- SGD_OPTIMIZER_OBJ

#### Trigger Types (RED)
- DATASTREAMER_DONE_TRIGGER
- SIM_DONE_TRIGGER
- TRAIN_STEP_DONE_TRIGGER
- TRAIN_STEP_TRIGGER

#### Schema Types (BROWN)
- CIFAR10_DATASET_SCHEMA
- MNIST_DATASET_SCHEMA
- SAMPLER_BATCH_SCHEMA


