"""
Subsystem constants for DNNE node categorization.

All exporters MUST use these constants when declaring their subsystem.
To add a new subsystem, update this file and document its purpose.
"""

# Core subsystems
SUBSYSTEM_TRAINING = "training"      # Training loop components (optimizers, loss, epochs)
SUBSYSTEM_DATA = "data"              # Data loading and batching
SUBSYSTEM_NETWORK = "network"        # Neural network architectures
SUBSYSTEM_RL = "rl"                  # Reinforcement learning components
SUBSYSTEM_ROBOTICS = "robotics"      # Robotics simulation and control
SUBSYSTEM_CONTROL = "control"        # Control flow (OR, balancer, etc.)
SUBSYSTEM_UTIL = "util"              # Utility nodes that don't fit elsewhere

# Framework subsystems
SUBSYSTEM_QUEUE = "queue"            # Queue framework internals
SUBSYSTEM_CHECKPOINT = "checkpoint"  # Checkpointing system
SUBSYSTEM_YIELD = "yield"            # Yield/concurrency management
SUBSYSTEM_BALANCING = "balancing"    # Load balancing

# Monitoring and metrics
SUBSYSTEM_TELEMETRY = "telemetry"    # Telemetry data collection
SUBSYSTEM_MONITORING = "monitoring"  # Active monitoring (subset of telemetry)

# RL-specific subsystems
SUBSYSTEM_PPO = "ppo"                # PPO algorithm specific

# All valid subsystems (for validation)
ALL_SUBSYSTEMS = {
    SUBSYSTEM_TRAINING,
    SUBSYSTEM_DATA,
    SUBSYSTEM_NETWORK,
    SUBSYSTEM_RL,
    SUBSYSTEM_ROBOTICS,
    SUBSYSTEM_CONTROL,
    SUBSYSTEM_UTIL,
    SUBSYSTEM_QUEUE,
    SUBSYSTEM_CHECKPOINT,
    SUBSYSTEM_YIELD,
    SUBSYSTEM_BALANCING,
    SUBSYSTEM_TELEMETRY,
    SUBSYSTEM_MONITORING,
    SUBSYSTEM_PPO,
}