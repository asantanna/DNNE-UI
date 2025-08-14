"""
DNNE Node Color Scheme

Defines standard colors for different node categories.
Each node type has a foreground (color) and background (bgcolor) color.
"""

# Color definitions matching ComfyUI's palette
NODE_COLORS = {
    # BROWN: Data nodes (datasets, batch samplers, get batch)
    "data": {
        "color": "#332922",
        "bgcolor": "#593930"
    },
    
    # YELLOW: Training nodes (loss, optimizer, training step, epoch tracker)
    "training": {
        "color": "#432",
        "bgcolor": "#653"
    },
    
    # BLUE: Network nodes
    "network": {
        "color": "#223",
        "bgcolor": "#335"
    },
    
    # PURPLE: Simulation nodes (IsaacGymSim)
    "simulation": {
        "color": "#323",
        "bgcolor": "#535"
    },
    
    # PALE BLUE: Utility nodes (IsaacGymEnvs, PPOConfig, OR, CustomComputation, BalancingNode, BalancingConfig)
    "utility": {
        "color": "#334455",
        "bgcolor": "#556677"
    },
    
    # GREEN: Layer nodes (LinearLayer, Conv2D, etc.)
    "layer": {
        "color": "#232",
        "bgcolor": "#353"
    },
    
    # RED: Non-async nodes (PPO nodes, RL agents)
    "rl": {
        "color": "#322",
        "bgcolor": "#533"
    },
    
    # Default color for uncategorized nodes
    "default": {
        "color": "#222",
        "bgcolor": "#333"
    }
}

def get_node_colors(category):
    """Get color and bgcolor for a node category"""
    return NODE_COLORS.get(category, NODE_COLORS["default"])