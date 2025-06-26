# Context class for maintaining state between nodes
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Context:
    """Shared context for stateful operations"""
    memory: Dict[str, Any] = field(default_factory=dict)
    training: bool = True
    step_count: int = 0
    episode_count: int = 0
    
    def reset(self):
        """Reset context for new episode"""
        self.memory.clear()
        self.step_count = 0
