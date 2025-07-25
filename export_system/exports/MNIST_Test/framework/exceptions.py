# Framework Exceptions

class TrainingCompleteException(Exception):
    """Raised when training is complete and the graph should stop"""
    def __init__(self, node_id, message="Training complete"):
        self.node_id = node_id
        self.message = message
        super().__init__(f"Node {node_id}: {message}")