# Framework Exceptions

class CauseExitException(Exception):
    """Raised when a node requests the graph to exit with a specific code"""
    def __init__(self, node_id, message="Exit requested", exit_code=0):
        self.node_id = node_id
        self.message = message
        self.exit_code = exit_code
        super().__init__(f"Node {node_id}: {message} (exit_code={exit_code})")