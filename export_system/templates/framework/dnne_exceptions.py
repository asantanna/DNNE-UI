"""
DNNE Exception Hierarchy

Provides specific exception types for fail-fast error handling.
All DNNE errors should inherit from DNNEError for easy identification.
"""


class DNNEError(Exception):
    """Base exception for all DNNE-specific errors"""
    pass


class NodeError(DNNEError):
    """Base exception for node-related errors"""
    pass


class NodeConfigurationError(NodeError):
    """Raised when a node is misconfigured or missing required parameters"""
    pass


class NodeConnectionError(NodeError):
    """Raised when required node connections are missing or invalid"""
    pass


class NodeExecutionError(NodeError):
    """Raised when a node fails during execution"""
    pass


class ExportError(DNNEError):
    """Base exception for export system errors"""
    pass


class WorkflowValidationError(ExportError):
    """Raised when workflow validation fails before export"""
    pass


class TemplateError(ExportError):
    """Raised when template processing fails"""
    pass


class ImportPathError(DNNEError):
    """Raised when required imports cannot be resolved"""
    pass


class CheckpointError(DNNEError):
    """Base exception for checkpoint-related errors"""
    pass


class CheckpointLoadError(CheckpointError):
    """Raised when checkpoint exists but cannot be loaded"""
    pass


class CheckpointSaveError(CheckpointError):
    """Raised when checkpoint cannot be saved"""
    pass


class EnvironmentError(DNNEError):
    """Raised when environment setup or configuration fails"""
    pass


class DeviceError(DNNEError):
    """Raised when device (GPU/CPU) configuration fails"""
    pass