"""Custom exceptions for AccOpsAgent."""


class AccOpsAgentError(Exception):
    """Base exception for AccOpsAgent."""

    pass


class BackendError(AccOpsAgentError):
    """Exception raised when backend operations fail."""

    pass


class DiagnosticReadError(BackendError):
    """Exception raised when reading diagnostics fails."""

    pass


class ParameterSetError(BackendError):
    """Exception raised when setting parameters fails."""

    pass


class ValidationError(AccOpsAgentError):
    """Exception raised when validation fails."""

    pass


class ConstraintViolationError(ValidationError):
    """Exception raised when a constraint is violated."""

    pass


class StateError(AccOpsAgentError):
    """Exception raised for invalid state operations."""

    pass


class GraphExecutionError(AccOpsAgentError):
    """Exception raised during graph execution."""

    pass
