"""Custom exceptions for dnn-benchmarking."""


class GraphLoadError(Exception):
    """Raised when a graph fails to load or validate."""

    pass


class ExecutionError(Exception):
    """Raised when graph execution fails."""

    pass


class ValidationError(Exception):
    """Raised when validation fails."""

    pass
