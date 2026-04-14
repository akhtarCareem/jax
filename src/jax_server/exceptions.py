class JaxServerError(RuntimeError):
    """Base runtime error."""


class ClientInputError(JaxServerError):
    """Raised when a request is invalid for the target model."""


class ModelLoadError(JaxServerError):
    """Raised when a model cannot be loaded."""


class ArtifactNotFoundError(ModelLoadError):
    """Raised when no compatible artifacts are found."""


class ExecutionError(JaxServerError):
    """Raised when inference fails due to a server/runtime problem."""
