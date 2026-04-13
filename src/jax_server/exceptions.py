class JaxServerError(RuntimeError):
    """Base runtime error."""


class ModelLoadError(JaxServerError):
    """Raised when a model cannot be loaded."""


class ArtifactNotFoundError(ModelLoadError):
    """Raised when no compatible artifacts are found."""


class PredictionError(JaxServerError):
    """Raised when inference fails."""
