class CapacityError(ValueError):
    """The redundant stream cannot hold the requested payload."""


class InvalidKeyError(ValueError):
    """A key produced invalid authenticated or structural data."""


class ExtractionError(ValueError):
    """A data-hider payload failed validation."""


class InsufficientSharesError(ValueError):
    """The requested recovery stage does not have enough shares."""


class InvalidShareError(ValueError):
    """A share failed its keyed integrity check."""
