class StatwrapError(Exception):
    """Base class for all statwrap exceptions with friendly output."""

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - simple string
        return self.message or self.__class__.__name__

    def __repr__(self) -> str:  # pragma: no cover - simple string
        return f"{self.__class__.__name__}({self.message!r})"


class SimplePlotError(StatwrapError):
    """Raised when RegressionLine.plot is called with unexpected columns."""

    def __str__(self) -> str:  # pragma: no cover - simple string
        if self.message:
            return self.message
        return (
            "RegressionLine.plot() requires simple linear regression with a single"
            " predictor column"
        )
