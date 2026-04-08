"""Nage SDK exceptions."""


class NageError(Exception):
    """Base exception for all Nage SDK errors."""
    def __init__(self, message: str, status_code: int = None, body: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


class AuthError(NageError):
    """Invalid or missing API key."""
    pass


class RateLimitError(NageError):
    """STRATUM tier rate limit exceeded."""
    def __init__(self, message: str, limit: int = None, used: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.limit = limit
        self.used = used


class NotFoundError(NageError):
    """Resource not found."""
    pass


class ServerError(NageError):
    """Server-side error."""
    pass
