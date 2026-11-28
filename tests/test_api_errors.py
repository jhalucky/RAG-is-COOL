
from src.agents.api_errors import is_rate_limit_error


class FakeAPIError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def test_detects_gemini_429():
    error = FakeAPIError(
        "429 RESOURCE_EXHAUSTED",
        code=429,
    )

    assert is_rate_limit_error(error) is True


def test_detects_quota_exceeded():
    error = FakeAPIError("Quota exceeded for free tier")

    assert is_rate_limit_error(error) is True


def test_ignores_other_errors():
    error = FakeAPIError("Invalid request", code=400)

    assert is_rate_limit_error(error) is False
