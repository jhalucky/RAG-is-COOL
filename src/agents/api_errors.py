
def is_rate_limit_error(error: Exception) -> bool:
    """
    Detect Gemini quota and rate-limit errors.

    Treat HTTP 429 as non-retryable within the current
    research run.
    """

    code = getattr(error, "code", None)
    status = getattr(error, "status", None)

    message = str(error).upper()

    return (
        str(code) == "429"
        or str(status) == "429"
        or "429 RESOURCE_EXHAUSTED" in message
        or "QUOTA EXCEEDED" in message
    )
