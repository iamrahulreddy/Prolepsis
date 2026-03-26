"""Triton kernels for accelerated operations."""

# Lazy import to avoid Triton dependency when not using kernels
def __getattr__(name: str):
    if name == "verify_speculative_tokens":
        from prolepsis.kernels.verify_kernel import verify_speculative_tokens
        return verify_speculative_tokens
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["verify_speculative_tokens"]
