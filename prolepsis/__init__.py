import logging

from prolepsis.config import SpeculativeConfig

__version__ = "1.0.0"
__all__ = [
    "SpeculativeConfig",
    "SpeculativeDecoder",
    "RejectionSampler",
    "__version__",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Lazy imports to avoid loading heavy dependencies at import time
def __getattr__(name: str):
    if name == "SpeculativeDecoder":
        from prolepsis.core.speculative_engine import SpeculativeDecoder
        return SpeculativeDecoder
    if name == "RejectionSampler":
        from prolepsis.core.rejection_sampler import RejectionSampler
        return RejectionSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
