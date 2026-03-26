"""Core components for speculative decoding."""

__all__ = ["RejectionSampler", "DualKVCacheManager", "apply_sampling_filters"]


# Lazy imports to avoid loading torch at import time
def __getattr__(name: str):
    if name == "RejectionSampler":
        from prolepsis.core.rejection_sampler import RejectionSampler
        return RejectionSampler
    if name == "DualKVCacheManager":
        from prolepsis.core.kv_cache import DualKVCacheManager
        return DualKVCacheManager
    if name == "apply_sampling_filters":
        from prolepsis.core.sampling_utils import apply_sampling_filters
        return apply_sampling_filters
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
