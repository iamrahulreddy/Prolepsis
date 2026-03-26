"""
KV cache management for speculative decoding.

Tracks logical cache lengths for both draft and target models,
handling synchronization after rejection.
"""

from dataclasses import dataclass


@dataclass
class CacheState:
    """State of a single model's KV cache."""
    length: int = 0  
    
    def advance(self, num_tokens: int):
        self.length += num_tokens
    
    def truncate(self, new_length: int):
        """Truncate to new_length."""
        if new_length < 0:
            raise ValueError(f"Cannot truncate to negative length: {new_length}")
        if new_length < self.length:
            self.length = new_length
    
    def reset(self):
        """Reset to empty."""
        self.length = 0


class DualKVCacheManager:
    
    def __init__(self):
        self.draft = CacheState()
        self.target = CacheState()
        self.sync_point: int = 0  # Last synchronized position
        
        # Statistics
        self.total_accepted: int = 0
        self.total_drafted: int = 0
    
    def prefill_complete(self, prompt_len: int):
        self.draft.length = prompt_len
        self.target.length = prompt_len
        self.sync_point = prompt_len
    
    def after_drafting(self, cache_tokens: int, drafted_tokens=None):
    
        self.draft.advance(cache_tokens)
        if drafted_tokens is None:
            drafted_tokens = cache_tokens
        self.total_drafted += drafted_tokens
    
    
    
    def sync_after_acceptance(
        self,
        num_accepted: int,
        draft_wrapper,  # DraftModelWrapper
        target_wrapper,  # TargetModelWrapper
        includes_bonus: bool = False,
    ):
       
        bonus_len = 1 if includes_bonus else 0
        new_sync_point = self.sync_point + bonus_len + num_accepted
        
        draft_wrapper.truncate_cache(new_sync_point)
        target_wrapper.truncate_cache(new_sync_point)
        
        self.draft.length = new_sync_point
        self.target.length = new_sync_point
        self.sync_point = new_sync_point
        
        self.total_accepted += num_accepted
    
    def get_acceptance_rate(self) -> float:
        """Get overall acceptance rate."""
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted
    
    def reset(self):
        """Reset manager for new generation."""
        self.draft.reset()
        self.target.reset()
        self.sync_point = 0
        self.total_accepted = 0
        self.total_drafted = 0
    
    def __repr__(self) -> str:
        return (
            f"DualKVCacheManager(sync={self.sync_point}, "
            f"draft={self.draft.length}, target={self.target.length}, "
            f"accept_rate={self.get_acceptance_rate():.2%})"
        )
