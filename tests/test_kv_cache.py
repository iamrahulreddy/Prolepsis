"""
Tests for KV cache management.
"""

import pytest


class TestDualKVCacheManager:
    """Test suite for DualKVCacheManager."""
    
    @pytest.fixture
    def manager(self):
        from prolepsis.core.kv_cache import DualKVCacheManager
        return DualKVCacheManager()
    
    def test_initial_state(self, manager):
        """Manager should start with empty state."""
        assert manager.sync_point == 0
        assert manager.draft.length == 0
        assert manager.target.length == 0
    
    def test_prefill_updates_all(self, manager):
        """Prefill should update all lengths."""
        manager.prefill_complete(128)
        
        assert manager.sync_point == 128
        assert manager.draft.length == 128
        assert manager.target.length == 128
    
    def test_drafting_only_updates_draft(self, manager):
        """Drafting should only update draft length."""
        manager.prefill_complete(100)
        manager.after_drafting(4)
        
        assert manager.draft.length == 104
        assert manager.target.length == 100  # Unchanged
        assert manager.sync_point == 100     # Unchanged
    
    def test_acceptance_rate(self, manager):
        """Acceptance rate should be calculated correctly."""
        manager.total_drafted = 100
        manager.total_accepted = 75
        
        assert manager.get_acceptance_rate() == 0.75
    
    def test_reset_clears_all(self, manager):
        """Reset should clear all state."""
        manager.prefill_complete(100)
        manager.after_drafting(10)
        manager.total_accepted = 50
        
        manager.reset()
        
        assert manager.sync_point == 0
        assert manager.draft.length == 0
        assert manager.total_accepted == 0


class TestCacheState:
    """Test suite for CacheState."""
    
    def test_advance(self):
        from prolepsis.core.kv_cache import CacheState
        
        state = CacheState(length=10)
        state.advance(5)
        assert state.length == 15
    
    def test_truncate(self):
        from prolepsis.core.kv_cache import CacheState
        
        state = CacheState(length=20)
        state.truncate(10)
        assert state.length == 10
    
    def test_truncate_noop_if_larger(self):
        from prolepsis.core.kv_cache import CacheState
        
        state = CacheState(length=10)
        state.truncate(20)  # Should be noop
        assert state.length == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
