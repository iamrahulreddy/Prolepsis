"""
Mock-based engine test for the speculative decoding loop.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Optional, Tuple

from prolepsis.core.rejection_sampler import RejectionSampler, RejectionResult
from prolepsis.core.kv_cache import DualKVCacheManager


# ════════════════════════════════════════════════════════════════
# Mock Model Wrappers
# ════════════════════════════════════════════════════════════════

class MockDraftWrapper:
  
    def __init__(self, vocab_size: int = 50, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.device = device
        self.past_key_values = None
        self.cache_len: int = 0
        self._last_token: Optional[torch.Tensor] = None
        
        # Mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = 0
        self.tokenizer.vocab_size = vocab_size
        self.tokenizer.encode = MagicMock(
            return_value=torch.tensor([[1, 2, 3]])
        )
        self.tokenizer.decode = MagicMock(return_value="mock output text")
        
        # Control what tokens the draft generates
        self._draft_queue = []
    
    def set_draft_tokens(self, tokens: list, probs: Optional[torch.Tensor] = None):
       
        if probs is None:
            probs = torch.zeros(len(tokens), self.vocab_size)
            for i, t in enumerate(tokens):
                # Concentrate probability on the configured token
                probs[i, t] = 0.8
                # Spread rest uniformly
                probs[i] += 0.2 / self.vocab_size
                probs[i] = probs[i] / probs[i].sum()
        
        self._draft_queue = list(zip(tokens, probs))
    
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.cache_len = input_ids.shape[1]
        self._last_token = input_ids[:, -1:]
        
        # Return logits (uniform)
        logits = torch.randn(1, self.vocab_size)
        return logits
    
    def decode_one(
        self,
        input_token=None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._draft_queue:
            token_id, prob = self._draft_queue.pop(0)
        else:
            # Default: random token
            token_id = torch.randint(1, self.vocab_size, (1,)).item()
            prob = torch.ones(self.vocab_size) / self.vocab_size
        
        token = torch.tensor([token_id])
        if input_token is not None:
            if input_token.dim() == 0:
                self.cache_len += 1
            elif input_token.dim() == 1:
                self.cache_len += input_token.shape[0]
            else:
                self.cache_len += input_token.shape[1]
        else:
            self.cache_len += 1
        
        self._last_token = token.unsqueeze(0)
        return token, prob
    
    def truncate_cache(self, new_len: int):
        """Truncate cache to new_len."""
        if new_len < self.cache_len:
            self.cache_len = new_len
    
    def reset(self):
        """Reset state."""
        self.past_key_values = None
        self.cache_len = 0
        self._last_token = None


class MockTargetWrapper:
    
    def __init__(self, vocab_size: int = 50, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.device = device
        self.past_key_values = None
        self.cache_len: int = 0
        
        # Mock model.config for vocab_size check
        self.model = MagicMock()
        self.model.config.vocab_size = vocab_size
        
        self._verify_logits = None
    
    def set_verify_logits(self, logits: torch.Tensor):
        self._verify_logits = logits
    
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Simulate prefill."""
        self.cache_len = input_ids.shape[1]
        return torch.randn(1, input_ids.shape[1], self.vocab_size)
    
    def verify_forward(self, draft_tokens: torch.Tensor) -> torch.Tensor:
        """Return pre-configured logits or uniform logits."""
        if draft_tokens.dim() == 1:
            gamma = draft_tokens.shape[0]
        else:
            gamma = draft_tokens.shape[1]
        
        self.cache_len += gamma
        
        if self._verify_logits is not None:
            return self._verify_logits
        
        # Default: uniform logits
        return torch.zeros(1, gamma, self.vocab_size)
    
    def truncate_cache(self, new_len: int):
        """Truncate cache."""
        if new_len < self.cache_len:
            self.cache_len = new_len
    
    def reset(self):
        """Reset state."""
        self.past_key_values = None
        self.cache_len = 0


# ════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════

class TestRejectionSamplerWithSync:
    
    @pytest.fixture
    def components(self):
        """Create sampler, KV manager, and mock wrappers."""
        vocab_size = 50
        return {
            "sampler": RejectionSampler(eos_token_id=0),
            "kv_manager": DualKVCacheManager(),
            "draft": MockDraftWrapper(vocab_size=vocab_size),
            "target": MockTargetWrapper(vocab_size=vocab_size),
            "vocab_size": vocab_size,
        }
    
    def test_sync_truncates_correctly_all_accepted(self, components):
        kv = components["kv_manager"]
        draft = components["draft"]
        target = components["target"]
        
        # Setup: pretend we prefilled 10 tokens
        kv.prefill_complete(10)
        draft.cache_len = 10
        target.cache_len = 10
        
        # Draft 4 tokens
        gamma = 4
        kv.after_drafting(gamma)
        draft.cache_len = 14  # 10 + 4
        target.cache_len = 14  # After verification
        
        # All 4 accepted
        kv.sync_after_acceptance(
            num_accepted=4,
            draft_wrapper=draft,
            target_wrapper=target,
            includes_bonus=False,
        )
        
        assert kv.sync_point == 14  # 10 + 4
        assert draft.cache_len == 14
        assert target.cache_len == 14
    
    def test_sync_truncates_on_partial_accept(self, components):
        """When 2 of 4 rejected, caches truncated to sync_point + 2."""
        kv = components["kv_manager"]
        draft = components["draft"]
        target = components["target"]
        
        kv.prefill_complete(10)
        draft.cache_len = 10
        target.cache_len = 10
        
        gamma = 4
        kv.after_drafting(gamma)
        draft.cache_len = 14
        target.cache_len = 14
        
        # Only 2 accepted
        kv.sync_after_acceptance(
            num_accepted=2,
            draft_wrapper=draft,
            target_wrapper=target,
            includes_bonus=False,
        )
        
        assert kv.sync_point == 12  # 10 + 2
        assert draft.cache_len == 12
        assert target.cache_len == 12
    
    def test_sync_includes_bonus_from_previous(self, components):
        kv = components["kv_manager"]
        draft = components["draft"]
        target = components["target"]
        
        # First iteration: prefill 10, accept 3
        kv.prefill_complete(10)
        kv.after_drafting(4)
        draft.cache_len = 14
        target.cache_len = 14
        
        kv.sync_after_acceptance(
            num_accepted=3,
            draft_wrapper=draft,
            target_wrapper=target,
            includes_bonus=False,
        )
        assert kv.sync_point == 13  # 10 + 3
        
        kv.after_drafting(4)
        draft.cache_len = 18  # 13 + 1(bonus) + 4
        target.cache_len = 18
        
        kv.sync_after_acceptance(
            num_accepted=2,
            draft_wrapper=draft,
            target_wrapper=target,
            includes_bonus=True,
        )
        
        # sync_point = 13 + 1(bonus) + 2(accepted) = 16
        assert kv.sync_point == 16
        assert draft.cache_len == 16
        assert target.cache_len == 16
    
    def test_acceptance_rate_accumulates(self, components):
        kv = components["kv_manager"]
        draft = components["draft"]
        target = components["target"]
        
        kv.prefill_complete(10)
        
        # Round 1: draft 4, accept 3
        kv.after_drafting(4)
        draft.cache_len = 14
        target.cache_len = 14
        kv.sync_after_acceptance(3, draft, target, includes_bonus=False)
        
        # Round 2: draft 4, accept 1
        kv.after_drafting(4)
        draft.cache_len = 18
        target.cache_len = 18
        kv.sync_after_acceptance(1, draft, target, includes_bonus=True)
        
        # Total: 4 accepted / 8 drafted = 0.5
        assert kv.get_acceptance_rate() == pytest.approx(0.5)


class TestRejectionSamplerEdgeCases:
    
    @pytest.fixture
    def sampler(self):
        return RejectionSampler(eos_token_id=0)
    
    def test_zero_draft_prob_accepts(self, sampler):
        """When draft prob ≈ 0 but target wants the token, should accept."""
        vocab_size = 10
        
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, 5] = 1e-15  # Essentially zero for token 5
        draft_probs[0, :5] = 0.2
        
        target_probs = torch.zeros(2, vocab_size)
        target_probs[0, 5] = 0.5
        target_probs[0, :5] = 0.1
        target_probs[1] = target_probs[0]
        
        draft_tokens = torch.tensor([5])
        
        result = sampler.sample(draft_tokens, draft_probs, target_probs)
        assert result.num_accepted == 1
    
    def test_identical_distributions_high_acceptance(self, sampler):
        vocab_size = 20
        gamma = 4
        
        torch.manual_seed(42)
        
        base_probs = torch.softmax(torch.randn(gamma, vocab_size), dim=-1)
        target_probs = torch.cat([base_probs, base_probs[-1:]], dim=0)
        
        acceptances = []
        for _ in range(50):
            draft_tokens = torch.multinomial(base_probs, num_samples=1).squeeze(-1)
            result = sampler.sample(draft_tokens, base_probs, target_probs)
            acceptances.append(result.num_accepted)
        
        avg_acceptance = sum(acceptances) / len(acceptances)
        assert avg_acceptance > gamma * 0.7  # ≥70% acceptance rate
    
    def test_single_token_gamma(self, sampler):
        """gamma=1 should still produce valid results."""
        vocab_size = 10
        
        draft_probs = torch.softmax(torch.randn(1, vocab_size), dim=-1)
        target_probs = torch.softmax(torch.randn(2, vocab_size), dim=-1)
        draft_tokens = torch.multinomial(draft_probs, num_samples=1).squeeze(-1)
        
        result = sampler.sample(draft_tokens, draft_probs, target_probs)
        
        assert result.num_accepted in [0, 1]
        if result.bonus_token is not None:
            assert result.bonus_token.shape == (1,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestTritonKernel:
    """Tests for the Triton kernel (skipped if Triton not available or if CUDA is unavailable)."""

    @pytest.fixture(autouse=True)
    def check_triton(self):
        """Skip all tests in this class if Triton is not available."""
        try:
            from prolepsis.kernels.verify_kernel import TRITON_AVAILABLE
            if not TRITON_AVAILABLE:
                pytest.skip("Triton not available")
        except ImportError:
            pytest.skip("Triton not available")

    def test_all_accepted(self):
        """Identical distributions → all accepted by kernel."""
        from prolepsis.kernels.verify_kernel import verify_speculative_tokens

        gamma = 4
        vocab_size = 100

        # Same distribution → ratio = 1 → always accept
        probs = torch.softmax(
            torch.randn(gamma, vocab_size, device="cuda"), dim=-1
        )
        target_probs = torch.cat(
            [probs, probs[-1:]], dim=0
        )

        draft_tokens = torch.multinomial(
            probs, num_samples=1
        ).squeeze(-1).to(torch.long)

        idx, bonus = verify_speculative_tokens(
            draft_tokens, probs, target_probs, eos_token_id=0
        )

        assert idx == gamma  # All accepted
        assert bonus is not None

    def test_all_rejected(self):
        from prolepsis.kernels.verify_kernel import verify_speculative_tokens

        gamma = 4
        vocab_size = 100

        # Draft concentrates on token 10
        draft_probs = torch.zeros(gamma, vocab_size, device="cuda")
        draft_probs[:, 10] = 1.0

        # Target concentrates on token 20
        target_probs = torch.zeros(gamma + 1, vocab_size, device="cuda")
        target_probs[:, 20] = 1.0

        draft_tokens = torch.full(
            (gamma,), 10, dtype=torch.long, device="cuda"
        )

        idx, bonus = verify_speculative_tokens(
            draft_tokens, draft_probs, target_probs, eos_token_id=0
        )

        assert idx == 0  # First position rejected
        assert bonus.item() == 20  # Residual is entirely token 20

    def test_matches_python_sampler(self):
        from prolepsis.kernels.verify_kernel import verify_speculative_tokens

        gamma = 4
        vocab_size = 50

        torch.manual_seed(123)

        # Create test distributions
        draft_probs = torch.softmax(
            torch.randn(gamma, vocab_size, device="cuda"), dim=-1
        )
        target_probs = torch.softmax(
            torch.randn(gamma + 1, vocab_size, device="cuda"), dim=-1
        )
        draft_tokens = torch.multinomial(
            draft_probs, num_samples=1
        ).squeeze(-1).to(torch.long)

        python_sampler = RejectionSampler(eos_token_id=0)

        kernel_rejections = []
        python_rejections = []

        for _ in range(100):
            k_idx, _ = verify_speculative_tokens(
                draft_tokens, draft_probs, target_probs, eos_token_id=0
            )
            kernel_rejections.append(k_idx)

            p_result = python_sampler.sample(
                draft_tokens.cpu(), draft_probs.cpu(), target_probs.cpu()
            )
            python_rejections.append(p_result.num_accepted)

        k_avg = sum(kernel_rejections) / len(kernel_rejections)
        p_avg = sum(python_rejections) / len(python_rejections)

        assert abs(k_avg - p_avg) < 1.5, (
            f"Kernel avg={k_avg:.2f} vs Python avg={p_avg:.2f} — too different"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])