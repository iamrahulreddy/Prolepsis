import pytest
import torch


class TestRejectionSampler:
    """Test suite for RejectionSampler."""
    
    @pytest.fixture
    def sampler(self):
        from prolepsis.core.rejection_sampler import RejectionSampler
        return RejectionSampler(eos_token_id=0)
    
    def test_all_accepted(self, sampler):
        """When draft == target, all tokens should be accepted."""
        gamma = 4
        vocab_size = 100
        
        # Create identical distributions
        probs = torch.softmax(torch.randn(gamma, vocab_size), dim=-1)
        
        # Generate tokens from the distribution
        draft_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Same probs for both (except target has gamma+1 for bonus)
        draft_probs = probs
        target_probs = torch.cat([probs, probs[-1:]], dim=0)
        
        result = sampler.sample(draft_tokens, draft_probs, target_probs)
        
        # All should be accepted (with high probability)
        assert result.num_accepted == gamma
        assert result.bonus_token is not None
    
    def test_all_rejected(self, sampler):
        """When target strongly disagrees, tokens should be rejected."""
        gamma = 4
        vocab_size = 100
        
        # Draft concentrates probability on token 10
        draft_probs = torch.zeros(gamma, vocab_size)
        draft_probs[:, 10] = 1.0
        
        # Target concentrates probability on token 20
        target_probs = torch.zeros(gamma + 1, vocab_size)
        target_probs[:, 20] = 1.0
        
        # Draft tokens are all 10
        draft_tokens = torch.full((gamma,), 10, dtype=torch.long)
        
        result = sampler.sample(draft_tokens, draft_probs, target_probs)
        
        # First token should be rejected (p_target[10] = 0)
        assert result.num_accepted == 0
        # Bonus should be from residual (which is token 20)
        assert result.bonus_token.item() == 20
    
    def test_eos_stops_generation(self, sampler):
        """EOS token acceptance should set bonus_token to None."""
        gamma = 4
        vocab_size = 100
        
        # Uniform distribution (high acceptance)
        probs = torch.ones(gamma, vocab_size) / vocab_size
        target_probs = torch.ones(gamma + 1, vocab_size) / vocab_size
        
        # First draft token is EOS (token 0)
        draft_tokens = torch.zeros(gamma, dtype=torch.long)
        draft_tokens[0] = 0  # EOS
        
        result = sampler.sample(draft_tokens, probs, target_probs)
        
        # EOS accepted, bonus_token should be None
        assert 0 in result.accepted_tokens
        assert result.bonus_token is None
    
    def test_residual_distribution_correct(self, sampler):
        """Residual distribution should be max(0, p_target - p_draft)."""
        vocab_size = 10
        
        # Draft: [0.8, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0]
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, :4] = torch.tensor([0.8, 0.1, 0.05, 0.05])
        
        # Target: [0.2, 0.3, 0.2, 0.2, 0.1, 0, 0, 0, 0, 0]
        target_probs = torch.zeros(2, vocab_size)
        target_probs[0, :5] = torch.tensor([0.2, 0.3, 0.2, 0.2, 0.1])
        target_probs[1, :5] = torch.tensor([0.2, 0.3, 0.2, 0.2, 0.1])
        
        # Force rejection by having draft sample high-prob-draft token
        draft_tokens = torch.tensor([0])  # Token 0 has 0.8 draft, 0.2 target
        
        # Run many times to check residual distribution
        residual_samples = []
        for _ in range(100):
            result = sampler.sample(draft_tokens, draft_probs, target_probs)
            if result.bonus_token is not None:
                residual_samples.append(result.bonus_token.item())
        
        # Residual = max(0, [0.2-0.8, 0.3-0.1, 0.2-0.05, 0.2-0.05, 0.1-0])
        #          = [0, 0.2, 0.15, 0.15, 0.1]
        # Token 0 should never be sampled from residual
        assert 0 not in residual_samples or len([x for x in residual_samples if x == 0]) < 5

    def test_zero_distribution_fallback_avoids_eos_bias(self, sampler):
        """Degenerate all-zero distributions should not deterministically pick EOS."""
        probs = torch.zeros(8)

        samples = {sampler._sample_from_probs(probs).item() for _ in range(20)}

        assert 0 not in samples
        assert samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
