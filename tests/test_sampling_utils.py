import pytest
import torch

from prolepsis.core.sampling_utils import apply_sampling_filters


class TestApplySamplingFilters:
    """Test suite for the shared apply_sampling_filters utility."""

    def test_identity_with_defaults(self):
        """Default params (temp=1, no top-k, no top-p) should just be softmax."""
        logits = torch.randn(1, 50)
        probs = apply_sampling_filters(logits)

        expected = torch.softmax(logits, dim=-1)
        torch.testing.assert_close(probs, expected)

    def test_temperature_sharpens(self):
        """Lower temperature should make distribution more peaked."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])

        probs_hot = apply_sampling_filters(logits, temperature=2.0)
        probs_cold = apply_sampling_filters(logits, temperature=0.1)

        # Cold temperature should make argmax token dominate
        assert probs_cold[0, 2] > probs_hot[0, 2]
        # Entropy should be lower with cold temperature
        entropy_hot = -(probs_hot * probs_hot.log()).sum()
        entropy_cold = -(probs_cold * probs_cold.clamp(min=1e-10).log()).sum()
        assert entropy_cold < entropy_hot

    def test_top_k_masks_tokens(self):
        """Top-k should zero out all but top k tokens."""
        logits = torch.tensor([[5.0, 3.0, 1.0, 0.5, 0.1]])
        probs = apply_sampling_filters(logits, top_k=2)

        # Only the top 2 should have nonzero probability
        assert probs[0, 0] > 0  # highest logit
        assert probs[0, 1] > 0  # second highest
        assert probs[0, 2].item() == pytest.approx(0.0, abs=1e-7)
        assert probs[0, 3].item() == pytest.approx(0.0, abs=1e-7)
        assert probs[0, 4].item() == pytest.approx(0.0, abs=1e-7)

    def test_top_p_nucleus(self):
        """Top-p should keep the smallest set exceeding the threshold."""
        # Create distribution where token 0 has ~0.9 probability
        logits = torch.tensor([[10.0, 1.0, 0.5, 0.1, 0.01]])
        probs = apply_sampling_filters(logits, top_p=0.95)

        # Token 0 should definitely be kept, low-probability tokens may be masked
        assert probs[0, 0] > 0
        # Sum should be 1.0
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_top_p_always_keeps_top_1(self):
        """Top-p=0.01 should still keep at least the top token."""
        logits = torch.tensor([[5.0, 3.0, 1.0]])
        probs = apply_sampling_filters(logits, top_p=0.01)

        # The top token must survive
        assert probs[0, 0] > 0
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_combined_top_k_and_top_p(self):
        """Both top-k and top-p should apply in sequence."""
        logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        probs = apply_sampling_filters(logits, top_k=3, top_p=0.8)

        # top-k=3 keeps tokens 0,1,2; then top-p further narrows
        assert probs[0, 3].item() == pytest.approx(0.0, abs=1e-7)
        assert probs[0, 4].item() == pytest.approx(0.0, abs=1e-7)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_batched_input(self):
        """Should work with batched inputs [..., vocab_size]."""
        logits = torch.randn(4, 100)
        probs = apply_sampling_filters(logits, temperature=0.8, top_k=10)

        assert probs.shape == (4, 100)
        # Each row should sum to 1
        for i in range(4):
            assert probs[i].sum().item() == pytest.approx(1.0, abs=1e-5)
            # Exactly 10 nonzero entries per row
            assert (probs[i] > 1e-8).sum().item() == 10

    def test_2d_batched_input(self):
        """Should work with [batch, seq, vocab] shaped inputs."""
        logits = torch.randn(2, 5, 50)
        probs = apply_sampling_filters(logits, top_k=5)

        assert probs.shape == (2, 5, 50)
        # Each position should sum to 1
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)

    def test_probabilities_sum_to_one(self):
        """Output should always be a valid probability distribution."""
        logits = torch.randn(1, 1000)
        probs = apply_sampling_filters(logits, temperature=0.5, top_k=50, top_p=0.9)

        assert probs.sum().item() == pytest.approx(1.0, abs=1e-4)
        assert (probs >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
