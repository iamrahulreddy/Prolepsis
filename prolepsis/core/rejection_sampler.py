"""
Modified rejection sampling for speculative decoding.

Implements the core sampling algorithm from:
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding"
- Chen et al. "Accelerating LLM Decoding with Speculative Sampling"
    
"""

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class RejectionResult(NamedTuple):
    """Result of rejection sampling."""

    accepted_tokens: Tensor
    num_accepted: int
    bonus_token: Optional[Tensor]  # [1] or None if EOS was accepted


class RejectionSampler:

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def sample(
        self,
        draft_tokens: Tensor,
        draft_probs: Tensor,
        target_probs: Tensor,
    ) -> RejectionResult:
    
        gamma = len(draft_tokens)
        device = draft_tokens.device

        draft_vocab = draft_probs.shape[-1]
        target_vocab = target_probs.shape[-1]
        if draft_vocab != target_vocab:
            max_vocab = max(draft_vocab, target_vocab)
            if draft_vocab < max_vocab:
                draft_probs = F.pad(
                    draft_probs, (0, max_vocab - draft_vocab), value=0.0
                )
            if target_vocab < max_vocab:
                target_probs = F.pad(
                    target_probs, (0, max_vocab - target_vocab), value=0.0
                )

        positions = torch.arange(gamma, device=device)
        token_ids = draft_tokens.long()

        # Vectorize accept/reject math to avoid repeated .item() syncs.
        p_draft = draft_probs[positions, token_ids]
        p_target = target_probs[positions, token_ids]
        accept_probs = torch.where(
            p_draft < 1e-10,
            torch.ones_like(p_draft),
            torch.clamp(p_target / p_draft.clamp(min=1e-10), max=1.0),
        )

        rejected_mask = torch.rand(gamma, device=device) >= accept_probs
        eos_accept_mask = (token_ids == self.eos_token_id) & (~rejected_mask)

        reject_positions = rejected_mask.nonzero(as_tuple=False)
        eos_positions = eos_accept_mask.nonzero(as_tuple=False)

        first_rejection_idx = (
            int(reject_positions[0].item()) if reject_positions.numel() > 0 else gamma
        )
        first_eos_idx = int(eos_positions[0].item()) if eos_positions.numel() > 0 else gamma

        if first_eos_idx < first_rejection_idx:
            accepted = draft_tokens[: first_eos_idx + 1]
            return RejectionResult(
                accepted_tokens=accepted,
                num_accepted=first_eos_idx + 1,
                bonus_token=None,
            )

        accepted_count = first_rejection_idx
        if accepted_count > 0:
            accepted_tokens = draft_tokens[:accepted_count]
        else:
            accepted_tokens = torch.tensor([], device=device, dtype=torch.long)

        if first_rejection_idx < gamma:
            bonus_token = self._sample_residual(
                draft_probs[first_rejection_idx],
                target_probs[first_rejection_idx],
            )
        else:
            bonus_token = self._sample_from_probs(target_probs[gamma])

        return RejectionResult(
            accepted_tokens=accepted_tokens,
            num_accepted=accepted_count,
            bonus_token=bonus_token.unsqueeze(0),
        )

    def _sample_residual(
        self,
        draft_probs: Tensor,
        target_probs: Tensor,
    ) -> Tensor:
        residual = torch.clamp(target_probs - draft_probs, min=0.0)
        residual_sum = residual.sum()

        if not torch.isfinite(residual_sum) or residual_sum <= 0:
            return self._sample_from_probs(target_probs)

        residual = residual / residual_sum
        return torch.multinomial(residual, num_samples=1).squeeze(-1)

    def _sample_from_probs(self, probs: Tensor) -> Tensor:
        """Sample a single token from a probability distribution."""
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        probs = probs.clamp(min=0.0)

        total = probs.sum()
        if total <= 0:

            fallback = torch.ones_like(probs, dtype=torch.float32)
            vocab_size = fallback.shape[-1]
            if 0 <= self.eos_token_id < vocab_size and vocab_size > 1:
                fallback[..., self.eos_token_id] = 0.0

            fallback_sum = fallback.sum(dim=-1, keepdim=True)
            if torch.any(fallback_sum <= 0):
                fallback = torch.ones_like(fallback)
                fallback_sum = fallback.sum(dim=-1, keepdim=True)

            fallback = fallback / fallback_sum
            return torch.multinomial(fallback, num_samples=1).squeeze(-1)

        probs = probs / total
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


