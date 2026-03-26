"""
Shared sampling utilities for speculative decoding.
"""

import torch
from torch import Tensor


def apply_sampling_filters(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tensor:

    if temperature == 0:
        indices = logits.argmax(dim=-1, keepdim=True)
        one_hot = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
        return one_hot

    scaled = logits if temperature == 1.0 else logits / temperature

    if top_k <= 0 and top_p >= 1.0:
        return torch.softmax(scaled, dim=-1)

    vocab_size = scaled.shape[-1]
    top_k = min(top_k, vocab_size) if top_k > 0 else 0

    top_k_logits = None
    top_k_indices = None

    if top_k > 0:
        if top_p < 1.0 and top_k < vocab_size:
            top_k_logits, top_k_indices = torch.topk(scaled, top_k, dim=-1)
        else:
            top_k_values = torch.topk(scaled, top_k, dim=-1)[0]
            threshold = top_k_values[..., -1:]
            indices_to_remove = scaled < threshold
            scaled = scaled.masked_fill(indices_to_remove, float("-inf"))

    if top_p < 1.0:
        if top_k_logits is not None and top_k_indices is not None:
            sorted_logits, sorted_positions = torch.sort(
                top_k_logits, descending=True, dim=-1
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            top_k_remove_mask = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_positions, src=sorted_indices_to_remove
            )
            top_k_logits = top_k_logits.masked_fill(top_k_remove_mask, float("-inf"))

            scaled = torch.full_like(scaled, float("-inf"))
            scaled.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
        else:
            sorted_logits, sorted_indices = torch.sort(
                scaled, descending=True, dim=-1
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            scaled = scaled.masked_fill(indices_to_remove, float("-inf"))

    return torch.softmax(scaled, dim=-1)
