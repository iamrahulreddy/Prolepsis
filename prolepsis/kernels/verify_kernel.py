from typing import Tuple
import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _rejection_sampling_kernel(
        draft_tokens_ptr,
        draft_probs_ptr,
        target_probs_ptr,
        random_values_ptr,    
        gumbel_noise_ptr,     
        
        first_rejection_ptr,  
        bonus_token_ptr,      
        
        gamma: tl.constexpr,
        vocab_size: tl.constexpr,
        eos_token_id: tl.constexpr,
        
        BLOCK_VOCAB: tl.constexpr = 1024,
    ):
        first_rejection_idx = gamma
        eos_accepted = 0
        found = 0 
        
        for i in range(gamma):
            if found == 0:
                token_i = tl.load(draft_tokens_ptr + i)
                
                draft_prob = tl.load(draft_probs_ptr + i * vocab_size + token_i)
                target_prob = tl.load(target_probs_ptr + i * vocab_size + token_i)
                
                # compute min(1, P_target / P_draft) with numerical stability
                safe_draft = tl.maximum(draft_prob, 1e-10)
                accept_prob = tl.where(
                    draft_prob < 1e-10,
                    1.0,
                    tl.minimum(1.0, target_prob / safe_draft),
                )
                
                r = tl.load(random_values_ptr + i)
                
                if r >= accept_prob:
                    first_rejection_idx = i
                    found = 1
                
                # Halt if we accept an EOS token
                elif token_i == eos_token_id:
                    first_rejection_idx = i + 1 
                    eos_accepted = 1
                    found = 1
                
        tl.store(first_rejection_ptr, first_rejection_idx)
        
        if eos_accepted == 1:
            tl.store(bonus_token_ptr, eos_token_id)
            return
        
        if first_rejection_idx < gamma:
            sample_pos = first_rejection_idx
            use_residual = True
        else:
            sample_pos = gamma
            use_residual = False
            
        max_perturbed = -1e30
        max_token = 0
        
        max_target_p = -1.0
        fallback_token = 0
        
        offsets = tl.arange(0, BLOCK_VOCAB)
        
        for block_start in range(0, vocab_size, BLOCK_VOCAB):
            v_offsets = block_start + offsets
            mask = v_offsets < vocab_size
            
            target_p = tl.load(target_probs_ptr + sample_pos * vocab_size + v_offsets, mask=mask, other=0.0)
            
            if use_residual:
                draft_p = tl.load(draft_probs_ptr + sample_pos * vocab_size + v_offsets, mask=mask, other=0.0)
                prob = tl.maximum(0.0, target_p - draft_p)
            else:
                prob = target_p
                
            block_max_target = tl.max(target_p)
            if block_max_target > max_target_p:
                max_target_p = block_max_target
                fallback_token = block_start + tl.argmax(target_p, axis=0)
                
            is_valid = (prob > 0.0) & mask
            log_prob = tl.log(tl.maximum(prob, 1e-20))
            gumbel = tl.load(gumbel_noise_ptr + v_offsets, mask=mask, other=0.0)
            
            perturbed = tl.where(is_valid, log_prob + gumbel, -1e30)
            
            block_max_perturbed = tl.max(perturbed)
            if block_max_perturbed > max_perturbed:
                max_perturbed = block_max_perturbed
                max_token = block_start + tl.argmax(perturbed, axis=0)
        
        if max_perturbed == -1e30:
            max_token = fallback_token
            
        tl.store(bonus_token_ptr, max_token)

    def verify_speculative_tokens(
        draft_tokens: Tensor,
        draft_probs: Tensor,
        target_probs: Tensor,
        eos_token_id: int,
    ) -> Tuple[int, Tensor]:
        """
        Launch the fused Triton kernel to verify proposed tokens.
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available. Fallback to CPU/PyTorch RejectionSampler.")
        
        gamma = draft_tokens.shape[0]
        vocab_size = draft_probs.shape[1]
        device = draft_tokens.device
        
        random_values = torch.rand(gamma, device=device, dtype=torch.float32)
        
        uniform = torch.rand(vocab_size, device=device, dtype=torch.float32)
        uniform = uniform.clamp(min=1e-10, max=1.0 - 1e-7)
        gumbel_noise = -torch.log(-torch.log(uniform))
        
        first_rejection = torch.zeros(1, device=device, dtype=torch.int32)
        bonus_token = torch.zeros(1, device=device, dtype=torch.int64)
        
        _rejection_sampling_kernel[(1,)](
            draft_tokens,
            draft_probs,
            target_probs,
            random_values,
            gumbel_noise,
            first_rejection,
            bonus_token,
            gamma=gamma,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
        )
        
        return first_rejection.item(), bonus_token

else:
    def verify_speculative_tokens(
        draft_tokens: Tensor,
        draft_probs: Tensor,
        target_probs: Tensor,
        eos_token_id: int,
    ) -> Tuple[int, Tensor]:
        """Fallback stub when Triton is unavailable."""
        raise RuntimeError(
            "Triton is not available. Install with: `pip install triton>=2.1.0` "
            "or fallback to standard PyTorch sampling."
        )