from typing import Optional, Tuple

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from prolepsis.core.sampling_utils import apply_sampling_filters


def _normalize_device(device: str) -> str:
    return "cuda:0" if device == "cuda" else device


def _validate_runtime_device(device: str) -> str:
    normalized = _normalize_device(device)
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but torch.cuda.is_available() is False. "
            "Switch device to 'cpu' or ensure CUDA drivers/toolkits are visible."
        )
    return normalized


def _load_causal_lm(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    quantization: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, str]:
    target_device = _validate_runtime_device(device)
    
    kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    
    if quantization:
        q_mode = quantization.lower()
        if q_mode == "int8":
            kwargs["load_in_8bit"] = True
        elif q_mode == "int4":
            kwargs["load_in_4bit"] = True
        else:
            raise ValueError(f"Unsupported quantization precision: {quantization}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": target_device},
            **kwargs,
        )
    except (TypeError, ValueError):
        if quantization:
            kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            model.to(target_device)

    return model, target_device


class _KVCacheMixin:
    
    past_key_values: Optional[Tuple] = None
    cache_len: int = 0
    
    def truncate_cache(self, keep_len: int):
        if keep_len >= self.cache_len or self.past_key_values is None:
            return 
        
        # Modern HF DynamicCache (transformers >= 4.36)
        if hasattr(self.past_key_values, "crop"):
            self.past_key_values.crop(keep_len)
        else:
            # Legacy tuple-based cache
            # Expected shape per layer: [batch_size, num_heads, seq_len, head_dim]
            self.past_key_values = tuple(
                (k[:, :, :keep_len, :], v[:, :, :keep_len, :]) 
                for k, v in self.past_key_values
            )
        
        self.cache_len = keep_len
    
    def reset(self):
        """Purge execution state."""
        self.past_key_values = None
        self.cache_len = 0


class DraftModelWrapper(_KVCacheMixin):

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        quantization: Optional[str] = None,
    ):
        self.device = _normalize_device(device)
        self.dtype = dtype
        
        self.model, self.device = _load_causal_lm(model_name, self.device, dtype, quantization)
        self.model.eval()
        
        if hasattr(torch, "compile"):
            # Fuse operations to minimize Python overhead during the tight draft loop
            self.model = torch.compile(self.model, mode="default", dynamic=True)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._last_token: Optional[Tensor] = None
    
    @torch.inference_mode()
    def prefill(self, input_ids: Tensor) -> Tensor:
        """Process the prompt and establish the initial KV cache."""
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        
        self.past_key_values = outputs.past_key_values
        self.cache_len = input_ids.shape[1]
        self._last_token = input_ids[:, -1:]
        
        # Return logits strictly for the next decoding step
        return outputs.logits[:, -1, :]
    
    @torch.inference_mode()
    def decode_one(
        self,
        input_token: Optional[Tensor] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        token_to_process = input_token if input_token is not None else self._last_token
        
        if token_to_process.dim() == 0:
            token_to_process = token_to_process.view(1, 1)
        elif token_to_process.dim() == 1:
            token_to_process = token_to_process.unsqueeze(0)
            
        seq_len = token_to_process.shape[1]
        
        outputs = self.model(
            input_ids=token_to_process,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        
        self.past_key_values = outputs.past_key_values
        self.cache_len += seq_len
        
        logits = outputs.logits[:, -1, :]
        
        probs = apply_sampling_filters(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        
        sampled_token = torch.multinomial(probs.squeeze(0), num_samples=1)
        self._last_token = sampled_token.unsqueeze(0)
        
        return sampled_token, probs.squeeze(0)
    
    def reset(self):
        super().reset()
        self._last_token = None


class TargetModelWrapper(_KVCacheMixin):
  
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        quantization: Optional[str] = None,
    ):
        self.device = _normalize_device(device)
        self.dtype = dtype
        
        self.model, self.device = _load_causal_lm(model_name, self.device, dtype, quantization)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    @torch.inference_mode()
    def prefill(self, input_ids: Tensor) -> Tensor:
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        
        self.past_key_values = outputs.past_key_values
        self.cache_len = input_ids.shape[1]
        
        return outputs.logits
    
    @torch.inference_mode()
    def verify_forward(self, draft_tokens: Tensor) -> Tensor:
        if draft_tokens.dim() == 1:
            draft_tokens = draft_tokens.unsqueeze(0)
            
        gamma = draft_tokens.shape[1]
        
        outputs = self.model(
            input_ids=draft_tokens,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        
        self.past_key_values = outputs.past_key_values
        self.cache_len += gamma
        
        return outputs.logits