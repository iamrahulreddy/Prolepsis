import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeculativeConfig:
    
    draft_model_name: str = "Qwen/Qwen3-1.7B"
    target_model_name: str = "Qwen/Qwen3-8B"
    
    gamma: int = 4
    
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    
    eos_token_id: Optional[int] = None
    max_new_tokens: int = 256
    
    device: str = "cuda"
    dtype: str = "float16"
    
    draft_quantization: Optional[str] = None
    target_quantization: Optional[str] = None
    
    # System & Ops
    use_triton_kernel: bool = False
    log_file_path: Optional[str] = None
    enable_visualizer: bool = False
    synchronize_timing: bool = False
    
    def __post_init__(self):
        """Sanity check hyperparams before kicking off."""
        if self.gamma < 1:
            raise ValueError(f"Gamma must be >= 1. Got {self.gamma}.")
            
        if self.gamma > 16:
            warnings.warn(
                f"gamma={self.gamma} is unusually high. You'll likely hit diminishing returns. "
                "The sweet spot is usually between 4 and 8 depending on draft-target alignment."
            )
            
        if self.temperature < 0:
            raise ValueError(f"Temperature cannot be negative. Got {self.temperature}.")
            
        if not (0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1]. Got {self.top_p}.")
            
        if self.top_k < 0:
            raise ValueError(f"top_k cannot be negative. Got {self.top_k}.")

        # Quantization checks
        valid_quants = {None, "int8", "int4"}
        if self.draft_quantization not in valid_quants:
            raise ValueError(f"Unsupported draft quant: {self.draft_quantization}. Expected one of {valid_quants}.")
        if self.target_quantization not in valid_quants:
            raise ValueError(f"Unsupported target quant: {self.target_quantization}. Expected one of {valid_quants}.")

        if self.max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0. Got {self.max_new_tokens}.")
            
        valid_dtypes = {"float16", "bfloat16", "float32"}
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}. Got {self.dtype}.")
            
    def get_torch_dtype(self):
        import torch 
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map[self.dtype]
    
    def __repr__(self) -> str:
        return (
            f"SpeculativeConfig("
            f"draft='{self.draft_model_name}', "
            f"target='{self.target_model_name}', "
            f"gamma={self.gamma}, temp={self.temperature})"
        )