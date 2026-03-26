"""
Main speculative decoding engine. This is the orchestrator that combines draft model, target model, rejection sampler, and KV cache management into a complete
speculative decoding pipeline.
"""

import logging
import time
import warnings
import torch
from torch import Tensor
from typing import Optional, List

from prolepsis.config import SpeculativeConfig
from prolepsis.models.wrapper import DraftModelWrapper, TargetModelWrapper
from prolepsis.core.rejection_sampler import RejectionSampler, RejectionResult
from prolepsis.core.kv_cache import DualKVCacheManager
from prolepsis.core.sampling_utils import apply_sampling_filters
from prolepsis.utils.logger import SpeculativeEventLogger as EventLogger

logger = logging.getLogger(__name__)

_TRITON_AVAILABLE = None

def _check_triton():
    """Lazy check for Triton availability."""
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            from prolepsis.kernels.verify_kernel import verify_speculative_tokens, TRITON_AVAILABLE
            _TRITON_AVAILABLE = TRITON_AVAILABLE
        except ImportError:
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


class SpeculativeDecoder:
    
    def __init__(self, config: SpeculativeConfig):
        self.config = config
        
        # Load models
        dtype = config.get_torch_dtype()
        
        logger.info("Loading draft model: %s", config.draft_model_name)
        self.draft = DraftModelWrapper(
            config.draft_model_name,
            device=config.device,
            dtype=dtype,
            quantization=config.draft_quantization,
        )
        
        logger.info("Loading target model: %s", config.target_model_name)
        self.target = TargetModelWrapper(
            config.target_model_name,
            device=config.device,
            dtype=dtype,
            quantization=config.target_quantization,
        )

        self.device = self.draft.device
        if self.target.device != self.device:
            raise RuntimeError(
                "Draft and target models were loaded onto different devices. "
                "Use a shared runtime device for speculative decoding."
            )
        
        if self.draft.tokenizer.get_vocab() != self.target.tokenizer.get_vocab():
            raise ValueError(
                "Draft and target tokenizers do not match. Speculative decoding "
                "requires identical token-to-id mappings."
            )

        target_vocab = self.target.model.config.vocab_size
        draft_vocab = self.draft.tokenizer.vocab_size
        if draft_vocab != target_vocab:
            warnings.warn(
                f"Draft vocab size ({draft_vocab}) != Target vocab size ({target_vocab}). "
                f"Using max vocab size ({max(draft_vocab, target_vocab)}) for sampling. "
                "This is expected when embedding matrices are padded differently."
            )
            self._effective_vocab_size = max(draft_vocab, target_vocab)
        else:
            self._effective_vocab_size = target_vocab

        self.tokenizer = self.draft.tokenizer
        
        if config.eos_token_id is not None:
            eos_token_id = config.eos_token_id
        else:
            eos_token_id = self.tokenizer.eos_token_id

        if eos_token_id is None:
            warnings.warn(
                "Tokenizer does not define an eos_token_id. "
                "EOS-based early stopping is disabled for this decoder."
            )
            eos_token_id = -1

        self.eos_token_id = int(eos_token_id)
        
        self.sampler = RejectionSampler(eos_token_id=self.eos_token_id)
        self.kv_manager = DualKVCacheManager()
        
        # Resolve Triton kernel usage
        self._use_triton = False
        if config.use_triton_kernel:
            if _check_triton():
                from prolepsis.kernels.verify_kernel import verify_speculative_tokens
                self._triton_verify = verify_speculative_tokens
                self._use_triton = True
            else:
                warnings.warn(
                    "use_triton_kernel=True but Triton is not available. "
                    "Falling back to Python RejectionSampler."
                )
        
        self.step_count: int = 0
        self._last_target_logits: Optional[Tensor] = None
        self.draft_prefill_logits: Optional[Tensor] = None
        
        self.total_generated: int = 0
        self.total_forward_passes: int = 0
        self.total_accepted_draft_tokens: int = 0
        self.total_drafted_tokens: int = 0
        
        self.logger: Optional[EventLogger] = None
        if config.log_file_path:
            self.logger = EventLogger(config.log_file_path)
            self.logger.log_config(
                draft_model=config.draft_model_name,
                target_model=config.target_model_name,
                gamma=config.gamma,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
            )
        
        self._generation_time_sec: float = 0.0
        self._sync_timing: bool = bool(config.synchronize_timing)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate decoded text using speculative decoding."""
        output_ids = self.generate_ids(prompt, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    @torch.inference_mode()
    def generate_ids(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        *,
        reset_logger: bool = True,
        render_visualizer: bool = True,
    ) -> Tensor:
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        elif max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

        if self.logger and reset_logger:
            # Default interactive usage treats each generate() call as an
            # independent run. Benchmark harnesses can disable this to collect
            # one aggregate telemetry stream across many prompts.
            self.logger.reset()
        
        # Tokenize prompt
        input_ids = self._prepare_input_ids(prompt)
        
        # Generate tokens with wall-clock timing
        if self._should_sync_timing():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        output_ids = self._generate_tokens(input_ids, max_new_tokens)
        
        if self._should_sync_timing():
            torch.cuda.synchronize()
        
        self._generation_time_sec += time.perf_counter() - start_time
    
        if render_visualizer:
            self.render_visualizer()

        if self.logger:
            self.logger.log_summary()

        return output_ids

    def render_visualizer(self):
        """Render charts for the currently buffered event log, if enabled."""
        if not self.logger or not self.config.enable_visualizer:
            return None

        from prolepsis.utils.visualizer import SpeculativeVisualizer

        config_label = (
            f"{self.config.draft_model_name} → {self.config.target_model_name}"
            f"  |  γ={self.config.gamma}  |  T={self.config.temperature}"
        )
        vis = SpeculativeVisualizer(
            self.logger.iterations,
            self.config.log_file_path,
            config_label=config_label,
        )
        return vis.generate_dashboard()

    def _prepare_input_ids(self, prompt: str) -> Tensor:
        """Tokenize a prompt using the model's chat template.

        Wraps the raw prompt string in ChatML format so instruction-tuned
        models (e.g. Qwen3) produce answers rather than text completions.
        Thinking mode is disabled to avoid wasting output tokens on
        chain-of-thought reasoning.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError as e:
            if "enable_thinking" in str(e) or "unexpected keyword" in str(e):
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    text = prompt
            else:
                text = prompt
        except Exception:
            text = prompt

        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        if input_ids.shape[1] == 0:
            bos_token_id = self.tokenizer.bos_token_id
            if bos_token_id is None:
                raise ValueError(
                    "Prompt tokenized to an empty sequence and tokenizer has no "
                    "bos_token_id. Provide a non-empty prompt or use a tokenizer "
                    "that defines a BOS token."
                )
            input_ids = torch.tensor([[bos_token_id]], dtype=torch.long)

        return input_ids.to(self.device)

    def _should_sync_timing(self) -> bool:
        """Whether to synchronize CUDA for strict timing measurements."""
        return (
            self._sync_timing
            and self._device_is_cuda()
            and torch.cuda.is_available()
        )

    def _device_is_cuda(self) -> bool:
        """Return whether the resolved runtime device is CUDA-backed."""
        if isinstance(self.device, torch.device):
            return self.device.type == "cuda"
        return str(self.device).startswith("cuda")

    def _decode_token_batch(self, token_ids: Tensor) -> List[str]:
        """Decode a 1D tensor of token ids in one tokenizer call."""
        if token_ids.numel() == 0:
            return []
        token_batch = token_ids.reshape(-1, 1).detach().to("cpu").tolist()
        return self.tokenizer.batch_decode(token_batch, skip_special_tokens=False)
    
    def _generate_tokens(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
    ) -> Tensor:

        prompt_len = input_ids.shape[1]
        
        # Reset state
        self.draft.reset()
        self.target.reset()
        self.kv_manager.reset()
        self.step_count = 0
        
        # Both models process the prompt
        t0_draft = time.perf_counter()
        self.draft_prefill_logits = self.draft.prefill(input_ids)
        t1_draft = time.perf_counter()
        
        t0_target = time.perf_counter()
        target_logits = self.target.prefill(input_ids)
        t1_target = time.perf_counter()
        
        self.kv_manager.prefill_complete(prompt_len)
        
        if self.logger:
            self.logger.log_prefill(
                prompt_len, 
                (t1_target - t0_target) * 1000, 
                (t1_draft - t0_draft) * 1000
            )
        
        # Store logits from prefill for the FIRST verification step only.
        self._last_target_logits = target_logits[:, -1, :]
        
        # Track last token for drafting (initially last prompt token)
        last_token = input_ids[0, -1:]  # [1] (1D, not scalar)
        
        # Output accumulator
        generated = input_ids.clone()
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            # Calculate how many tokens we can still generate
            remaining = max_new_tokens - tokens_generated
            gamma = min(self.config.gamma, remaining)
            
            if self.logger:
                self.logger.log_step_start(self.step_count)
                
            t0_draft = time.perf_counter()
            draft_tokens, draft_probs = self._draft_tokens(gamma, last_token, generated)
            t1_draft = time.perf_counter()
            draft_time_ms = (t1_draft - t0_draft) * 1000
            
            if self.logger:
                draft_str_list = self._decode_token_batch(draft_tokens)
                self.logger.log_draft_results(draft_str_list, draft_time_ms)
            
            if self.step_count == 0:
                self.kv_manager.after_drafting(gamma - 1, drafted_tokens=gamma)
            else:
                self.kv_manager.after_drafting(gamma, drafted_tokens=gamma)
            
            t0_verify = time.perf_counter()
            target_probs = self._verify_tokens(
                draft_tokens, last_token
            )
            t1_verify = time.perf_counter()
            verify_time_ms = (t1_verify - t0_verify) * 1000
            
            self.step_count += 1
            self.total_forward_passes += 1
            
            result = self._run_rejection_sampling(
                draft_tokens, draft_probs, target_probs
            )
            
            # Combine accepted tokens + bonus token
            if result.bonus_token is not None:
                new_tokens = torch.cat([result.accepted_tokens, result.bonus_token])
            else:
                # EOS accepted - no bonus token
                new_tokens = result.accepted_tokens

            emitted_tokens = new_tokens[:remaining]
            
            if self.logger:
                accepted_strs = self._decode_token_batch(result.accepted_tokens)
                rejected_str = None
                if (
                    result.bonus_token is not None
                    and result.num_accepted < gamma
                    and result.num_accepted < len(draft_tokens)
                ):
                    rejected_tokens = draft_tokens[result.num_accepted : result.num_accepted + 1]
                    rejected_str = self._decode_token_batch(rejected_tokens)[0]
                bonus_str = None
                if result.bonus_token is not None:
                    bonus_str = self._decode_token_batch(result.bonus_token)[0]
                
                self.logger.log_verify_results(
                    accepted_tokens=accepted_strs,
                    rejected_token=rejected_str,
                    bonus_token=bonus_str,
                    bonus_from_residual=(
                        result.bonus_token is not None and result.num_accepted < gamma
                    ),
                    verify_time_ms=verify_time_ms
                )
            
            # Update generated sequence
            generated = torch.cat([
                generated,
                emitted_tokens.unsqueeze(0)
            ], dim=1)
            tokens_generated += len(emitted_tokens)
            
            if len(emitted_tokens) > 0:
                last_token = emitted_tokens[-1:]  # [1] (keep 1D, not scalar)
            
            # Update statistics
            self.total_generated += len(emitted_tokens)
            self.total_accepted_draft_tokens += result.num_accepted
            self.total_drafted_tokens += gamma
            
            self.kv_manager.sync_after_acceptance(
                num_accepted=result.num_accepted,
                draft_wrapper=self.draft,
                target_wrapper=self.target,
                includes_bonus=(self.step_count > 1),
            )
            
            if self.logger:
                self.logger.log_step_end(len(emitted_tokens), self.kv_manager.sync_point)
                    
            # After that, we always pass [bonus, draft_tokens] to target
            # and get fresh logits. Clear to free memory.
            if self.step_count == 1:
                self._last_target_logits = None
                self.draft_prefill_logits = None
            
            if result.bonus_token is None:
                # EOS was accepted - stop generation
                break
            
            if len(emitted_tokens) > 0 and last_token.item() == self.eos_token_id:
                break
        
        return generated[:, :prompt_len + max_new_tokens]
    
    def _draft_tokens(
        self,
        gamma: int,
        last_token: Tensor,
        generated: Tensor,
    ) -> tuple[Tensor, Tensor]:

        tokens: List[Tensor] = []
        probs: List[Tensor] = []
        
        if self.step_count == 0:
            p = apply_sampling_filters(
                self.draft_prefill_logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            if self.config.temperature == 0:
                token = p.squeeze(0).argmax(dim=-1, keepdim=True)
            else:
                token = torch.multinomial(p.squeeze(0), num_samples=1)
            
            self.draft._last_token = token.unsqueeze(0) if token.dim() == 0 else token
            
            tokens.append(token.squeeze())
            probs.append(p.squeeze())
            
            current_token = token
            num_decodes = gamma - 1
        else:
            current_token = generated[:, self.draft.cache_len:]
            num_decodes = gamma
        
        for _ in range(num_decodes):
            token, prob = self.draft.decode_one(
                input_token=current_token,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
            
            tokens.append(token.squeeze())
            probs.append(prob.squeeze())
            current_token = token
            
        return torch.stack(tokens), torch.stack(probs)
    
    def _verify_tokens(
        self,
        draft_tokens: Tensor,
        last_token: Tensor,
    ) -> Tensor:

        if self.step_count == 0:
            draft_logits = self.target.verify_forward(draft_tokens)
            
            full_logits = torch.cat([
                self._last_target_logits.unsqueeze(1),
                draft_logits
            ], dim=1)
        else:
            full_input = torch.cat([
                last_token.view(1),
                draft_tokens
            ])
            
            full_logits = self.target.verify_forward(full_input)
        
        full_logits = full_logits.squeeze(0)
        
        probs = apply_sampling_filters(
            full_logits,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )
        
        return probs
    
    def _run_rejection_sampling(
        self,
        draft_tokens: Tensor,
        draft_probs: Tensor,
        target_probs: Tensor,
    ) -> RejectionResult:
        
        if self._use_triton:
            draft_probs, target_probs = self._align_vocab_for_triton(
                draft_probs,
                target_probs,
            )
            first_rejection_idx, bonus_token = self._triton_verify(
                draft_tokens=draft_tokens,
                draft_probs=draft_probs,
                target_probs=target_probs,
                eos_token_id=self.eos_token_id,
            )
            
            if first_rejection_idx > 0:
                accepted_tokens = draft_tokens[:first_rejection_idx]
            else:
                accepted_tokens = torch.tensor(
                    [], device=draft_tokens.device, dtype=torch.long
                )
            
            eos_accepted = (
                first_rejection_idx > 0
                and accepted_tokens[-1].item() == self.eos_token_id
            )
            
            if eos_accepted:
                return RejectionResult(
                    accepted_tokens=accepted_tokens,
                    num_accepted=first_rejection_idx,
                    bonus_token=None,
                )
            
            return RejectionResult(
                accepted_tokens=accepted_tokens,
                num_accepted=first_rejection_idx,
                bonus_token=bonus_token,
            )
        else:
            return self.sampler.sample(
                draft_tokens=draft_tokens,
                draft_probs=draft_probs,
                target_probs=target_probs,
            )

    @staticmethod
    def _align_vocab_for_triton(
        draft_probs: Tensor,
        target_probs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pad vocab dimensions for Triton kernels that require exact alignment."""
        import torch.nn.functional as F

        draft_vocab = draft_probs.shape[-1]
        target_vocab = target_probs.shape[-1]
        if draft_vocab == target_vocab:
            return draft_probs, target_probs

        max_vocab = max(draft_vocab, target_vocab)
        if draft_vocab < max_vocab:
            draft_probs = F.pad(
                draft_probs,
                (0, max_vocab - draft_vocab),
                value=0.0,
            )
        if target_vocab < max_vocab:
            target_probs = F.pad(
                target_probs,
                (0, max_vocab - target_vocab),
                value=0.0,
            )
        return draft_probs, target_probs
    
    def get_acceptance_rate(self) -> float:
        """Get overall acceptance rate."""
        if self.total_drafted_tokens == 0:
            return 0.0
        return self.total_accepted_draft_tokens / self.total_drafted_tokens
    
    def get_tokens_per_pass(self) -> float:
        
        if self.total_forward_passes == 0:
            return 1.0
        return self.total_generated / self.total_forward_passes
    

    def get_generation_time(self) -> float:
        """
        Get total wall-clock generation time in seconds.
        On CUDA, this is strictly synchronized only when config.synchronize_timing is enabled.
        """
        return self._generation_time_sec
    
    def get_stats(self) -> dict:
        """Get generation statistics."""
        return {
            "total_generated": self.total_generated,
            "total_forward_passes": self.total_forward_passes,
            "acceptance_rate": self.get_acceptance_rate(),
            "tokens_per_pass": self.get_tokens_per_pass(),
            "generation_time_sec": self._generation_time_sec,
            "synchronize_timing": self._sync_timing,
            "gamma": self.config.gamma,
            "use_triton_kernel": self._use_triton,
        }
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.total_generated = 0
        self.total_forward_passes = 0
        self.total_accepted_draft_tokens = 0
        self.total_drafted_tokens = 0
        self._generation_time_sec = 0.0
        self.kv_manager.reset()
