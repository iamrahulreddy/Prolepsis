"""
Benchmark harness for speculative decoding.

Provides utilities for measuring throughput, latency, and memory consumption 
across baseline autoregressive generation, standard HuggingFace assisted 
generation, and custom speculative decoding implementations.
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class BenchmarkResult:
    """Metrics and metadata for a single benchmark pass."""

    method: str
    num_prompts: int
    total_tokens: int
    total_time_sec: float
    median_time_sec: float
    time_std_sec: float
    tokens_per_sec: float
    throughput_std_tok_s: float
    avg_latency_ms: float
    
    memory_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None
    acceptance_rate: Optional[float] = None
    
    per_run_times_sec: List[float] = field(default_factory=list)
    per_run_tokens: List[int] = field(default_factory=list)
    per_run_throughput_tok_s: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"{self.method}: {self.tokens_per_sec:.1f} tok/s, {self.avg_latency_ms:.1f}ms latency"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics for downstream logging/dashboards."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PromptResponseRecord:
    """Captures I/O and timings for a specific prompt to verify transcript correctness."""

    prompt_index: int
    prompt: str
    response_text: str
    elapsed_sec: float
    generated_tokens: int


class BenchmarkHarness:
    """Orchestrates comparative benchmarking across generation strategies."""

    def __init__(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        warmup_runs: int = 2,
        num_runs: int = 3,
    ):
        if not prompts:
            raise ValueError("Must provide at least one prompt for benchmarking.")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0. Got {max_new_tokens}.")
        if warmup_runs < 0:
            raise ValueError(f"warmup_runs must be >= 0. Got {warmup_runs}.")
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1. Got {num_runs}.")

        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        
        self.prompt_response_records: Dict[str, List[PromptResponseRecord]] = {}
        self.comparison_seed_base = 1337

    def run_speculative(self, decoder: Any) -> BenchmarkResult:
        """Profile the custom speculative decoding implementation."""
        warmup_prompts = self.prompts[:3]
        
        # Warmup phase: pre-compile kernels and populate caches
        for _ in range(self.warmup_runs):
            decoder.reset_stats()
            for prompt in warmup_prompts:
                self._run_decoder_generation(
                    decoder,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    reset_logger=False,
                    render_visualizer=False,
                )

        decoder_logger = getattr(decoder, "logger", None)
        if decoder_logger:
            decoder_logger.reset()
            decoder_logger.log_benchmark_start(
                total_runs=self.num_runs,
                num_prompts=len(self.prompts),
                max_new_tokens=self.max_new_tokens,
            )

        self._reset_gpu_memory_stats()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        all_times: List[float] = []
        all_tokens: List[int] = []
        all_acceptance_rates: List[float] = []

        # Benchmark phase
        for run_index in range(self.num_runs):
            decoder.reset_stats()
            if decoder_logger:
                decoder_logger.log_run_start(run_index + 1, self.num_runs)

            start = time.perf_counter()
            for prompt_index, prompt in enumerate(self.prompts, start=1):
                if decoder_logger:
                    decoder_logger.log_prompt_start(prompt_index, len(self.prompts), prompt)
                
                self._run_decoder_generation(
                    decoder,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    reset_logger=False,
                    render_visualizer=False,
                )

            # Block CPU until GPU operations finish for accurate wall-clock timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()

            all_times.append(end - start)
            all_tokens.append(decoder.total_generated)
            all_acceptance_rates.append(decoder.get_acceptance_rate())

        render_visualizer = getattr(decoder, "render_visualizer", None)
        if decoder_logger and callable(render_visualizer):
            render_visualizer()

        return self._build_result(
            method="Prolepsis (speculative)",
            all_times=all_times,
            all_tokens=all_tokens,
            num_prompts=len(self.prompts),
            memory_stats=self._get_gpu_memory_stats(),
            acceptance_rates=all_acceptance_rates,
        )

    def run_baseline(
        self,
        model: Any,
        tokenizer: Any,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> BenchmarkResult:
        """Profile standard autoregressive generation."""
        warmup_prompts = self.prompts[:3]
        
        for _ in range(self.warmup_runs):
            for prompt in warmup_prompts:
                chat_prompt = self._apply_chat_template(tokenizer, prompt)
                inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
                model.generate(
                    **inputs,
                    **self._generation_kwargs(
                        max_new_tokens=self.max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    ),
                )

        self._reset_gpu_memory_stats()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        all_times: List[float] = []
        all_tokens: List[int] = []

        for _ in range(self.num_runs):
            total_new = 0
            start = time.perf_counter()
            
            for prompt in self.prompts:
                chat_prompt = self._apply_chat_template(tokenizer, prompt)
                inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[1]

                outputs = model.generate(
                    **inputs,
                    **self._generation_kwargs(
                        max_new_tokens=self.max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    ),
                )
                total_new += outputs.shape[1] - input_len

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            all_times.append(time.perf_counter() - start)
            all_tokens.append(total_new)

        return self._build_result(
            method="Baseline (autoregressive)",
            all_times=all_times,
            all_tokens=all_tokens,
            num_prompts=len(self.prompts),
            memory_stats=self._get_gpu_memory_stats(),
        )

    def run_assisted_generation(
        self,
        target_model: Any,
        assistant_model: Any,
        tokenizer: Any,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> BenchmarkResult:
        """Profile HuggingFace's native assisted generation."""
        warmup_prompts = self.prompts[:3]
        
        for _ in range(self.warmup_runs):
            for prompt in warmup_prompts:
                chat_prompt = self._apply_chat_template(tokenizer, prompt)
                inputs = tokenizer(chat_prompt, return_tensors="pt").to(target_model.device)
                target_model.generate(
                    **inputs,
                    **self._generation_kwargs(
                        max_new_tokens=self.max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        assistant_model=assistant_model,
                    ),
                )

        self._reset_gpu_memory_stats()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        all_times: List[float] = []
        all_tokens: List[int] = []

        for _ in range(self.num_runs):
            total_new = 0
            start = time.perf_counter()
            
            for prompt in self.prompts:
                chat_prompt = self._apply_chat_template(tokenizer, prompt)
                inputs = tokenizer(chat_prompt, return_tensors="pt").to(target_model.device)
                input_len = inputs["input_ids"].shape[1]

                outputs = target_model.generate(
                    **inputs,
                    **self._generation_kwargs(
                        max_new_tokens=self.max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        assistant_model=assistant_model,
                    ),
                )
                total_new += outputs.shape[1] - input_len

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            all_times.append(time.perf_counter() - start)
            all_tokens.append(total_new)

        return self._build_result(
            method="HuggingFace (assisted_generation)",
            all_times=all_times,
            all_tokens=all_tokens,
            num_prompts=len(self.prompts),
            memory_stats=self._get_gpu_memory_stats(),
        )

    @staticmethod
    def _reset_gpu_memory_stats():
        """Clear CUDA allocator High-Water Marks to isolate memory footprints per method."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _run_decoder_generation(
        decoder: Any,
        prompt: str,
        max_new_tokens: int,
        *,
        reset_logger: bool = True,
        render_visualizer: bool = True,
    ):
        """Dispatch generation using internal API if available, otherwise fallback to generic."""
        generate_ids = getattr(decoder, "generate_ids", None)
        if callable(generate_ids):
            return generate_ids(
                prompt,
                max_new_tokens=max_new_tokens,
                reset_logger=reset_logger,
                render_visualizer=render_visualizer,
            )
        return decoder.generate(prompt, max_new_tokens=max_new_tokens)

    @staticmethod
    def _build_result(
        method: str,
        all_times: List[float],
        all_tokens: List[int],
        num_prompts: int,
        memory_stats: Dict[str, Optional[float]],
        acceptance_rates: Optional[List[float]] = None,
    ) -> BenchmarkResult:
        """Aggregate cross-run metrics into a final BenchmarkResult object."""
        avg_time = statistics.mean(all_times)
        median_time = statistics.median(all_times)
        time_std = statistics.stdev(all_times) if len(all_times) > 1 else 0.0

        avg_tokens = statistics.mean(all_tokens)
        throughputs = [(t / e) if e > 0 else 0.0 for t, e in zip(all_tokens, all_times)]
        
        avg_throughput = statistics.mean(throughputs)
        throughput_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
        
        avg_latency_ms = (avg_time / avg_tokens) * 1000 if avg_tokens > 0 else 0.0
        avg_acceptance_rate = statistics.mean(acceptance_rates) if acceptance_rates else None

        return BenchmarkResult(
            method=method,
            num_prompts=num_prompts,
            total_tokens=int(avg_tokens),
            total_time_sec=avg_time,
            median_time_sec=median_time,
            time_std_sec=time_std,
            tokens_per_sec=avg_throughput,
            throughput_std_tok_s=throughput_std,
            avg_latency_ms=avg_latency_ms,
            memory_mb=memory_stats["allocated_mb"],
            memory_reserved_mb=memory_stats["reserved_mb"],
            acceptance_rate=avg_acceptance_rate,
            per_run_times_sec=list(all_times),
            per_run_tokens=list(all_tokens),
            per_run_throughput_tok_s=throughputs,
        )

    def _get_gpu_memory_stats(self) -> Dict[str, Optional[float]]:
        """Fetch peak VRAM allocated and reserved since the last reset."""
        if torch.cuda.is_available():
            mb = 1024 * 1024
            return {
                "allocated_mb": torch.cuda.max_memory_allocated() / mb,
                "reserved_mb": torch.cuda.max_memory_reserved() / mb,
            }
        return {"allocated_mb": None, "reserved_mb": None}

    @staticmethod
    def _generation_kwargs(
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        assistant_model: Any = None,
    ) -> Dict[str, Any]:
        """Construct consistent generation kwargs to ensure apples-to-apples comparisons."""
        kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        
        if assistant_model is not None:
            kwargs["assistant_model"] = assistant_model

        if temperature > 0:
            kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        else:
            kwargs["do_sample"] = False

        return kwargs

    def capture_baseline_responses(
        self,
        model: Any,
        tokenizer: Any,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> None:
        """Run a deterministic transcript pass to serve as the oracle for correctness."""
        records: List[PromptResponseRecord] = []

        for prompt_index, prompt in enumerate(self.prompts, start=1):
            self._set_comparison_seed(prompt_index)
            chat_prompt = self._apply_chat_template(tokenizer, prompt)
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                **self._generation_kwargs(
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            new_tokens = outputs.shape[1] - input_len
            records.append(
                PromptResponseRecord(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    response_text=self._decode_continuation(tokenizer, outputs[0], input_len),
                    elapsed_sec=elapsed,
                    generated_tokens=new_tokens,
                )
            )

        self.prompt_response_records["Baseline (autoregressive)"] = records

    def capture_speculative_responses(self, decoder: Any) -> None:
        """Run a seeded transcript pass for side-by-side comparison with the baseline."""
        records: List[PromptResponseRecord] = []
        original_logger = getattr(decoder, "logger", None)

        try:
            # Disable noisy logging during the transcript pass
            if hasattr(decoder, "logger"):
                decoder.logger = None

            for prompt_index, prompt in enumerate(self.prompts, start=1):
                self._set_comparison_seed(prompt_index)
                chat_prompt = self._apply_chat_template(decoder.tokenizer, prompt)
                input_len = decoder.tokenizer(chat_prompt, return_tensors="pt")["input_ids"].shape[1]

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                output_ids = self._run_decoder_generation(
                    decoder,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    reset_logger=False,
                    render_visualizer=False,
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                generated_tokens = output_ids.shape[1] - input_len
                records.append(
                    PromptResponseRecord(
                        prompt_index=prompt_index,
                        prompt=prompt,
                        response_text=self._decode_continuation(decoder.tokenizer, output_ids[0], input_len),
                        elapsed_sec=elapsed,
                        generated_tokens=generated_tokens,
                    )
                )
        finally:
            if hasattr(decoder, "logger"):
                decoder.logger = original_logger

        self.prompt_response_records["Prolepsis (speculative)"] = records

    def format_prompt_response_log(self) -> str:
        """Generate a scannable ASCII transcript comparing method outputs."""
        if not self.prompt_response_records:
            return ""

        ordered_methods = [
            m for m in (
                "Baseline (autoregressive)",
                "Prolepsis (speculative)",
                "HuggingFace (assisted_generation)",
            ) if m in self.prompt_response_records
        ]

        record_lookup = {
            method: {record.prompt_index: record for record in records}
            for method, records in self.prompt_response_records.items()
        }

        num_prompts = len(self.prompts)
        lines = [
            "=" * 78,
            "PROMPT & RESPONSE COMPARISON",
            "=" * 78,
            "",
            "Reproducible comparison pass per prompt. Profiled separately from the",
            "aggregate throughput benchmark to prevent transcript contamination.",
            f"Total prompts: {num_prompts}",
            "",
        ]

        for prompt_index, prompt in enumerate(self.prompts, start=1):
            header = f"  PROMPT {prompt_index} / {num_prompts}"
            lines.extend([
                "╔" + "═" * 76 + "╗",
                f"║{header:<76}║",
                "╚" + "═" * 76 + "╝",
                "",
                f"  \"{prompt}\"",
                ""
            ])

            baseline_record = record_lookup.get("Baseline (autoregressive)", {}).get(prompt_index)
            speculative_record = record_lookup.get("Prolepsis (speculative)", {}).get(prompt_index)

            for method in ordered_methods:
                record = record_lookup.get(method, {}).get(prompt_index)
                if not record:
                    continue

                tok_s = (record.generated_tokens / record.elapsed_sec) if record.elapsed_sec > 0 else 0.0
                method_label = method.upper()
                
                lines.append(f"  ── {method_label} " + "─" * max(0, 72 - len(method_label) - 5))
                lines.append(
                    f"  Time: {record.elapsed_sec:.3f}s  │  "
                    f"Tokens: {record.generated_tokens}  │  "
                    f"Tok/s: {tok_s:.1f}"
                )
                lines.append("")
                
                response_text = record.response_text or "(empty)"
                lines.extend(f"  {resp_line}" for resp_line in response_text.splitlines())
                lines.append("")

            if baseline_record and speculative_record and speculative_record.elapsed_sec > 0:
                speedup = baseline_record.elapsed_sec / speculative_record.elapsed_sec
                exact_match = baseline_record.response_text == speculative_record.response_text
                
                lines.extend([
                    "  ── COMPARISON " + "─" * 61,
                    f"  Speedup: {speedup:.2f}×  │  Exact match: {'yes' if exact_match else 'no'}",
                    "", ""
                ])

        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _decode_continuation(tokenizer: Any, token_ids: torch.Tensor, input_len: int) -> str:
        """Slice off the prompt and decode only the generated continuation."""
        continuation_ids = token_ids[input_len:].tolist()
        return tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
        """
        Safely inject the user prompt into the model's native chat template.
        Disables chain-of-thought tokens on newer unified models if possible.
        """
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "tokenize": False, 
            "add_generation_prompt": True
        }

        try:
            # First attempt: Pass enable_thinking flag (useful for models like Qwen3)
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError as e:
            # Fallback: Tokenizer doesn't support enable_thinking
            if "enable_thinking" in str(e) or "unexpected keyword" in str(e):
                try:
                    return tokenizer.apply_chat_template(messages, **kwargs)
                except Exception:
                    pass
        except Exception:
            pass
            
        # Hard fallback to raw prompt
        return prompt

    def _set_comparison_seed(self, prompt_index: int) -> None:
        """Pin the RNG state to ensure greedy sampling is fully deterministic."""
        seed = self.comparison_seed_base + prompt_index
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def format_results(results: List[BenchmarkResult]) -> str:
        """Compile a clean ASCII summary of the throughput benchmarks."""
        width = 70
        lines = ["=" * width, "Prolepsis Benchmark Results", "=" * width]

        for result in results:
            lines.extend([
                "",
                result.method,
                "-" * width,
                f"  Tokens           {result.total_tokens}",
                f"  Mean time        {result.total_time_sec:.2f} s",
                f"  Median time      {result.median_time_sec:.2f} s",
                f"  Time std         {result.time_std_sec:.3f} s",
                f"  Throughput       {result.tokens_per_sec:.1f} +/- {result.throughput_std_tok_s:.1f} tok/s",
                f"  Latency          {result.avg_latency_ms:.2f} ms/tok",
            ])
            if result.memory_mb is not None:
                lines.append(f"  GPU allocated    {result.memory_mb:.0f} MB")
            if result.memory_reserved_mb is not None:
                lines.append(f"  GPU reserved     {result.memory_reserved_mb:.0f} MB")
            if result.acceptance_rate is not None:
                lines.append(f"  Acceptance       {result.acceptance_rate:.1%}")

        baseline = next((r for r in results if "Baseline" in r.method), None)
        speculative = next((r for r in results if "speculative" in r.method), None)
        assisted = next((r for r in results if "assisted" in r.method), None)

        if len(results) >= 2:
            lines.extend(["", "-" * width, "Speedup Summary", "-" * width])

            if baseline and speculative and baseline.tokens_per_sec > 0:
                speedup = speculative.tokens_per_sec / baseline.tokens_per_sec
                verdict = "PASS" if speedup >= 1.0 else "CHECK"
                lines.append(f"  Prolepsis vs Baseline    {speedup:.2f}x  {verdict}")

            if baseline and assisted and baseline.tokens_per_sec > 0:
                speedup = assisted.tokens_per_sec / baseline.tokens_per_sec
                lines.append(f"  HF Assisted vs Baseline  {speedup:.2f}x")

            if speculative and assisted and assisted.tokens_per_sec > 0:
                ratio = speculative.tokens_per_sec / assisted.tokens_per_sec
                label = "faster" if ratio >= 1.0 else "slower"
                lines.append(f"  Prolepsis vs HF Assisted {ratio:.2f}x  ({label})")

        lines.extend(["", "=" * width])
        return "\n".join(lines)

    @staticmethod
    def print_results(results: List[BenchmarkResult]):
        """Print benchmark results to stdout."""
        print(BenchmarkHarness.format_results(results))