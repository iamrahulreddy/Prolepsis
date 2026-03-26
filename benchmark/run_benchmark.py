"""CLI entry point for Prolepsis benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import torch


_PROMPT_BANK = {
    "systems": [
        "Explain how GPU memory bandwidth, KV-cache reuse, and kernel launch overhead interact during LLM inference.",
        (
            "You are debugging a serving regression. Symptoms: p95 latency doubled after a model rollout, "
            "GPU utilization dropped, and request volume stayed flat. Summarize likely causes and next checks."
        ),
        (
            "Compare static batching, continuous batching, and speculative decoding for an API that serves "
            "mixed prompt lengths."
        ),
    ],
    "coding": [
        "Write a Python function to merge overlapping intervals and explain its time and space complexity.",
        (
            "Review a function that mutates shared state from multiple threads and describe the most likely "
            "correctness risks."
        ),
        (
            "Design unit tests for a cache manager that supports prefill, truncate, and reset operations "
            "for two transformer models."
        ),
    ],
    "data": [
        (
            "Given quarterly revenue values of 42, 47, 51, and 58 million dollars, describe the trend and "
            "estimate the next quarter."
        ),
        (
            "Write a SQL query to return the top three customers by lifetime spending, including ties and the "
            "total spend per customer."
        ),
        (
            "Convert the following order records into JSON with fields order_id, region, total, and high_value: "
            "A17 west 129.50; B02 south 49.00; C81 west 310.25."
        ),
    ],
    "reasoning": [
        "Solve this step by step: if a model emits 128 tokens in 2.5 seconds, what is the throughput in tokens per second?",
        (
            "A service processes 900 requests per minute and fails 1.5 percent of them. Estimate how many failures "
            "occur in two hours and show the arithmetic."
        ),
        (
            "Compare the trade-offs between optimizing for throughput and optimizing for p95 latency in a GPU-backed "
            "inference service."
        ),
    ],
    "science": [
        "Compare photosynthesis and cellular respiration in a concise, structured explanation.",
        "Explain how CRISPR differs from traditional selective breeding in biology.",
        (
            "Summarize the greenhouse effect, then explain why aerosols and greenhouse gases do not affect climate "
            "in the same way."
        ),
    ],
    "communication": [
        "Draft a short customer-support reply to a user who was billed twice for the same subscription.",
        (
            "Write a concise release note announcing faster inference, cleaner artifacts, and improved benchmark "
            "reporting for an internal ML platform."
        ),
        (
            "Turn these meeting notes into a professional follow-up email with owners and next steps: migrate the "
            "benchmark runner, verify plots in cloud, and document remaining eval work."
        ),
    ],
    "planning": [
        "Outline a three-day study plan for preparing for a machine learning systems interview.",
        (
            "Create a one-week execution plan for hardening an inference prototype before sharing it in job "
            "applications."
        ),
        (
            "Given a deadline in ten days, propose a realistic sequence for benchmarking, validating outputs, "
            "and packaging results."
        ),
    ],
    "language": [
        "Translate this sentence into Spanish and explain one grammar choice: We finished the migration ahead of schedule.",
        (
            "Rewrite this sentence in simpler English without changing meaning: The deployment was deferred pending "
            "completion of dependency remediation."
        ),
        (
            "Classify the tone of this message and suggest a more professional alternative: This bug is killing us, "
            "can someone fix it today?"
        ),
    ],
    "history_and_policy": [
        "Explain the significance of the Battle of Plassey in Indian history.",
        (
            "Summarize the main causes and downstream effects of the 2008 global financial crisis."
        ),
        (
            "Write a short briefing on how rate limiting and authentication affect access to hosted model "
            "repositories."
        ),
    ],
    "long_context": [
        (
            "You are given release notes for an internal inference platform. Notes: benchmark runner now saves a "
            "dashboard image, JSON summary, and a per-prompt comparison log; telemetry is grouped under a dedicated "
            "directory; tests pass except a skipped real-model integration case; and the mixed prompt suite now "
            "covers systems, coding, planning, science, and support workflows. Summarize the release for engineers "
            "in four bullet points."
        ),
        (
            "Read these incident notes and produce a short postmortem summary. Notes: GPU utilization dropped after "
            "deploying a new decoding path; latency increased on mixed prompts but not on short prompts; model "
            "weights and driver versions were unchanged; visualizer outputs were saved successfully; and engineers "
            "suspect verification overhead is offsetting speculative gains. Include likely root cause, impact, and "
            "next action."
        ),
        (
            "A team compared three inference modes on the same model pair: standard autoregressive generation, a "
            "custom speculative decoder, and an assisted-generation baseline. The speculative path showed healthy "
            "token acceptance but only a small end-to-end speedup on a broad prompt set. Write a concise technical "
            "interpretation that explains why acceptance rate and throughput improvement are related but not identical."
        ),
    ],
    "product_and_ops": [
        "Write a short product description for a reusable steel water bottle aimed at college students.",
        (
            "Summarize the operational risks and mitigations when deploying an LLM service behind an API."
        ),
        (
            "Draft a troubleshooting checklist for a benchmark run that shows low GPU utilization and no speedup "
            "over baseline."
        ),
    ],
}


def _build_default_prompt_pool() -> List[str]:
    """Interleave prompt categories so the first N prompts stay domain-balanced."""
    prompt_groups = list(_PROMPT_BANK.values())
    max_group_size = max(len(group) for group in prompt_groups)

    prompts: List[str] = []
    for index in range(max_group_size):
        for group in prompt_groups:
            if index < len(group):
                prompts.append(group[index])
    return prompts


DEFAULT_BENCHMARK_PROMPTS = _build_default_prompt_pool()


def _normalize_device(device: str) -> str:
    """Normalize shorthand device strings for transformers loaders."""
    return "cuda:0" if device == "cuda" else device


def _cuda_index_from_device(device: str) -> Optional[int]:
    """Parse the CUDA device index from a normalized device string."""
    normalized_device = _normalize_device(device)
    if not normalized_device.startswith("cuda"):
        return None

    _, _, suffix = normalized_device.partition(":")
    if not suffix:
        return 0

    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Invalid CUDA device {device!r}. Expected 'cuda' or 'cuda:<index>'."
        ) from exc


def _validate_runtime_device(device: str) -> str:
    """Fail fast with a clear error when CUDA is requested but unavailable."""
    normalized_device = _normalize_device(device)
    if normalized_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested, but torch.cuda.is_available() is False. "
                "Run on a CUDA-enabled machine or set --device cpu."
            )

        device_index = _cuda_index_from_device(normalized_device)
        device_count = torch.cuda.device_count()
        if device_index is None or device_index < 0 or device_index >= device_count:
            raise RuntimeError(
                f"CUDA device {normalized_device!r} is unavailable. "
                f"Found {device_count} CUDA device(s)."
            )
    return normalized_device


def _parse_dtype(dtype_name: str) -> torch.dtype:
    """Convert CLI dtype name to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def _resolve_runtime_settings(
    device: str,
    dtype_name: str,
) -> tuple[str, str, torch.dtype, Optional[str]]:
    """Resolve device and dtype settings used for model loading."""
    resolved_device = _validate_runtime_device(device)
    resolved_dtype_name = dtype_name
    note = None

    if resolved_device.startswith("cpu") and dtype_name == "float16":
        resolved_dtype_name = "float32"
        note = (
            "CPU benchmarks use float32 by default because float16 model loading "
            "on CPU is not broadly supported."
        )

    return resolved_device, resolved_dtype_name, _parse_dtype(resolved_dtype_name), note


def _load_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load a causal LM with conservative transformers kwargs."""
    from transformers import AutoModelForCausalLM

    normalized_device = _validate_runtime_device(device)
    base_kwargs = {
        "trust_remote_code": True,
    }
    dtype_key_candidates = ("dtype", "torch_dtype")

    model = None
    last_error: Optional[Exception] = None
    for dtype_key in dtype_key_candidates:
        model_kwargs = dict(base_kwargs)
        model_kwargs[dtype_key] = dtype

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": normalized_device},
                **model_kwargs,
            )
            break
        except (TypeError, ValueError) as exc:
            last_error = exc

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            model.to(normalized_device)
            break
        except (TypeError, ValueError) as exc:
            last_error = exc

    if model is None:
        raise RuntimeError(
            f"Unable to load model {model_name!r} with a compatible dtype argument."
        ) from last_error

    model.eval()
    return model


def _try_command_output(command: List[str]) -> Optional[str]:
    """Run a lightweight system command and return its stdout if available."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _collect_system_info(
    requested_device: str,
    resolved_device: str,
    requested_dtype: str,
    resolved_dtype: str,
) -> Dict[str, object]:
    """Collect lightweight environment metadata for benchmark provenance."""
    info: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "requested_dtype": requested_dtype,
        "resolved_dtype": resolved_dtype,
    }

    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        device_index = _cuda_index_from_device(resolved_device)
        props = torch.cuda.get_device_properties(device_index)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["gpu_count"] = torch.cuda.device_count()
        info["selected_gpu_index"] = device_index

    nvidia_smi = _try_command_output(["nvidia-smi"])
    if nvidia_smi:
        info["nvidia_smi"] = nvidia_smi

    return info


def _format_system_info(info: Dict[str, object]) -> str:
    """Render system metadata as a readable text artifact."""
    lines = [
        f"Timestamp (UTC): {info.get('timestamp_utc', 'n/a')}",
        f"Platform: {info.get('platform', 'n/a')}",
        f"Python: {info.get('python_version', 'n/a')}",
        f"PyTorch: {info.get('torch_version', 'n/a')}",
        f"CUDA available: {info.get('cuda_available', 'n/a')}",
        f"CUDA version: {info.get('cuda_version', 'n/a')}",
        f"Requested device: {info.get('requested_device', 'n/a')}",
        f"Resolved device: {info.get('resolved_device', 'n/a')}",
        f"Requested dtype: {info.get('requested_dtype', 'n/a')}",
        f"Resolved dtype: {info.get('resolved_dtype', 'n/a')}",
    ]

    if "gpu_name" in info:
        lines.extend(
            [
                f"GPU index: {info['selected_gpu_index']}",
                f"GPU: {info['gpu_name']}",
                f"GPU memory (GB): {info['gpu_memory_gb']}",
                f"GPU count: {info['gpu_count']}",
            ]
        )

    if "nvidia_smi" in info:
        lines.extend(["", "nvidia-smi", "-----------", str(info["nvidia_smi"])])

    return "\n".join(lines) + "\n"


def _serialize_config(
    args: argparse.Namespace,
    prompts: List[str],
    save_dir: Path,
    trace_base: Optional[str],
    resolved_device: str,
    resolved_dtype: str,
) -> Dict[str, object]:
    """Capture the benchmark configuration for saved JSON artifacts."""
    return {
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "gamma": args.gamma,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "device": resolved_device,
        "dtype": resolved_dtype,
        "requested_device": args.device,
        "requested_dtype": args.dtype,
        "resolved_device": resolved_device,
        "resolved_dtype": resolved_dtype,
        "visualize_trace": args.visualize,
        "synchronize_timing": args.synchronize_timing,
        "include_assisted": args.include_assisted,
        "skip_baseline": args.skip_baseline,
        "prompt_file": args.prompt_file,
        "save_dir": str(save_dir),
        "trace_base": trace_base,
        "theme": args.theme,
        "detailed_plots": args.detailed_plots,
        "prompts": prompts,
    }


def _resolve_trace_base(
    log_file: Optional[str],
    visualize: bool,
    save_dir: Path,
) -> Optional[str]:
    """Resolve the speculative trace path relative to the benchmark save dir."""
    telemetry_dir = save_dir / "telemetry"
    if log_file:
        base_path = Path(log_file)
        if not base_path.is_absolute():
            base_path = save_dir / base_path
    elif visualize:
        base_path = telemetry_dir / "speculative_trace"
    else:
        return None

    return str(base_path)


def _get_trace_artifacts(trace_base: Optional[str]) -> List[str]:
    """Return the telemetry artifact paths produced for the benchmark run."""
    if not trace_base:
        return []

    base_path = Path(trace_base)
    if base_path.suffix in (".txt", ".jsonl"):
        base_path = base_path.with_suffix("")

    candidates = [
        Path(f"{base_path}.txt"),
        Path(f"{base_path}_acceptance.png"),
        Path(f"{base_path}_cumulative.png"),
    ]

    return [str(path) for path in candidates if path.exists()]


def _calculate_speedups(results: List[object]) -> Dict[str, Optional[float]]:
    """Calculate speedup ratios relative to the baseline throughput."""
    baseline = next((result for result in results if "Baseline" in result.method), None)
    if baseline is None or baseline.tokens_per_sec <= 0:
        return {}

    speedups: Dict[str, Optional[float]] = {}
    for result in results:
        if result.method == baseline.method:
            continue
        speedups[result.method] = result.tokens_per_sec / baseline.tokens_per_sec
    return speedups


def _save_text(path: Path, contents: str):
    """Save a UTF-8 text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _save_json(path: Path, payload: Dict[str, object]):
    """Save a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_prompts(prompt_file: Optional[str], num_prompts: int) -> List[str]:
    """Load prompts from a file or fall back to the built-in benchmark set."""
    if prompt_file is None:
        prompts = DEFAULT_BENCHMARK_PROMPTS
    else:
        path = Path(prompt_file)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                prompts = [str(item).strip() for item in payload if str(item).strip()]
            elif isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
                prompts = [
                    str(item).strip() for item in payload["prompts"] if str(item).strip()
                ]
            else:
                raise ValueError(
                    "JSON prompt file must be a list of strings or an object with a 'prompts' list."
                )
        else:
            prompts = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    if not prompts:
        raise ValueError("At least one prompt is required for benchmarking.")
    if num_prompts < 1:
        raise ValueError(f"num_prompts must be >= 1, got {num_prompts}")
    if len(prompts) < num_prompts:
        raise ValueError(
            f"Requested {num_prompts} prompts, but only {len(prompts)} prompt(s) are available."
        )

    return prompts[:num_prompts]


def _remove_stale_artifacts(save_dir: Path, detailed_plots: bool):
    """Remove old artifacts that are no longer produced by the current config."""
    stale_paths = [
        save_dir / "prompt_and_responses.txt",
        save_dir / "speculative_trace.txt",
        save_dir / "speculative_trace_acceptance.png",
        save_dir / "speculative_trace_cumulative.png",
    ]

    if not detailed_plots:
        stale_paths.extend(
            [
                save_dir / "benchmark_throughput.png",
                save_dir / "benchmark_latency.png",
                save_dir / "benchmark_memory.png",
            ]
        )

    for path in stale_paths:
        if path.exists():
            path.unlink()

    telemetry_dir = save_dir / "telemetry"
    if telemetry_dir.exists():
        shutil.rmtree(telemetry_dir)


def _run(args: argparse.Namespace, save_dir: Path):
    """Run the benchmark and write result artifacts."""
    _remove_stale_artifacts(save_dir, detailed_plots=args.detailed_plots)

    prompts = _load_prompts(args.prompt_file, args.num_prompts)

    trace_base = _resolve_trace_base(args.log_file, args.visualize, save_dir)
    resolved_device, resolved_dtype_name, torch_dtype, runtime_note = (
        _resolve_runtime_settings(args.device, args.dtype)
    )

    print("=" * 70)
    print("Prolepsis Benchmark")
    print("=" * 70)
    print(f"Draft model: {args.draft_model}")
    print(f"Target model: {args.target_model}")
    if args.device != resolved_device or args.dtype != resolved_dtype_name:
        print(f"Requested: device={args.device}  dtype={args.dtype}")
    print(f"Runtime: device={resolved_device}  dtype={resolved_dtype_name}")
    print(f"Sampling: gamma={args.gamma}  temp={args.temperature}  top_p={args.top_p}  top_k={args.top_k}")
    print(f"Prompts: {len(prompts)}  output_len={args.output_len}")
    if args.prompt_file:
        print(f"Prompt source: {args.prompt_file}")
    else:
        print(f"Prompt source: built-in mixed-domain set ({len(DEFAULT_BENCHMARK_PROMPTS)} prompts)")
    print(f"Runs per config: {args.runs} (warmup: {args.warmup})")
    print(f"Strict sync timing: {args.synchronize_timing}")
    print(f"Artifacts dir: {save_dir}")
    print("=" * 70)

    from prolepsis import SpeculativeConfig, SpeculativeDecoder
    from benchmark.harness import BenchmarkHarness

    harness = BenchmarkHarness(
        prompts=prompts,
        max_new_tokens=args.output_len,
        warmup_runs=args.warmup,
        num_runs=args.runs,
    )

    config = SpeculativeConfig(
        draft_model_name=args.draft_model,
        target_model_name=args.target_model,
        gamma=args.gamma,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.output_len,
        device=resolved_device,
        dtype=resolved_dtype_name,
        draft_quantization=args.draft_quantization,
        target_quantization=args.target_quantization,
        log_file_path=trace_base,
        enable_visualizer=args.visualize,
        synchronize_timing=args.synchronize_timing,
    )

    results = []

    if not args.skip_baseline:
        print("\nBenchmarking baseline (autoregressive)...")
        from transformers import AutoTokenizer

        baseline_model = _load_model(
            args.target_model,
            device=resolved_device,
            dtype=torch_dtype,
        )
        baseline_tokenizer = AutoTokenizer.from_pretrained(
            args.target_model,
            trust_remote_code=True,
        )

        baseline_result = harness.run_baseline(
            baseline_model,
            baseline_tokenizer,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
        results.append(baseline_result)
        harness.capture_baseline_responses(
            baseline_model,
            baseline_tokenizer,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nBenchmarking Prolepsis (speculative)...")
    decoder = SpeculativeDecoder(config)
    speculative_result = harness.run_speculative(decoder)
    results.append(speculative_result)
    harness.capture_speculative_responses(decoder)

    if args.include_assisted:
        print("\nBenchmarking HuggingFace assisted generation...")
        from transformers import AutoTokenizer

        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        target_hf = _load_model(
            args.target_model,
            device=resolved_device,
            dtype=torch_dtype,
        )
        tokenizer_hf = AutoTokenizer.from_pretrained(
            args.target_model,
            trust_remote_code=True,
        )
        assistant_hf = _load_model(
            args.draft_model,
            device=resolved_device,
            dtype=torch_dtype,
        )

        assisted_result = harness.run_assisted_generation(
            target_hf,
            assistant_hf,
            tokenizer_hf,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
        results.append(assisted_result)

        del target_hf, assistant_hf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_text = BenchmarkHarness.format_results(results)
    print("\n" + summary_text)

    system_info = _collect_system_info(
        requested_device=args.device,
        resolved_device=resolved_device,
        requested_dtype=args.dtype,
        resolved_dtype=resolved_dtype_name,
    )
    speedups = _calculate_speedups(results)

    output_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": _serialize_config(
            args,
            prompts,
            save_dir,
            trace_base,
            resolved_device,
            resolved_dtype_name,
        ),
        "system": system_info,
        "results": [result.to_dict() for result in results],
        "speedups": speedups,
    }

    summary_path = save_dir / "benchmark_summary.txt"
    system_info_path = save_dir / "system_info.txt"
    json_path = save_dir / "benchmark_results.json"
    prompt_response_path = save_dir / "prompt_and_responses.txt"

    _save_text(summary_path, summary_text + "\n")
    _save_text(system_info_path, _format_system_info(system_info))
    _save_json(json_path, output_payload)
    prompt_response_log = harness.format_prompt_response_log()
    if prompt_response_log:
        _save_text(prompt_response_path, prompt_response_log)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved system info to: {system_info_path}")
    print(f"Saved benchmark JSON to: {json_path}")
    if prompt_response_log:
        print(f"Saved prompt/response comparison to: {prompt_response_path}")

    telemetry_artifacts = _get_trace_artifacts(trace_base)
    if telemetry_artifacts:
        print("Saved telemetry artifacts:")
        for path in telemetry_artifacts:
            print(f"  {path}")

    if not args.no_plots:
        from prolepsis.utils.benchmark_visualization import create_benchmark_plots

        plot_paths = create_benchmark_plots(
            results,
            save_dir,
            theme=args.theme,
            detailed=args.detailed_plots,
        )
        if plot_paths:
            print("\nSaved benchmark plots:")
            for path in plot_paths:
                print(f"  {path}")
        else:
            print("\nBenchmark plots skipped (matplotlib unavailable).")


def main():
    parser = argparse.ArgumentParser(
        description="Run Prolepsis benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--draft-model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Draft model name or path",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Target model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision for benchmark loading; CPU float16 is promoted to float32",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=4,
        help="Speculation length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 disables it)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to benchmark",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional prompt file (.txt or .json) to use instead of the built-in prompt set",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Tokens to generate per prompt",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="light",
        choices=["light", "dark"],
        help="Theme for benchmark plots",
    )
    parser.add_argument(
        "--detailed-plots",
        action="store_true",
        help="Save individual throughput/latency/memory charts in addition to the dashboard",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline comparison",
    )
    parser.add_argument(
        "--include-assisted",
        action="store_true",
        help="Include HuggingFace assisted_generation comparison",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Base path for speculative event logs (relative paths go inside save-dir)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate speculative execution trace plots",
    )
    parser.add_argument(
        "--synchronize-timing",
        action="store_true",
        help="Synchronize CUDA around decoder generation for strict internal timing",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip benchmark comparison plot generation",
    )
    parser.add_argument(
        "--draft-quantization",
        type=str,
        default=None,
        choices=["int8", "int4"],
        help="Quantize the draft model (requires bitsandbytes)",
    )
    parser.add_argument(
        "--target-quantization",
        type=str,
        default=None,
        choices=["int8", "int4"],
        help="Quantize the target model (requires bitsandbytes)",
    )

    args = parser.parse_args()

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    benchmark_log_path = save_dir / "benchmark_log.txt"

    _run(args, save_dir)

if __name__ == "__main__":
    main()
