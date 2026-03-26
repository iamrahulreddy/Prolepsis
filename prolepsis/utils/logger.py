import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

_W = 72  # column width for separators


class SpeculativeEventLogger:

    def __init__(self, base_file_path: str):
        path = Path(base_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in (".txt", ".jsonl"):
            base = str(path.with_suffix(""))
        else:
            base = str(path)

        self.txt_path = f"{base}.txt"

        self._log = logging.getLogger(f"prolepsis.event.{base}")
        self._log.setLevel(logging.INFO)
        self._log.propagate = False

        self.iterations: List[Dict[str, Any]] = []
        self._current: Dict[str, Any] = {}
        self._run_index: Optional[int] = None
        self._prompt_index: Optional[int] = None
        self._prompt_preview: Optional[str] = None

        # Running counters for summary
        self._total_steps = 0
        self._total_accepted = 0
        self._total_rejected = 0
        self._total_bonus = 0
        self._total_draft_ms = 0.0
        self._total_verify_ms = 0.0

        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Start a fresh telemetry log."""
        for handler in list(self._log.handlers):
            self._log.removeHandler(handler)
            handler.close()

        handler = logging.FileHandler(self.txt_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._log.addHandler(handler)

        self._log.info("=" * _W)
        self._log.info("  SPECULATIVE DECODING  —  Execution Trace")
        self._log.info("=" * _W)
        self._log.info("")

        self.iterations = []
        self._current = {}
        self._run_index = None
        self._prompt_index = None
        self._prompt_preview = None
        self._total_steps = 0
        self._total_accepted = 0
        self._total_rejected = 0
        self._total_bonus = 0
        self._total_draft_ms = 0.0
        self._total_verify_ms = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_prompt_preview(prompt: str, max_chars: int = 80) -> str:
        """Collapse whitespace and trim prompt text for readable log headers."""
        preview = " ".join(prompt.split())
        if not preview:
            return "(empty prompt)"
        if len(preview) <= max_chars:
            return preview
        return preview[: max_chars - 3].rstrip() + "..."

    # ------------------------------------------------------------------
    # Config header (called once per benchmark or generation)
    # ------------------------------------------------------------------

    def log_config(
        self,
        draft_model: str,
        target_model: str,
        gamma: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """Write a configuration header so the log is self-describing."""
        self._log.info("  Config")
        self._log.info("  " + "-" * (_W - 2))
        self._log.info(f"  Draft model   : {draft_model}")
        self._log.info(f"  Target model  : {target_model}")
        self._log.info(f"  Gamma         : {gamma}")
        self._log.info(f"  Temperature   : {temperature}")
        self._log.info(f"  Top-k         : {top_k}")
        self._log.info(f"  Top-p         : {top_p}")
        self._log.info("")

    # ------------------------------------------------------------------
    # Benchmark-level markers
    # ------------------------------------------------------------------

    def log_benchmark_start(
        self,
        total_runs: int,
        num_prompts: int,
        max_new_tokens: int,
    ):
        """Log the scope of an aggregated benchmark session."""
        self._log.info(
            "  BENCHMARK  |  runs %d  |  prompts/run %d  |  max_new_tokens %d",
            total_runs,
            num_prompts,
            max_new_tokens,
        )
        self._log.info("")

    def log_run_start(self, run_index: int, total_runs: int):
        """Mark the start of a timed benchmark run."""
        self._run_index = run_index
        self._prompt_index = None
        self._prompt_preview = None
        self._log.info("")
        self._log.info("=" * _W)
        self._log.info("  RUN %d / %d", run_index, total_runs)
        self._log.info("=" * _W)

    def log_prompt_start(self, prompt_index: int, total_prompts: int, prompt: str):
        """Mark the start of a prompt within the current run."""
        self._prompt_index = prompt_index
        self._prompt_preview = self._make_prompt_preview(prompt)
        self._log.info("")
        self._log.info("-" * _W)
        self._log.info(
            "  PROMPT %d / %d  |  %r",
            prompt_index,
            total_prompts,
            self._prompt_preview,
        )
        self._log.info("-" * _W)

    # ------------------------------------------------------------------
    # Per-generation events
    # ------------------------------------------------------------------

    def log_prefill(self, prompt_len: int, target_time_ms: float, draft_time_ms: float):
        total = target_time_ms + draft_time_ms
        self._log.info("")
        self._log.info(
            "  PREFILL  |  %d tokens  |  target %.1f ms  |  draft %.1f ms  |  total %.1f ms",
            prompt_len,
            target_time_ms,
            draft_time_ms,
            total,
        )

    def log_step_start(self, step: int):
        self._log.info("")
        self._log.info("  STEP %d  %s", step, "-" * (_W - 12))
        self._current = {"step": step}
        if self._run_index is not None:
            self._current["run_index"] = self._run_index
        if self._prompt_index is not None:
            self._current["prompt_index"] = self._prompt_index
        if self._prompt_preview is not None:
            self._current["prompt_preview"] = self._prompt_preview

    def log_draft_results(self, draft_tokens: List[str], draft_time_ms: float):
        self._log.info("    draft    %s", draft_tokens)
        self._log.info("    time     %.1f ms", draft_time_ms)
        self._current["draft_tokens"] = draft_tokens
        self._current["draft_time_ms"] = draft_time_ms
        self._total_draft_ms += draft_time_ms

    def log_verify_results(
        self,
        accepted_tokens: List[str],
        rejected_token: Optional[str],
        bonus_token: Optional[str],
        bonus_from_residual: bool,
        verify_time_ms: float,
    ):
        self._current["accepted_tokens"] = accepted_tokens
        self._current["rejected_token"] = rejected_token
        self._current["bonus_token"] = bonus_token
        self._current["bonus_is_residual"] = bonus_from_residual
        self._current["verify_time_ms"] = verify_time_ms
        self._total_verify_ms += verify_time_ms

        n_acc = len(accepted_tokens)
        self._total_accepted += n_acc

        if accepted_tokens:
            self._log.info("    accept   %s  (%d)", accepted_tokens, n_acc)
        else:
            self._log.info("    accept   (none)")

        if rejected_token:
            self._total_rejected += 1
            self._log.info("    reject   '%s'", rejected_token)

        if bonus_token is None:
            self._log.info("    bonus    —  (EOS accepted)")
        else:
            self._total_bonus += 1
            source = "residual" if bonus_from_residual else "target"
            self._log.info("    bonus    '%s'  <- %s", bonus_token, source)

        # Compute step time if draft time is available
        step_ms = verify_time_ms
        draft_ms = self._current.get("draft_time_ms", 0.0)
        if draft_ms:
            step_ms += draft_ms
        self._current["step_time_ms"] = step_ms

        self._log.info(
            "    verify   %.1f ms  |  step total %.1f ms",
            verify_time_ms,
            step_ms,
        )

    def log_step_end(self, total_emitted: int, sync_point: int):
        self._total_steps += 1
        self._log.info("    emit     %d  |  cache -> %d", total_emitted, sync_point)

        self._current["total_emitted"] = total_emitted
        self._current["sync_point"] = sync_point

        self.iterations.append(self._current)
        self._current = {}

    # ------------------------------------------------------------------
    # Summary (called after generation or benchmark is complete)
    # ------------------------------------------------------------------

    def log_summary(self):
        """Write a summary block with aggregate statistics."""
        if self._total_steps == 0:
            return

        avg_accepted = self._total_accepted / self._total_steps
        avg_draft = self._total_draft_ms / self._total_steps if self._total_draft_ms else 0
        avg_verify = self._total_verify_ms / self._total_steps if self._total_verify_ms else 0
        total_time = self._total_draft_ms + self._total_verify_ms

        self._log.info("")
        self._log.info("=" * _W)
        self._log.info("  SUMMARY")
        self._log.info("=" * _W)
        self._log.info("  Total iterations      : %d", self._total_steps)
        self._log.info("  Tokens accepted       : %d", self._total_accepted)
        self._log.info("  Tokens rejected       : %d", self._total_rejected)
        self._log.info("  Bonus tokens          : %d", self._total_bonus)
        self._log.info(
            "  Avg accepted / step   : %.2f",
            avg_accepted,
        )
        if self._total_accepted + self._total_rejected > 0:
            rate = self._total_accepted / (self._total_accepted + self._total_rejected) * 100
            self._log.info("  Acceptance rate       : %.1f%%", rate)
        self._log.info("  Avg draft time        : %.1f ms", avg_draft)
        self._log.info("  Avg verify time       : %.1f ms", avg_verify)
        self._log.info("  Total decode time     : %.1f ms", total_time)
        self._log.info("=" * _W)
