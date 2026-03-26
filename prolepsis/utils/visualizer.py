import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

_C_RED = "#6c2b2b"
_C_DARK_BLUE = "#334b6c"
_C_LIGHT_BLUE = "#c6d0dd"
_C_BG = "#fcfcfc"
_GRAY = "gray"
_LIGHT_GRAY = "lightgray"

def _apply_global_style():
    plt.rcParams.update({
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
    })


def _save(fig, path: str):
    fig.savefig(
        path,
        facecolor=_C_BG,
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.3,
    )
    plt.close(fig)


class SpeculativeVisualizer:

    def __init__(
        self,
        iterations: List[Dict[str, Any]],
        output_base: str,
        config_label: Optional[str] = None,
    ):
        self._it = iterations
        base = str(Path(output_base).with_suffix(""))
        self._out_acceptance = f"{base}_acceptance.png"
        self._out_cumulative = f"{base}_cumulative.png"
        self._out_dashboard = f"{base}_dashboard.png"
        self._label = config_label or ""

    def generate_dashboard(self) -> Optional[str]:
        """Render all charts. Returns the dashboard path or None."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib unavailable — skipping charts.")
            return None
        if not self._it:
            logger.warning("No iteration data to visualise.")
            return None

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="matplotlib.font_manager",
        )
        _apply_global_style()

        steps = np.arange(len(self._it))

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.patch.set_facecolor(_C_BG)
        for ax in axes:
            ax.set_facecolor(_C_BG)

        fig.suptitle(
            "Speculative decoding \u2014 execution profile",
            fontsize=18,
            fontweight="bold",
            y=0.99,
        )
        if self._label:
            fig.text(
                0.5, 0.94,
                self._label,
                ha="center",
                color=_GRAY,
                fontsize=11,
            )

        self._chart_acceptance(axes[0], steps)
        self._chart_timing(axes[1], steps)
        self._chart_cumulative(axes[2], steps)

        fig.tight_layout(rect=[0, 0.02, 1, 0.90], h_pad=3.5)
        _save(fig, self._out_dashboard)

        fig1, ax1 = plt.subplots(figsize=(14, 4.5))
        fig1.patch.set_facecolor(_C_BG)
        ax1.set_facecolor(_C_BG)
        self._chart_acceptance(ax1, steps)
        _save(fig1, self._out_acceptance)

        fig2, ax2 = plt.subplots(figsize=(14, 4.5))
        fig2.patch.set_facecolor(_C_BG)
        ax2.set_facecolor(_C_BG)
        self._chart_cumulative(ax2, steps)
        _save(fig2, self._out_cumulative)

        return self._out_dashboard

    def _chart_acceptance(self, ax, steps):
        accepted = np.array(
            [len(r.get("accepted_tokens", [])) for r in self._it], dtype=np.float64
        )
        drafted = np.array(
            [len(r.get("draft_tokens", [])) for r in self._it], dtype=np.float64
        )

        ax.bar(steps, accepted, color=_C_LIGHT_BLUE, width=1.0, zorder=2)

        window = min(10, max(3, len(steps) // 20))
        if len(steps) > window:
            try:
                import pandas as pd
                accepted_ma = pd.Series(accepted).rolling(window=window, min_periods=1).mean()
            except ImportError:
                kernel = np.ones(window) / window
                accepted_ma = np.convolve(accepted, kernel, mode="same")
            ax.plot(steps, accepted_ma, color=_C_RED, linewidth=2, zorder=3)

        if len(drafted) > 0:
            gamma_val = int(np.max(drafted))
            ax.axhline(gamma_val, color=_LIGHT_GRAY, linestyle="--", linewidth=1, zorder=1)

        ax.set_title("Token acceptance", fontsize=13, fontweight="bold", loc="left", pad=10)
        ax.set_ylabel("Tokens", color=_GRAY, fontsize=11)
        ax.set_ylim(0, (max(drafted) + 2.5) if len(drafted) > 0 else 7.5)

        gamma_label = f"Drafted (γ={int(np.max(drafted))})" if len(drafted) > 0 else "Drafted"
        leg_handles = [
            mpatches.Patch(facecolor=_C_LIGHT_BLUE, edgecolor="none", label="Accepted"),
            Line2D([0], [0], color=_C_RED, linewidth=2, label=f"Moving avg ({window})"),
            Line2D([0], [0], color=_LIGHT_GRAY, linewidth=1, linestyle="--", label=gamma_label),
        ]
        ax.legend(handles=leg_handles, loc="upper left", ncol=3, frameon=False, fontsize=9)

        safe_drafted = np.maximum(drafted, 1)
        rate = float(np.mean(accepted / safe_drafted)) * 100
        avg_per_step = float(np.mean(accepted))
        ax.annotate(
            f"Acceptance: {rate:.0f}%  |  Avg {avg_per_step:.1f} tok/step",
            xy=(0.98, 0.88),
            xycoords="axes fraction",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc=_C_BG, ec=_C_DARK_BLUE, lw=1),
            color=_C_DARK_BLUE,
            fontweight="bold",
            fontsize=10,
        )

    def _chart_timing(self, ax, steps):
        draft_ms = np.array(
            [r.get("draft_time_ms", 0.0) for r in self._it], dtype=np.float64
        )
        verify_ms = np.array(
            [r.get("verify_time_ms", 0.0) for r in self._it], dtype=np.float64
        )

        if draft_ms.sum() == 0 and verify_ms.sum() == 0:
            ax.text(
                0.5, 0.5,
                "No timing data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=_GRAY, style="italic",
            )
            ax.set_title("Step timing", fontsize=13, fontweight="bold", loc="left", pad=10)
            return

        ax.fill_between(steps, 0, verify_ms, color=_C_LIGHT_BLUE, zorder=2)
        ax.fill_between(steps, verify_ms, verify_ms + draft_ms, color=_C_DARK_BLUE, zorder=2)

        total_ms = verify_ms + draft_ms
        if len(total_ms) > 10:
            p98 = float(np.percentile(total_ms, 98))
            if p98 > 0:
                ax.set_ylim(0, p98 * 1.15)

        ax.set_title("Step timing", fontsize=13, fontweight="bold", loc="left", pad=10)
        ax.set_ylabel("Time (ms)", color=_GRAY, fontsize=11)

        # Legend
        leg_handles = [
            mpatches.Patch(facecolor=_C_LIGHT_BLUE, edgecolor="#8fa0b5", linewidth=1.2, label="Verify"),
            mpatches.Patch(facecolor=_C_DARK_BLUE, edgecolor="none", label="Draft"),
        ]
        ax.legend(handles=leg_handles, loc="lower left", ncol=2, frameon=False, fontsize=9)

        total_draft = draft_ms.sum()
        total_verify = verify_ms.sum()
        total_all = total_draft + total_verify
        if total_all > 0:
            draft_pct = total_draft / total_all * 100
            verify_pct = total_verify / total_all * 100
            ax.annotate(
                f"\u25a0 Draft {draft_pct:.0f}%",
                xy=(0.99, 0.93),
                xycoords="axes fraction",
                ha="right",
                color=_C_DARK_BLUE,
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_C_DARK_BLUE, lw=1),
            )
            ax.annotate(
                f"\u25a0 Verify {verify_pct:.0f}%",
                xy=(0.99, 0.78),
                xycoords="axes fraction",
                ha="right",
                color="#5a7a9a",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#8fa0b5", lw=1),
            )

    def _chart_cumulative(self, ax, steps):
        emitted = np.array([r.get("total_emitted", 0) for r in self._it])
        cumulative = np.cumsum(emitted)
        baseline = np.arange(1, len(steps) + 1)

        ax.plot(steps, cumulative, color=_C_RED, linewidth=2, label="Speculative output", zorder=3)
        ax.fill_between(steps, cumulative, color=_C_RED, alpha=0.08, zorder=2)
        ax.plot(steps, baseline, color=_GRAY, linestyle="--", label="1 token / iter (baseline)", zorder=2)

        ax.set_title("Cumulative tokens", fontsize=13, fontweight="bold", loc="left", pad=10)
        ax.set_ylabel("Total tokens", color=_GRAY, fontsize=11)
        ax.set_xlabel("Iteration", color=_GRAY, fontsize=11, labelpad=8)
        ax.set_xlim(0, len(steps))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

        total_tokens = int(cumulative[-1]) if len(cumulative) > 0 else 0
        total_iters = len(steps)
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=9,
            title=f"Total output: {total_tokens:,} tokens in {total_iters:,} iterations",
            title_fontsize=9,
        )

        if len(steps) > 10 and baseline[-1] > 0:
            ratio = cumulative[-1] / baseline[-1]
            midpoint = int(len(steps) * 0.67)
            mid_y = int(cumulative[midpoint])
            ax.annotate(
                f"{ratio:.2f}× tokens / iter",
                xy=(midpoint, mid_y),
                xytext=(midpoint - int(len(steps) * 0.16), mid_y + int(cumulative[-1] * 0.14)),
                arrowprops=dict(
                    arrowstyle="->",
                    color=_C_RED,
                    lw=1.4,
                    connectionstyle="arc3,rad=-0.2",
                ),
                bbox=dict(boxstyle="round,pad=0.35", fc=_C_BG, ec=_C_RED, lw=1),
                color=_C_RED,
                fontweight="bold",
                fontsize=10,
            )
