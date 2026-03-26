from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn(
        "Matplotlib not installed. Benchmark visualization features are unavailable."
    )


_COLOR_BASELINE = "#444444"
_COLOR_PROLEPSIS = "#1f4e79"
_COLOR_ASSISTED = "#6a6a6a"
_BG = "#FFFFFF"
_GRAY_TEXT = "gray"
_BLACK = "black"


def _check_matplotlib():
    """Ensure matplotlib is available before plotting."""
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "Matplotlib is required for benchmark plots. Install it to enable "
            "benchmark visualization."
        )


def _apply_global_style():
    """Set global rcParams to match the monospace / clean-grid aesthetic."""
    plt.rcParams.update({
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
    })


def _method_color(method: str) -> str:
    """Choose a stable color for a benchmarked method."""
    normalized = method.lower()
    if "baseline" in normalized:
        return _COLOR_BASELINE
    if "prolepsis" in normalized or "speculative" in normalized:
        return _COLOR_PROLEPSIS
    if "assisted" in normalized:
        return _COLOR_ASSISTED
    return _COLOR_ASSISTED


def _method_hatch(method: str) -> str:
    """Return a hatch pattern for non-baseline methods."""
    normalized = method.lower()
    if "baseline" in normalized:
        return ""
    return "//"


def _display_method_name(method: str) -> str:
    """Return a short display label for chart axes."""
    normalized = method.lower()
    if "baseline" in normalized:
        return "Baseline"
    if "assisted" in normalized:
        return "HF Assisted"
    if "prolepsis" in normalized or "speculative" in normalized:
        return "Prolepsis"
    return method


def _finite_values(values: Sequence[float]) -> List[float]:
    """Return the finite values in the provided sequence."""
    return [value for value in values if not np.isnan(value)]


def _metric_formatter(label: str):
    """Build a value formatter from the metric label."""
    normalized = label.lower()
    if "speedup" in normalized:
        return lambda value: f"{value:.2f}"
    if "millisecond" in normalized or "latency" in normalized:
        return lambda value: f"{value:.2f}"
    if "allocated" in normalized or "memory" in normalized or "mb" in normalized:
        return lambda value: f"{value:,.0f}"
    return lambda value: f"{value:.1f}"


def _is_lower_better(label: str) -> bool:
    """Return whether smaller values are preferable for this metric."""
    normalized = label.lower()
    return any(
        token in normalized
        for token in ("latency", "millisecond", "memory", "allocated", "reserved")
    )


def _baseline_index(methods: Sequence[str]) -> Optional[int]:
    """Return the baseline index when present."""
    for index, method in enumerate(methods):
        if "baseline" in method.lower():
            return index
    return None


def _delta_label(value: float, baseline: float, lower_is_better: bool) -> Optional[str]:
    """Return a formatted delta relative to the baseline."""
    if np.isnan(value) or np.isnan(baseline) or baseline == 0:
        return None
    if np.isclose(value, baseline):
        return None

    if lower_is_better:
        delta_pct = ((baseline - value) / baseline) * 100.0
    else:
        delta_pct = ((value - baseline) / baseline) * 100.0

    sign = "+" if delta_pct >= 0 else ""
    return f"{sign}{delta_pct:.0f}%"


def _plot_bar_panel(
    ax,
    methods: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    ylim: Optional[tuple] = None,
    annotations: Optional[Sequence[str]] = None,
    reference_line: Optional[float] = None,
    panel_note: Optional[str] = None,
):
    num_methods = len(methods)
    display_methods = [_display_method_name(m) for m in methods]

    if num_methods == 2:
        positions = [-0.3, 0.3]
        bar_width = 0.45
        xlim = (-0.8, 0.8)
    elif num_methods == 3:
        positions = [-0.5, 0.0, 0.5]
        bar_width = 0.35
        xlim = (-0.9, 0.9)
    else:
        positions = list(range(num_methods))
        bar_width = 0.45
        xlim = (-0.6, num_methods - 0.4)

    colors = [_method_color(m) for m in methods]
    hatches = [_method_hatch(m) for m in methods]

    bars = ax.bar(
        positions,
        values,
        color=colors,
        width=bar_width,
        edgecolor=_BLACK,
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax.set_xticks(positions)
    ax.set_xticklabels(display_methods)
    ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)
    else:
        finite = _finite_values(values)
        if finite:
            ax.set_ylim(0, max(finite) * 1.25)

    formatter = _metric_formatter(ylabel)
    for bar_obj, val in zip(bars, values):
        if np.isnan(val):
            continue
        h = bar_obj.get_height()
        if annotations:
            idx = list(bars).index(bar_obj)
            label_text = annotations[idx]
        else:
            label_text = formatter(val) if val < 100 else f"{int(val):,}"
        ax.annotate(
            label_text,
            xy=(bar_obj.get_x() + bar_obj.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    if reference_line is not None:
        ax.axhline(
            y=reference_line,
            color="lightgray",
            linestyle="--",
            linewidth=1,
            zorder=1,
        )

        if panel_note:
            ax.text(
                1.0, 1.03, panel_note,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=_GRAY_TEXT,
            )


class BenchmarkDashboard:

    def __init__(
        self,
        figsize=(14, 9),
        dpi: int = 300,
        theme: str = "light",
        config_subtitle: Optional[str] = None,
    ):
        _check_matplotlib()
        _apply_global_style()

        self.figsize = figsize
        self.dpi = dpi
        self.config_subtitle = config_subtitle
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.patch.set_facecolor(_BG)

    def _draw_header(self):
        self.fig.suptitle(
            "Benchmark Performance Summary",
            fontsize=20,
            fontweight="bold",
            ha="center",
            y=0.99,
        )
        subtitle = self.config_subtitle or "Throughput, latency, memory, and baseline-relative speed"
        self.fig.text(
            0.5, 0.93,
            subtitle,
            ha="center",
            color=_GRAY_TEXT,
            fontsize=11,
        )

    def plot(
        self,
        methods: Sequence[str],
        throughput_tok_s: Sequence[float],
        latency_ms: Sequence[float],
        memory_mb: Sequence[float],
        speedups: Sequence[float],
        acceptance_labels: Optional[Sequence[str]] = None,
        acceptance_rates: Optional[Sequence[float]] = None,
        show_speedup_panel: bool = True,
        speedup_note: Optional[str] = None,
    ):
        self._draw_header()

        _plot_bar_panel(
            self.axes[0, 0],
            methods=methods,
            values=throughput_tok_s,
            title="Throughput",
            ylabel="Tokens / second",
            annotations=[f"{v:.2f}" for v in throughput_tok_s],
        )
        _plot_bar_panel(
            self.axes[0, 1],
            methods=methods,
            values=latency_ms,
            title="Latency",
            ylabel="ms / token",
            annotations=[f"{v:.2f}" for v in latency_ms],
        )
        _plot_bar_panel(
            self.axes[1, 0],
            methods=methods,
            values=memory_mb,
            title="Peak GPU Memory",
            ylabel="Allocated (MB)",
            annotations=[
                f"{v:,.0f}" if not np.isnan(v) else "n/a"
                for v in memory_mb
            ],
        )

        if show_speedup_panel:
            _plot_bar_panel(
                self.axes[1, 1],
                methods=methods,
                values=speedups,
                title="Speedup vs. Baseline",
                ylabel="Speedup (×)",
                annotations=acceptance_labels,
                reference_line=1.0,
                panel_note=speedup_note,
            )
        else:
            rates = acceptance_rates or []
            if any(not np.isnan(rate) for rate in rates):
                percent = [
                    rate * 100.0 if not np.isnan(rate) else float("nan")
                    for rate in rates
                ]
                _plot_bar_panel(
                    self.axes[1, 1],
                    methods=methods,
                    values=percent,
                    title="Acceptance Rate",
                    ylabel="Accepted Tokens (%)",
                    annotations=[
                        f"{v:.0f}%" if not np.isnan(v) else "n/a"
                        for v in percent
                    ],
                )
            else:
                ax = self.axes[1, 1]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.text(
                    0.5, 0.5,
                    "Baseline skipped.\nSpeedup is unavailable for this run.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=_GRAY_TEXT, style="italic",
                )

        self.fig.tight_layout(rect=[0, 0.02, 1, 0.89], h_pad=3.5, w_pad=3)

    def save(self, filepath: Union[str, Path]):
        """Save dashboard to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(
            filepath,
            dpi=self.dpi,
            bbox_inches="tight",
            pad_inches=0.3,
            facecolor=_BG,
        )

    def close(self):
        """Close the figure."""
        plt.close(self.fig)

class BenchmarkBarPlot:

    def __init__(self, title: str, ylabel: str, theme: str = "light", **kwargs):
        _check_matplotlib()
        _apply_global_style()

        self.ylabel = ylabel
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.patch.set_facecolor(_BG)
        self.title = title

    def plot_metric(
        self,
        methods: Sequence[str],
        values: Sequence[float],
        annotations: Optional[Sequence[str]] = None,
        *,
        show_delta: bool = True,
    ):
        """Plot one metric as a styled bar chart."""
        _plot_bar_panel(
            self.ax,
            methods=methods,
            values=values,
            title=self.title,
            ylabel=self.ylabel,
            annotations=annotations,
        )
        self.fig.tight_layout()

    def save(self, filepath: Union[str, Path]):
        """Save plot to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
            facecolor=_BG,
        )

    def close(self):
        """Close the figure."""
        plt.close(self.fig)

def create_benchmark_plots(
    results: Sequence[object],
    save_dir: Union[str, Path],
    theme: str = "light",
    detailed: bool = False,
    config_subtitle: Optional[str] = None,
) -> List[str]:
    if not HAS_MATPLOTLIB:
        return []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    methods = [result.method for result in results]
    throughput = [result.tokens_per_sec for result in results]
    latency = [result.avg_latency_ms for result in results]
    memory = [
        result.memory_mb if result.memory_mb is not None else float("nan")
        for result in results
    ]
    acceptance_rates = [
        result.acceptance_rate if result.acceptance_rate is not None else float("nan")
        for result in results
    ]

    baseline_index = _baseline_index(methods)
    baseline_throughput = (
        throughput[baseline_index]
        if baseline_index is not None and throughput[baseline_index] > 0
        else None
    )
    baseline_available = baseline_throughput is not None

    speedups = []
    speedup_labels = []
    for method, throughput_value, acceptance_rate in zip(
        methods,
        throughput,
        acceptance_rates,
    ):
        if baseline_available:
            speedup = throughput_value / baseline_throughput
        else:
            speedup = float("nan")
        speedups.append(speedup)

        if "baseline" in method.lower():
            speedup_labels.append("1.00")
        elif baseline_available:
            speedup_labels.append(f"{speedup:.2f}")
        else:
            speedup_labels.append("n/a")

    acceptance_note = None
    acceptance_values = [
        rate for rate in acceptance_rates if not np.isnan(rate)
    ]
    if len(acceptance_values) == 1:
        acceptance_note = f"Acceptance {acceptance_values[0]:.0%}"
    elif acceptance_values:
        acceptance_note = "Acceptance: " + ", ".join(
            f"{_display_method_name(method)} {rate:.0%}"
            for method, rate in zip(methods, acceptance_rates)
            if not np.isnan(rate)
        )

    saved_paths: List[str] = []

    if detailed:
        for title, ylabel, values, annotations, filename in [
            (
                "Throughput Comparison",
                "Tokens / second",
                throughput,
                [f"{value:.1f}" for value in throughput],
                "benchmark_throughput.png",
            ),
            (
                "Latency Comparison",
                "ms / token",
                latency,
                [f"{value:.2f}" for value in latency],
                "benchmark_latency.png",
            ),
            (
                "Peak GPU Memory",
                "Allocated (MB)",
                memory,
                [
                    f"{value:,.0f}" if not np.isnan(value) else "n/a"
                    for value in memory
                ],
                "benchmark_memory.png",
            ),
        ]:
            if filename == "benchmark_memory.png" and not any(
                not np.isnan(value) for value in values
            ):
                continue
            plot = BenchmarkBarPlot(title=title, ylabel=ylabel, theme=theme)
            plot.plot_metric(methods, values, annotations=annotations)
            output_path = save_dir / filename
            plot.save(output_path)
            plot.close()
            saved_paths.append(str(output_path))

    dashboard = BenchmarkDashboard(theme=theme, config_subtitle=config_subtitle)
    dashboard.plot(
        methods=methods,
        throughput_tok_s=throughput,
        latency_ms=latency,
        memory_mb=memory,
        speedups=speedups,
        acceptance_labels=speedup_labels,
        acceptance_rates=acceptance_rates,
        show_speedup_panel=baseline_available,
        speedup_note=acceptance_note,
    )
    dashboard_path = save_dir / "benchmark_dashboard.png"
    dashboard.save(dashboard_path)
    dashboard.close()
    saved_paths.append(str(dashboard_path))

    return saved_paths


__all__ = [
    "HAS_MATPLOTLIB",
    "BenchmarkBarPlot",
    "BenchmarkDashboard",
    "create_benchmark_plots",
]
