"""Visualização dos resultados do benchmark (desacoplada da lógica de execução).

Gera gráficos comparativos entre os modos de execução (GA_ONLY, LS_ONLY, HYBRID).
Pode produzir múltiplos arquivos de saída:

- ``resultados/<instancia>/comparison.png``: comparação detalhada por instância
  (distância best/mean, distribuição, gap do ótimo e tempo de execução).
- ``resultados/summary_comparison.png``: comparação do gap entre todas as
  instâncias (gerado apenas quando há mais de uma instância).

Este módulo cuida SOMENTE de visualização; recebe os resultados já calculados e
não conhece detalhes do algoritmo ou do orquestrador.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # backend headless (benchmark roda sem GUI)
import matplotlib.pyplot as plt
import numpy as np


MODE_COLORS: Dict[str, str] = {
    "GA_ONLY": "#1f77b4",
    "LS_ONLY": "#ff7f0e",
    "HYBRID": "#2ca02c",
}
DEFAULT_COLOR = "#7f7f7f"
DPI = 150


def _color_for(mode: str) -> str:
    return MODE_COLORS.get(mode, DEFAULT_COLOR)


def _valid(results: Sequence) -> List:
    return [r for r in results if not r.timed_out and r.error is None and r.best_distance != float("inf")]


def _distances(results: Sequence) -> List[float]:
    return [r.best_distance for r in _valid(results)]


def _times(results: Sequence) -> List[float]:
    return [r.execution_time for r in _valid(results)]


def _gaps(results: Sequence) -> List[float]:
    return [r.gap_from_optimal for r in _valid(results) if r.gap_from_optimal is not None]


def _optimal(results: Sequence):
    for r in results:
        if r.optimal_solution:
            return r.optimal_solution
    return None


def _annotate_bars(ax, bars, values, fmt: str) -> None:
    for bar, value in zip(bars, values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_instance_comparison(
    instance_name: str,
    per_mode_results: Dict[str, List],
    output_path: Path,
) -> Path:
    """Gera o gráfico comparativo detalhado de uma instância."""
    modes = list(per_mode_results.keys())
    colors = [_color_for(m) for m in modes]
    optimal = None
    for results in per_mode_results.values():
        optimal = optimal or _optimal(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Distância: best e mean (com barra de erro = std) por modo.
    ax = axes[0, 0]
    best_values = [min(_distances(r)) if _distances(r) else np.nan for r in per_mode_results.values()]
    mean_values = [float(np.mean(_distances(r))) if _distances(r) else np.nan for r in per_mode_results.values()]
    std_values = [float(np.std(_distances(r))) if _distances(r) else 0.0 for r in per_mode_results.values()]
    x = np.arange(len(modes))
    width = 0.38
    bars_best = ax.bar(x - width / 2, best_values, width, label="Best", color=colors, alpha=0.95)
    bars_mean = ax.bar(x + width / 2, mean_values, width, yerr=std_values, capsize=4,
                       label="Mean +/- std", color=colors, alpha=0.5)
    if optimal:
        ax.axhline(optimal, color="black", linestyle="--", linewidth=1, label=f"Optimal ({optimal:,})")
    ax.set_title("Distance by mode", fontweight="bold")
    ax.set_ylabel("Distance")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    _annotate_bars(ax, bars_best, best_values, "{:,.0f}")
    _annotate_bars(ax, bars_mean, mean_values, "{:,.0f}")

    # (2) Distribuição das distâncias (boxplot) por modo.
    ax = axes[0, 1]
    box_data = [_distances(r) for r in per_mode_results.values()]
    non_empty = [(m, d, _color_for(m)) for m, d in zip(modes, box_data) if d]
    if non_empty:
        labels = [m for m, _, _ in non_empty]
        data = [d for _, d, _ in non_empty]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        for patch, (_, _, c) in zip(bp["boxes"], non_empty):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    else:
        ax.text(0.5, 0.5, "No valid runs", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Distance distribution across runs", fontweight="bold")
    ax.set_ylabel("Distance")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # (3) Gap do ótimo (%) por modo: best e mean.
    ax = axes[1, 0]
    if optimal:
        best_gap = [min(_gaps(r)) if _gaps(r) else np.nan for r in per_mode_results.values()]
        mean_gap = [float(np.mean(_gaps(r))) if _gaps(r) else np.nan for r in per_mode_results.values()]
        bars_bg = ax.bar(x - width / 2, best_gap, width, label="Best gap", color=colors, alpha=0.95)
        bars_mg = ax.bar(x + width / 2, mean_gap, width, label="Mean gap", color=colors, alpha=0.5)
        ax.set_title("Gap from optimal (%)", fontweight="bold")
        ax.set_ylabel("Gap (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        _annotate_bars(ax, bars_bg, best_gap, "{:.2f}%")
        _annotate_bars(ax, bars_mg, mean_gap, "{:.2f}%")
    else:
        ax.text(0.5, 0.5, "Optimal unknown\n(gap unavailable)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Gap from optimal (%)", fontweight="bold")

    # (4) Tempo médio de execução por modo.
    ax = axes[1, 1]
    mean_time = [float(np.mean(_times(r))) if _times(r) else np.nan for r in per_mode_results.values()]
    time_std = [float(np.std(_times(r))) if _times(r) else 0.0 for r in per_mode_results.values()]
    bars_t = ax.bar(x, mean_time, width=0.6, yerr=time_std, capsize=4, color=colors, alpha=0.85)
    ax.set_title("Mean execution time", fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    _annotate_bars(ax, bars_t, mean_time, "{:.1f}s")

    fig.suptitle(f"Benchmark comparison - {instance_name}", fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    return output_path


def plot_summary_across_instances(summary: Dict[str, Dict], output_path: Path) -> Path:
    """Gera um gráfico agrupado do gap best (%) por modo entre instâncias."""
    instances = list(summary.keys())
    modes = sorted({m for inst in summary.values() for m in inst.keys()})

    x = np.arange(len(instances))
    total_width = 0.8
    bar_width = total_width / max(1, len(modes))

    fig, ax = plt.subplots(figsize=(max(8, len(instances) * 2.2), 6))
    for idx, mode in enumerate(modes):
        values = []
        for instance in instances:
            entry = summary[instance].get(mode, {})
            gap = entry.get("best_gap_percent")
            values.append(gap if gap is not None else np.nan)
        offset = (idx - (len(modes) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=mode, color=_color_for(mode), alpha=0.9)
        for bar, value in zip(bars, values):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", va="bottom", fontsize=7)

    ax.set_title("Best gap from optimal (%) by instance and mode", fontweight="bold")
    ax.set_ylabel("Best gap (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(instances)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    return output_path


def generate_benchmark_reports(
    summary: Dict[str, Dict],
    detailed: Dict[str, Dict[str, List]],
    results_root: Path,
) -> List[Path]:
    """Gera todos os gráficos do benchmark e retorna os caminhos criados."""
    generated: List[Path] = []

    for instance_name, per_mode_results in detailed.items():
        output_path = results_root / instance_name / "comparison.png"
        generated.append(plot_instance_comparison(instance_name, per_mode_results, output_path))

    if len(summary) > 1:
        summary_path = results_root / "summary_comparison.png"
        generated.append(plot_summary_across_instances(summary, summary_path))

    return generated
