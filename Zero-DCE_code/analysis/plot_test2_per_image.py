import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_uneven_lighting import evaluate_pair, write_csv  # noqa: E402


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def evaluate_directories(
    orig_dir: Path, enh_dir: Path
) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []

    origin_files = sorted([p for p in orig_dir.iterdir() if p.is_file()])
    for orig_path in origin_files:
        enh_path = enh_dir / orig_path.name
        if not enh_path.exists():
            continue

        original = load_image(orig_path)
        enhanced = load_image(enh_path)
        row = evaluate_pair(original, enhanced)
        row["image"] = orig_path.name
        metrics.append(row)

    return metrics


def _valid_subset(names: List[str], arrays: List[np.ndarray]):
    mask = np.ones(len(names), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    subset_names = [name for idx, name in enumerate(names) if mask[idx]]
    subset_arrays = [arr[mask] for arr in arrays]
    return subset_names, subset_arrays


def _group_bar(
    ax,
    names: List[str],
    values_a: np.ndarray,
    values_b: np.ndarray,
    title: str,
    ylabel: str,
    labels=("Original", "Enhanced"),
):
    idx_names, (vals_a, vals_b) = _valid_subset(
        names, [values_a, values_b]
    )
    if not idx_names:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    indices = np.arange(len(idx_names))
    width = 0.35

    ax.bar(
        indices - width / 2,
        vals_a,
        width,
        label=labels[0],
        color="#6baed6",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.bar(
        indices + width / 2,
        vals_b,
        width,
        label=labels[1],
        color="#31a354",
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(indices)
    ax.set_xticklabels(idx_names, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    if not ax.get_legend():
        ax.legend(fontsize=8)


def _single_bar(
    ax,
    names: List[str],
    values: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
):
    idx_names, (vals,) = _valid_subset(names, [values])
    if not idx_names:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    indices = np.arange(len(idx_names))
    ax.bar(
        indices,
        vals,
        color=color,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(indices)
    ax.set_xticklabels(idx_names, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_metrics(
    metrics: List[Dict[str, float]], output_path: Path
) -> None:
    names = [m["image"] for m in metrics]

    def arr(column: str) -> np.ndarray:
        return np.array([m.get(column, np.nan) for m in metrics], dtype=float)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
    fig.suptitle("Zero-DCE Test-2 Per-Image Metrics", fontsize=16)

    _group_bar(
        axes[0, 0],
        names,
        arr("orig_grid_std"),
        arr("enh_grid_std"),
        "Patch Contrast (4×4 std)",
        "Std Dev",
    )
    _group_bar(
        axes[0, 1],
        names,
        arr("orig_dark_frac"),
        arr("enh_dark_frac"),
        "Dark Pixel Fraction",
        "Fraction",
    )
    _group_bar(
        axes[0, 2],
        names,
        arr("orig_bright_frac"),
        arr("enh_bright_frac"),
        "Bright Pixel Fraction",
        "Fraction",
    )
    _group_bar(
        axes[1, 0],
        names,
        arr("orig_niqe"),
        arr("enh_niqe"),
        "NIQE (lower is better)",
        "Score",
    )
    _group_bar(
        axes[1, 1],
        names,
        arr("orig_nima"),
        arr("enh_nima"),
        "NIMA (higher is better)",
        "Score",
    )
    _single_bar(
        axes[1, 2],
        names,
        arr("loe"),
        "Lightness Order Error",
        "Error",
        "#756bb1",
    )
    _single_bar(
        axes[2, 0],
        names,
        arr("grid_std_delta"),
        "Δ Patch Contrast",
        "Change",
        "#3182bd",
    )
    _single_bar(
        axes[2, 1],
        names,
        arr("psnr"),
        "PSNR vs Original",
        "dB",
        "#31a354",
    )
    _single_bar(
        axes[2, 2],
        names,
        arr("ssim"),
        "SSIM vs Original",
        "Score",
        "#fd8d3c",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-image bar charts for Zero-DCE test-2 comparisons."
    )
    parser.add_argument(
        "--orig_dir",
        type=Path,
        default=PROJECT_ROOT / "data/test_data/test-2",
        help="Directory containing original low-light images.",
    )
    parser.add_argument(
        "--enh_dir",
        type=Path,
        default=PROJECT_ROOT / "data/result_cpu/test-2",
        help="Directory containing enhanced outputs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "analysis/test2_per_image_metrics.csv",
        help="Path to save per-image metrics table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "analysis/test2_per_image_bars.png",
        help="Where to save the bar-chart figure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.orig_dir.exists():
        raise SystemExit(f"Original directory not found: {args.orig_dir}")
    if not args.enh_dir.exists():
        raise SystemExit(f"Enhanced directory not found: {args.enh_dir}")

    metrics = evaluate_directories(args.orig_dir, args.enh_dir)
    if not metrics:
        raise SystemExit("No overlapping images found between directories.")

    write_csv(metrics, args.csv)
    plot_metrics(metrics, args.output)


if __name__ == "__main__":
    main()


