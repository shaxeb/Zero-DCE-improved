import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _finite(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    return finite


def load_metrics(csv_path: Path) -> List[Dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find metrics CSV at {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, float]] = []
        for row in reader:
            numeric_row = {}
            for key, value in row.items():
                if key == "image":
                    numeric_row[key] = value
                else:
                    numeric_row[key] = float(value)
            rows.append(numeric_row)

    if not rows:
        raise ValueError(f"Metrics CSV {csv_path} is empty.")

    return rows


def scatter_patch_variance(ax, orig: np.ndarray, enh: np.ndarray) -> None:
    max_val = float(max(orig.max(), enh.max()))
    ax.scatter(orig, enh, s=16, alpha=0.7, color="#2f74c0", edgecolors="none")
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlabel("Original 4×4 patch std")
    ax.set_ylabel("Enhanced 4×4 patch std")
    ax.set_title("Patch Contrast Before vs After")
    ax.grid(alpha=0.3, linestyle=":")


def histogram_dark_delta(ax, deltas: np.ndarray) -> None:
    ax.hist(deltas, bins=20, color="#7f8c8d", alpha=0.85)
    mean_val = float(deltas.mean())
    ax.axvline(mean_val, color="#e67e22", linestyle="--", label=f"mean {mean_val:.2f}")
    ax.set_xlabel("Δ dark pixel fraction")
    ax.set_ylabel("Image count")
    ax.set_title("Dark Region Reduction")
    ax.legend(loc="upper right")


def bar_brightness_mix(ax, metrics: Dict[str, np.ndarray]) -> None:
    categories = ["Dark Fraction", "Bright Fraction"]
    orig_values = [
        metrics["orig_dark_frac"].mean(),
        metrics["orig_bright_frac"].mean(),
    ]
    enh_values = [
        metrics["enh_dark_frac"].mean(),
        metrics["enh_bright_frac"].mean(),
    ]
    indices = np.arange(len(categories))
    width = 0.35

    ax.bar(indices - width / 2, orig_values, width, label="Original", color="#bdc3c7")
    ax.bar(indices + width / 2, enh_values, width, label="Enhanced", color="#27ae60")
    ax.set_xticks(indices)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel("Fraction of pixels")
    ax.set_ylim(0, 1)
    ax.set_title("Average Pixel Distribution")
    ax.legend()


def histogram_loe(ax, loe_values: np.ndarray) -> None:
    ax.hist(loe_values, bins=15, color="#9b59b6", alpha=0.8)
    mean_val = float(loe_values.mean())
    ax.axvline(mean_val, color="#2c3e50", linestyle="--", label=f"mean {mean_val:.3f}")
    ax.set_xlabel("Lightness order error")
    ax.set_ylabel("Image count")
    ax.set_title("Lightness Consistency")
    ax.legend(loc="upper right")


def bar_niqe(ax, orig: np.ndarray, enh: np.ndarray) -> None:
    means = [float(orig.mean()), float(enh.mean())]
    labels = ["Original", "Enhanced"]
    width = 0.5
    ax.bar(labels, means, width=width, color=["#95a5a6", "#16a085"])
    for idx, value in enumerate(means):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    delta = means[1] - means[0]
    ax.set_ylabel("NIQE (lower is better)")
    ax.set_title(f"Naturalness (Δ {delta:+.2f})")
    ax.set_ylim(0, max(means) * 1.2 if max(means) > 0 else 1)


def histogram_psnr(ax, psnr_values: np.ndarray) -> None:
    values = _finite(psnr_values)
    if values.size == 0:
        ax.text(0.5, 0.5, "No PSNR data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.hist(values, bins=20, color="#1abc9c", alpha=0.8)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("Image count")
    mean_val = float(values.mean())
    ax.axvline(mean_val, color="#2c3e50", linestyle="--", label=f"mean {mean_val:.2f}")
    ax.set_title("PSNR vs Original")
    ax.legend(loc="upper right")


def histogram_ssim(ax, ssim_values: np.ndarray) -> None:
    values = _finite(ssim_values)
    if values.size == 0:
        ax.text(0.5, 0.5, "No SSIM data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.hist(values, bins=20, color="#f39c12", alpha=0.8)
    mean_val = float(values.mean())
    ax.axvline(mean_val, color="#2c3e50", linestyle="--", label=f"mean {mean_val:.3f}")
    ax.set_xlabel("SSIM")
    ax.set_ylabel("Image count")
    ax.set_title("SSIM vs Original")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")


def bar_nima(ax, orig: np.ndarray, enh: np.ndarray) -> None:
    means = [float(np.nanmean(orig)), float(np.nanmean(enh))]
    labels = ["Original", "Enhanced"]
    width = 0.5
    ax.bar(labels, means, width=width, color=["#8e44ad", "#e74c3c"])
    for idx, value in enumerate(means):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    delta = means[1] - means[0]
    ax.set_ylabel("NIMA (higher is better)")
    ax.set_title(f"Perceptual Appeal (Δ {delta:+.2f})")
    ax.set_ylim(0, max(means) * 1.2 if max(means) > 0 else 1)


def plot_metrics(csv_path: Path, output_path: Path) -> None:
    rows = load_metrics(csv_path)
    metric_keys = [k for k in rows[0] if k != "image"]

    metrics: Dict[str, np.ndarray] = {
        key: np.array([row[key] for row in rows], dtype=np.float32) for key in metric_keys
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    flat_axes = axes.flatten()
    scatter_patch_variance(
        flat_axes[0], metrics["orig_grid_std"], metrics["enh_grid_std"]
    )
    histogram_dark_delta(flat_axes[1], metrics["dark_frac_delta"])
    bar_brightness_mix(flat_axes[2], metrics)
    histogram_loe(flat_axes[3], metrics["loe"])
    bar_niqe(flat_axes[4], metrics["orig_niqe"], metrics["enh_niqe"])
    histogram_psnr(flat_axes[5], metrics["psnr"])
    histogram_ssim(flat_axes[6], metrics["ssim"])
    bar_nima(flat_axes[7], metrics["orig_nima"], metrics["enh_nima"])
    flat_axes[8].axis("off")

    fig.suptitle("Zero-DCE Uneven Lighting Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    print(f"Saved plots to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize uneven lighting metrics collected from evaluate_uneven_lighting.py"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="uneven_lighting_metrics.csv",
        help="Path to the metrics CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="uneven_lighting_summary.png",
        help="Where to write the summary visualization image.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (base_dir / csv_path).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()
    plot_metrics(csv_path, output_path)


if __name__ == "__main__":
    main()


