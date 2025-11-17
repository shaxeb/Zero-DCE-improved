import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch

import model

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    import pyiqa
except ImportError:  # pragma: no cover - optional dependency
    pyiqa = None

try:  # pragma: no cover - optional dependency
    from skimage.metrics import niqe as skimage_niqe
except ImportError:
    skimage_niqe = None

try:  # pragma: no cover - optional dependency
    from skimage.metrics import peak_signal_noise_ratio
except ImportError:
    peak_signal_noise_ratio = None

try:  # pragma: no cover - optional dependency
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None

_NIQE_METRIC = None
_NIMA_METRIC = None


def _sanitize_image(arr: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(cleaned, 0.0, 1.0)


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    net = model.enhance_net_nopool().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).to(device)
    return tensor


def tensor_to_npimg(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if hasattr(torch, "nan_to_num"):
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    tensor = tensor.clamp(0, 1)
    arr = tensor.permute(1, 2, 0).numpy()
    return _sanitize_image(arr)


def enhance_image(net: torch.nn.Module, img_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, enhanced, _ = net(img_tensor.unsqueeze(0))
    return enhanced.squeeze(0)


def compute_luminance(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def downsample_luma_for_loe(luma: np.ndarray, max_dim: int = 64) -> np.ndarray:
    h, w = luma.shape
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    if step <= 1:
        return luma
    return luma[::step, ::step]


def compute_lightness_order_error(orig_luma: np.ndarray, enh_luma: np.ndarray) -> float:
    orig_ds = downsample_luma_for_loe(orig_luma)
    enh_ds = downsample_luma_for_loe(enh_luma)

    orig_flat = orig_ds.flatten()
    enh_flat = enh_ds.flatten()

    if orig_flat.size == 0 or enh_flat.size == 0:
        return float("nan")

    orig_diff = orig_flat[:, None] - orig_flat[None, :]
    enh_diff = enh_flat[:, None] - enh_flat[None, :]

    orig_order = (orig_diff >= 0).astype(np.uint8)
    enh_order = (enh_diff >= 0).astype(np.uint8)

    return float(np.abs(orig_order - enh_order).mean())


def compute_niqe_score(luma: np.ndarray) -> float:
    global _NIQE_METRIC
    if pyiqa is not None:
        if _NIQE_METRIC is None:
            _NIQE_METRIC = pyiqa.create_metric("niqe", device="cpu")
        tensor = torch.from_numpy(luma).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            score = _NIQE_METRIC(tensor)
        return float(score.item())
    if skimage_niqe is not None:
        return float(skimage_niqe(luma.astype(np.float32)))
    return float("nan")


def compute_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    if peak_signal_noise_ratio is None:
        return float("nan")
    original = _sanitize_image(original.astype(np.float32, copy=False))
    enhanced = _sanitize_image(enhanced.astype(np.float32, copy=False))
    value = float(peak_signal_noise_ratio(original, enhanced, data_range=1.0))
    if np.isinf(value):
        return 100.0
    return value


def compute_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    if structural_similarity is None:
        return float("nan")
    original = _sanitize_image(original.astype(np.float32, copy=False))
    enhanced = _sanitize_image(enhanced.astype(np.float32, copy=False))
    score = float(
        structural_similarity(
            original,
            enhanced,
            channel_axis=2,
            data_range=1.0,
        )
    )
    return score


def compute_nima_score(image: np.ndarray) -> float:
    global _NIMA_METRIC
    if pyiqa is None:
        return float("nan")
    if _NIMA_METRIC is None:
        _NIMA_METRIC = pyiqa.create_metric("nima", device="cpu")
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        score = _NIMA_METRIC(tensor)
    return float(score.item())


def grid_patch_stats(luma: np.ndarray, grid: int = 4) -> np.ndarray:
    h, w = luma.shape
    patch_h = h // grid
    patch_w = w // grid
    patches: List[float] = []
    for i in range(grid):
        for j in range(grid):
            patch = luma[
                i * patch_h : (i + 1) * patch_h,
                j * patch_w : (j + 1) * patch_w,
            ]
            patches.append(float(patch.mean()))
    return np.array(patches)


def evaluate_pair(original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
    original = _sanitize_image(original)
    enhanced = _sanitize_image(enhanced)
    orig_luma = compute_luminance(original)
    enh_luma = compute_luminance(enhanced)

    orig_patches = grid_patch_stats(orig_luma)
    enh_patches = grid_patch_stats(enh_luma)

    orig_grid_std = float(orig_patches.std())
    enh_grid_std = float(enh_patches.std())
    orig_dark = float((orig_luma < 0.2).mean())
    enh_dark = float((enh_luma < 0.2).mean())
    orig_bright = float((orig_luma > 0.9).mean())
    enh_bright = float((enh_luma > 0.9).mean())
    loe = compute_lightness_order_error(orig_luma, enh_luma)
    orig_niqe = compute_niqe_score(orig_luma)
    enh_niqe = compute_niqe_score(enh_luma)
    psnr = compute_psnr(original, enhanced)
    ssim = compute_ssim(original, enhanced)
    orig_nima = compute_nima_score(original)
    enh_nima = compute_nima_score(enhanced)

    return {
        "orig_grid_std": orig_grid_std,
        "enh_grid_std": enh_grid_std,
        "orig_dark_frac": orig_dark,
        "enh_dark_frac": enh_dark,
        "orig_bright_frac": orig_bright,
        "enh_bright_frac": enh_bright,
        "grid_std_delta": enh_grid_std - orig_grid_std,
        "dark_frac_delta": enh_dark - orig_dark,
        "bright_frac_delta": enh_bright - orig_bright,
        "loe": loe,
        "orig_niqe": orig_niqe,
        "enh_niqe": enh_niqe,
        "niqe_delta": enh_niqe - orig_niqe,
        "psnr": psnr,
        "ssim": ssim,
        "orig_nima": orig_nima,
        "enh_nima": enh_nima,
        "nima_delta": enh_nima - orig_nima,
    }


def save_tensor_image(tensor: torch.Tensor, save_path: Path) -> None:
    arr = (tensor_to_npimg(tensor) * 255.0).astype(np.uint8)
    image = Image.fromarray(arr)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def process_dataset(
    data_dir: Path,
    output_dir: Path,
    weights_path: Path,
    device: torch.device,
    save_images: bool,
) -> List[Dict[str, float]]:
    net = load_model(str(weights_path), device)

    metrics: List[Dict[str, float]] = []
    image_paths = sorted([p for p in data_dir.glob("*") if p.is_file()])
    for img_path in image_paths:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            tensor = pil_to_tensor(img, device)
            enhanced_tensor = enhance_image(net, tensor)

            if save_images:
                relative = img_path.relative_to(data_dir)
                save_tensor_image(enhanced_tensor, output_dir / relative)

            original_np = tensor_to_npimg(tensor)
            enhanced_np = tensor_to_npimg(enhanced_tensor)
            row = evaluate_pair(original_np, enhanced_np)
            row["image"] = img_path.name
            metrics.append(row)

    return metrics


def write_csv(metrics: List[Dict[str, float]], csv_path: Path) -> None:
    if not metrics:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["image"] + [k for k in metrics[0] if k != "image"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metrics)


def summarize(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not metrics:
        return summary
    keys = [k for k in metrics[0] if k != "image"]
    for key in keys:
        values = np.array([m[key] for m in metrics], dtype=np.float32)
        summary[f"mean_{key}"] = float(np.nanmean(values))
        summary[f"median_{key}"] = float(np.nanmedian(values))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Zero-DCE uneven lighting behavior."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_ROOT / "data/test_data/test-2"),
        help="Folder with low-light inputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data/result_cpu/test-2"),
        help="Where to save enhanced images.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "snapshots/Epoch99.pth"),
        help="Path to Zero-DCE weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu or cuda).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(PROJECT_ROOT / "analysis/uneven_lighting_metrics.csv"),
        help="Where to write metric table.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving enhanced images.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    weights_path = Path(args.weights)
    csv_path = Path(args.csv)

    metrics = process_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        weights_path=weights_path,
        device=device,
        save_images=not args.no_save,
    )
    write_csv(metrics, csv_path)

    summary = summarize(metrics)
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

