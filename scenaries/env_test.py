# scenaries/env_test.py - реконструкция envencoder (depth + odom)

import os

import cv2
import numpy as np

from teach_ml.envencoder import run_envencoder_reconstruct


def _load_depth_map_hw(depth_npy_path: str) -> np.ndarray:
    d = np.load(depth_npy_path).astype(np.float32)
    if d.ndim == 3 and d.shape[0] == 1:
        d = d[0]
    if d.ndim != 2:
        raise ValueError(
            f"Ожидалась глубина (H, W) или (1,H,W), получено {d.shape}: {depth_npy_path}"
        )
    return d


def _depth_float_to_png_u8(depth_hw: np.ndarray) -> np.ndarray:
    d = depth_hw.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    norm = (d - d_min) / (d_max - d_min + 1e-6)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def _save_envencoder_recon_outputs(
    out_dir: str,
    depth: np.ndarray,
    odom_raw: np.ndarray,
    original_depth_hw: np.ndarray | None = None,
) -> dict[str, str]:
    out = os.path.abspath(out_dir)
    os.makedirs(out, exist_ok=True)
    npy_path = os.path.join(out, "depth_recon.npy")
    png_path = os.path.join(out, "depth_recon.png")
    log_path = os.path.join(out, "odom_recon.log")
    np.save(npy_path, depth.astype(np.float32))
    u8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(png_path, u8)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{float(odom_raw[0])} {float(odom_raw[1])}\n")
    result = {
        "depth_npy": npy_path,
        "depth_png": png_path,
        "odom_log": log_path,
    }
    if original_depth_hw is not None:
        orig_png = os.path.join(out, "depth_original.png")
        cv2.imwrite(orig_png, _depth_float_to_png_u8(original_depth_hw))
        result["depth_original_png"] = orig_png
    return result


def run_env_test(
    depth_npy: str,
    odom_log: str,
    encoder_weights: str,
    decoder_weights: str,
    out_dir: str = "test",
    device: str = "cpu",
    imu_scale: float = 1e4,
    stochastic: bool = False,
    seed: int | None = None,
):
    print("=" * 60)
    print("env-test: реконструкция envencoder")
    print("=" * 60)
    depth_hw = _load_depth_map_hw(depth_npy)
    recon = run_envencoder_reconstruct(
        depth_npy=depth_npy,
        odom_log=odom_log,
        encoder_path=encoder_weights,
        decoder_path=decoder_weights,
        device=device,
        imu_scale=imu_scale,
        use_mu=not stochastic,
        seed=seed,
    )
    paths = _save_envencoder_recon_outputs(
        out_dir, recon["depth"], recon["odom_raw"], original_depth_hw=depth_hw
    )
    for k, p in paths.items():
        print(f"  {k}: {p}")
