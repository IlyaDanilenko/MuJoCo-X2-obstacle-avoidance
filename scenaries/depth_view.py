# scenaries/depth_view.py - сохранение карт глубины FastDepth

import os

import cv2
import numpy as np
import torch

from rl.environment import DroneObstacleAvoidanceEnv
from teach_ml.fastdepth import FastDepthWrapper


def run_depth_view(
    weights_path=None,
    out_dir="test_depth",
    steps=40,
    device="cpu",
    seed=42,
    num_obstacles=30,
):
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print("Сценарий depth-view (FastDepth --> test_depth)")
    print("=" * 60)
    fd = FastDepthWrapper(weights_path)
    print(f"  Веса: {fd.weights_path}")
    print(f"  Выход: {out_dir}, шагов: {steps}, устройство: {device}")
    fd.to(device)
    fd.eval()
    env = DroneObstacleAvoidanceEnv(
        render_mode=None,
        max_velocity=0.4,
        forward_velocity=0.0,
        max_episode_steps=max(10000, steps + 50),
        num_obstacles=num_obstacles,
        obstacle_path_length=5.0,
    )
    obs, info = env.reset(seed=seed)
    print(f"  После взлёта позиция: {info['position']}")
    saved = 0
    for t in range(steps):
        depth_u8, _ = fd.numpy_rgb_to_depth_uint8(obs["image"], device=torch.device(device))
        path_png = os.path.join(out_dir, f"depth_{t:04d}.png")
        d_save = np.asarray(depth_u8, dtype=np.uint8)
        d_save = np.squeeze(d_save)
        if d_save.ndim == 3:
            d_save = d_save[..., 0]
        if d_save.ndim != 2:
            raise ValueError(f"глубина для PNG должна быть HxW, получено shape={d_save.shape}")
        cv2.imwrite(path_png, d_save)
        saved += 1
        obs, _, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated or truncated:
            print(f"  Эпизод оборван на шаге {t} (terminated={terminated}, truncated={truncated})")
            break
    env.close()
    print(f"Сохранено кадров: {saved} --> {out_dir}")
