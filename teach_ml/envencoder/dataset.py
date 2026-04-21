# teach_ml/envencoder/dataset.py - датасет depth+odom для envencoder

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_odom_log(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip().split()
    if len(text) < 2:
        raise ValueError(f"Ожидались 2 числа в {path}")
    return np.array([float(text[0]), float(text[1])], dtype=np.float32)


def count_depth_imu_pairs(root: Path) -> int:
    depth_dir = root / "depth"
    imu_dir = root / "imu"
    if not depth_dir.is_dir() or not imu_dir.is_dir():
        return 0
    n = 0
    for dp in depth_dir.glob("depth_*.npy"):
        try:
            idx = int(dp.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (imu_dir / f"odom_{idx}.log").is_file():
            n += 1
    return n


def _attempt_dir_sort_key(path: Path) -> tuple:
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def list_attempt_roots(datasets_dir: Path) -> list[Path]:
    """
    Список корней попыток (каждый с depth/ и imu/).

    Если сам datasets_dir уже содержит depth/ и imu/ — одна попытка [datasets_dir].
    Иначе — все прямые подпапки с depth/ и imu/, отсортированные по имени.
    """
    d = Path(datasets_dir).resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"Нет каталога: {d}")
    if (d / "depth").is_dir() and (d / "imu").is_dir():
        return [d]
    candidates: list[Path] = []
    for sub in sorted(d.iterdir(), key=_attempt_dir_sort_key):
        if not sub.is_dir():
            continue
        if (sub / "depth").is_dir() and (sub / "imu").is_dir():
            candidates.append(sub)
    if not candidates:
        raise ValueError(
            f"В {d} нет попыток с depth/ и imu/ "
            "(ни в корне, ни в подпапках). Укажите --env-dataset-root или корректный --datasets-dir."
        )
    return candidates


def _depth_to_1hw(depth: np.ndarray) -> np.ndarray:
    """Карта глубины --> float32 [1, H, W] для энкодера."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 1:
        n = int(d.size)
        s = int(round(np.sqrt(n)))
        if s * s != n:
            raise ValueError(
                f"depth: {n} элементов — не H*W квадрата (нужен плоский вектор длины s²)"
            )
        return d.reshape(1, s, s)
    if d.ndim == 2:
        h, w = int(d.shape[0]), int(d.shape[1])
        if h == 1 and w > 1:
            n = w
            s = int(round(np.sqrt(n)))
            if s * s == n:
                return d.reshape(s, s).reshape(1, s, s)
        if w == 1 and h > 1:
            n = h
            s = int(round(np.sqrt(n)))
            if s * s == n:
                return d.reshape(s, s).reshape(1, s, s)
        return d.reshape(1, h, w)
    if d.ndim == 3:
        if d.shape[0] == 1:
            return d.reshape(1, int(d.shape[1]), int(d.shape[2]))
        if d.shape[-1] == 1:
            return d.reshape(1, int(d.shape[0]), int(d.shape[1]))
        return d[:, :, 0].reshape(1, int(d.shape[0]), int(d.shape[1]))
    raise ValueError(f"depth: ожидались 1–3 измерения, получено shape={d.shape}")


def _load_pair(depth_path: Path, odom_path: Path, imu_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    depth = np.load(depth_path)
    arr = _depth_to_1hw(depth)
    d_min, d_max = float(arr.min()), float(arr.max())
    arr = (arr - d_min) / (d_max - d_min + 1e-6)
    imu = read_odom_log(odom_path) * imu_scale
    return torch.from_numpy(arr).float(), torch.from_numpy(imu).float()


class EnvEncoderMergedDataset(Dataset):
    """Все попытки в одном списке сэмплов; порядок в эпохе задаёт DataLoader(shuffle=True)."""

    def __init__(self, attempt_roots: list[Path], imu_scale: float = 1e4):
        self.imu_scale = float(imu_scale)
        self._pairs: list[tuple[Path, Path]] = []
        for root in attempt_roots:
            root = Path(root).resolve()
            depth_dir = root / "depth"
            imu_dir = root / "imu"
            if not depth_dir.is_dir() or not imu_dir.is_dir():
                raise ValueError(f"Ожидались {depth_dir} и {imu_dir}")
            for dp in sorted(
                depth_dir.glob("depth_*.npy"),
                key=lambda p: int(p.stem.split("_")[1]),
            ):
                idx = int(dp.stem.split("_")[1])
                odom_p = imu_dir / f"odom_{idx}.log"
                if odom_p.is_file():
                    self._pairs.append((dp, odom_p))
        if not self._pairs:
            raise ValueError("Не найдены пары depth_*.npy / odom_*.log ни в одной попытке")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, i: int):
        dp, op = self._pairs[i]
        return _load_pair(dp, op, self.imu_scale)


class EnvEncoderDataset(Dataset):
    def __init__(self, root: Path, imu_scale: float = 1e4):
        self._merged = EnvEncoderMergedDataset([Path(root)], imu_scale=imu_scale)

    def __len__(self) -> int:
        return len(self._merged)

    def __getitem__(self, i: int):
        return self._merged[i]
