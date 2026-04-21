# rl/envencoder_bridge.py - мост RGB+IMU --> depth+dXY --> латент z (EnvEncoder)

from pathlib import Path
import typing

import numpy as np
import torch

from teach_ml.envencoder.infer import latent_dim_from_encoder_state_dict
from teach_ml.envencoder.model import EnvEncoder
from teach_ml.fastdepth import FastDepthWrapper
from utils.imu_odometry import IMUStrapdownOdometry


class RLEnvEncoderBridge:
    """
    На каждом наблюдении: odom.update(accel, gyro) --> (dX, dY), FastDepth(RGB) --> depth 224,
    энкодер(depth, imu_scaled) --> z (μ по умолчанию).
    """

    def __init__(
        self,
        encoder_path: str | Path,
        device: str = "cpu",
        fastdepth_weights: typing.Optional[str | Path] = None,
        imu_scale: float = 1e4,
    ):
        self.imu_scale = float(imu_scale)
        dev = device
        if dev == "mps" and not torch.backends.mps.is_available():
            dev = "cpu"
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device_t = torch.device(dev)

        self.fastdepth = FastDepthWrapper(fastdepth_weights)
        self.fastdepth.depth_net.to(self.device_t)
        self.fastdepth.depth_net.eval()

        enc_path = Path(encoder_path).resolve()
        enc_sd = torch.load(enc_path, map_location=self.device_t, weights_only=True)
        latent_dim = latent_dim_from_encoder_state_dict(enc_sd)
        self.latent_dim = latent_dim
        self.encoder = EnvEncoder(latent_dim=latent_dim).to(self.device_t)
        self.encoder.load_state_dict(enc_sd)
        self.encoder.eval()

        self.odom: typing.Optional[IMUStrapdownOdometry] = None

    def reset(self, env) -> None:
        """Вызывать после env.reset(): новая сессия одометрии."""
        dt = float(env.drone.m.opt.timestep)
        q = env.drone.d.qpos[3:7].astype(np.float64).copy()
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            q = q / n
        self.odom = IMUStrapdownOdometry(dt=dt)
        self.odom.reset(position_world=np.zeros(3, dtype=np.float64), quaternion_body_to_world=q)

    def pack_odom_state(self) -> dict:
        if self.odom is None:
            raise RuntimeError("Вызовите reset(env) перед pack_odom_state")
        return self.odom.pack_internal_state()

    def unpack_odom_state(self, state: dict) -> None:
        if self.odom is None:
            raise RuntimeError("Вызовите reset(env) перед unpack_odom_state")
        self.odom.unpack_internal_state(state)

    def _encode_depth_dxy_tensor(self, depth_hw: np.ndarray, dxy: np.ndarray) -> torch.Tensor:
        """Как _encode_depth_dxy, но μ на device_t без round-trip на CPU ([1, latent_dim])."""
        arr = np.asarray(depth_hw, dtype=np.float32).reshape(224, 224)
        d_min, d_max = float(arr.min()), float(arr.max())
        arr_norm = (arr - d_min) / (d_max - d_min + 1e-6)
        depth_t = torch.from_numpy(arr_norm).float().view(1, 1, 224, 224).to(self.device_t)
        imu_t = torch.from_numpy(np.asarray(dxy, dtype=np.float32).reshape(1, 2) * self.imu_scale).float().to(
            self.device_t
        )
        with torch.inference_mode():
            mu, _ = self.encoder(depth_t, imu_t)
        return mu.float()

    def _encode_depth_dxy(self, depth_hw: np.ndarray, dxy: np.ndarray) -> np.ndarray:
        """depth_hw: float32 (224,224) сырые выходы FastDepth; dxy: (2,) приращение XY за шаг."""
        return (
            self._encode_depth_dxy_tensor(depth_hw, dxy)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def encode_observation_tensor(self, obs: dict) -> torch.Tensor:
        """
        Как encode_observation, но латент на device_t ([1, latent_dim]).
        FastDepth --> нормализованная глубина остаётся на GPU (без полного depth на CPU как в numpy_rgb_to_depth_uint8).
        """
        if self.odom is None:
            raise RuntimeError("Вызовите reset(env) перед encode_observation_tensor")
        depth_norm = self.fastdepth.numpy_hwc_to_depth_normalized_tensor(
            obs["image"], device=self.device_t
        )
        delta = self.odom.update(obs["accelerometer"], obs["gyroscope"])
        dxy = np.array([float(delta[0]), float(delta[1])], dtype=np.float32)
        imu_t = torch.from_numpy(dxy.reshape(1, 2) * self.imu_scale).float().to(self.device_t)
        with torch.inference_mode():
            mu, _ = self.encoder(depth_norm, imu_t)
        return mu.float()

    def encode_observation(self, obs: dict) -> np.ndarray:
        """
        Один шаг цепочки: обновление страпдауна по текущему IMU из obs, затем z.
        Не вызывать дважды на одном и том же obs без отката odom.
        """
        return (
            self.encode_observation_tensor(obs)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
