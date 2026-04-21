# teach_ml/envencoder/infer.py - инференс envencoder (encode/decode/reconstruct)

from pathlib import Path
import typing

import numpy as np
import torch

from teach_ml.envencoder.dataset import read_odom_log
from teach_ml.envencoder.model import EnvDecoder, EnvEncoder, reparameterize

LatentInput = typing.Union[str, Path, np.ndarray]


def latent_dim_from_encoder_state_dict(state_dict: dict) -> int:
    w = state_dict.get("fc_mu.weight")
    if w is None:
        raise ValueError("В чекпойнте энкодера нет ключа fc_mu.weight")
    return int(w.shape[0])


def latent_dim_from_decoder_state_dict(state_dict: dict) -> int:
    w = state_dict.get("fc.weight")
    if w is None:
        raise ValueError("В чекпойнте декодера нет ключа fc.weight")
    return int(w.shape[1])


def preprocess_depth_imu_tensors(
    depth_npy: str | Path,
    odom_log: str | Path,
    imu_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Как в датасете: depth --> [0,1] по кадру, imu x imu_scale.
    Возвращает depth [1,1,H,W], imu [1,2] на device.
    """
    depth = np.load(Path(depth_npy).resolve()).astype(np.float32)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Ожидалась карта глубины (H, W) или (1,H,W), получено {depth.shape}")
    h, w = depth.shape
    if h != 224 or w != 224:
        raise ValueError(f"Модель обучена на 224x224, получено {h}x{w}")

    arr = depth.reshape(1, h, w)
    d_min, d_max = float(arr.min()), float(arr.max())
    arr_norm = (arr - d_min) / (d_max - d_min + 1e-6)
    imu_raw = read_odom_log(Path(odom_log).resolve())
    imu_t = torch.from_numpy(imu_raw * imu_scale).float().view(1, 2).to(device)
    depth_t = torch.from_numpy(arr_norm).float().view(1, 1, h, w).to(device)
    return depth_t, imu_t


def run_envencoder_encode(
    depth_npy: str | Path,
    odom_log: str | Path,
    encoder_path: str | Path,
    device: str = "cpu",
    imu_scale: float = 1e4,
    use_mu: bool = True,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Энкодер: depth + odom --> μ, log σ², z.

    Размер латента берётся из весов энкодера (fc_mu в state_dict).

    use_mu=True: z = μ; иначе один сэмпл из q(z|x).

    Возвращает: z, mu, logvar — float32, форма (latent_dim,).
    """
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device)
    if device == "mps" and not torch.backends.mps.is_available():
        dev = torch.device("cpu")

    depth_t, imu_t = preprocess_depth_imu_tensors(depth_npy, odom_log, imu_scale, dev)

    enc_sd = torch.load(Path(encoder_path).resolve(), map_location=dev)
    latent_dim = latent_dim_from_encoder_state_dict(enc_sd)
    enc = EnvEncoder(latent_dim=latent_dim).to(dev)
    enc.load_state_dict(enc_sd)
    enc.eval()

    with torch.no_grad():
        mu, logvar = enc(depth_t, imu_t)
        if use_mu:
            z = mu
        else:
            z = reparameterize(mu, logvar)

    z_np = z.squeeze(0).float().cpu().numpy().astype(np.float32)
    mu_np = mu.squeeze(0).float().cpu().numpy().astype(np.float32)
    logvar_np = logvar.squeeze(0).float().cpu().numpy().astype(np.float32)

    return {"z": z_np, "mu": mu_np, "logvar": logvar_np}


def run_envencoder_decode(
    latent: LatentInput,
    decoder_path: str | Path,
    device: str = "cpu",
    imu_scale: float = 1e4,
) -> dict[str, np.ndarray]:
    """
    Декодер: z --> реконструированная depth и imu в «сырых» единицах (как в odom.log).

    Размер z берётся из весов декодера (fc.in_features).

    latent: путь к .npy с z или массив numpy длины latent_dim.

    Возвращает:
      depth — float32 (H, W), значения в [0, 1];
      odom_raw — float32 (2,), dX/dY без imu_scale.
    """
    dev = torch.device(device)
    if device == "mps" and not torch.backends.mps.is_available():
        dev = torch.device("cpu")

    dec_sd = torch.load(Path(decoder_path).resolve(), map_location=dev)
    latent_dim = latent_dim_from_decoder_state_dict(dec_sd)

    if isinstance(latent, (str, Path)):
        z_np = np.load(Path(latent).resolve()).astype(np.float32)
    else:
        z_np = np.asarray(latent, dtype=np.float32)
    z_np = z_np.reshape(-1)
    if z_np.shape[0] != latent_dim:
        raise ValueError(f"Ожидался латент размера {latent_dim}, получено {z_np.shape[0]}")
    z = torch.from_numpy(z_np).float().view(1, latent_dim).to(dev)

    dec = EnvDecoder(latent_dim=latent_dim).to(dev)
    dec.load_state_dict(dec_sd)
    dec.eval()

    with torch.no_grad():
        rec_d, rec_i = dec(z)

    rec_depth = rec_d.squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32)
    rec_imu_scaled = rec_i.squeeze(0).float().cpu().numpy()
    rec_imu_raw = (rec_imu_scaled / float(imu_scale)).astype(np.float32)

    return {"depth": rec_depth, "odom_raw": rec_imu_raw}


def run_envencoder_reconstruct(
    depth_npy: str | Path,
    odom_log: str | Path,
    encoder_path: str | Path,
    decoder_path: str | Path,
    device: str = "cpu",
    imu_scale: float = 1e4,
    use_mu: bool = True,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    encode --> decode. Возвращает тот же словарь, что run_envencoder_decode: depth, odom_raw.
    """
    enc_out = run_envencoder_encode(
        depth_npy=depth_npy,
        odom_log=odom_log,
        encoder_path=encoder_path,
        device=device,
        imu_scale=imu_scale,
        use_mu=use_mu,
        seed=seed,
    )
    return run_envencoder_decode(
        latent=enc_out["z"],
        decoder_path=decoder_path,
        device=device,
        imu_scale=imu_scale,
    )
