# teach_ml/envencoder/train.py - обучение envencoder (VAE)

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from teach_ml.envencoder.dataset import EnvEncoderMergedDataset, EnvEncoderDataset, list_attempt_roots
from teach_ml.envencoder.model import EnvDecoder, EnvEncoder, kl_divergence, reparameterize


def _next_run_dir(model_root: str) -> tuple[str, str]:
    model_root = os.path.abspath(model_root)
    os.makedirs(model_root, exist_ok=True)
    existing = []
    if os.path.isdir(model_root):
        for name in os.listdir(model_root):
            path = os.path.join(model_root, name)
            if os.path.isdir(path) and (
                name == "run" or (name.startswith("run") and len(name) > 3 and name[3:].isdigit())
            ):
                existing.append(name)
    run_numbers = [0 if n == "run" else int(n[3:]) for n in existing]
    next_num = max(run_numbers) + 1 if run_numbers else 0
    run_name = "run" if next_num == 0 else f"run{next_num}"
    run_dir = os.path.join(model_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name


def _mean_vae_loss_on_loader(
    enc: EnvEncoder,
    dec: EnvDecoder,
    loader: DataLoader,
    device: torch.device,
    beta_kl: float,
) -> float:
    """Средний VAE-loss (recon + beta_kl * KL) по всем сэмплам лоадера, без градиентов."""
    enc.eval()
    dec.eval()
    total = 0.0
    n = 0
    with torch.inference_mode():
        for depth, imu in loader:
            depth = depth.to(device)
            imu = imu.to(device)
            mu, logvar = enc(depth, imu)
            z = reparameterize(mu, logvar)
            rec_d, rec_i = dec(z)
            kl = kl_divergence(mu, logvar)
            loss_recon_d = F.l1_loss(rec_d, depth)
            loss_recon_i = F.l1_loss(rec_i, imu)
            loss_recon = loss_recon_d + 0.1 * loss_recon_i
            loss_vae = loss_recon + beta_kl * kl
            bs = depth.size(0)
            total += float(loss_vae) * bs
            n += bs
    return total / max(1, n)


def train_envencoder(
    datasets_dir: str | Path | None = None,
    dataset_root: str | Path | None = None,
    model_root: str | Path = "models/envencoder",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 2e-4,
    beta_kl: float = 1e-3,
    imu_scale: float = 1e4,
    latent_dim: int = 256,
    num_workers: int = 0,
    device: str = "mps",
    log_dir: str | None = None,
    save_every: int = 5,
    seed: int | None = 42,
    grad_clip_max_norm: float = 10.0,
) -> str:
    """
    Обучает VAE (encoder + decoder). Возвращает путь к каталогу запуска (run / runN).

    datasets_dir: корень datasets (по умолчанию <repo>/datasets): все подпапки с depth/imu сшиваются в один пул
    dataset_root: одна попытка (depth/, imu/); если задан — только она, без слияния

    grad_clip_max_norm: клип градиентов E+D (0 = выкл.).
    Данные делятся на train/val (~85%/15%); encoder_best/decoder_best — по минимуму val VAE-loss.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    dev = torch.device(device)
    if device == "mps" and not torch.backends.mps.is_available():
        dev = torch.device("cpu")
        print("MPS недоступен, используем CPU")

    repo_root = Path(__file__).resolve().parent.parent.parent
    if dataset_root is not None:
        root = Path(dataset_root).resolve()
        ds = EnvEncoderDataset(root, imu_scale=imu_scale)
        print(f"Датасет: одна попытка {root}")
    else:
        ddir = Path(datasets_dir).resolve() if datasets_dir else (repo_root / "datasets")
        attempt_roots = list_attempt_roots(ddir)
        ds = EnvEncoderMergedDataset(attempt_roots, imu_scale=imu_scale)
        names = ", ".join(p.name for p in attempt_roots[:8])
        extra = f" ... (+{len(attempt_roots) - 8})" if len(attempt_roots) > 8 else ""
        print(f"Датасет: {len(attempt_roots)} попыток сшиты в один пул ({names}{extra})")
    n_total = len(ds)
    val_fraction = 0.15
    val_n = max(1, int(round(n_total * val_fraction)))
    val_n = min(val_n, n_total - 1)
    train_n = n_total - val_n
    if train_n < 1:
        raise ValueError(
            f"Недостаточно сэмплов для train/val: всего {n_total}, нужно минимум 2"
        )
    split_gen = torch.Generator()
    split_gen.manual_seed(int(seed) if seed is not None else 0)
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=split_gen)
    print(
        f"Разбиение train/val: {train_n} сэмплов / {val_n} сэмплов (~{100 * val_n / n_total:.0f}% val)"
    )

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    enc = EnvEncoder(latent_dim=latent_dim).to(dev)
    dec = EnvDecoder(latent_dim=latent_dim).to(dev)

    ed_params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(ed_params, lr=lr, betas=(0.5, 0.999))

    run_dir, _ = _next_run_dir(str(model_root))
    tb_dir = log_dir if log_dir else os.path.join(run_dir, "tensorboard")
    tb_dir = os.path.abspath(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"Запуск: {run_dir}")
    print(f"TensorBoard: {tb_dir}")
    print(f"VAE | LR (E+D)={lr}")
    if grad_clip_max_norm > 0:
        print(f"Grad clip (L2 norm): {grad_clip_max_norm}")

    global_step = 0
    best_val_vae = float("inf")
    best_enc_path = os.path.join(run_dir, "encoder_best.pth")
    best_dec_path = os.path.join(run_dir, "decoder_best.pth")

    for epoch in range(epochs):
        enc.train()
        dec.train()
        epoch_vae_sum = 0.0
        epoch_batches = 0
        pbar = tqdm(loader, desc=f"Эпоха {epoch + 1}/{epochs}", leave=False)
        for depth, imu in pbar:
            depth = depth.to(dev)
            imu = imu.to(dev)

            mu, logvar = enc(depth, imu)
            z = reparameterize(mu, logvar)
            rec_d, rec_i = dec(z)
            kl = kl_divergence(mu, logvar)

            loss_recon_d = F.l1_loss(rec_d, depth)
            loss_recon_i = F.l1_loss(rec_i, imu)
            loss_recon = 0.5 * loss_recon_d + 0.5 * loss_recon_i
            loss_vae = loss_recon + beta_kl * kl

            opt.zero_grad()
            loss_vae.backward()
            if grad_clip_max_norm > 0:
                nn_utils.clip_grad_norm_(ed_params, grad_clip_max_norm)
            opt.step()

            pbar.set_postfix(vae=float(loss_vae.detach()))
            writer.add_scalar("loss/recon", loss_recon.item(), global_step)
            writer.add_scalar("loss/recon_depth", loss_recon_d.item(), global_step)
            writer.add_scalar("loss/recon_imu", loss_recon_i.item(), global_step)
            writer.add_scalar("loss/kl", kl.item(), global_step)
            writer.add_scalar("loss/vae_total", loss_vae.item(), global_step)
            epoch_vae_sum += float(loss_vae.detach())
            epoch_batches += 1
            global_step += 1

        mean_vae_train = epoch_vae_sum / max(1, epoch_batches)
        writer.add_scalar("epoch/mean_vae_train", mean_vae_train, epoch)

        enc.train()
        dec.train()
        mean_vae_val = _mean_vae_loss_on_loader(enc, dec, val_loader, dev, beta_kl)
        writer.add_scalar("epoch/mean_vae_val", mean_vae_val, epoch)

        if mean_vae_val < best_val_vae:
            best_val_vae = mean_vae_val
            torch.save(enc.state_dict(), best_enc_path)
            torch.save(dec.state_dict(), best_dec_path)
            writer.add_scalar("best/val_vae", best_val_vae, epoch)
            print(
                f"Лучшая модель по val (mean VAE {best_val_vae:.6f}): {best_enc_path}, {best_dec_path}"
            )

        if (epoch + 1) % max(1, save_every) == 0 or epoch == epochs - 1:
            ep = epoch + 1
            enc_path = os.path.join(run_dir, f"encoder_epoch_{ep}.pth")
            dec_path = os.path.join(run_dir, f"decoder_epoch_{ep}.pth")
            torch.save(enc.state_dict(), enc_path)
            torch.save(dec.state_dict(), dec_path)
            print(f"Сохранено (эпоха {ep}): {enc_path}, {dec_path}")

        enc.eval()
        dec.eval()
        with torch.no_grad():
            depth_v, imu_v = next(iter(val_loader))
            depth_v = depth_v.to(dev)
            imu_v = imu_v.to(dev)
            mu_v, logv_v = enc(depth_v, imu_v)
            z_v = reparameterize(mu_v, logv_v)
            rd, _ = dec(z_v)
            n = min(4, depth_v.size(0))
            writer.add_images("viz/depth_real", depth_v[:n].cpu(), epoch)
            writer.add_images("viz/depth_recon", rd[:n].cpu().clamp(0, 1), epoch)

        if dev.type == "mps" and (epoch + 1) % 10 == 0:
            torch.mps.empty_cache()

    writer.close()
    print(f"Готово. Каталог: {run_dir}")
    return run_dir
