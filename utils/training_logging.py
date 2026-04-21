# utils/training_logging.py - логирование обучения в каталог run: копия YAML наград + train_run_config.yaml

import argparse
import os
from pathlib import Path
import typing

import numpy as np
import yaml

from rl.environment import DroneObstacleAvoidanceEnv
from rl.envencoder_bridge import RLEnvEncoderBridge
from utils.reward_config import copy_reward_config_to_run, resolve_reward_config_path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


RUN_CONFIG_FILENAME = "train_run_config.yaml"


def copy_reward_yaml_into_run(
    run_dir: str,
    reward_config_path: str | None = None,
) -> tuple[str, str]:
    """
    Разрешает путь к YAML наград/curriculum и копирует его в run (имя как в utils.reward_config.RUN_COPY_NAME).

    Returns:
        (абсолютный_путь_к_файлу_в_run, абсолютный_путь_исходника).
    """
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    src = resolve_reward_config_path(reward_config_path)
    dst = copy_reward_config_to_run(src, run_dir)
    return os.path.abspath(dst), src


# Геометрия, динамика, карта, сенсоры (коэффициенты наград в run не дублируем).
_ENV_PHYSICAL_ATTRS: tuple[str, ...] = (
    "max_velocity",
    "forward_velocity",
    "max_episode_steps",
    "initial_height",
    "takeoff_steps",
    "num_obstacles",
    "obstacle_path_length",
    "min_passage_width",
    "map_type",
    "single_center_radius",
    "show_training_target_pillar",
    "drone_xy_half_span",
    "sensor_buffer_len",
)


def _to_yaml_value(x: typing.Any) -> typing.Any:
    if x is None or isinstance(x, (bool, str, int)):
        return x
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return str(x)
        return float(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x.resolve())
    if isinstance(x, tuple):
        return [_to_yaml_value(v) for v in x]
    if isinstance(x, list):
        return [_to_yaml_value(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_yaml_value(v) for k, v in x.items()}
    if isinstance(x, typing.Mapping):
        return {str(k): _to_yaml_value(v) for k, v in x.items()}
    return str(x)


_CURRICULUM_SUMMARY_KEYS: tuple[str, ...] = (
    "name",
    "map_type",
    "path_length",
    "num_obstacles",
    "single_center_radius",
    "goal_xy_tolerance",
    "reward_goal_reached",
    "reward_goal_time_coef",
    "reward_goal_distance_progress_coef",
    "reward_survival",
    "reward_collision",
    "reward_exit_failure",
)


def summarize_curriculum_stages(stages: list[dict[str, typing.Any]] | None) -> list[dict[str, typing.Any]] | None:
    if not stages:
        return None
    rows: list[dict[str, typing.Any]] = []
    for st in stages:
        rows.append({k: _to_yaml_value(st[k]) for k in _CURRICULUM_SUMMARY_KEYS if k in st})
    return rows


def snapshot_environment_dict(env: DroneObstacleAvoidanceEnv) -> dict[str, typing.Any]:
    """Физика, карта, сенсоры (без коэффициентов наград из отдельного YAML в run)."""
    out: dict[str, typing.Any] = {}
    for name in _ENV_PHYSICAL_ATTRS:
        if not hasattr(env, name):
            continue
        out[name] = _to_yaml_value(getattr(env, name))
    out["initial_position_range"] = _to_yaml_value(getattr(env, "initial_position_range", None))
    try:
        low = np.asarray(env.action_space.low, dtype=np.float64).reshape(-1).tolist()
        high = np.asarray(env.action_space.high, dtype=np.float64).reshape(-1).tolist()
        out["action_space_low"] = low
        out["action_space_high"] = high
    except Exception:
        pass
    if getattr(env, "drone", None) is not None:
        out["camera_height"] = int(env.drone.camera_height)
        out["camera_width"] = int(env.drone.camera_width)
        out["mj_timestep"] = _to_yaml_value(float(env.drone.m.opt.timestep))
    if "obstacle_path_length" in out:
        out["path_length"] = out["obstacle_path_length"]
    return out


def write_train_run_config_yaml(
    *,
    run_dir: str,
    algorithm: str,
    env: DroneObstacleAvoidanceEnv,
    bridge: RLEnvEncoderBridge,
    encoder_path: str,
    device: str,
    seed: int | None,
    policy_inference: dict[str, typing.Any],
    training_hyperparams: dict[str, typing.Any] | None = None,
    curriculum_stages: list[dict[str, typing.Any]] | None = None,
    use_curriculum: bool | None = None,
) -> str:
    """
    Пишет train_run_config.yaml: среда, пути энкодера/FastDepth/IMU drift, блок инференса политики, опционально curriculum и гиперпараметры обучения.
    """
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    enc_abs = os.path.abspath(os.path.expanduser(str(encoder_path)))
    fd_path = getattr(getattr(bridge, "fastdepth", None), "weights_path", None)
    fd_abs = str(Path(fd_path).resolve()) if fd_path is not None else None

    doc: dict[str, typing.Any] = {
        "schema_version": 1,
        "algorithm": algorithm,
        "run_directory": run_dir,
        "device_train": str(device),
        "seed": seed,
        "paths": {
            "encoder_weights": enc_abs,
            "fastdepth_weights": fd_abs,
            "imu_scale": float(bridge.imu_scale),
        },
        "environment": snapshot_environment_dict(env),
        "policy_inference": _to_yaml_value(policy_inference),
    }
    if use_curriculum is not None:
        doc["curriculum_enabled"] = bool(use_curriculum)
    if curriculum_stages:
        doc["curriculum_stages"] = _to_yaml_value(curriculum_stages)
    if training_hyperparams:
        doc["training_hyperparams"] = _to_yaml_value(training_hyperparams)

    out_path = os.path.join(run_dir, RUN_CONFIG_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            doc,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    return out_path


_TRAIN_ALGOS_PLAY = frozenset({"mcts", "ppo", "dqn"})


def resolve_play_train_config_dir(args: typing.Any) -> str | None:
    """
    Каталог, где искать train_run_config.yaml для play:
    --run, иначе родительский каталог --policy, если там есть файл.
    """
    run_v = getattr(args, "run", None)
    if run_v and str(run_v).strip():
        return os.path.abspath(os.path.expanduser(str(run_v)))
    pol = getattr(args, "policy", None)
    if pol and str(pol).strip():
        d = os.path.dirname(os.path.abspath(os.path.expanduser(str(pol))))
        if os.path.isfile(os.path.join(d, RUN_CONFIG_FILENAME)):
            return d
    return None


def _play_parser_dest_defaults(play_parser: argparse.ArgumentParser) -> dict[str, typing.Any]:
    out: dict[str, typing.Any] = {}
    for a in play_parser._actions:
        if a.dest in (None, "help"):
            continue
        if a.default is argparse.SUPPRESS:
            continue
        out[a.dest] = a.default
    return out


def load_train_run_config_for_play(
    args: typing.Any,
    play_parser: argparse.ArgumentParser,
    cfg_dir: str | None,
) -> tuple[str | None, dict[str, typing.Any] | None]:
    """
    Подставляет в args значения из train_run_config.yaml, если соответствующий аргумент
    совпадает с дефолтом play_parser (явные флаги CLI не перезаписываются).

    Returns:
        (абсолютный путь к yaml, блок environment для доп. полей конструктора среды) или (None, None).
    """
    if not cfg_dir:
        return None, None
    cfg_dir = os.path.abspath(cfg_dir)
    cfg_path = os.path.join(cfg_dir, RUN_CONFIG_FILENAME)
    if not os.path.isfile(cfg_path):
        return None, None
    with open(cfg_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        return None, None
    defs = _play_parser_dest_defaults(play_parser)
    env = doc.get("environment")
    env = env if isinstance(env, dict) else None
    pi = doc.get("policy_inference")
    pi = pi if isinstance(pi, dict) else {}
    disc = pi.get("discretization")
    disc = disc if isinstance(disc, dict) else {}
    th = doc.get("training_hyperparams")
    th = th if isinstance(th, dict) else {}

    def _unchanged(val: typing.Any, dest: str) -> bool:
        return val == defs.get(dest)

    if env:
        if "max_episode_steps" in env and _unchanged(getattr(args, "max_steps", None), "max_steps"):
            args.max_steps = int(env["max_episode_steps"])
        if "num_obstacles" in env and _unchanged(getattr(args, "num_obstacles", None), "num_obstacles"):
            args.num_obstacles = int(env["num_obstacles"])
        path_len = env.get("obstacle_path_length", env.get("path_length"))
        if path_len is not None and _unchanged(getattr(args, "path_length", None), "path_length"):
            args.path_length = float(path_len)
        if "map_type" in env and _unchanged(getattr(args, "map_type", None), "map_type"):
            args.map_type = str(env["map_type"])
        if "max_velocity" in env and _unchanged(
            getattr(args, "play_max_velocity", None), "play_max_velocity"
        ):
            args.play_max_velocity = float(env["max_velocity"])
        if "forward_velocity" in env and _unchanged(
            getattr(args, "play_forward_velocity", None), "play_forward_velocity"
        ):
            args.play_forward_velocity = float(env["forward_velocity"])

    if disc:
        if "n_vx_bins" in disc and _unchanged(getattr(args, "latent_vx_bins", None), "latent_vx_bins"):
            args.latent_vx_bins = int(disc["n_vx_bins"])
        if "n_vy_bins" in disc and _unchanged(getattr(args, "latent_vy_bins", None), "latent_vy_bins"):
            args.latent_vy_bins = int(disc["n_vy_bins"])
        if "vy_min" in disc and _unchanged(getattr(args, "latent_vy_min", None), "latent_vy_min"):
            args.latent_vy_min = float(disc["vy_min"])
        if "vy_max" in disc and _unchanged(getattr(args, "latent_vy_max", None), "latent_vy_max"):
            args.latent_vy_max = float(disc["vy_max"])

    if pi:
        if "n_vx_bins" in pi and _unchanged(getattr(args, "latent_vx_bins", None), "latent_vx_bins"):
            args.latent_vx_bins = int(pi["n_vx_bins"])
        if "n_vy_bins" in pi and _unchanged(getattr(args, "latent_vy_bins", None), "latent_vy_bins"):
            args.latent_vy_bins = int(pi["n_vy_bins"])
        if "vy_min" in pi and _unchanged(getattr(args, "latent_vy_min", None), "latent_vy_min"):
            args.latent_vy_min = float(pi["vy_min"])
        if "vy_max" in pi and _unchanged(getattr(args, "latent_vy_max", None), "latent_vy_max"):
            args.latent_vy_max = float(pi["vy_max"])

    if th and "gamma" in th and _unchanged(getattr(args, "gamma", None), "gamma"):
        args.gamma = float(th["gamma"])

    algo = doc.get("algorithm")
    if isinstance(algo, str) and algo in _TRAIN_ALGOS_PLAY and _unchanged(
        getattr(args, "play_algo", None), "play_algo"
    ):
        args.play_algo = algo

    if doc.get("seed") is not None and getattr(args, "seed", None) is None:
        args.seed = int(doc["seed"])

    paths = doc.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    enc = paths.get("encoder_weights")
    if enc and getattr(args, "encoder_path", None) is None:
        enc_abs = os.path.abspath(os.path.expanduser(str(enc)))
        if os.path.isfile(enc_abs):
            args.encoder_path = enc_abs
    fd = paths.get("fastdepth_weights")
    if fd and getattr(args, "fastdepth_weights", None) is None:
        fd_abs = os.path.abspath(os.path.expanduser(str(fd)))
        if os.path.isfile(fd_abs):
            args.fastdepth_weights = fd_abs

    return os.path.abspath(cfg_path), (dict(env) if env else None)
