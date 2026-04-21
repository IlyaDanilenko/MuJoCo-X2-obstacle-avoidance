# utils/reward_config.py - награды и curriculum из YAML

import os
import shutil
import typing

import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_REWARD_YAML = os.path.join(_REPO_ROOT, "config", "reward_config.yaml")
RUN_COPY_NAME = "reward_config.yaml"

ENV_REWARD_KEYS: frozenset[str] = frozenset(
    {
        "reward_collision",
        "reward_exit_failure",
        "reward_survival",
        "reward_progress",
        "reward_progress_max_per_step",
        "reward_deviation_lateral_coef",
        "reward_lateral_deviation_gate_low",
        "reward_lateral_deviation_gate_high",
        "reward_safe_goal_scale_min",
        "reward_clearance_gate_low",
        "reward_clearance_gate_high",
        "reward_barrier_coef",
        "reward_barrier_scale",
        "reward_goal_ungated_fraction",
        "reward_goal_distance_progress_coef",
        "reward_goal_reached",
        "reward_goal_time_coef",
        "reward_clearance_coef",
        "reward_clearance_threshold",
        "reward_orbit_deficit_coef",
        "reward_orbit_clearance_margin",
        "reward_action_l2_coef",
        "reward_action_smooth_coef",
        "goal_xy_tolerance",
    }
)


def default_reward_config_path() -> str:
    return DEFAULT_REWARD_YAML


def resolve_reward_config_path(path: str | None) -> str:
    if path is None or str(path).strip() == "":
        p = DEFAULT_REWARD_YAML
    else:
        p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Файл настроек наград не найден: {p}")
    return p


def load_reward_config(path: str) -> dict[str, typing.Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Ожидается YAML-объект в корне, получено: {type(raw)}")
    if "env" not in raw or not isinstance(raw["env"], dict):
        raw["env"] = {}
    if "curriculum" in raw and raw["curriculum"] is not None:
        if not isinstance(raw["curriculum"], list):
            raise ValueError("reward_config: curriculum должен быть списком этапов")
    else:
        raw["curriculum"] = []
    if "play" not in raw or raw["play"] is None:
        raw["play"] = {}
    elif not isinstance(raw["play"], dict):
        raise ValueError("reward_config: play должен быть объектом")
    if "map" not in raw or raw["map"] is None:
        raw["map"] = {}
    elif not isinstance(raw["map"], dict):
        raise ValueError("reward_config: map должен быть объектом")
    return raw


def dqn_gamma_from_reward_cfg(raw: dict[str, typing.Any]) -> float | None:
    """
    Опционально: dqn.gamma или корневой dqn_gamma.
    Используется DQN / MCTS / PPO, если дисконт не задан в CLI и профиле.
    """
    dqn = raw.get("dqn")
    if isinstance(dqn, dict):
        v = dqn.get("gamma")
        if v is not None:
            return float(v)
    v = raw.get("dqn_gamma")
    if v is not None:
        return float(v)
    return None


def merge_mcts_train_args_from_reward_yaml(
    reward_cfg: dict[str, typing.Any],
    base: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """
    Секция ``mcts:`` в YAML эксперимента переопределяет аргументы ``train_mcts`` (поверх CLI и profile).
    Ключи YAML: gamma, mcts_sim | n_mcts_simulations, policy_updates, c_puct, root_dirichlet_*,
    visit_temperature, prior_label_smoothing, lr, max_grad_norm, value_coef, prior_coef, gae_lambda,
    n_vx_bins | latent_vx_bins, n_vy_bins | latent_vy_bins, vy_min, vy_max,
    curriculum_*, early_stop_*, curriculum_use_fallback.
    """
    m = reward_cfg.get("mcts")
    if not isinstance(m, dict) or not m:
        return dict(base)
    out = dict(base)
    pairs = [
        ("gamma", "gamma", float),
        ("mcts_sim", "n_mcts_simulations", int),
        ("n_mcts_simulations", "n_mcts_simulations", int),
        ("policy_updates", "policy_updates", int),
        ("c_puct", "mcts_c_puct", float),
        ("root_dirichlet_alpha", "root_dirichlet_alpha", float),
        ("root_dirichlet_eps", "root_dirichlet_eps", float),
        ("visit_temperature", "visit_temperature", float),
        ("prior_label_smoothing", "prior_label_smoothing", float),
        ("lr", "lr", float),
        ("max_grad_norm", "max_grad_norm", float),
        ("value_coef", "value_coef", float),
        ("prior_coef", "prior_coef", float),
        ("gae_lambda", "gae_lambda", float),
        ("n_vx_bins", "n_vx_bins", int),
        ("latent_vx_bins", "n_vx_bins", int),
        ("n_vy_bins", "n_vy_bins", int),
        ("latent_vy_bins", "n_vy_bins", int),
        ("vy_min", "vy_min", float),
        ("vy_max", "vy_max", float),
        ("curriculum_window", "curriculum_window", int),
        ("curriculum_min_episodes_per_stage", "curriculum_min_episodes_per_stage", int),
        ("curriculum_success_threshold", "curriculum_success_threshold", float),
        ("curriculum_fallback_max_mean_dist", "curriculum_fallback_max_mean_dist", float),
        ("curriculum_fallback_max_collision_rate", "curriculum_fallback_max_collision_rate", float),
        ("early_stop_patience", "early_stop_patience", int),
        ("early_stop_min_episodes", "early_stop_min_episodes", int),
    ]
    for yk, pk, typ in pairs:
        if yk in m and m[yk] is not None:
            v = m[yk]
            out[pk] = typ(v)
    if "curriculum_use_fallback" in m and m["curriculum_use_fallback"] is not None:
        out["curriculum_use_fallback"] = bool(m["curriculum_use_fallback"])
    return out


def merge_ppo_train_args_from_reward_yaml(
    reward_cfg: dict[str, typing.Any],
    base: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """
    Секция ``ppo:`` в YAML эксперимента переопределяет аргументы ``train_ppo`` (поверх CLI и profile).
    Ключи: gamma, policy_updates, gae_lambda, ppo_clip_eps, entropy_coef, entropy_coef_final,
    entropy_decay_fraction, value_coef, lr, max_grad_norm,
    n_vx_bins | latent_vx_bins, n_vy_bins | latent_vy_bins, vy_min, vy_max,
    use_curriculum,
    curriculum_*, early_stop_*, curriculum_use_fallback,
    entropy_floor_until_goal, entropy_high_out_rate_threshold, entropy_floor_when_high_out_rate.
    """
    m = reward_cfg.get("ppo")
    if not isinstance(m, dict) or not m:
        return dict(base)
    out = dict(base)
    pairs = [
        ("gamma", "gamma", float),
        ("policy_updates", "policy_updates", int),
        ("gae_lambda", "gae_lambda", float),
        ("ppo_clip_eps", "ppo_clip_eps", float),
        ("entropy_coef", "entropy_coef", float),
        ("entropy_coef_final", "entropy_coef_final", float),
        ("entropy_decay_fraction", "entropy_decay_fraction", float),
        ("value_coef", "value_coef", float),
        ("lr", "lr", float),
        ("max_grad_norm", "max_grad_norm", float),
        ("n_vx_bins", "n_vx_bins", int),
        ("latent_vx_bins", "n_vx_bins", int),
        ("n_vy_bins", "n_vy_bins", int),
        ("latent_vy_bins", "n_vy_bins", int),
        ("vy_min", "vy_min", float),
        ("vy_max", "vy_max", float),
        ("curriculum_window", "curriculum_window", int),
        ("curriculum_min_episodes_per_stage", "curriculum_min_episodes_per_stage", int),
        ("curriculum_success_threshold", "curriculum_success_threshold", float),
        ("curriculum_fallback_max_mean_dist", "curriculum_fallback_max_mean_dist", float),
        ("curriculum_fallback_max_collision_rate", "curriculum_fallback_max_collision_rate", float),
        ("early_stop_patience", "early_stop_patience", int),
        ("early_stop_min_episodes", "early_stop_min_episodes", int),
        ("entropy_floor_until_goal", "entropy_floor_until_goal", float),
        ("entropy_high_out_rate_threshold", "entropy_high_out_rate_threshold", float),
        ("entropy_floor_when_high_out_rate", "entropy_floor_when_high_out_rate", float),
    ]
    for yk, pk, typ in pairs:
        if yk in m and m[yk] is not None:
            out[pk] = typ(m[yk])
    if "curriculum_use_fallback" in m and m["curriculum_use_fallback"] is not None:
        out["curriculum_use_fallback"] = bool(m["curriculum_use_fallback"])
    if "use_curriculum" in m and m["use_curriculum"] is not None:
        out["use_curriculum"] = bool(m["use_curriculum"])
    return out


def merge_map_train_args_from_reward_yaml(
    reward_cfg: dict[str, typing.Any],
    base: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """
    Секция ``map:`` в YAML эксперимента переопределяет базовые параметры среды/дискретизации
    (поверх CLI и profile): map_type, path_length, num_obstacles, vy_*, n_v*_bins.
    """
    m = reward_cfg.get("map")
    if not isinstance(m, dict) or not m:
        return dict(base)
    out = dict(base)
    pairs = [
        ("map_type", "map_type", str),
        ("path_length", "path_length", float),
        ("obstacle_path_length", "path_length", float),
        ("num_obstacles", "num_obstacles", int),
        ("vy_min", "vy_min", float),
        ("vy_max", "vy_max", float),
        ("n_vx_bins", "n_vx_bins", int),
        ("latent_vx_bins", "n_vx_bins", int),
        ("n_vy_bins", "n_vy_bins", int),
        ("latent_vy_bins", "n_vy_bins", int),
    ]
    for yk, pk, typ in pairs:
        if yk in m and m[yk] is not None:
            out[pk] = typ(m[yk])
    return out


def dqn_epsilon_decay_steps_from_reward_cfg(raw: dict[str, typing.Any]) -> int | None:
    """
    Опционально в YAML эксперимента: секция dqn.epsilon_decay_steps или корневой dqn_epsilon_decay_steps.
    Используется train DQN, если CLI --dqn-epsilon-decay-steps не задан.
    """
    dqn = raw.get("dqn")
    if isinstance(dqn, dict):
        v = dqn.get("epsilon_decay_steps")
        if v is not None:
            return int(v)
    v = raw.get("dqn_epsilon_decay_steps")
    if v is not None:
        return int(v)
    return None


def copy_reward_config_to_run(src_path: str, run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    dest = os.path.join(run_dir, RUN_COPY_NAME)
    shutil.copy2(src_path, dest)
    return dest


def _geometry_for_stage(name: str, pl: float, base_num_obstacles: int) -> dict[str, float | int]:
    pl = float(pl)
    n = int(base_num_obstacles)
    if name == "single_center_easy":
        return {
            "path_length": min(1.5, max(0.9, 0.45 * pl)),
            "num_obstacles": 1,
        }
    if name == "single_center_mid":
        return {
            "path_length": min(2.4, max(1.3, 0.75 * pl)),
            "num_obstacles": 1,
        }
    if name == "random_small":
        return {
            "path_length": min(3.0, max(2.0, pl)),
            "num_obstacles": max(2, min(4, n)),
        }
    if name == "target_final":
        return {
            "path_length": pl,
            "num_obstacles": n,
        }
    return {}


def build_curriculum_stages(
    raw_stages: list[dict[str, typing.Any]],
    path_length: float,
    num_obstacles: int,
) -> list[dict[str, typing.Any]]:
    pl = float(path_length)
    n_obs = int(num_obstacles)
    out: list[dict[str, typing.Any]] = []
    for s in raw_stages:
        if not isinstance(s, dict):
            raise ValueError("каждый этап curriculum должен быть объектом YAML")
        name = s.get("name")
        if not name:
            raise ValueError("у этапа curriculum нужно поле name")
        geom = _geometry_for_stage(str(name), pl, n_obs)
        merged = {**geom, **s}
        if "path_length" not in merged or "num_obstacles" not in merged:
            raise ValueError(
                f"этап {name!r}: задайте path_length и num_obstacles в YAML либо используйте известное name "
                f"(single_center_easy|single_center_mid|random_small|target_final)"
            )
        out.append(merged)
    return out


def apply_env_reward_dict(env: typing.Any, env_cfg: dict[str, typing.Any]) -> None:
    for k, v in env_cfg.items():
        if k not in ENV_REWARD_KEYS:
            continue
        if not hasattr(env, k):
            continue
        if isinstance(v, bool):
            setattr(env, k, v)
        elif isinstance(v, (int, float)):
            setattr(env, k, float(v))
        else:
            setattr(env, k, v)


def apply_curriculum_stage_full(env: typing.Any, stage: dict[str, typing.Any]) -> None:
    """Как в scenaries.train_ppo: геометрия карты + награды этапа."""
    env.map_type = str(stage["map_type"])
    env.obstacle_path_length = float(stage["path_length"])
    env.num_obstacles = int(stage["num_obstacles"])
    env.single_center_radius = float(stage.get("single_center_radius", env.single_center_radius))
    env.goal_xy_tolerance = float(stage["goal_xy_tolerance"])
    env.reward_goal_reached = float(stage["reward_goal_reached"])
    env.reward_goal_time_coef = float(stage["reward_goal_time_coef"])
    env.reward_goal_distance_progress_coef = float(stage["reward_goal_distance_progress_coef"])
    env.reward_survival = float(stage["reward_survival"])
    if "reward_collision" in stage:
        env.reward_collision = float(stage["reward_collision"])
    if "reward_exit_failure" in stage:
        env.reward_exit_failure = float(stage["reward_exit_failure"])


def apply_play_reward_overrides(env: typing.Any, cfg: dict[str, typing.Any]) -> None:
    """
    Для play: секция env + при необходимости награды/goal из последнего этапа curriculum
    (карта задаётся CLI play, не перезаписывается этапом).
    """
    apply_env_reward_dict(env, cfg.get("env", {}))
    play = cfg.get("play") or {}
    use_last = bool(play.get("apply_last_curriculum_stage_rewards", True))
    stages = cfg.get("curriculum") or []
    if use_last and stages:
        st = dict(stages[-1])
        for k, v in st.items():
            if k.startswith("reward_") and hasattr(env, k) and isinstance(v, (int, float)):
                setattr(env, k, float(v))
        if "goal_xy_tolerance" in st:
            env.goal_xy_tolerance = float(st["goal_xy_tolerance"])


def resolve_play_reward_config_path(
    explicit: str | None,
    run_dir: str | None,
    policy_path: str,
) -> str:
    if explicit and str(explicit).strip():
        return resolve_reward_config_path(explicit)
    candidates = []
    if run_dir:
        candidates.append(os.path.join(os.path.abspath(run_dir), RUN_COPY_NAME))
    policy_dir = os.path.dirname(os.path.abspath(policy_path))
    candidates.append(os.path.join(policy_dir, RUN_COPY_NAME))
    candidates.append(DEFAULT_REWARD_YAML)
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Не найден reward_config.yaml: укажите --reward-config или положите файл рядом с весами / в каталог run."
    )
