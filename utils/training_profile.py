# utils/training_profile.py - YAML-профили гиперпараметров PPO / DQN / MCTS

import os
import typing

import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_training_profile(path: str | None) -> dict[str, typing.Any]:
    """
    Читает плоский dict из YAML. Пустой/отсутствующий путь --> {}.
    Неизвестные ключи вызывающий код может игнорировать.
    """
    if path is None or str(path).strip() == "":
        return {}
    p = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Файл профиля обучения не найден: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Профиль обучения: ожидается объект в корне YAML, получено {type(raw)}")
    return raw


def default_profiles_dir() -> str:
    return os.path.join(_REPO_ROOT, "config", "profiles")


def merge_ppo_train_hyperparams(
    profile: typing.Mapping[str, typing.Any],
    *,
    policy_updates: int,
    gamma: typing.Optional[float],
    entropy_coef: float,
    entropy_coef_final: float,
    entropy_decay_fraction: float,
    curriculum_success_threshold: typing.Optional[float],
    curriculum_window: typing.Optional[int],
    curriculum_min_episodes_per_stage: typing.Optional[int],
    entropy_floor_until_goal: float,
    early_stop_patience: int,
    early_stop_min_episodes: int,
) -> dict[str, typing.Any]:
    """
    Сначала дефолты curriculum из CLI (None --> 0.8 / 40 / 30), затем поверх — ключи из profile
    (curriculum из профиля только если пользователь не задал явный CLI).
    gamma: None — не задан в CLI; тогда может взяться из profile; иначе train_ppo подставит из YAML наград.
    """
    gamma_user = gamma
    cst_user = curriculum_success_threshold
    cw_user = curriculum_window
    cmin_user = curriculum_min_episodes_per_stage
    cst = 0.8 if cst_user is None else float(cst_user)
    cw = 40 if cw_user is None else int(cw_user)
    cmin = 30 if cmin_user is None else int(cmin_user)
    pu = int(policy_updates)
    g_res: typing.Optional[float] = float(gamma_user) if gamma_user is not None else None
    ec = float(entropy_coef)
    ecf = float(entropy_coef_final)
    edf = float(entropy_decay_fraction)
    efg = float(entropy_floor_until_goal)
    esp = int(early_stop_patience)
    esm = int(early_stop_min_episodes)
    p = dict(profile)
    if p:
        if "policy_updates" in p:
            pu = int(p["policy_updates"])
        if g_res is None and "gamma" in p:
            g_res = float(p["gamma"])
        if "entropy_coef" in p:
            ec = float(p["entropy_coef"])
        if "entropy_coef_final" in p:
            ecf = float(p["entropy_coef_final"])
        if "entropy_decay_fraction" in p:
            edf = float(p["entropy_decay_fraction"])
        if cw_user is None and "curriculum_window" in p:
            cw = int(p["curriculum_window"])
        if cmin_user is None and "curriculum_min_episodes_per_stage" in p:
            cmin = int(p["curriculum_min_episodes_per_stage"])
        if cst_user is None and "curriculum_success_threshold" in p:
            cst = float(p["curriculum_success_threshold"])
        if "entropy_floor_until_goal" in p:
            efg = float(p["entropy_floor_until_goal"])
        if "early_stop_patience" in p:
            esp = int(p["early_stop_patience"])
        if "early_stop_min_episodes" in p:
            esm = int(p["early_stop_min_episodes"])
    return {
        "policy_updates": pu,
        "gamma": g_res,
        "entropy_coef": ec,
        "entropy_coef_final": ecf,
        "entropy_decay_fraction": edf,
        "curriculum_window": cw,
        "curriculum_min_episodes_per_stage": cmin,
        "curriculum_success_threshold": cst,
        "entropy_floor_until_goal": efg,
        "early_stop_patience": esp,
        "early_stop_min_episodes": esm,
    }


def merge_dqn_train_hyperparams(
    profile: typing.Mapping[str, typing.Any],
    *,
    early_stop_patience: int,
    early_stop_min_episodes: int,
    curriculum_success_threshold: typing.Optional[float],
    curriculum_window: typing.Optional[int],
    curriculum_min_episodes_per_stage: typing.Optional[int],
    batch_size: int,
    buffer_capacity: int,
    learning_starts: int,
    train_frequency: int,
    target_update_every: int,
) -> dict[str, typing.Any]:
    cst_user = curriculum_success_threshold
    cw_user = curriculum_window
    cmin_user = curriculum_min_episodes_per_stage
    cst = 0.8 if cst_user is None else float(cst_user)
    cw = 40 if cw_user is None else int(cw_user)
    cmin = 30 if cmin_user is None else int(cmin_user)
    esp = int(early_stop_patience)
    esm = int(early_stop_min_episodes)
    bs = int(batch_size)
    buf = int(buffer_capacity)
    ls = int(learning_starts)
    tf = int(train_frequency)
    tu = int(target_update_every)
    p = dict(profile)
    if p:
        if "early_stop_patience" in p:
            esp = int(p["early_stop_patience"])
        if "early_stop_min_episodes" in p:
            esm = int(p["early_stop_min_episodes"])
        if cw_user is None and "curriculum_window" in p:
            cw = int(p["curriculum_window"])
        if cmin_user is None and "curriculum_min_episodes_per_stage" in p:
            cmin = int(p["curriculum_min_episodes_per_stage"])
        if cst_user is None and "curriculum_success_threshold" in p:
            cst = float(p["curriculum_success_threshold"])
        if "batch_size" in p:
            bs = int(p["batch_size"])
        if "buffer_capacity" in p:
            buf = int(p["buffer_capacity"])
        if "learning_starts" in p:
            ls = int(p["learning_starts"])
        if "train_frequency" in p:
            tf = int(p["train_frequency"])
        if "target_update_every" in p:
            tu = int(p["target_update_every"])
    return {
        "early_stop_patience": esp,
        "early_stop_min_episodes": esm,
        "curriculum_success_threshold": cst,
        "curriculum_window": cw,
        "curriculum_min_episodes_per_stage": cmin,
        "batch_size": bs,
        "buffer_capacity": buf,
        "learning_starts": ls,
        "train_frequency": tf,
        "target_update_every": tu,
    }


def merge_mcts_train_hyperparams(
    profile: typing.Mapping[str, typing.Any],
    *,
    policy_updates: int,
    n_mcts_simulations: int,
    mcts_c_puct: float,
    root_dirichlet_alpha: float,
    root_dirichlet_eps: float,
    visit_temperature: float,
    prior_label_smoothing: float,
    lr: float,
    curriculum_success_threshold: typing.Optional[float],
    curriculum_window: typing.Optional[int],
    curriculum_min_episodes_per_stage: typing.Optional[int],
    early_stop_patience: int,
    early_stop_min_episodes: int,
) -> dict[str, typing.Any]:
    """Профиль YAML (config/profiles/*.yaml) накладывается поверх значений CLI (как у PPO/DQN)."""
    cst_user = curriculum_success_threshold
    cw_user = curriculum_window
    cmin_user = curriculum_min_episodes_per_stage
    cst = 0.8 if cst_user is None else float(cst_user)
    cw = 40 if cw_user is None else int(cw_user)
    cmin = 30 if cmin_user is None else int(cmin_user)
    pu = int(policy_updates)
    nsim = int(n_mcts_simulations)
    cp = float(mcts_c_puct)
    rda = float(root_dirichlet_alpha)
    rde = float(root_dirichlet_eps)
    vt = float(visit_temperature)
    pls = float(prior_label_smoothing)
    lr_v = float(lr)
    esp = int(early_stop_patience)
    esm = int(early_stop_min_episodes)
    mgn = 0.5
    vc = 1.0
    prc = 0.5
    gl = 0.9
    p = dict(profile)
    if p:
        if "policy_updates" in p:
            pu = int(p["policy_updates"])
        if "mcts_sim" in p:
            nsim = int(p["mcts_sim"])
        elif "n_mcts_simulations" in p:
            nsim = int(p["n_mcts_simulations"])
        if "c_puct" in p:
            cp = float(p["c_puct"])
        if "root_dirichlet_alpha" in p:
            rda = float(p["root_dirichlet_alpha"])
        if "root_dirichlet_eps" in p:
            rde = float(p["root_dirichlet_eps"])
        if "visit_temperature" in p:
            vt = float(p["visit_temperature"])
        if "prior_label_smoothing" in p:
            pls = float(p["prior_label_smoothing"])
        if "lr" in p:
            lr_v = float(p["lr"])
        if "max_grad_norm" in p:
            mgn = float(p["max_grad_norm"])
        if "value_coef" in p:
            vc = float(p["value_coef"])
        if "prior_coef" in p:
            prc = float(p["prior_coef"])
        if "gae_lambda" in p:
            gl = float(p["gae_lambda"])
        if cw_user is None and "curriculum_window" in p:
            cw = int(p["curriculum_window"])
        if cmin_user is None and "curriculum_min_episodes_per_stage" in p:
            cmin = int(p["curriculum_min_episodes_per_stage"])
        if cst_user is None and "curriculum_success_threshold" in p:
            cst = float(p["curriculum_success_threshold"])
        if "early_stop_patience" in p:
            esp = int(p["early_stop_patience"])
        if "early_stop_min_episodes" in p:
            esm = int(p["early_stop_min_episodes"])
    return {
        "policy_updates": pu,
        "n_mcts_simulations": nsim,
        "c_puct": cp,
        "root_dirichlet_alpha": rda,
        "root_dirichlet_eps": rde,
        "visit_temperature": vt,
        "prior_label_smoothing": pls,
        "lr": lr_v,
        "max_grad_norm": mgn,
        "value_coef": vc,
        "prior_coef": prc,
        "gae_lambda": gl,
        "curriculum_success_threshold": cst,
        "curriculum_window": cw,
        "curriculum_min_episodes_per_stage": cmin,
        "early_stop_patience": esp,
        "early_stop_min_episodes": esm,
    }
