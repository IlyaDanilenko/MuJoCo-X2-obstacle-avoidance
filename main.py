# main.py - точка входа CLI, конфиги и профили; сценарии в scenaries/

import argparse
import os
import typing

from scenaries.train_dqn import train as train_dqn
from scenaries.train_mcts import train as train_mcts
from scenaries.train_ppo import train as train_ppo
from scenaries.depth_view import run_depth_view
from scenaries.env_test import run_env_test
from scenaries.imu_test import run_imu_test
from scenaries.play import run_play
from scenaries.random_policy import run_random_policy
from scenaries.view_environment import view_environment
from teach_ml.envencoder import train_envencoder
from utils.reward_config import (
    load_reward_config,
    merge_map_train_args_from_reward_yaml,
    resolve_reward_config_path,
)
from utils.training_profile import (
    load_training_profile,
    merge_dqn_train_hyperparams,
    merge_mcts_train_hyperparams,
    merge_ppo_train_hyperparams,
)
from utils.training_logging import load_train_run_config_for_play, resolve_play_train_config_dir


def _optional_profile_abspath(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    s = str(path_str).strip()
    if not s:
        return None
    return os.path.abspath(os.path.expanduser(s))


_MAP_TYPES = ("random", "single_center")
_DEVICE = ("cpu", "mps", "cuda")
_TRAIN_ALGOS = ("mcts", "ppo", "dqn")
_PLAY_ALGOS = ("mcts", "ppo", "dqn")


def _add_scene_map_path(
    ap: typing.Any,
    *,
    num_obstacles: int,
    map_default: str,
    path_length: float,
    num_help: str | None = None,
) -> None:
    ap.add_argument(
        "--num-obstacles",
        type=int,
        default=num_obstacles,
        help=num_help or "Максимальное количество препятствий",
    )
    ap.add_argument(
        "--map-type",
        type=str,
        default=map_default,
        choices=_MAP_TYPES,
        help="Тип карты: random или single_center (столб по центру траектории)",
    )
    ap.add_argument(
        "--path-length",
        type=float,
        default=path_length,
        help="Длина участка генерации препятствий (меньше — плотнее сцена)",
    )


def _add_device(ap: typing.Any, *, default: str, help_text: str) -> None:
    ap.add_argument("--device", type=str, default=default, choices=_DEVICE, help=help_text)


def _add_fastdepth_weights(ap: typing.Any, *, help_text: str) -> None:
    ap.add_argument("--fastdepth-weights", type=str, default=None, help=help_text)


if __name__ == "__main__":
    DEFAULT_MAX_OBSTACLE = 50

    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Команда для выполнения",
        required=True,
    )
    
    # Парсер для сценария случайного перемещения
    random_parser = subparsers.add_parser('random', help='Сценарий случайного перемещения дрона (случайные действия)')
    random_parser.add_argument('--no-render', action='store_true',
                             help='Отключить визуализацию')
    random_parser.add_argument('--steps', type=int, default=500,
                             help='Максимум шагов за эпизод')
    random_parser.add_argument('--episodes', type=int, default=1,
                             help='Количество эпизодов')
    random_parser.add_argument('--num-obstacles', type=int, default=DEFAULT_MAX_OBSTACLE,
                             help='Максимальное количество препятствий')
    random_parser.add_argument('--seed', type=int, default=None,
                             help='Семя для воспроизводимости (опционально)')
    
    view_parser = subparsers.add_parser("view", help="Просмотр environment с препятствиями")
    view_parser.set_defaults(render=True)
    view_parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Отключить визуализацию (по умолчанию окно включено)",
    )
    view_parser.add_argument("--steps", type=int, default=1000, help="Количество шагов симуляции")
    view_parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Без лимита по шагам: до столкновения, границы среды или Ctrl+C (--steps игнорируется)",
    )
    _add_scene_map_path(
        view_parser,
        num_obstacles=DEFAULT_MAX_OBSTACLE,
        map_default="random",
        path_length=5.0,
        num_help="Максимальное количество препятствий",
    )
    
    train_parser = subparsers.add_parser(
        "train",
        help="Обучение в латенте: MCTS, PPO или DQN (--algo)",
    )
    tr_common = train_parser.add_argument_group("общее")
    tr_common.add_argument(
        "--algo",
        type=str,
        default="mcts",
        choices=list(_TRAIN_ALGOS),
        help="mcts | ppo | dqn",
    )
    tr_common.add_argument(
        "--encoder-path",
        type=str,
        required=True,
        help="Веса EnvEncoder (.pth), напр. models/envencoder/run/encoder_epoch_50.pth",
    )
    tr_common.add_argument("--episodes", type=int, default=1000, help="Эпизоды обучения")
    tr_common.add_argument("--max-steps", type=int, default=1000, help="Макс. шагов в эпизоде")
    tr_common.add_argument("--save-interval", type=int, default=10, help="Сохранять чекпойнт каждые N эпизодов")
    tr_common.add_argument(
        "--model-dir",
        type=str,
        default="models/mcts-latent",
        help="Корень RL-чекпойнтов (run, run1, ...)",
    )
    _add_device(tr_common, default="mps", help_text="Устройство для обучения")
    tr_common.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard (по умолчанию <model_dir>/tensorboard)",
    )
    tr_common.add_argument("--seed", type=int, default=42, help="Сид воспроизводимости")
    tr_common.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Дисконт (PPO/DQN/MCTS); по умолчанию 0.8; для DQN/MCTS без флага — из YAML наград (dqn.gamma), иначе 0.8",
    )
    tr_common.add_argument(
        "--profile-path",
        type=str,
        default=None,
        help="YAML профиль (config/profiles/*.yaml): mcts / ppo / dqn",
    )
    _add_fastdepth_weights(
        tr_common,
        help_text="FastDepth .pth; по умолчанию pretrained_model/FastDepth_L1GN_Best.pth",
    )

    tr_scene = train_parser.add_argument_group("сцена")
    _add_scene_map_path(
        tr_scene,
        num_obstacles=50,
        map_default="single_center",
        path_length=3.0,
        num_help="Макс. препятствий в сцене",
    )

    tr_mcts = train_parser.add_argument_group("MCTS (--algo mcts)")
    tr_mcts.add_argument("--mcts-sim", type=int, default=11, help="Симуляций MCTS на шаг")
    tr_mcts.add_argument(
        "--policy-updates",
        type=int,
        default=2,
        help="Шагов Adam на батче одного эпизода (MCTS; для PPO подмешивается из профиля)",
    )
    tr_mcts.add_argument("--c-puct", type=float, default=1.5, help="PUCT c")
    tr_mcts.add_argument("--root-dirichlet-alpha", type=float, default=0.3, help="Dirichlet в корне (0 — выкл.)")
    tr_mcts.add_argument("--root-dirichlet-eps", type=float, default=0.25, help="Вес Dirichlet в корне (0..1)")
    tr_mcts.add_argument("--visit-temp", type=float, default=1.0, help="Температура по visit counts (0 — argmax)")
    tr_mcts.add_argument(
        "--prior-label-smoothing",
        type=float,
        default=0.05,
        help="Сглаживание таргета prior по visit counts (0..1)",
    )

    tr_latent = train_parser.add_argument_group("латент: сетка vx/vy и curriculum (ppo, dqn, mcts)")
    tr_latent.add_argument("--latent-vx-bins", type=int, default=11, help="Уровней vx в дискретной сетке")
    tr_latent.add_argument("--latent-vy-bins", type=int, default=5, help="Уровней vy (1 = только vx)")
    tr_latent.add_argument("--latent-vy-min", type=float, default=0.01, help="Нижняя граница vy")
    tr_latent.add_argument("--latent-vy-max", type=float, default=0.75, help="Верхняя граница vy")
    tr_latent.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (ppo/dqn/mcts); профиль YAML может переопределить",
    )
    tr_latent.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Эпизодов без улучшения avg_reward до остановки",
    )
    tr_latent.add_argument(
        "--early-stop-min-episodes",
        type=int,
        default=30,
        help="Не раньше этого эпизода включать early stop",
    )
    tr_latent.add_argument("--no-curriculum", action="store_true", help="Выключить curriculum")
    tr_latent.add_argument(
        "--curriculum-success-threshold",
        type=float,
        default=None,
        help="Порог success-rate; иначе из YAML или 0.8",
    )
    tr_latent.add_argument(
        "--curriculum-window",
        type=int,
        default=None,
        help="Окно эпизодов для success-rate; иначе из YAML или 40",
    )
    tr_latent.add_argument(
        "--curriculum-min-episodes-per-stage",
        type=int,
        default=None,
        help="Мин. эпизодов на stage; иначе из YAML или 30",
    )
    tr_latent.add_argument(
        "--no-curriculum-fallback",
        action="store_true",
        help="Только success_rate для смены stage (без mean_dist/collision)",
    )
    tr_latent.add_argument(
        "--curriculum-fallback-max-mean-dist",
        type=float,
        default=0.55,
        help="Порог mean_dist_to_goal_end для fallback stage",
    )
    tr_latent.add_argument(
        "--curriculum-fallback-max-collision",
        type=float,
        default=0.38,
        help="Макс. collision_rate в окне для fallback",
    )

    tr_reward = train_parser.add_argument_group("награды и curriculum (ppo, dqn, mcts)")
    tr_reward.add_argument(
        "--reward-config",
        type=str,
        default=None,
        help="YAML наград/curriculum; по умолчанию config/reward_config.yaml (копия в каталог run)",
    )

    tr_ppo = train_parser.add_argument_group("PPO (--algo ppo)")
    tr_ppo.add_argument("--ppo-clip-eps", type=float, default=0.2, help="PPO clip ε")
    tr_ppo.add_argument("--ppo-entropy-coef", type=float, default=0.01, help="Коэфф. энтропии")
    tr_ppo.add_argument("--ppo-entropy-coef-final", type=float, default=0.003, help="Финальный коэфф. энтропии")
    tr_ppo.add_argument(
        "--ppo-entropy-decay-fraction",
        type=float,
        default=0.6,
        help="Доля обучения до decay энтропии до final",
    )
    tr_ppo.add_argument("--ppo-value-coef", type=float, default=0.5, help="Вес value-loss")
    tr_ppo.add_argument("--ppo-gae-lambda", type=float, default=0.95, help="GAE λ")
    tr_ppo.add_argument(
        "--ppo-entropy-floor-until-goal",
        type=float,
        default=0.012,
        help="Мин. энтропия, пока не было goal_reached",
    )
    tr_ppo.add_argument(
        "--ppo-entropy-high-out-threshold",
        type=float,
        default=0.5,
        help="При out_rate выше — поднять энтропию (см. --ppo-entropy-floor-high-out)",
    )
    tr_ppo.add_argument(
        "--ppo-entropy-floor-high-out",
        type=float,
        default=0.02,
        help="Мин. энтропия при высоком out_rate",
    )

    tr_dqn = train_parser.add_argument_group("DQN (--algo dqn)")
    tr_dqn.add_argument("--dqn-batch-size", type=int, default=128, help="Размер мини-батча")
    tr_dqn.add_argument("--dqn-buffer-capacity", type=int, default=200_000, help="Replay buffer")
    tr_dqn.add_argument("--dqn-learning-starts", type=int, default=10_000, help="Шагов до первых градиентов")
    tr_dqn.add_argument("--dqn-train-frequency", type=int, default=4, help="Градиент каждые N шагов среды")
    tr_dqn.add_argument("--dqn-target-update-every", type=int, default=1000, help="Копировать target каждые N шагов")
    tr_dqn.add_argument("--dqn-epsilon-start", type=float, default=1.0, help="Начальное ε")
    tr_dqn.add_argument("--dqn-epsilon-end", type=float, default=0.05, help="Конечное ε")
    tr_dqn.add_argument(
        "--dqn-epsilon-decay-steps",
        type=int,
        default=None,
        help="Шагов на спад ε; иначе из YAML (dqn.epsilon_decay_steps / dqn_epsilon_decay_steps), иначе episodesxmax-steps",
    )
    tr_dqn.add_argument("--dqn-max-grad-norm", type=float, default=10.0, help="Клип градиента Q-сети")

    play_parser = subparsers.add_parser(
        "play",
        help="Инференс: EnvEncoder-->z, MCTS / PPO / DQN",
    )
    pl_w = play_parser.add_argument_group("веса")
    pl_w.add_argument(
        "--encoder-path",
        type=str,
        default=None,
        help="EnvEncoder .pth (обязательно, если не в run)",
    )
    pl_w.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Путь к .pth политики / Q-сети",
    )
    pl_w.add_argument(
        "--run",
        type=str,
        default=None,
        help="Каталог run: latent_policy_final.pth или latent_dqn_final.pth",
    )
    _add_fastdepth_weights(
        pl_w,
        help_text="FastDepth; по умолчанию как при train",
    )

    pl_run = play_parser.add_argument_group("эпизод")
    pl_run.add_argument("--episodes", type=int, default=3, help="Число эпизодов")
    pl_run.add_argument("--max-steps", type=int, default=500, help="Макс. шагов за эпизод")
    pl_run.add_argument("--no-render", action="store_true", help="Без окна MuJoCo")
    pl_run.add_argument("--seed", type=int, default=None, help="Сид")
    pl_run.add_argument(
        "--no-limit",
        action="store_true",
        help="До столкновения или Ctrl+C",
    )
    _add_device(pl_run, default="mps", help_text="Устройство инференса")

    pl_scene = play_parser.add_argument_group("сцена")
    _add_scene_map_path(
        pl_scene,
        num_obstacles=50,
        map_default="random",
        path_length=5.0,
        num_help="Макс. препятствий",
    )

    pl_mcts = play_parser.add_argument_group("MCTS / play")
    pl_mcts.add_argument("--mcts-sim", type=int, default=24, help="Симуляций MCTS на шаг")
    pl_mcts.add_argument("--c-puct", type=float, default=1.5, help="PUCT c")
    pl_mcts.add_argument("--gamma", type=float, default=0.8, help="Дисконт листа MCTS")
    pl_mcts.add_argument(
        "--play-algo",
        type=str,
        default="mcts",
        choices=list(_PLAY_ALGOS),
        help="mcts | ppo | dqn",
    )
    pl_mcts.add_argument("--latent-vx-bins", type=int, default=11, help="Бинов vx (латент ppo/dqn)")
    pl_mcts.add_argument("--latent-vy-bins", type=int, default=5, help="Бинов vy (латент ppo/dqn)")
    pl_mcts.add_argument("--latent-vy-min", type=float, default=0.01, help="min vy (латент ppo/dqn)")
    pl_mcts.add_argument("--latent-vy-max", type=float, default=0.75, help="max vy (латент ppo/dqn)")
    pl_mcts.add_argument(
        "--max-velocity",
        type=float,
        default=3.0,
        dest="play_max_velocity",
        help="Лимит скорости в среде",
    )
    pl_mcts.add_argument(
        "--forward-velocity",
        type=float,
        default=0.25,
        dest="play_forward_velocity",
        help="Базовый |vy| для действия из одного числа",
    )
    pl_mcts.add_argument(
        "--reward-config",
        type=str,
        default=None,
        help="YAML наград",
    )

    depth_view_parser = subparsers.add_parser(
        'depth-view',
        help='Взлёт (как в reset), зависание, сохранение карт глубины FastDepth в test_depth',
    )
    depth_view_parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Путь к .pth; по умолчанию pretrained_model/FastDepth_L1GN_Best.pth',
    )
    depth_view_parser.add_argument('--out-dir', type=str, default='test_depth',
                                   help='Папка для PNG глубины')
    depth_view_parser.add_argument('--steps', type=int, default=40,
                                   help='Число шагов симуляции после reset (кадр на шаг)')
    _add_device(depth_view_parser, default="cpu", help_text="Устройство для FastDepth")
    depth_view_parser.add_argument('--seed', type=int, default=42)
    depth_view_parser.add_argument('--num-obstacles', type=int, default=30)

    imu_test_parser = subparsers.add_parser(
        "imu-test",
        help="Пустая сцена, квадрат 1x1 м, график XY: одометрия IMU vs симулятор --> папка test",
    )
    imu_test_parser.add_argument(
        "--out-dir",
        type=str,
        default="test",
        help="Каталог для imu_square_xy.png",
    )
    imu_test_parser.add_argument(
        "--render",
        action="store_true",
        help="Визуализация MuJoCo (human)",
    )
    imu_test_parser.add_argument(
        "--max-velocity",
        type=float,
        default=0.4,
        dest="max_velocity",
        help="Лимит скорости планировщика (м/с); выше ~0.5 квадрат часто уходит в срыв",
    )

    env_train_parser = subparsers.add_parser(
        "env-train",
        help="Обучение envencoder (VAE): depth + IMU --> латент 256; датасет в datasets/",
    )
    env_train_parser.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        help="Корень datasets (по умолчанию <корень_репо>/datasets): все подпапки с depth/imu "
        "сшиваются в один датасет, батчи перемешиваются между попытками",
    )
    env_train_parser.add_argument(
        "--env-dataset-root",
        type=str,
        default=None,
        dest="env_dataset_root",
        help="Только эта попытка (depth/, imu/), без слияния с остальными; иначе используется --datasets-dir",
    )
    env_train_parser.add_argument(
        "--model-root",
        type=str,
        default="models/envencoder",
        help="Корень для сохранения run / runN: encoder_epoch_K.pth, decoder_epoch_K.pth",
    )
    env_train_parser.add_argument("--epochs", type=int, default=50)
    env_train_parser.add_argument("--batch-size", type=int, default=16)
    env_train_parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Adam LR для encoder+decoder (VAE)",
    )
    env_train_parser.add_argument("--beta-kl", type=float, default=1e-3, dest="beta_kl")
    env_train_parser.add_argument(
        "--imu-scale",
        type=float,
        default=1e4,
        help="Масштаб dX,dY из odom_*.log перед обучением (симметрично в лоссе)",
    )
    env_train_parser.add_argument("--latent-dim", type=int, default=256)
    env_train_parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Сохранять encoder_epoch_K.pth / decoder_epoch_K.pth каждые K эпох и на последней",
    )
    env_train_parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard (по умолчанию <run_dir>/tensorboard)",
    )
    _add_device(env_train_parser, default="mps", help_text="Устройство обучения энкодера")
    env_train_parser.add_argument("--num-workers", type=int, default=0)
    env_train_parser.add_argument("--seed", type=int, default=42)
    env_train_parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        dest="grad_clip_max_norm",
        help="L2-клип градиентов E+D (0 = отключить)",
    )

    env_test_parser = subparsers.add_parser(
        "env-test",
        help="Инференс envencoder: depth .npy + odom .log --> depth_recon и odom_recon в папку test",
    )
    env_test_parser.add_argument(
        "--depth-npy",
        type=str,
        required=True,
        help="Путь к карте глубины (float32, 224x224), как в датасете",
    )
    env_test_parser.add_argument(
        "--odom-log",
        type=str,
        required=True,
        help="Путь к odom_*.log (две координаты смещения за шаг)",
    )
    env_test_parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Веса энкодера (например encoder_epoch_10.pth)",
    )
    env_test_parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Веса декодера (например decoder_epoch_10.pth)",
    )
    env_test_parser.add_argument(
        "--out-dir",
        type=str,
        default="test",
        help="Каталог: depth_original.png, depth_recon.npy/.png, odom_recon.log",
    )
    _add_device(env_test_parser, default="cpu", help_text="Устройство инференса")
    env_test_parser.add_argument(
        "--imu-scale",
        type=float,
        default=1e4,
        dest="imu_scale",
        help="Как при env-train: масштаб входного/выходного odom",
    )
    env_test_parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Сэмплировать z из q(z|x) вместо детерминированного μ",
    )
    env_test_parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.command == 'play':
        play_cfg_dir = resolve_play_train_config_dir(args)
        train_yaml_applied, train_env_snapshot = load_train_run_config_for_play(
            args, play_parser, play_cfg_dir
        )
        if train_yaml_applied:
            print(f"Play: дефолты из train_run_config — {train_yaml_applied} (CLI переопределяет совпадающие с дефолтом)")
        if args.policy:
            policy_path = os.path.abspath(args.policy)
        elif args.run:
            run_dir = os.path.abspath(args.run)
            if args.play_algo == "dqn":
                policy_path = os.path.join(run_dir, "latent_dqn_final.pth")
            else:
                policy_path = os.path.join(run_dir, "latent_policy_final.pth")
        else:
            parser.error("Для play укажите --policy <path.pth> или --run <папка_run>")
        if not args.encoder_path:
            parser.error(
                "Для play укажите --encoder-path к весам EnvEncoder (.pth) "
                "или --run/--policy рядом с train_run_config.yaml (encoder_weights в нём)"
            )
        encoder_path = os.path.abspath(args.encoder_path)
        play_run_dir = os.path.abspath(args.run) if args.run else play_cfg_dir
        run_play(
            policy_path=policy_path,
            encoder_path=encoder_path,
            device=args.device,
            episodes=args.episodes,
            max_steps=args.max_steps,
            num_obstacles=args.num_obstacles,
            path_length=args.path_length,
            map_type=args.map_type,
            render=not args.no_render,
            seed=args.seed,
            no_limit=args.no_limit,
            fastdepth_weights=args.fastdepth_weights,
            n_mcts_simulations=args.mcts_sim,
            latent_vx_bins=args.latent_vx_bins,
            latent_vy_bins=args.latent_vy_bins,
            latent_vy_min=args.latent_vy_min,
            latent_vy_max=args.latent_vy_max,
            max_velocity=args.play_max_velocity,
            forward_velocity=args.play_forward_velocity,
            mcts_c_puct=args.c_puct,
            gamma=args.gamma,
            play_algo=args.play_algo,
            reward_config_path=args.reward_config,
            run_dir=play_run_dir,
            train_environment_snapshot=train_env_snapshot,
        )
    elif args.command == 'train':
        print("=" * 60)
        print(
            "Train: генератор препятствий (SceneGenerator / DroneObstacleAvoidanceEnv): "
            f"map_type={args.map_type!r}, path_length={args.path_length}, "
            f"num_obstacles={args.num_obstacles}"
        )
        if args.algo in ("ppo", "dqn", "mcts"):
            _rc_train = resolve_reward_config_path(args.reward_config)
            print(f"Train: YAML наград и curriculum ({args.algo.upper()}) — {_rc_train}")
            try:
                _rc_cfg = load_reward_config(_rc_train)
                _map_eff = merge_map_train_args_from_reward_yaml(
                    _rc_cfg,
                    {
                        "map_type": args.map_type,
                        "path_length": args.path_length,
                        "num_obstacles": args.num_obstacles,
                        "vy_min": args.latent_vy_min,
                        "vy_max": args.latent_vy_max,
                        "n_vx_bins": args.latent_vx_bins,
                        "n_vy_bins": args.latent_vy_bins,
                    },
                )
                print(
                    "Train: эффективная сцена после merge(map из reward YAML поверх CLI): "
                    f"map_type={_map_eff['map_type']!r}, path_length={_map_eff['path_length']}, "
                    f"num_obstacles={_map_eff['num_obstacles']}"
                )
            except Exception as e:
                print(f"Train: не удалось показать merge(map): {e}")
        print("=" * 60)
        if args.algo == 'mcts':
            mcts_prof_path = _optional_profile_abspath(args.profile_path)
            mcts_prof = load_training_profile(mcts_prof_path)
            mcts_hp = merge_mcts_train_hyperparams(
                mcts_prof,
                policy_updates=args.policy_updates,
                n_mcts_simulations=args.mcts_sim,
                mcts_c_puct=args.c_puct,
                root_dirichlet_alpha=args.root_dirichlet_alpha,
                root_dirichlet_eps=args.root_dirichlet_eps,
                visit_temperature=args.visit_temp,
                prior_label_smoothing=args.prior_label_smoothing,
                lr=args.lr,
                curriculum_success_threshold=args.curriculum_success_threshold,
                curriculum_window=args.curriculum_window,
                curriculum_min_episodes_per_stage=args.curriculum_min_episodes_per_stage,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_episodes=args.early_stop_min_episodes,
            )
            if mcts_prof:
                print(
                    f"✓ MCTS профиль YAML: {mcts_prof_path}\n"
                    f"  updates={mcts_hp['policy_updates']}, sims={mcts_hp['n_mcts_simulations']}, "
                    f"curriculum window={mcts_hp['curriculum_window']}, "
                    f"success>={mcts_hp['curriculum_success_threshold']:.2f}"
                )
            train_mcts(
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                save_interval=args.save_interval,
                model_dir=args.model_dir,
                log_dir=args.log_dir,
                device=args.device,
                num_obstacles=args.num_obstacles,
                path_length=args.path_length,
                map_type=args.map_type,
                encoder_path=os.path.abspath(args.encoder_path),
                fastdepth_weights=args.fastdepth_weights,
                n_mcts_simulations=mcts_hp["n_mcts_simulations"],
                n_vx_bins=args.latent_vx_bins,
                n_vy_bins=args.latent_vy_bins,
                vy_min=args.latent_vy_min,
                vy_max=args.latent_vy_max,
                mcts_c_puct=mcts_hp["c_puct"],
                policy_updates=mcts_hp["policy_updates"],
                root_dirichlet_alpha=mcts_hp["root_dirichlet_alpha"],
                root_dirichlet_eps=mcts_hp["root_dirichlet_eps"],
                visit_temperature=mcts_hp["visit_temperature"],
                prior_label_smoothing=mcts_hp["prior_label_smoothing"],
                gamma=args.gamma,
                lr=mcts_hp["lr"],
                max_grad_norm=mcts_hp["max_grad_norm"],
                value_coef=mcts_hp["value_coef"],
                prior_coef=mcts_hp["prior_coef"],
                gae_lambda=mcts_hp["gae_lambda"],
                seed=args.seed,
                reward_config_path=args.reward_config,
                use_curriculum=not args.no_curriculum,
                curriculum_success_threshold=mcts_hp["curriculum_success_threshold"],
                curriculum_window=mcts_hp["curriculum_window"],
                curriculum_min_episodes_per_stage=mcts_hp["curriculum_min_episodes_per_stage"],
                curriculum_use_fallback=not args.no_curriculum_fallback,
                curriculum_fallback_max_mean_dist=args.curriculum_fallback_max_mean_dist,
                curriculum_fallback_max_collision_rate=args.curriculum_fallback_max_collision,
                early_stop_patience=mcts_hp["early_stop_patience"],
                early_stop_min_episodes=mcts_hp["early_stop_min_episodes"],
            )
        elif args.algo == 'ppo':
            ppo_model_dir = args.model_dir
            if ppo_model_dir == 'models/mcts-latent':
                ppo_model_dir = 'models/ppo-latent'
            ppo_prof_path = _optional_profile_abspath(args.profile_path)
            ppo_prof = load_training_profile(ppo_prof_path)
            ppo_hp = merge_ppo_train_hyperparams(
                ppo_prof,
                policy_updates=args.policy_updates,
                gamma=args.gamma,
                entropy_coef=args.ppo_entropy_coef,
                entropy_coef_final=args.ppo_entropy_coef_final,
                entropy_decay_fraction=args.ppo_entropy_decay_fraction,
                curriculum_success_threshold=args.curriculum_success_threshold,
                curriculum_window=args.curriculum_window,
                curriculum_min_episodes_per_stage=args.curriculum_min_episodes_per_stage,
                entropy_floor_until_goal=args.ppo_entropy_floor_until_goal,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_episodes=args.early_stop_min_episodes,
            )
            if ppo_prof:
                _g = ppo_hp["gamma"]
                _g_s = f"{_g:.3f}" if _g is not None else "из YAML наград / 0.8"
                print(
                    f"✓ PPO профиль YAML: {ppo_prof_path}\n"
                    f"  updates={ppo_hp['policy_updates']}, γ={_g_s}, "
                    f"entropy {ppo_hp['entropy_coef']}->{ppo_hp['entropy_coef_final']}, "
                    f"curriculum window={ppo_hp['curriculum_window']}, "
                    f"curriculum success>={ppo_hp['curriculum_success_threshold']:.2f}"
                )
            train_ppo(
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                save_interval=args.save_interval,
                model_dir=ppo_model_dir,
                log_dir=args.log_dir,
                device=args.device,
                num_obstacles=args.num_obstacles,
                path_length=args.path_length,
                map_type=args.map_type,
                encoder_path=os.path.abspath(args.encoder_path),
                fastdepth_weights=args.fastdepth_weights,
                n_vx_bins=args.latent_vx_bins,
                n_vy_bins=args.latent_vy_bins,
                vy_min=args.latent_vy_min,
                vy_max=args.latent_vy_max,
                policy_updates=ppo_hp["policy_updates"],
                gamma=ppo_hp["gamma"],
                gae_lambda=args.ppo_gae_lambda,
                ppo_clip_eps=args.ppo_clip_eps,
                entropy_coef=ppo_hp["entropy_coef"],
                entropy_coef_final=ppo_hp["entropy_coef_final"],
                entropy_decay_fraction=ppo_hp["entropy_decay_fraction"],
                value_coef=args.ppo_value_coef,
                lr=args.lr,
                seed=args.seed,
                early_stop_patience=ppo_hp["early_stop_patience"],
                early_stop_min_episodes=ppo_hp["early_stop_min_episodes"],
                use_curriculum=not args.no_curriculum,
                curriculum_success_threshold=ppo_hp["curriculum_success_threshold"],
                curriculum_window=ppo_hp["curriculum_window"],
                curriculum_min_episodes_per_stage=ppo_hp["curriculum_min_episodes_per_stage"],
                curriculum_use_fallback=not args.no_curriculum_fallback,
                curriculum_fallback_max_mean_dist=args.curriculum_fallback_max_mean_dist,
                curriculum_fallback_max_collision_rate=args.curriculum_fallback_max_collision,
                entropy_floor_until_goal=ppo_hp["entropy_floor_until_goal"],
                reward_config_path=args.reward_config,
                entropy_high_out_rate_threshold=args.ppo_entropy_high_out_threshold,
                entropy_floor_when_high_out_rate=args.ppo_entropy_floor_high_out,
            )
        elif args.algo == 'dqn':
            dqn_model_dir = args.model_dir
            if dqn_model_dir == 'models/mcts-latent':
                dqn_model_dir = 'models/dqn-latent'
            dqn_prof_path = _optional_profile_abspath(args.profile_path)
            dqn_prof = load_training_profile(dqn_prof_path)
            dqn_hp = merge_dqn_train_hyperparams(
                dqn_prof,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_episodes=args.early_stop_min_episodes,
                curriculum_success_threshold=args.curriculum_success_threshold,
                curriculum_window=args.curriculum_window,
                curriculum_min_episodes_per_stage=args.curriculum_min_episodes_per_stage,
                batch_size=args.dqn_batch_size,
                buffer_capacity=args.dqn_buffer_capacity,
                learning_starts=args.dqn_learning_starts,
                train_frequency=args.dqn_train_frequency,
                target_update_every=args.dqn_target_update_every,
            )
            if dqn_prof:
                print(
                    f"✓ DQN профиль YAML: {dqn_prof_path}\n"
                    f"  batch={dqn_hp['batch_size']}, buf={dqn_hp['buffer_capacity']}, "
                    f"learn_start={dqn_hp['learning_starts']}, train_freq={dqn_hp['train_frequency']}, "
                    f"tgt_upd={dqn_hp['target_update_every']}, curriculum window={dqn_hp['curriculum_window']}"
                )
            train_dqn(
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                save_interval=args.save_interval,
                model_dir=dqn_model_dir,
                log_dir=args.log_dir,
                device=args.device,
                num_obstacles=args.num_obstacles,
                path_length=args.path_length,
                map_type=args.map_type,
                encoder_path=os.path.abspath(args.encoder_path),
                fastdepth_weights=args.fastdepth_weights,
                n_vx_bins=args.latent_vx_bins,
                n_vy_bins=args.latent_vy_bins,
                vy_min=args.latent_vy_min,
                vy_max=args.latent_vy_max,
                gamma=args.gamma,
                lr=args.lr,
                max_grad_norm=args.dqn_max_grad_norm,
                seed=args.seed,
                early_stop_patience=dqn_hp["early_stop_patience"],
                early_stop_min_episodes=dqn_hp["early_stop_min_episodes"],
                use_curriculum=not args.no_curriculum,
                curriculum_success_threshold=dqn_hp["curriculum_success_threshold"],
                curriculum_window=dqn_hp["curriculum_window"],
                curriculum_min_episodes_per_stage=dqn_hp["curriculum_min_episodes_per_stage"],
                curriculum_use_fallback=not args.no_curriculum_fallback,
                curriculum_fallback_max_mean_dist=args.curriculum_fallback_max_mean_dist,
                curriculum_fallback_max_collision_rate=args.curriculum_fallback_max_collision,
                reward_config_path=args.reward_config,
                batch_size=dqn_hp["batch_size"],
                buffer_capacity=dqn_hp["buffer_capacity"],
                learning_starts=dqn_hp["learning_starts"],
                train_frequency=dqn_hp["train_frequency"],
                target_update_every=dqn_hp["target_update_every"],
                epsilon_start=args.dqn_epsilon_start,
                epsilon_end=args.dqn_epsilon_end,
                epsilon_decay_steps=args.dqn_epsilon_decay_steps,
            )
    elif args.command == 'random':
        env_render_mode = "human" if not args.no_render else None
        run_random_policy(
            render_mode=env_render_mode,
            steps=args.steps,
            num_obstacles=args.num_obstacles,
            episodes=args.episodes,
            seed=args.seed,
        )
    elif args.command == 'view':
        env_render_mode = "human" if args.render else None
        view_environment(
            env_render_mode,
            args.steps,
            args.num_obstacles,
            no_limit=args.no_limit,
            path_length=args.path_length,
            map_type=args.map_type,
        )
    elif args.command == 'depth-view':
        run_depth_view(
            weights_path=args.weights,
            out_dir=args.out_dir,
            steps=args.steps,
            device=args.device,
            seed=args.seed,
            num_obstacles=args.num_obstacles,
        )
    elif args.command == "imu-test":
        run_imu_test(
            out_dir=args.out_dir,
            render=args.render,
            max_velocity=args.max_velocity,
        )
    elif args.command == "env-train":
        train_envencoder(
            datasets_dir=args.datasets_dir,
            dataset_root=args.env_dataset_root,
            model_root=args.model_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta_kl=args.beta_kl,
            imu_scale=args.imu_scale,
            latent_dim=args.latent_dim,
            num_workers=args.num_workers,
            device=args.device,
            log_dir=args.log_dir,
            save_every=args.save_every,
            seed=args.seed,
            grad_clip_max_norm=args.grad_clip_max_norm,
        )
    elif args.command == "env-test":
        run_env_test(
            depth_npy=args.depth_npy,
            odom_log=args.odom_log,
            encoder_weights=args.encoder,
            decoder_weights=args.decoder,
            out_dir=args.out_dir,
            device=args.device,
            imu_scale=args.imu_scale,
            stochastic=args.stochastic,
            seed=args.seed,
        )
