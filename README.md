# Skydio X2 в MuJoCo: obstacle avoidance

Проект с симуляцией квадрокоптера **Skydio X2** в **MuJoCo** и Gymnasium-средой для задач **obstacle avoidance**. БПЛА летит вперёд с фиксированной линейной скоростью, а агент управляет боковым смещением, стараясь пролетать между процедурно сгенерированными препятствиями (столбами) без столкновений.

## Создание среды (Conda)

В репозитории есть `enviroment.yml` (Conda environment):

```bash
conda env create -f enviroment.yml
conda activate mujoco-x2-obstacle-avoidance
```

## Структура репозитория

- `main.py` — основной CLI (`train`, `play`, `view`, `random`, `env-train`, `env-test`, `imu-test`, `depth-view`).
- `rl/`
  - `environment.py` — Gymnasium-среда `DroneObstacleAvoidanceEnv`.
  - `map_generator.py` — генерация карты препятствий.
  - `mcts.py`, `ppo.py`, `dqn.py` — RL-алгоритмы.
  - `envencoder_bridge.py` — мост RGB+IMU --> depth+dXY --> латент `z`.
- `sim/`
  - `scene.py` — генерация/управление сценой MuJoCo, добавление препятствий.
  - `uav.py` — модель дрона, сенсоры, камера и интерфейс управления скоростями.
  - `controllers.py` — контроллеры (PID и т.п.).
- `scenaries/`
  - `train_ppo.py`, `train_dqn.py`, `train_mcts.py` — обучение RL-алгоритмов.
  - `play.py` — инференс обученной политики в среде.
  - `view_environment.py`, `random_policy.py` — быстрые тесты среды.
  - `imu_test.py`, `imu_test_trajectory.py`, `depth_view.py`, `env_test.py` — диагностические сценарии сенсоров и энкодера.
- `teach_ml/`
  - `envencoder/` — VAE энкодер среды.
  - `fastdepth/` — обертка FastDepth.
- `utils/`
  - `reward_config.py` — загрузка/объединение конфигураций YAML и curriculum.
  - `training_logging.py` — формирование `train_run_config.yaml` и служебные снимки run.
  - `training_profile.py` — профильные гиперпараметры PPO/DQN/MCTS.
  - `imu_odometry.py` — страпдаун-одометрия IMU.