# scenaries/imu_test_trajectory.py - траектория квадрата 1x1 м без графиков

import typing

import numpy as np

from rl.environment import DroneObstacleAvoidanceEnv
from utils.imu_odometry import IMUStrapdownOdometry

IMU_SQUARE_LEG_TOLERANCE = 0.08


def collect_imu_square_trajectory(
    seed: int = 42,
    max_velocity: float = 0.4,
    leg_tolerance: float = IMU_SQUARE_LEG_TOLERANCE,
    hover_steps: int = 50,
    odom_kwargs: typing.Optional[dict] = None,
) -> typing.Dict[str, typing.Any]:
    """
    Тот же цикл, что в scenaries.imu_test.run_imu_test: взлёт, зависание, квадрат 1x1 м.
    Возвращает временной ряд и GT / одометрию (относительно старта манёвра).
    """
    kw = dict(odom_kwargs or {})
    env = DroneObstacleAvoidanceEnv(
        render_mode=None,
        max_velocity=max_velocity,
        forward_velocity=0.0,
        max_episode_steps=15000,
        num_obstacles=0,
        obstacle_path_length=5.0,
    )
    env.reset(seed=seed)
    dt = float(env.drone.m.opt.timestep)

    for _ in range(hover_steps):
        env.drone.clear_target_navigation()
        env.forward_velocity = 0.0
        obs, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        if terminated or truncated:
            env.close()
            raise RuntimeError("Прервано при стабилизации после взлёта")

    pos0 = env.drone.sensor.get_position()[:3].copy()
    q_body = env.drone.d.qpos[3:7].astype(np.float64).copy()
    q_body = q_body / max(np.linalg.norm(q_body), 1e-12)

    odom = IMUStrapdownOdometry(dt=dt, **kw)
    odom.reset(position_world=np.zeros(3, dtype=np.float64), quaternion_body_to_world=q_body)

    p0x, p0y = float(pos0[0]), float(pos0[1])
    leg_corners: typing.List[typing.Tuple[float, float]] = [
        (p0x - 1.0, p0y),
        (p0x - 1.0, p0y + 1.0),
        (p0x, p0y + 1.0),
        (p0x, p0y),
    ]

    gt_x: typing.List[float] = []
    gt_y: typing.List[float] = []
    odom_x: typing.List[float] = []
    odom_y: typing.List[float] = []

    max_steps_leg = max(500, int(4.0 / max(max_velocity * dt, 1e-9)))

    for tx, ty in leg_corners:
        env.drone.set_target(
            (tx, ty, float(pos0[2])),
            xy_tolerance=leg_tolerance,
            z_tolerance=0.25,
        )
        for _ in range(max_steps_leg):
            if env.drone.point_reached:
                break
            obs, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
            if terminated or truncated:
                env.close()
                return {
                    "dt": dt,
                    "t": np.array([], dtype=np.float64),
                    "gt_x": np.array([], dtype=np.float64),
                    "gt_y": np.array([], dtype=np.float64),
                    "odom_x": np.array([], dtype=np.float64),
                    "odom_y": np.array([], dtype=np.float64),
                    "truncated": True,
                }
            odom.update(obs["accelerometer"], obs["gyroscope"])
            pos = env.drone.sensor.get_position()[:3]
            gt_x.append(float(pos[0] - p0x))
            gt_y.append(float(pos[1] - p0y))
            odom_x.append(float(odom.position_world[0]))
            odom_y.append(float(odom.position_world[1]))

    env.drone.clear_target_navigation()
    env.close()

    n = len(gt_x)
    t = np.arange(n, dtype=np.float64) * dt
    return {
        "dt": dt,
        "t": t,
        "gt_x": np.asarray(gt_x, dtype=np.float64),
        "gt_y": np.asarray(gt_y, dtype=np.float64),
        "odom_x": np.asarray(odom_x, dtype=np.float64),
        "odom_y": np.asarray(odom_y, dtype=np.float64),
        "truncated": False,
    }
