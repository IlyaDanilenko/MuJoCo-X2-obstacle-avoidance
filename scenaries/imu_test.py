# scenaries/imu_test.py - квадрат 1x1 м: IMU-одометрия vs симулятор

import os

import numpy as np

from rl.environment import DroneObstacleAvoidanceEnv
from utils.imu_odometry import IMUStrapdownOdometry
from .imu_test_trajectory import IMU_SQUARE_LEG_TOLERANCE


def run_imu_test(
    out_dir="test",
    seed=42,
    render=False,
    max_velocity=0.4,
    hover_steps=50,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Сценарий imu-test (квадрат 1 м, сравнение одометрия / симулятор)")
    print("=" * 60)

    env = DroneObstacleAvoidanceEnv(
        render_mode="human" if render else None,
        max_velocity=max_velocity,
        forward_velocity=0.0,
        max_episode_steps=8000,
        num_obstacles=0,
        obstacle_path_length=5.0,
    )

    obs, info = env.reset(seed=seed)
    dt = float(env.drone.m.opt.timestep)

    for _ in range(hover_steps):
        env.drone.clear_target_navigation()
        env.forward_velocity = 0.0
        obs, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        if render:
            env.render()
        if terminated or truncated:
            print("Прервано при стабилизации после взлёта")
            env.close()
            return

    pos0 = env.drone.sensor.get_position()[:3].copy()
    q_body = env.drone.d.qpos[3:7].astype(np.float64).copy()
    q_body = q_body / max(np.linalg.norm(q_body), 1e-12)

    odom = IMUStrapdownOdometry(dt=dt)
    odom.reset(position_world=np.zeros(3, dtype=np.float64), quaternion_body_to_world=q_body)

    p0x, p0y = float(pos0[0]), float(pos0[1])
    leg_corners = [
        (p0x - 1.0, p0y),
        (p0x - 1.0, p0y + 1.0),
        (p0x, p0y + 1.0),
        (p0x, p0y),
    ]

    gt_x, gt_y, odom_x, odom_y = [], [], [], []

    def record_frame():
        pos = env.drone.sensor.get_position()[:3]
        gt_x.append(float(pos[0] - p0x))
        gt_y.append(float(pos[1] - p0y))
        odom_x.append(float(odom.position_world[0]))
        odom_y.append(float(odom.position_world[1]))

    max_steps_leg = max(500, int(4.0 / max(max_velocity * dt, 1e-9)))

    for tx, ty in leg_corners:
        env.drone.set_target(
            (tx, ty, float(pos0[2])),
            xy_tolerance=IMU_SQUARE_LEG_TOLERANCE,
            z_tolerance=0.25,
        )
        for _ in range(max_steps_leg):
            if env.drone.point_reached:
                break
            obs, _, terminated, truncated, _ = env.step(
                np.array([0.0], dtype=np.float32)
            )
            odom.update(obs["accelerometer"], obs["gyroscope"])
            record_frame()
            if render:
                env.render()
            if terminated or truncated:
                print(f"Эпизод прерван: terminated={terminated}, truncated={truncated}")
                env.drone.clear_target_navigation()
                env.close()
                return

    env.drone.clear_target_navigation()
    env.close()

    gt_x = np.asarray(gt_x, dtype=np.float64)
    gt_y = np.asarray(gt_y, dtype=np.float64)
    odom_x = np.asarray(odom_x, dtype=np.float64)
    odom_y = np.asarray(odom_y, dtype=np.float64)
    err_x = odom_x - gt_x
    err_y = odom_y - gt_y
    err_norm = np.sqrt(err_x**2 + err_y**2)
    rmse_xy = float(np.sqrt(np.mean(err_norm**2)))
    max_xy = float(np.max(err_norm))
    mae_x = float(np.mean(np.abs(err_x)))
    mae_y = float(np.mean(np.abs(err_y)))
    final_ex = float(err_x[-1]) if len(err_x) else 0.0
    final_ey = float(err_y[-1]) if len(err_y) else 0.0
    final_norm = float(err_norm[-1]) if len(err_norm) else 0.0

    print("Погрешность одометрии относительно симулятора (позиция XY, м):")
    print(f"  RMSE по плоскости: {rmse_xy:.4f}")
    print(f"  макс. ‖Δp‖₂:        {max_xy:.4f}")
    print(f"  среднее |ΔX|:       {mae_x:.4f}")
    print(f"  среднее |ΔY|:       {mae_y:.4f}")
    print(f"  финал: ΔX={final_ex:+.4f}, ΔY={final_ey:+.4f}, ‖Δp‖₂={final_norm:.4f}")

    t = np.arange(len(gt_x), dtype=np.float64) * dt
    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    ax_tx, ax_ty, ax_xy = axes
    ax_tx.plot(t, gt_x, label="симулятор X", color="C0")
    ax_tx.plot(t, odom_x, "--", label="одометрия X", color="C1")
    ax_tx.set_ylabel("X, м")
    ax_tx.legend(loc="upper right")
    ax_tx.grid(True, alpha=0.3)
    ax_ty.plot(t, gt_y, label="симулятор Y", color="C0")
    ax_ty.plot(t, odom_y, "--", label="одометрия Y", color="C1")
    ax_ty.set_ylabel("Y, м")
    ax_ty.set_xlabel("t, с")
    ax_ty.legend(loc="upper right")
    ax_ty.grid(True, alpha=0.3)
    ax_ty.sharex(ax_tx)

    ax_xy.plot(gt_x, gt_y, label="симулятор", color="C0", lw=1.5)
    ax_xy.plot(odom_x, odom_y, "--", label="одометрия", color="C1", lw=1.5)
    ax_xy.set_xlabel("X, м")
    ax_xy.set_ylabel("Y, м")
    ax_xy.set_title("Траектория в плоскости XY (относительно старта манёвра)")
    ax_xy.legend(loc="upper right")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal", adjustable="box")

    fig.suptitle("Квадрат 1x1 м: координаты относительно старта манёвра")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "imu_square_xy.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"График сохранён: {out_path}")
