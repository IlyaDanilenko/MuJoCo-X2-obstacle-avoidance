# utils/imu_odometry.py - страпдаун IMU: ориентация и позиция в мировой СК

import numpy as np

ArrayLike = np.ndarray | list


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (q / n).astype(np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Гамильтон: q1 * q2, оба (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_integrate_body_rates(q: np.ndarray, omega_body: np.ndarray, h: float) -> np.ndarray:
    """Шаг ориентации: q' = normalize(q * dq), dq из omega в связанной СК (устойчивее, чем Эйлер по qdot)."""
    w = np.asarray(omega_body, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(w) * h)
    if th < 1e-14:
        return _normalize_quat(q)
    axis = w / (np.linalg.norm(w) + 1e-15)
    half = 0.5 * th
    s = np.sin(half)
    dq = np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)
    return _normalize_quat(_quat_mul(_normalize_quat(q), dq))


def _quat_to_R_body_to_world(q: np.ndarray) -> np.ndarray:
    """Матрица поворота R: v_world = R @ v_body. Кватернион (w, x, y, z)."""
    w, x, y, z = _normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class IMUStrapdownOdometry:
    """
    Страпдаун: гироскоп --> ориентация (кватернион через экспоненту), акселерометр --> мир,
    учёт g, двойное интегрирование.

    ``position_world`` / ``velocity_world``: при выравнивании тела с миром (кватернион identity)
    оси X, Y совпадают с высокоуровневыми осями дрона (как у ``Drone.set_velocity``): +X вправо, +Y вперёд.

    Единицы: гироскоп - рад/с (MuJoCo), акселерометр - м/с2, dt - с.
    """

    def __init__(
        self,
        dt: float,
        gravity_world: ArrayLike | None = None,
    ):
        """
        Args:
            dt: Шаг времени интегрирования (с), напр. model.opt.timestep или 1/render_fps.
            gravity_world: Ускорение свободного падения в мировой СК, м/с2.
                По умолчанию MuJoCo z-up: [0, 0, -9.81] - тогда при покое на земле
                линейное ускорение после проекции обнуляется.
        """
        self._dt = float(dt)
        if gravity_world is None:
            self._g = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        else:
            self._g = np.asarray(gravity_world, dtype=np.float64).reshape(3)

        self._t_elapsed = 0.0

        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._vel = np.zeros(3, dtype=np.float64)
        self._pos = np.zeros(3, dtype=np.float64)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def position_world(self) -> np.ndarray:
        """Накопленная позиция в мировой СК с момента последнего reset (копия)."""
        return self._pos.copy()

    @property
    def velocity_world(self) -> np.ndarray:
        """Накопленная скорость в мировой СК (копия)."""
        return self._vel.copy()

    @property
    def quaternion_body_to_world(self) -> np.ndarray:
        """Кватернион (w,x,y,z) body-->world, единичный (копия)."""
        return _normalize_quat(self._q).copy()

    def reset(
        self,
        position_world: ArrayLike | None = None,
        velocity_world: ArrayLike | None = None,
        quaternion_body_to_world: ArrayLike | None = None,
    ) -> None:
        """
        Сброс накопленного состояния. Ориентацию по умолчанию - без поворота относительно мира.
        """
        self._pos = (
            np.zeros(3, dtype=np.float64)
            if position_world is None
            else np.asarray(position_world, dtype=np.float64).reshape(3).copy()
        )
        self._vel = (
            np.zeros(3, dtype=np.float64)
            if velocity_world is None
            else np.asarray(velocity_world, dtype=np.float64).reshape(3).copy()
        )
        if quaternion_body_to_world is None:
            self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self._q = _normalize_quat(np.asarray(quaternion_body_to_world, dtype=np.float64).reshape(4))

        self._t_elapsed = 0.0
    def pack_internal_state(self) -> dict:
        """Снимок внутреннего состояния для отката (MCTS / форк симуляции)."""
        return {
            "q": self._q.copy(),
            "vel": self._vel.copy(),
            "pos": self._pos.copy(),
            "t_elapsed": float(self._t_elapsed),
        }

    def unpack_internal_state(self, d: dict) -> None:
        self._q[:] = np.asarray(d["q"], dtype=np.float64).reshape(4)
        self._vel[:] = np.asarray(d["vel"], dtype=np.float64).reshape(3)
        self._pos[:] = np.asarray(d["pos"], dtype=np.float64).reshape(3)
        self._t_elapsed = float(d["t_elapsed"])

    def update(
        self,
        accelerometer: ArrayLike,
        gyroscope: ArrayLike,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Один шаг интегрирования по текущим показаниям IMU.

        Args:
            accelerometer: [ax, ay, az] в связанной СК, м/с2 (MuJoCo body_linacc).
            gyroscope: [wx, wy, wz] в связанной СК, рад/с.
            dt: Если задан, подменяет dt из конструктора на этот шаг.

        Returns:
            np.ndarray формы (3,), float64 - приращение смещения в мировой СК за этот шаг.
        """
        h = float(self._dt if dt is None else dt)
        a_b = np.asarray(accelerometer, dtype=np.float64).reshape(3)
        omega_b = np.asarray(gyroscope, dtype=np.float64).reshape(3)

        q = _normalize_quat(self._q)
        self._q = _quat_integrate_body_rates(q, omega_b, h)

        R = _quat_to_R_body_to_world(self._q)
        a_w = R @ a_b + self._g

        delta_p = self._vel * h + 0.5 * a_w * (h * h)
        self._vel = self._vel + a_w * h
        self._pos = self._pos + delta_p

        self._t_elapsed += h

        return delta_p.astype(np.float64)
