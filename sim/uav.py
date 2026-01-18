# sim/uav.py - код взаимодействия с моделью БПЛА, функции управления взяты из оригинального репозитория модели

import mujoco
import cv2
import numpy as np
from simple_pid import PID
from .controllers import DummyPlanner, DummySensor

class Drone:
  def __init__(self, target=np.array((0,0,0)), scene=None):
    """
    Инициализация БПЛА
    
    Args:
        target: Целевая позиция БПЛА
        scene: Объект Scene, в котором находится дрон. Если None, загружается только модель коптера
    """
    self.m = scene.m
    self.d = scene.d

    self.planner = DummyPlanner(target=target)
    self.sensor = DummySensor(self.d)

    # instantiate controllers

    # inner control to stabalize inflight dynamics
    self.pid_alt = PID(5.50844,0.57871, 1.2,setpoint=0,) # PIDController(0.050844,0.000017871, 0, 0) # thrust
    self.pid_roll = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) ) #PID(11.0791,2.5263, 0.10513,setpoint=0, output_limits = (-1,1) )
    self.pid_pitch = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) )
    self.pid_yaw =  PID(0.54, 0, 5.358333, setpoint=1, output_limits = (-3,3) )# PID(0.11046, 0.0, 15.8333, setpoint=1, output_limits = (-2,2) )

    # outer control loops
    self.pid_v_x = PID(0.1, 0.003, 0.02, setpoint = 0,
                output_limits = (-0.1, 0.1))
    self.pid_v_y = PID(0.1, 0.003, 0.02, setpoint = 0,
                  output_limits = (-0.1, 0.1))
    
    # Управление скоростями
    self.manual_control = False  # Режим ручного управления
    self.manual_velocities = np.array([0.0, 0.0, 0.0])  # Желаемые скорости в ручном режиме
    self.max_velocity = 2.0  # Максимальная скорость
    self.velocity_step = 0.1  # Шаг изменения скорости при управлении
    
    # Камера дрона
    self.camera_id = "drone_camera"
    self.camera_width = 640
    self.camera_height = 480
    self.renderer = mujoco.Renderer(self.m, height=self.camera_height, width=self.camera_width)
    
    # Получаем индексы сенсоров
    self.gyro_id = mujoco.mj_name2id(
        self.m, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro"
    )
    self.accel_id = mujoco.mj_name2id(
        self.m, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc"
    )
    
    # Проверяем, что сенсоры найдены
    if self.gyro_id < 0:
        raise ValueError("Сенсор 'body_gyro' не найден в модели!")
    if self.accel_id < 0:
        raise ValueError("Сенсор 'body_linacc' не найден в модели!")

  def set_velocity(self, vx, vy, vz=None):
    """Устанавливает желаемые скорости напрямую
    
    Args:
        vx (float): Желаемая скорость по оси X
        vy (float): Желаемая скорость по оси Y
        vz (float, optional): Желаемая скорость по оси Z (высота). Если None, используется планировщик.
    """
    self.manual_control = True
    self.manual_velocities[0] = np.clip(vx, -self.max_velocity, self.max_velocity)
    self.manual_velocities[1] = np.clip(vy, -self.max_velocity, self.max_velocity)
    if vz is not None:
      self.manual_velocities[2] = np.clip(vz, -self.max_velocity, self.max_velocity)

  def set_auto_mode(self):
    """Переключает в автоматический режим (использует планировщик)"""
    self.manual_control = False

  def get_current_velocity(self):
    """Возвращает текущие скорости дрона"""
    return self.sensor.get_velocity()[:3]

  def get_camera_image(self):
    # Обновляем сцену перед рендерингом
    self.renderer.update_scene(self.d, camera=self.camera_id)
    image = self.renderer.render()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image_bgr

  def get_gyroscope_data(self):
    """
    Получает данные гироскопа (угловая скорость).
    
    Returns:
        np.array: Данные гироскопа [wx, wy, wz] в формате float32
    """
    gyro_start_idx = self.m.sensor_adr[self.gyro_id]
    gyro_dim = self.m.sensor_dim[self.gyro_id]
    gyro_data = self.d.sensordata[
        gyro_start_idx : gyro_start_idx + gyro_dim
    ].copy()
    return gyro_data.astype(np.float32)

  def get_accelerometer_data(self):
    """
    Получает данные акселерометра (линейное ускорение).
    
    Returns:
        np.array: Данные акселерометра [ax, ay, az] в формате float32
    """
    accel_start_idx = self.m.sensor_adr[self.accel_id]
    accel_dim = self.m.sensor_dim[self.accel_id]
    accel_data = self.d.sensordata[
        accel_start_idx : accel_start_idx + accel_dim
    ].copy()
    return accel_data.astype(np.float32)

  def update_outer_conrol(self):
    """Updates outer control loop for trajectory planning"""
    v = self.sensor.get_velocity()
    location = self.sensor.get_position()[:3]

    if self.manual_control:
      # Ручное управление скоростями
      self.pid_v_x.setpoint = self.manual_velocities[0]
      self.pid_v_y.setpoint = self.manual_velocities[1]
      
      # Если задана скорость по Z, используем её, иначе используем планировщик для высоты
      if self.manual_velocities[2] != 0:
        current_alt = location[2]
        self.pid_alt.setpoint = current_alt + self.manual_velocities[2] * 0.1
      else:
        self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
    else:
      # Автоматический режим - используем планировщик (оригинальный)
      # Compute velocites to target
      velocites = self.planner(loc=location)
      
      # In this example the altitude is directly controlled by a PID
      self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
      self.pid_v_x.setpoint = velocites[0]
      self.pid_v_y.setpoint = velocites[1]

    # Compute angles and set inner controllers accordingly
    angle_pitch = self.pid_v_x(v[0])
    angle_roll = - self.pid_v_y(v[1])

    self.pid_pitch.setpoint= angle_pitch
    self.pid_roll.setpoint = angle_roll

  def update_inner_control(self):
    """Upates inner control loop and sets actuators to stabilize flight
    dynamics"""
    alt = self.sensor.get_position()[2]
    angles = self.sensor.get_position()[3:] # roll, yaw, pitch
    
    # apply PID
    cmd_thrust = self.pid_alt(alt) + 3.2495
    cmd_roll = - self.pid_roll(angles[1])
    cmd_pitch = self.pid_pitch(angles[2])
    cmd_yaw = - self.pid_yaw(angles[0])

    #transfer to motor control
    out = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
    self.d.ctrl[:4] = out

  #  as the drone is underactuated we set
  def compute_motor_control(self, thrust, roll, pitch, yaw):
    motor_control = [
      thrust + roll + pitch - yaw,
      thrust - roll + pitch + yaw,
      thrust - roll -  pitch - yaw,
      thrust + roll - pitch + yaw
    ]
    return motor_control