# sim/controllers.py - код управляения БПЛА, модифицированный из оригинальной модели

import numpy as np
from simple_pid import PID

def pid_to_thrust(input: np.array):
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [-0.25, 0.25, 0.25, -0.25],
      [0.25, 0.25, -0.25, -0.25],
      [-0.25, 0.25, -0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

def outer_pid_to_thrust(input: np.array):
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [0.25, 0.25, -0.25, -0.25],
      [0.25, -0.25, -0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

class PDController:
  def __init__(self, kp, kd, setpoint):
    self.kp = kp
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.kd * derivative)
    self.prev_error = error
    return output

class PIDController:
  def __init__(self, kp, ki, kd, setpoint):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0
    self.integral = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    self.integral += error
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
    self.prev_error = error
    return output

class DummyPlanner:
  """Generate Path from 1 point directly to another"""

  def __init__(self, target, vel_limit = 2) -> None:
    self.target = target  
    self.vel_limit = vel_limit
    # setpoint target location, controller output: desired velocity.
    self.pid_x = PID(2, 0.15, 1.5, setpoint = self.target[0],
                output_limits = (-vel_limit, vel_limit),)
    self.pid_y = PID(2, 0.15, 1.5, setpoint = self.target[1],
                output_limits = (-vel_limit, vel_limit))
  
  def __call__(self, loc: np.array):
    """Calls planner at timestep to update cmd_vel"""
    velocites = np.array([0,0,0])
    velocites[0] = self.pid_x(loc[0])
    velocites[1] = self.pid_y(loc[1])
    return velocites

  def get_velocities(self,loc: np.array, target: np.array,
                     time_to_target: float = None,
                     flight_speed: float = 0.5) -> np.array:
    """Compute

    Args:
        loc (np.array): Current location in world coordinates.
        target (np.array): Desired location in world coordinates
        time_to_target (float): If set, adpats length of velocity vector.

    Returns:
        np.array: returns velocity vector in world coordinates.
    """

    direction = target - loc
    distance = np.linalg.norm(direction)
    if distance > 1:
        velocities = flight_speed * direction / distance

    else:
        velocities =  direction * distance

    return velocities

  def get_alt_setpoint(self, loc: np.array) -> float:

    target = self.target
    distance = target[2] - loc[2]
    
    if distance > 0.5:
        time_sample = 1/4
        time_to_target =  distance / self.vel_limit
        number_steps = int(time_to_target/time_sample)
        # compute distance for next update
        delta_alt = distance / number_steps

        # 2 times for smoothing
        alt_set = loc[2] + 2 * delta_alt
    
    else:
        alt_set = target[2]

    return alt_set

  def update_target(self, target):
    """Update targets"""
    self.target = target  
    # setpoint target location, controller output: desired velocity.
    self.pid_x.setpoint = self.target[0]
    self.pid_y.setpoint = self.target[1]

class DummySensor:
  """Dummy sensor data. So the control code remains intact."""
  def __init__(self, d):
    self.position = d.qpos
    self.velocity = d.qvel
    self.acceleration = d.qacc

  def get_position(self):
    return self.position
  
  def get_velocity(self):
    return self.velocity
  
  def get_acceleration(self):
    return self.acceleration