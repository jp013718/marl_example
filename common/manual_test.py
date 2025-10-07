import sys
sys.path.append("..")
import hid
import threading
from time import sleep

from environment.envs.marl_env import MarlEnvironment

def scale_analog_output(analog_val, min_in, max_in, min_out, max_out, deadzone=0):
  scale = (max_in-min_in)/(max_out-min_out)
  val = analog_val/scale+min_out
  if val > (min_out+max_out)/2 - deadzone and val < (min_out+max_out)/2 + deadzone:
    val = (min_out+max_out)/2

  return val

class DualShock(object):
  def __init__(self):
    self.controller = None
    while self.controller is None:
      print("Connecting to DUALSHOCK 4 Wireless Controller")
      sleep(1)
      for device in hid.enumerate():
        if device['product_string'] == "DUALSHOCK 4 Wireless Controller":
          vendor = device['vendor_id']
          product = device['product_id']

          self.controller = hid.device()
          self.controller.open(vendor, product)
          self.controller.set_nonblocking(True)
          print("Connected to DUALSHOCK 4 Controller")
          break

    self.LeftAnalogX = 0
    self.LeftAnalogY = 0
    self.RightAnalogX = 0
    self.RightAnalogY = 0
    self.Triangle = 0
    self.Circle = 0
    self.X = 0
    self.Square = 0
    self.DPadUp = 0
    self.DPadRight = 0
    self.DPadDown = 0
    self.DPadLeft = 0
    self.L1 = 0
    self.R1 = 0
    self.L2 = 0
    self.R2 = 0

    self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
    self._monitor_thread.daemon = True
    self._monitor_thread.start()

  def read(self):
    lay = self.LeftAnalogY
    rax = self.RightAnalogX
    l1 = self.L1
    r1 = self.R1
    circle = self.Circle

    return (lay, rax, l1, r1, circle)

  def _assign_DPad(self, up, right, down, left):
    self.DPadUp = up
    self.DPadRight = right
    self.DPadDown = down
    self.DPadLeft = left

  def _monitor_controller(self):
    while True:
      report = self.controller.read(64)
      if report:
        self.LeftAnalogX = report[1]
        self.LeftAnalogY = report[2]
        self.RightAnalogX = report[3]
        self.RightAnalogY = report[4]
        self.Triangle = report[5] & 0b10000000
        self.Circle = report[5] & 0b01000000
        self.X = report[5] & 0b00100000
        self.Square = report[5] & 0b00010000
        self.L1 = report[6] & 0b01
        self.R1 = report[6] & 0b10
        self.L2 = report[8]
        self.R2 = report[9]

        d_pad_val = report[5] & 0b00001111
        if d_pad_val == 0:
          self._assign_DPad(1, 0, 0, 0)
        elif d_pad_val == 1:
          self._assign_DPad(1, 1, 0, 0)
        elif d_pad_val == 2:
          self._assign_DPad(0, 1, 0, 0)
        elif d_pad_val == 3:
          self._assign_DPad(0, 1, 1, 0)
        elif d_pad_val == 4:
          self._assign_DPad(0, 0, 1, 0)
        elif d_pad_val == 5:
          self._assign_DPad(0, 0, 1, 1)
        elif d_pad_val == 6:
          self._assign_DPad(0, 0, 0, 1)
        elif d_pad_val == 7:
          self._assign_DPad(1, 0, 0, 1)
        elif d_pad_val == 8:
          self._assign_DPad(0, 0, 0, 0)

if __name__ == "__main__":
  controller = DualShock()
  
  env = MarlEnvironment(
    n_agents=1,
    num_near_agents=0,
    mapsize=50
  )

  active_agent = 0
  done = False  
  
  while not done:
    observations, infos = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    swap_timer = 0

    while not (all(terminated.values()) or all(truncated.values())):
      env.render()

      inputs = controller.read()
      if inputs[2] and swap_timer == 0:
        active_agent -= 1
        active_agent %= env.num_agents
        swap_timer = env.metadata['render_fps']//2
      elif inputs[3] and swap_timer == 0:
        active_agent += 1
        active_agent %= env.num_agents
        swap_timer = env.metadata['render_fps']//2
      
      done = True if inputs[4] else done
      if inputs[4]:
        break

      swap_timer -= 1 if swap_timer > 0 else 0
      
      actions = {}
      for i, agent in enumerate(env.agents):
        if i == active_agent:
          speed = -scale_analog_output(inputs[0], 0, 255, -1, 1, 0.03)*env.max_speed
          speed = max(0, min(speed, env.max_speed))
          omega = -scale_analog_output(inputs[1], 0, 255, -1, 1, 0.03)*env.max_angular_speed

          actions[agent] = [omega, speed]
        
        else:
          actions[agent] = [0, 0]

      observations, rewards, terminated, truncated, infos = env.step(actions)
      print(observations)
      print(rewards)

  env.close()
