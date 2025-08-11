import torch
from pettingzoo import ParallelEnv
import numpy as np

def flatten_dict_to_list(d: dict):
  lst = []
  for item in d.values():
    if type(item) is dict:
      contents = flatten_dict_to_list(item)
      for val in contents:
        lst.append(val)
    else:
      lst.append(item)

  return lst

def flatten_dict_without_nest(d: dict):
  lst = []
  for item in d.values():
    if not type(item) is dict:
      lst.append(item)

  return lst

class Model:
  def __init__(self):
    raise NotImplementedError()

  def get_action(self, observation):
    return NotImplementedError()

class Network:
  def __init__(self, input_size: int, hidden_size: int|list[int], output_size: int):
    pass