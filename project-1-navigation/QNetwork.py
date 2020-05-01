import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
   """Simple network to optimize Q-values for best next possible action"""
   def __init__(self, state_size, action_size):
      super(QNetwork, self).__init__()

      self.model = nn.Sequential(
         nn.Linear(state_size, 256),
         nn.ReLU(),
         nn.Linear(256, 256),
         nn.ReLU(),
         nn.Linear(256, 256),
         nn.ReLU(),
         nn.Linear(256, action_size)
      )

   def forward(self, state):
      return self.model(state)