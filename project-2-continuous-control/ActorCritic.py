import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

def linear_block(in_f, out_f):
   return nn.Sequential(
      nn.Linear(in_f, out_f),
      nn.ReLU()
   )

class ActorCritic(nn.Module):
   def __init__(self, state_size, action_size, hidden_size=64):
      super(ActorCritic, self).__init__()
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

      self.std = nn.Parameter(torch.zeros(1, action_size))
      
      self.base = nn.Sequential(
         linear_block(state_size, hidden_size),
         linear_block(hidden_size, hidden_size)
      )

      self.actor_lin = nn.Linear(hidden_size, action_size)
      self.critic_lin = nn.Linear(hidden_size, 1)

   def forward(self, state):
      x = self.base(state)

      # Critic
      value = self.critic_lin(x)

      # Actor
      actor_output = self.actor_lin(x)
      mean = torch.tanh(actor_output)
      dist = Normal(mean, F.softplus(self.std))
      
      return value, dist