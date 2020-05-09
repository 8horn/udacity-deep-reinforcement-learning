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
      super().__init__()      
      
      self.base = nn.Sequential(
         linear_block(state_size, hidden_size),
         linear_block(hidden_size, hidden_size),
         linear_block(hidden_size, hidden_size)
      )

      self.actor_lin = nn.Linear(hidden_size, action_size)
      self.critic_lin = nn.Linear(hidden_size, 1)

      self.std = nn.Parameter(torch.zeros(1, action_size))

   def forward(self, state):
      x = self.base(state)

      # Critic
      value = self.critic_lin(x)

      # Actor
      actor_output = self.actor_lin(x)
      mean = torch.tanh(actor_output)
      dist = Normal(mean, F.softplus(self.std))
      
      return value, dist


class Batcher():
   def __init__(self, states, actions, old_log_probs, returns, advantages):
      self.states = states
      self.actions = actions
      self.old_log_probs = old_log_probs
      self.returns = returns
      self.advantages = advantages
      
   def __len__(self):
      return len(self.returns)
   
   def __getitem__(self, index):
      return (self.states[index], 
              self.actions[index], 
              self.old_log_probs[index], 
              self.returns[index],
              self.advantages[index]
             )