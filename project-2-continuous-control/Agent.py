import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from collections import namedtuple

trajectory = namedtuple('Trajectory', field_names=['state', 'action', 'reward', 'mask', 'log_prob', 'value'])

def act(env, model, state):
   # Pick an action
   value, dist = model(torch.FloatTensor(state))
   action = dist.sample()
   log_prob = dist.log_prob(action)

   # Convert to numpy
   log_prob = log_prob.detach().cpu().numpy()
   value = value.detach().cpu().numpy()
   action = action.detach().cpu().numpy()
   
   # Act and see response
   env_info = env.step(np.clip(action, -1, 1))[brain_name]
   next_state = env_info.vector_observations
   reward = np.array(env_info.rewards).reshape(-1, 1)
   done = np.array(env_info.local_done)
   mask = (1-done).reshape(-1, 1)
   
   t = trajectory(state, action, reward, mask, log_prob, value)
   return next_state, done, t


def calculate_gae_returns(trajectories, next_value, num_agents, gamma=0.99, gae_tau=0.95):
   processed_trajectories = [None] * (len(trajectories)-1)
   gae = 0.
   
   for i in reversed(range(len(trajectories) - 1)):
      state, action, reward, mask, log_prob, value = trajectories[i] 
      next_value = trajectories[i+1].value

      delta = reward + gamma * next_value * mask - value
      gae = delta + gamma * gae_tau * mask * gae
      discounted_return = gae + value
      advantage = discounted_return - value
      
      processed_trajectories[i] = (
         state, 
         action, 
         log_prob, 
         discounted_return, 
         advantage
      )
      
   return processed_trajectories


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


def learn(model, optimizer, batcher, epsilon_clip=0.2, beta=0.001, gradient_clip=10, critic_discount=1., num_epochs=5):
   for _ in range(num_epochs):
      for states, actions, old_log_probs, returns, advantages in batcher:
         # Get updated values from policy
         values, dist = model(states) 
         new_log_probs = dist.log_prob(actions)
         entropy = dist.entropy()

         # Calculate ratio and clip, so that learning doesn't change new policy much from old 
         ratio = (new_log_probs - old_log_probs).exp()
         clip = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip)
         clipped_surrogate = torch.min(ratio*advantages, clip*advantages)

         # Get losses
         actor_loss = -torch.mean(clipped_surrogate) - beta * entropy.mean()
         critic_loss = torch.mean(torch.square((returns - values)))
         losses = critic_loss * critic_discount + actor_loss

         # Do the optimizer step
         optimizer.zero_grad()
         losses.backward()
         nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
         optimizer.step()
