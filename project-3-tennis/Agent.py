import torch
import torch.nn as nn
import numpy as np
from torch import FloatTensor
from collections import namedtuple
from ActorCritic import ActorCritic

class PPO_Agent():
   def __init__(self):
      self.trajectory = namedtuple('Trajectory', field_names=['state', 'action', 'reward', 'mask', 'log_prob', 'value'])
      self.reset()

   def reset(self):
      self.trajectories = []
      self.processed_trajectories = []

   def chooce_action(self, model, state):
      # Pick an action
      value, dist = model(FloatTensor(state))
      action = dist.sample()
      log_prob = dist.log_prob(action)

      # Convert to numpy
      log_prob = log_prob.detach().cpu().numpy()
      value = value.detach().cpu().numpy()
      action = action.detach().cpu().numpy()

      return action, log_prob, value

   def register_trajectories(self, state, action, reward, mask, log_prob, value, is_terminal=False):
      if not is_terminal:
         reward = np.array(reward).reshape(-1)
         mask = np.array((1-mask)).reshape(-1)
         value = value.reshape(-1)
      t = self.trajectory(state, action, reward, mask, log_prob, value)
      self.trajectories.append(t)

   def calculate_gae_returns(self, gamma=0.99, gae_tau=0.95):
      current_trajectories = [None] * (len(self.trajectories)-1)
      gae = 0.
      
      for i in reversed(range(len(self.trajectories) - 1)):
         state, action, reward, mask, log_prob, value = self.trajectories[i] 
         next_value = self.trajectories[i+1].value

         delta = reward + gamma * next_value * mask - value
         gae = delta + gamma * gae_tau * mask * gae
         discounted_return = gae + value
         advantage = discounted_return - value
         
         current_trajectories[i] = (
            state, 
            action, 
            log_prob, 
            discounted_return, 
            advantage
         )
      
      # Reset collected raw trajectories and accumulate processed experiences
      self.trajectories = []
      self.processed_trajectories += current_trajectories

   # def learn(self, batcher, epsilon_clip=0.2, gradient_clip=10, beta=0.001, critic_discount=1., num_epochs=5):
   #    self.model.train()
   #    for _ in range(num_epochs):
   #       for states, actions, old_log_probs, returns, advantages in batcher:
   #          # Get updated values from policy
   #          values, dist = self.model(states) 
   #          new_log_probs = dist.log_prob(actions)
   #          entropy = dist.entropy()

   #          # Calculate ratio and clip, so that learning doesn't change new policy much from old 
   #          ratio = (new_log_probs - old_log_probs).exp()
   #          clip = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip)
   #          clipped_surrogate = torch.min(ratio*advantages, clip*advantages)

   #          # Get losses
   #          actor_loss = -torch.mean(clipped_surrogate) - beta * entropy.mean()
   #          critic_loss = torch.mean(torch.square((returns - values)))
   #          losses = critic_loss * critic_discount + actor_loss

   #          # Do the optimizer step
   #          self.optimizer.zero_grad()
   #          losses.backward()
   #          nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
   #          self.optimizer.step()