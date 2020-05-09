from ActorCritic import ActorCritic, Batcher
from Agent import PPO_Agent

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np

class MultiAgent():
   def __init__(self, state_size, action_size, num_agents, hidden_size=64, lr=1e-4):
      self.model = ActorCritic(state_size, action_size, hidden_size=hidden_size)
      self.optimizer = Adam(self.model.parameters(), lr=lr)
      self.agents = [PPO_Agent() for _ in range(num_agents)]
      
   def save_checkpoint(self, filepath=None):
      if filepath is None: filepath = 'checkpoint.pth'
      torch.save(self.model.state_dict(), filepath)
      
   def load_checkpoint(self, filepath):
      self.model.load_state_dict(torch.load(filepath))
      
   def act(self, states):
      results = zip(*[agent.chooce_action(self.model, state) for agent, state in zip(self.agents, states)])
      actions, log_probs, values = map(lambda x: np.array(x).squeeze(1), results)
      return actions, log_probs, values
   
   def step(self, states, actions, rewards, dones, log_probs, values, is_terminal=False):
      for i, agent in enumerate(self.agents):
         if is_terminal:
            agent.register_trajectories(states[i], None, None, None, None, values[i], is_terminal=is_terminal)
         else:
            agent.register_trajectories(states[i], actions[i], rewards[i], dones[i], log_probs[i], values[i])
            
   def process_trajectories(self, gamma=0.99, gae_tau=0.95):
      for agent in self.agents:
         agent.calculate_gae_returns(gamma=gamma, gae_tau=gae_tau)
         
   def maybe_learn(self, i_episode, update_every=4):
      if i_episode % update_every == 0:
         accumulated_trajectories = []
         for agent in self.agents:
            accumulated_trajectories += agent.processed_trajectories
            
         self.learn(accumulated_trajectories)
         
   def learn(self, accumulated_trajectories, batch_size=64, epsilon_clip=0.2, gradient_clip=10, beta=0.001, critic_discount=1., num_epochs=5):
      # Unroll and convert accumulated trajectories to tensors
      states, actions, old_log_probs, returns, advantages = map(torch.FloatTensor, zip(*accumulated_trajectories))
         
      # Normalized advantages
      advantages = (advantages - advantages.mean())  / (advantages.std() + 1e-7)
      
      # Get random batches from accumulated trajectories
      batcher = DataLoader(
         Batcher(states, actions, old_log_probs, returns, advantages),
         batch_size=batch_size,
         shuffle=True
      )
      
      self.model.train()
      for _ in range(num_epochs):
         for states, actions, old_log_probs, returns, advantages in batcher:
            # Get updated values from policy
            values, dist = self.model(states) 
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
            self.optimizer.zero_grad()
            losses.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.optimizer.step()
      
      # Reset collected trajectories
      for agent in self.agents: agent.reset()