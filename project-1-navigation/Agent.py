from ReplayBuffer import ReplayBuffer
from QNetwork import QNetwork
import torch
import torch.nn.functional as F
import random
import numpy as np

# Torch device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

class Agent():
   def __init__(self, capacity, state_size, action_size, pretrained_model_path=None,
                tau=1e-3, gamma=0.99, batch_size=32, lr=1e-4, learn_every_n_steps=4):
      # Environment variables
      self.state_size = state_size
      self.action_size = action_size
      
      # Create Qnetworks
      self.qnetwork_local = QNetwork(state_size, action_size).to(device)
      self.qnetwork_target = QNetwork(state_size, action_size).to(device)
      self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)
      self.batch_size = batch_size
      self.gamma = gamma
      self.tau = tau

      if pretrained_model_path is not None:
         self.qnetwork_local.load_state_dict(torch.load(pretrained_model_path))
      
      # Initialize memory buffer
      self.memory = ReplayBuffer(capacity, batch_size)
      
      # Initialize time step for updating target network every q steps
      self.learn_every_n_steps = learn_every_n_steps
      self.t_step = 0
      
   def step(self, state, action, reward, next_state, done):
      """Learn from the action and environments reponse."""
      self.memory.add(state, action, reward, next_state, done)
      
      # Maybe learn if learn_every_n_steps has passed
      self.t_step = (self.t_step +1) % self.learn_every_n_steps
      if self.t_step == 0:
         if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
      
   def act(self, state, eps=1.):
      """
      Returns actions for given state as per current policy.
        
      Params
      ======
         state (array_like): current state
         eps (float): epsilon, for epsilon-greedy action selection
      """
      state = torch.from_numpy(state).float().unsqueeze(0).to(device)
      self.qnetwork_local.eval()
      with torch.no_grad():
         action_values = self.qnetwork_local(state)
      self.qnetwork_local.train()
      
      # epsilon-greedy action selection
      if random.random() > eps:
         return np.argmax(action_values.cpu().data.numpy()).astype(int)
      else:
         return random.choice(np.arange(self.action_size)).astype(int)
    
   def learn(self, experiences):
      """Update network parameters"""
      states, actions, rewards, next_states, dones = experiences

      # Get best score according to the target network and evaluate it against the local network
      next_action_values = self.qnetwork_target(next_states).detach().max(dim=1)[0].unsqueeze(1)
      y = rewards + (self.gamma*next_action_values*(1-dones))
      yhat = self.qnetwork_local(states).gather(1, actions)
      loss = F.mse_loss(yhat, y)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      
      # Update target network
      self.soft_update()

   def soft_update(self):
      """Performs soft update of frozen target network as per double-DQN"""
      for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
         target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)