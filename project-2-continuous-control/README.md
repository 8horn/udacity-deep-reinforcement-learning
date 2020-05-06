# Unity Reacher 

Project 2 from Udacity cource on Deep Reinforcement Learning.

### Aim

In this environment, a double-jointed arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

![Gif showing the environment and how agent acts](env-view.gif)

### Environment

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque applicable to two joints. 
Every entry in the action vector should be a number between -1 and 1.

There are 20 identical agents, each with its own copy of the environment - the key of having multiple agents within the same environment
is that they can learn from each other and therefore speed up the training process.

The environment si considered solved when the average score over 100 consecutive episodes is +30, where average score per episode represents 
an average score across all 20 agents.

### Getting Started

Ensure ```Python 3.6``` and ```PyTorch 1.5``` along with ```UnityML-Agents``` and ```OpenAI Gym``` are installed in your environment. 
What is also required is the Banana.exe application built in Unity - please refer to Udacity course for more details.

### Instructions

There are 2 python scripts and 1 Jupyter Notebook in this project. Jupyter notebook contains all the code necessary to train and
test the agent, while the python scripts are divided based by their functions:

- ```ActorCritic.py``` - Torch module holding Actor-Critic model that helps agent pick the most optimal action given environment state
- ```Agent.py``` - Collections of functions that build up the agent

### Implementation

Implementation is described in ```Report.md```

### Inspirations

Inspiried (and heavily adopted to suit my coding style) by the following git repositories and blog posts:

- https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
- https://github.com/nikhilbarhate99/PPO-PyTorch
