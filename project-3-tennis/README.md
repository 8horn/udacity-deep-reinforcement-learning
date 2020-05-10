# Unity Tennis 

Project 3 from Udacity cource on Deep Reinforcement Learning.

### Aim

In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.

![Gif showing the environment and how agent acts](tennis-env.gif)

### Environment

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. 
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents):

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Getting Started

It is recommended to create new environment with all necessary dependencies to run and train the agents. The instructions below demonstrate the steps how to do this, as advised by Udacity:

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

### Instructions

There are 2 python scripts and 1 Jupyter Notebook in this project. Jupyter notebook contains all the code necessary to train and
test the agent, while the python scripts are divided based by their functions:

- ```ActorCritic.py``` - Torch module holding Actor-Critic model that helps agent pick the most optimal action given environment state
- ```Agent.py``` - Proximal Policy Optimization agent, deals with choosing appropriate action given state, processes environment reaction and stores trajectories
- ```MultiAgent.py``` - Main class that wraps multiple ```Agent``` and learns a single ```ActorCritic``` model based on collected, shared experiences across all agents.

To run the code in Jupyter notebook, it is necessary to provide required Unity environment. Built binaries for this project - tennis - can be downloaded from the link below:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Once downlaoded and unzipped, please ensure the the cell in the ```Tennis.ipynb``` notebook which loads the environment point the the unzipped .exe file.

### Implementation

Implementation is described in ```Report.md```

This is an adoptation of my PPO implementation I did for project 2 ([link](https://github.com/8horn/udacity-deep-reinforcement-learning/tree/master/project-2-continuous-control)) for multi agent environment.
