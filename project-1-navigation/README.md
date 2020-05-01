# Unity Banana Navigation

Project 1 from Udacity cource on Deep Reinforcement Learning.

### Aim

The aim is to train an agent to navigate around 3d environment and collect how many yellow bananas as possible,
Each yellow banana is worth 1 point. Picking up blue bananas esult in a penalty of -1.

![Gif showing the environment and how agent acts](banana-example.gif)

### Environment

The state has 37 dimensions and contains agent's velocity, along with ray-based perception of objects around the agent's forward direction.

There are 4 discrete actions:

- ```0``` - move forward
- ```1``` - move backwards
- ```2``` - turn left
- ```3``` - turn right

The task is episodic and the environment is considered solved once the agent achieves average score of +13 over 100 consecutive episodes.

### Getting Started

Ensure ```Python 3.6``` and ```PyTorch 1.5``` along with ```UnityML-Agents``` and ```OpenAI Gym``` are installed in your environment. 
What is also required is the Banana.exe application built in Unity - please refer to Udacity course for more details.

### Instructions

There are 3 python scripts and 1 Jupyter Notebook in this project. Jupyter notebook contains all the code necessary to train and
test the agent, while the python scripts are divided based by their functions:

- ```Agent.py``` - Class representing Agent that learn how to behave in given environment
- ```QNetwork.py``` - Deep Learning-based architecture that acts as function approximation for an agent to learn which action to take current state
- ```ReplayBuffer.py``` - Storage where agent's past experiences are held

### Implementation

Implementation is described in ```Report.md```
