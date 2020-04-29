import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.9, gamma=1.):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state, eps):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if state in self.Q and eps < np.random.random():
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)
        
    def calculate_expected_reward(self, state, eps):
        probs = np.full(self.nA, eps/self.nA)
        probs[np.argmax(self.Q[state])] += 1 - eps
        return np.dot(self.Q[state], probs)

    def step(self, state, action, reward, next_state, done, eps):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        expected_reward = self.calculate_expected_reward(next_state, eps)
        self.Q[state][action] += self.alpha * (reward + self.gamma*expected_reward - self.Q[state][action])