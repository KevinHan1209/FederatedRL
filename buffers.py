import numpy as np

class OnPolicyBuffer:
    def __init__(self, gamma):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.log_probs = []
        self.dones = []
        self.gamma = gamma

    def store(self, state, action, reward, probs, log_probs, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(probs)
        self.log_probs.append(log_probs)
        self.dones.append(done)

    def compute_returns(self):
        returns = []
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        self.returns = returns

    def get(self, *args):
        results = []
        for arg in args:
            if hasattr(self, arg):
                results.append(getattr(self, arg))
            else:
                raise AttributeError(f"Attribute '{arg}' does not exist.")

        return tuple(results)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.log_probs = []
        self.dones = []
