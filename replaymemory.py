from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "probability"])


class ReplayMemory:
    def __init__(self, size, alpha: float = 0.7, beta: float = 0.5):
        self.size = size
        self.length = 0
        self.memory = np.empty(self.size, object)
        self.index = 0
        self.alpha = alpha
        self.beta = beta

    def add_sample(self, transition: Transition):

        if self.length < self.size:
            self.memory[self.length] = transition
            self.length += 1
            self.index += 1
        else:
            if self.index == self.length:
                self.index = 0

            self.memory[self.index] = transition
            self.index += 1

    def sample(self, size):
        arr = self.memory[:self.length]
        probabilities = np.array([i[5] for i in arr]) ** self.alpha
        sampling_probabilities = probabilities / np.sum(probabilities)
        indexes = np.random.choice(range(self.length), size, p=sampling_probabilities, replace=False)

        output = arr[indexes]
        states = [item.state for item in output]
        actions = [item.action for item in output]
        rewards = [item.reward for item in output]
        next_states = [item.next_state for item in output]
        dones = [item.done for item in output]

        output = (states, actions, rewards, next_states, dones)
        output = [torch.tensor(item, dtype=torch.double).cuda() if type(item[0]) != bool else torch.tensor(item).cuda()
                  for item in output]

        weights = np.power(self.length * sampling_probabilities[indexes], -self.beta)
        weights = weights / np.max(weights)
        weights = torch.tensor(weights).double().cuda()
        return output, weights

    def can_sample(self, size):
        if size > self.length:
            return False
        else:
            return True
