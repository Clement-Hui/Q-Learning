from collections import namedtuple, deque

import numpy as np
import torch

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
PERTransition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "probability"])


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.length = 0
        self.memory = deque(maxlen=int(size))
        self.index = 0

    def add_sample(self, transition):
        self.memory.append(transition)
        self.length = len(self.memory)

    def sample(self, size):
        # TODO: require optimization

        #
        indexes = np.random.choice(range(self.length), size, replace=False)

        output = [self.memory[i] for i in indexes]
        states = [item.state for item in output]
        actions = [item.action for item in output]
        rewards = [item.reward for item in output]
        next_states = [item.next_state for item in output]
        dones = [item.done for item in output]

        output = (states, actions, rewards, next_states, dones)
        output = [torch.tensor(np.array(item), dtype=torch.double).cuda() if type(item[0]) != bool else torch.tensor(
            np.array(item)).cuda()
                  for item in output]

        return output

    def can_sample(self, size):
        if size > self.length:
            return False
        else:
            return True


class PERReplayMemory(ReplayMemory):
    def __init__(self, size, alpha: float = 0.7, beta: float = 0.5):
        super().__init__(size)
        self.alpha = alpha
        self.beta = beta

    def add_sample(self, transition):
        super().add_sample(transition)

    def sample(self, size):
        # TODO: require optimization

        probabilities = np.array([i[5] for i in self.memory]) ** self.alpha
        sampling_probabilities = probabilities / (np.sum(probabilities))
        #
        indexes = np.random.choice(range(self.length), size, p=sampling_probabilities, replace=False)

        output = self.memory[indexes]
        states = [item.state for item in output]
        actions = [item.action for item in output]
        rewards = [item.reward for item in output]
        next_states = [item.next_state for item in output]
        dones = [item.done for item in output]

        output = (states, actions, rewards, next_states, dones)
        output = [torch.tensor(np.array(item), dtype=torch.double).cuda() if type(item[0]) != bool else torch.tensor(
            np.array(item)).cuda()
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
