import time
from random import random
from typing import *

import gym
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from replaymemory import ReplayMemory, Transition


class Agent:
    def __init__(self, model, network_dims: tuple, environment: str,
                 epsilon: float, gamma: float, lr, lr_decay: Callable, epsilon_decay: Callable,
                 optimizer: type(Optimizer), alpha: float, beta: float, memory_size: int, batch_size: int,
                 n_target_model_params_apply: int):
        self.env = gym.make(environment)

        self.network = model(network_dims).double().cuda()
        self.target_network = model(network_dims).double().cuda()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optimizer(self.network.parameters(), lr=lr)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_decay)

        self.writer = SummaryWriter()
        self.replay_memory = ReplayMemory(memory_size, alpha, beta)

        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_target_model_params_apply = n_target_model_params_apply

    def train(self, training_episodes, target_reward, episode_req, eval_freq):

        n = 0

        num_episode_reached_target_consecutive = 0
        for episode in range(training_episodes):
            state = self.env.reset()
            done = False
            episodic_reward = 0
            episode_len = 0
            for iteration in range(10000):
                if done:
                    break
                self.network.zero_grad()
                self.optimizer.zero_grad()
                episode_len += 1

                epsilon = self.epsilon_decay(n) * self.epsilon

                action, output = self.get_action(state)

                # epsilon greedy
                if random() < epsilon:
                    action = self.env.action_space.sample()

                next_state, reward, done, info = self.env.step(action)
                episodic_reward += reward * self.gamma ** iteration

                q_target = reward + self.gamma * torch.max(self.network(self.as_tensor(next_state)))
                q_prediction = output[action]
                error = abs((q_target - q_prediction)) + 1e-5
                error = error.cpu().item()
                self.replay_memory.add_sample(Transition(state, action, reward, next_state, done, 1))

                # training
                if not self.replay_memory.can_sample(self.batch_size):
                    continue

                data, weights = self.replay_memory.sample(self.batch_size)
                states, actions, rewards, next_states, dones = data

                dones = ~dones
                with torch.no_grad():
                    q_targets = self.target_network(next_states)

                q_targets = rewards + self.gamma * torch.max(q_targets, 1)[0] * dones

                q_predictions = self.network(states).gather(1, actions.long().reshape(actions.shape[0], 1))
                q_predictions = q_predictions.squeeze()
                loss = (q_predictions - q_targets) ** 2 / 2
                loss = loss
                loss = loss.mean()

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                state = next_state

                self.writer.add_scalar('Loss', loss, n)
                n += 1

                if n % self.n_target_model_params_apply == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            self.writer.add_scalar('Episodic reward', episodic_reward, episode)
            self.writer.add_scalar('Episode length', episode_len, episode)
            self.writer.flush()

            if episodic_reward >= target_reward:
                num_episode_reached_target_consecutive += 1
            else:
                num_episode_reached_target_consecutive = 0

            if num_episode_reached_target_consecutive >= episode_req:
                return

            if episode % eval_freq == 0:
                print(
                    f"Episode {episode} -- Episodic reward {episodic_reward} Episodic length {episode_len} Epsilon {self.epsilon_decay(n) * self.epsilon} LR {self.lr_scheduler.get_last_lr()}")
                self.env.reset()
                done = False
                video_recorder = VideoRecorder(self.env, f"videos/{episode}_lunar.mp4", enabled=True)

                while not done:
                    self.env.render(mode='rgb_array')
                    video_recorder.capture_frame()
                    with torch.no_grad():
                        action, output = self.get_action(state)
                    state, reward, done, info = self.env.step(action)
                    time.sleep(1 / 60)

                self.env.close()
                video_recorder.close()
                video_recorder.enabled = False

    def as_tensor(self, np_array):
        return torch.tensor(np_array, dtype=torch.double).cuda()

    def get_action(self, state):
        with torch.no_grad():
            output = self.network(self.as_tensor(state))
            action = torch.argmax(output, 0)
            action = int(action.item())
        return action, output
