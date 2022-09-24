import time
from random import random
from typing import *

import gym
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from replaymemory import ReplayMemory, Transition, PERReplayMemory, PERTransition


class Agent:
    def __init__(self, model, network_dims: tuple, environment: str,
                 epsilon: float, gamma: float, lr, lr_decay: Callable, epsilon_decay: Callable,
                 optimizer: type(Optimizer), PER: bool, alpha: float, beta: float, memory_size: int,
                 batch_size: int, train_every: int, update_every: int, tau: float):
        self.env = gym.make(environment)

        self.network = model(network_dims).double().cuda()
        self.target_network = model(network_dims).double().cuda()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optimizer(self.network.parameters(), lr=lr)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_decay)

        self.writer = SummaryWriter()

        self.PER = PER
        if PER:
            self.replay_memory = PERReplayMemory(memory_size, alpha, beta)
        else:
            self.replay_memory = ReplayMemory(memory_size)

        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_every = train_every
        self.tau = tau
        self.update_every = update_every

    def train(self, training_episodes, target_reward, episode_req, eval_freq):

        n = 0

        num_episode_reached_target_consecutive = 0
        avg_episodic_reward = 0
        avg_episodic_len = 0
        for episode in range(training_episodes):
            state = self.env.reset()
            done = False
            episodic_reward = 0
            episode_len = 0
            for iteration in range(10000):
                self.network.zero_grad()
                self.optimizer.zero_grad()
                self.target_network.zero_grad()
                if done:
                    break

                episode_len += 1

                epsilon = self.epsilon_decay(n) * self.epsilon

                action, output = self.get_action(state)

                # epsilon greedy
                if random() < epsilon:
                    action = self.env.action_space.sample()

                next_state, reward, done, info = self.env.step(action)
                episodic_reward += reward * self.gamma ** iteration

                if self.PER:
                    with torch.no_grad():
                        if done:
                            q_target = reward
                        else:
                            q_target = reward + self.gamma * torch.max(self.target_network(self.as_tensor(next_state)))
                        q_prediction = output[action]
                        error = abs((q_target - q_prediction)) + 1e-5
                        error = error.cpu().item()
                        self.replay_memory.add_sample(PERTransition(state, action, reward, next_state, done, error))
                else:
                    self.replay_memory.add_sample(Transition(state, action, reward, next_state, done))

                # training
                if n % self.train_every == 0:
                    if not self.replay_memory.can_sample(self.batch_size):
                        continue

                    if self.PER:
                        data, weights = self.replay_memory.sample(self.batch_size)
                    else:
                        data = self.replay_memory.sample(self.batch_size)
                    states, actions, rewards, next_states, dones = data

                    dones = ~dones
                    with torch.no_grad():
                        q_targets = self.target_network(next_states)

                    q_targets = rewards + self.gamma * torch.max(q_targets, 1)[0] * dones

                    q_predictions = self.network(states).gather(1, actions.long().reshape(actions.shape[0], 1))
                    q_predictions = q_predictions.squeeze()
                    loss = (q_predictions - q_targets) ** 2

                    if self.PER:
                        loss = loss * weights
                    loss = loss.mean() ** (1 / 2)
                    loss.backward()

                    # gradient clipping to avoid exploding gradients
                    torch.nn.utils.clip_grad_value_(self.network.parameters(), 1)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    state = next_state

                    self.writer.add_scalar('Loss', loss, n)
                n += 1

                # soft update target network
                if n % self.update_every == 0:
                    for target_param, local_param in zip(self.target_network.parameters(), self.network.parameters()):
                        target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

            self.writer.add_scalar('Episodic reward', episodic_reward, episode)
            self.writer.add_scalar('Episode length', episode_len, episode)
            self.writer.flush()

            avg_episodic_reward += episodic_reward
            avg_episodic_len += episode_len

            if episodic_reward >= target_reward:
                num_episode_reached_target_consecutive += 1
            else:
                num_episode_reached_target_consecutive = 0

            if num_episode_reached_target_consecutive >= episode_req:
                return

            if episode % eval_freq == 0:
                if episode > 0:
                    avg_episodic_reward /= eval_freq
                    avg_episodic_len /= eval_freq
                print(
                    f"Episode {episode} -- Episodic reward {avg_episodic_reward} Episodic length {avg_episodic_len} Epsilon {self.epsilon_decay(n) * self.epsilon} LR {self.lr_scheduler.get_last_lr()}")

                avg_episodic_reward = 0
                avg_episodic_len = 0
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
