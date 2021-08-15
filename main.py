import torch

from agent import Agent
from models import FCModel

agent = Agent(FCModel, (8, 32, 128, 4), "LunarLander-v2", 0.4, 0.99, 0.01,
              lambda x: max(0.998 ** x, 0.1), lambda x: (50000 - x) / 50000, torch.optim.Adam, 0.7, 0.5, 10000, 128,
              10000)

agent.train(10000, 150, 20, 25)
