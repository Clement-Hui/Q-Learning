import torch

from agent import Agent
from models import FCModel
from config import Config

config = Config("config.json")
"""
{
  "model": "FCModel",
  "model_size": [
    8,
    320,
    160,
    4
  ],
  "env": "LunarLander-v2",
  "epsilon": 0.5,
  "gamma": 0.995,
  "lr": 0.001,
  "lr_decay": "lambda x: max(0.999997 ** x, 0.5)",
  "epsilon_decay": "lambda x: max((50000 - x) / 50000, 0.1)",
  "optimizer": "adam",
  "alpha": 0.7,
  "beta": 0.5,
  "replay_size": 100000,
  "batch_size": 64,
  "update_every": 4,
  "tau": 0.004
}
"""
agent: Agent = Agent(config["model"],
                     config["model_size"],
                     config["env"],
                     config["epsilon"],
                     config["gamma"],
                     config["lr"],
                     config["lr_decay"],
                     config["epsilon_decay"],
                     config["optimizer"],
                     config["PER"],
                     config["alpha"],
                     config["beta"],
                     config["replay_size"],
                     config["batch_size"],
                     config["train_every"],
                     config["update_every"],
                     config["tau"])

agent.writer.add_text("Hyper-parameters", config.to_str())
agent.train(10000, 200, 20, 25)
