from agent import Agent
from config import Config

config = Config("config.json")

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
