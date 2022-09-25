import json
import warnings

import torch

from models import FCModel


class Config:

    def __init__(self, path):
        with open(path, "r") as f:
            self.text = f.read()
            self.raw_data = json.loads(self.text)

        self.data = self.raw_data.copy()
        self.data["model"] = {"FCModel": FCModel}[self.raw_data["model"]]
        self.data["lr_decay"] = eval(self.raw_data["lr_decay"])
        self.data["epsilon_decay"] = eval(self.raw_data["epsilon_decay"])
        self.data["optimizer"] = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}[
            self.raw_data["optimizer"]]

    def to_str(self):
        return self.text

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            warnings.warn("Invalid key provided")
            return None
