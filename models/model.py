import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from abc import ABC, abstractmethod

from utils import *

class Model(ABC):
    """
    A base class for all DVC models types in a PyTorch context
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def print_network_summary(self) -> None:
        """
        Print the network summary of this model
        """
        pass

    @abstractmethod
    def train_mode(self) -> None:
        """
        Set the mode to training
        """
        pass

    @abstractmethod
    def eval_mode(self) -> None:
        """
        Set the mode to evaluation
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Forward pass
        """
        pass

    @abstractmethod
    def initialize_weights(self) -> None:
        """
        Initialize weights
        """
        pass

    @abstractmethod
    def to(self, device):
        """
        To a specific device
        """
        pass

    @abstractmethod
    def parameters(self):
        """
        Return an iterator over module parameters

        This is typically passed to an optimizer
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Load a static dictionary of network
        """
        pass

    @abstractmethod
    def save_checkpoint(self):
        """
        Save the checkpoint of network
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train this model
        """
        pass

    @abstractmethod
    def eval(self):
        """
        Evaluate this model
        """
        pass

from models.model_cnn import ModelCNN
from models.model_volraft import ModelVolRAFT

class ModelFactory:
    @staticmethod
    def build_instance(patch_shape, flow_shape, config, ptdtype = torch.float32) -> Model:
        """
        Static method to create and return instances of models based on the model type string.

        Parameters:
        - model_type (str): The type of model to create. Supported values: 'CNN', 'FCN'.

        Returns:
        - An instance of the requested model.

        Raises:
        - ValueError: If an unsupported model type is requested.
        """
        model_type = str(config["network_name"]).lower()

        if model_type in ('cnn32', 'dvcnet'):
            return ModelCNN(patch_shape = patch_shape,
                            flow_shape = flow_shape,
                            config = config,
                            dtype = ptdtype)
        elif model_type == 'fcn':
            return ModelFCN()
        elif model_type == 'volraft':
            return ModelVolRAFT(patch_shape = patch_shape,
                                flow_shape = flow_shape,
                                config = config,
                                dtype = ptdtype)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # You can add more static methods here for other tasks related to models

class ModelFCN(Model):
    def __init__(self):
        super(ModelFCN, self).__init__()
