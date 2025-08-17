from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.utils.data import DataLoader

from mlp import MLP


class Predicatable(ABC):

    model: MLP
    dataset_loader: DataLoader
    loss_function: nn.CrossEntropyLoss

    @abstractmethod
    def predict(self, loss: int, correct: int) -> tuple[float, float]:
        pass