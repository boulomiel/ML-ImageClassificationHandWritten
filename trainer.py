import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

from mlp import MLP

class Trainer:

    def __init__(self,
                 model: MLP,
                 loss_function: nn.CrossEntropyLoss,
                 dataset_loader: DataLoader,
                 optimizer: optim.Optimizer
                 ):
        self.model = model
        self.dataset_loader = dataset_loader
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self):
        self.model.train()
        running_loss = 0
        running_correct = 0
        # x_train: We know images are 28 * 28
        # y_train: labels
        for (x_train, y_train) in self.dataset_loader:
            # Flatten the image since the input to the network is a 784 dimensional vector
            x_train = x_train.view(x_train.shape[0], -1)

            # Forward pass
            # Compute predicted y by passing x to the model
            # y: probability for each class
            y = self.model(x_train)

            # Compute and print loss
            loss = self.loss_function(y, y_train)
            running_loss += loss.item()

            # Compute Accuracy
            # y_predication: Corresponds to the class with the maximum prediction
            y_prediction: Tensor =  y.argmax(dim=1)
            correct = torch.sum(y_prediction == y_train)
            running_correct += correct

            # Zero gradients, perform a backward pass, and update the weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.predict(loss=running_loss, correct=running_correct)

    def predict(self, loss: float, correct: float) -> tuple[float, float]:
        return loss / len(self.dataset_loader), correct.item() / len(self.dataset_loader)