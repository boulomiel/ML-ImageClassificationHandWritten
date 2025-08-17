import torch
from functorch.dim import Tensor
from torch import nn, optim
from torch.utils.data import DataLoader

from mlp import MLP
from predictable import Predicatable


class Validator(Predicatable):

    def __init__(self, model: MLP,
                 loss_function: nn.CrossEntropyLoss,
                 dataset_loader: DataLoader
                 ):
        self.model = model
        self.dataset_loader = dataset_loader
        self.loss_function = loss_function

    def validate(self):
        self.model.eval()
        running_loss = 0
        running_correct = 0

        # When backward won't be call, saves computation memory by turning off gradients calculation
        with torch.no_grad():

            # x_train: We know images are 28 * 28
            # y_train: labels
            for (x_val, y_val) in self.dataset_loader:

                # Flatten the image since the input to the network is a 784 dimensional vector
                x_validated = x_val.view(x_val.shape[0], -1)

                # Compute raw score by passing x to the model
                # y: probability for each class
                y = self.model(x_validated)

                # Score to probability using softmax
                prob = nn.functional.softmax(y, dim=0)

                # Compute accuracy
                y_probability = prob.argmax(dim=1)

                correct =  torch.sum(y_probability == y_val)
                running_correct += correct

                # Compute print and loss
                loss = self.loss_function(y, y_val)
                running_loss += loss.item()

        return self.predict(running_loss, running_correct)

    def predict(self, loss:Tensor, correct:Tensor):
        return loss/len(self.dataset_loader), correct.item()/len(self.dataset_loader)
