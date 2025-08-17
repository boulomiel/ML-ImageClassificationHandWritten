import torch

from torchvision import  datasets
from torchvision import  transforms
from torch.utils.data import DataLoader

MNIST_ROOT = 'resource/lib/publicdata/data'

# Get reproductible results
torch.manual_seed(0)

class DatasetLoader:

    def __init__(self):
        self.train_set = datasets.MNIST(
            root=MNIST_ROOT,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        self.validation_set = datasets.MNIST(
            root=MNIST_ROOT,
            train=False,
            transform=transforms.ToTensor()
        )

        self.batch_size = 32

        self.train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.validation_loader = DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            shuffle=False,
        )

