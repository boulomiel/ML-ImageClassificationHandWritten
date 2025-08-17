import torch
from torch import nn, optim

from dataset_loader import DatasetLoader
from mlp import MLP
from trainer import Trainer
from validator import Validator

import matplotlib.pyplot as plt

def main() -> int:
    num_epochs: int = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)

    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dataset_loader = DatasetLoader()

    trainer = Trainer(model=model,
                      loss_function=loss_function,
                      dataset_loader=dataset_loader.train_loader,
                      optimizer=optimizer)

    validator = Validator(model=model,
                          loss_function=loss_function,
                          dataset_loader= dataset_loader.validation_loader)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    print("Starting Training...")
    for ep in range(num_epochs):
        train_loss, train_acc = trainer.train()
        val_loss, val_acc = validator.validate()
        print("Epoch: {}, Train Loss = {:.3f}, Train Acc = {:.3f} , Val Loss = {:.3f}, Val Acc = {:.3f}".
              format(ep, train_loss, train_acc, val_loss, val_acc))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

    plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.plot(train_loss_history, 'r')
    plt.plot(val_loss_history, 'b')
    plt.title("Loss Curve")
    plt.show()

    plt.subplot(122)
    plt.plot(train_acc_history, 'r')
    plt.plot(val_acc_history, 'b')
    plt.title("Accuracy Curve")
    plt.show()

    images, labels = next(iter(dataset_loader.validation_loader))
    plt.imshow(images[0][0], 'gray')
    plt.show()

    images.resize_(images.shape[0], 1, 784)
    score = model(images[0, :])
    prob = nn.functional.softmax(score[0], dim=0)
    y_prob = prob.argmax()
    print("Predicted class {} with probability {}".format(y_prob, prob[y_prob]))

    return 0

if __name__ == "__main__":
    main()