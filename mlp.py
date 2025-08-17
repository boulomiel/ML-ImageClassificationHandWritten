from torch import nn

IMG_SIZE = 28
HIDDEN_LAYER_SIZE = 512
OUTPUT_LAYER_SIZE = 10

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=IMG_SIZE*IMG_SIZE,
                      out_features=HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_LAYER_SIZE,
                      out_features=HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_LAYER_SIZE,
                      out_features=OUTPUT_LAYER_SIZE)
        )

    def forward(self, x):
        return self.model(x)