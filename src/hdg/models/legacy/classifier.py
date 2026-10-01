import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
