import torch
import torch.nn as nn


class TreeHealthModel(nn.Module):
    def __init__(self, input_size):
        super(TreeHealthModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 класса: Good, Fair, Poor

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x