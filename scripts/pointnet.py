from torch_geometric.nn import global_max_pool
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

# Define a quasi-PointNet model (https://arxiv.org/abs/1612.00593)
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1) # just 1 classes

    def forward(self, x, batch):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0] # max pooling
        x = torch.relu(self.fc1(x))
        x = global_max_pool(x, batch)  # [num_examples, hidden_channels]
        x = self.fc2(x)
        return x