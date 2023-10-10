from torch_geometric.nn import global_max_pool
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

# Define a quasi-PointnetCurv model (https://arxiv.org/abs/1612.00593)


class PointnetCurv(nn.Module):
    def __init__(self):
        super(PointnetCurv, self).__init__()
        self.conv1 = nn.Conv1d(6, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64+13, 1)  # just 1 classes
        # TODO
        # Concat additinal 9 + 4 features to fc3 layer
        # The 9 features are flattened from the 3x3 covariance matrix
        # The 4 features are the quaternion representation of the rotation matrix

    def forward(self, x, imu, batch):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]  # max pooling
        x = torch.relu(self.fc1(x))
        x = global_max_pool(x, batch)  # [num_examples, hidden_channels]
        x = self.fc2(x)

        # make imu view as batch size x 13
        imu = imu.view(-1, imu.shape[0])
        x = torch.cat((x, imu), dim=1)
        x = self.fc3(x)
        return x
