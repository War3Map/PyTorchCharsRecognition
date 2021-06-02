import torch
import torch.nn as nn
import torch.nn.functional as F


# Объявим класс для нашей нейронной сети
from torch import tensor


class SimpleEmnistConvNet(nn.Module):
    """
    NN architecture

    Attributes:

    need_resize(bool): if needs to resize input data

    """
    def __init__(self, device):
        super(SimpleEmnistConvNet, self).__init__()
        self.device = device
        # определяем слои нейросети
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.Conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.Conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(3 * 3 * 64, 200).to(device)
        self.fc2 = nn.Linear(200, 200).to(device)
        self.fc3 = nn.Linear(200, 62).to(device)
        self.need_resize = False

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.Conv1(x).to(self.device)
        x = self.Conv2(x).to(self.device)
        x = self.Conv3(x).to(self.device)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x)).to(self.device)
        x = F.relu(self.fc2(x)).to(self.device)
        x = self.fc3(x).to(self.device)
        return F.log_softmax(x, -1).to(self.device)

    # вывод архитектуры нейросети
    # def __str__(self):
    #     print(self)
