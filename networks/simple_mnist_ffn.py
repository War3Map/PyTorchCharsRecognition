import torch.nn as nn
import torch.nn.functional as F


class SimpleMnistFeedForward(nn.Module):
    def __init__(self, device):
        super(SimpleMnistFeedForward, self).__init__()
        self.device = device
        # определяем слои нейросети
        self.fc1 = nn.Linear(28 * 28, 784).to(device)
        self.fc2 = nn.Linear(784, 784).to(device)
        self.fc3 = nn.Linear(784, 10).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x)).to(self.device)
        x = F.relu(self.fc2(x)).to(self.device)
        x = self.fc3(x).to(self.device)
        return F.log_softmax(x).to(self.device)

    # вывод архитектуры нейросети
    def __str__(self):
        print(self)
