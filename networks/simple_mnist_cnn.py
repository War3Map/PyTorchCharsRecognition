import torch.nn as nn
import torch.nn.functional as F


class SimpleMnistConvNet(nn.Module):
    def __init__(self, device):
        super(SimpleMnistConvNet, self).__init__()
        self.device = device
        # Слои нейросети
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.Conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc3 = nn.Linear(7 * 7 * 16, 10).to(device)

    def forward(self, x):
        x = self.Conv1(x).to(self.device)
        x = self.Conv2(x).to(self.device)
        x = x.reshape(x.size(0), -1)
        x = self.fc3(x).to(self.device)
        return F.log_softmax(x).to(self.device)

    # вывод архитектуры нейросети
    def __str__(self):
        print(self)

