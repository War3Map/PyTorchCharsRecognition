# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:49:57 2020

@author: barce
"""

# стандартные модули
import os.path
import time
import sys
from datetime import timedelta

# импорт модулей pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd import Variable
from torchvision import datasets, transforms

import graphs_shower

DATAPATH = r"C:\Users\IVAN\Desktop\dataEMNIST"


# Объявим класс для нашей нейронной сети
class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.device = device
        # определяем слои нейросети
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2))
        self.Conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2))
        self.Conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2))
        self.fc1 = nn.Sequential(
            nn.Linear(3 * 3 * 64, 200),
            nn.Dropout(0.5))
        self.fc2 = nn.Linear(200, 62)

    def forward(self, x):
        x = self.Conv1(x).to(self.device)
        x = self.Conv2(x).to(self.device)
        x = self.Conv3(x).to(self.device)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x)).to(self.device)
        x = self.fc2(x).to(self.device)
        # x = self.fc3(x).to(self.device)
        return F.log_softmax(x, -1).to(self.device)

    # вывод архитектуры нейросети
    def __str__(self):
        print(self)


def load_traindata(batch_size):
    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(DATAPATH, split="byclass", train=True, download=True, transform=transformations),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    labels_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(DATAPATH, split="byclass", train=False, download=False, transform=transformations),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    dataset_test_len = len(labels_loader.dataset)
    dataset_train_len = len(train_loader.dataset)
    print("Длина обучающего датасета {}\n Длина трениро"
          "вочного датасета\n".format(dataset_train_len, dataset_test_len))
    return train_loader, labels_loader


def train_net(net, train_loader, optimizer, criterion, device, losses_info, acc_info):
    # запускаем главный тренировочный цикл
    # пройдёмся по батчам из наших тестовых наборов
    # каждый проход меняется эпоха
    loss = 0
    correct = 0
    total = 0
    maxbatch_count = len(train_loader.dataset) // batch_size

    for epoch in range(epochs):
        final_loss = 0
        correct = 0
        batch_count = 0
        for batch_id, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            # # изменим размер с (batch_size, 1, 28, 28) на (batch_size, 28*28)
            # data = data.view(-1, 28*28)
            # оптимизатор
            # для начала обнулим градиенты перед работой
            optimizer.zero_grad()
            # считаем выход нейросети
            net_out = net(data)
            # оптимизировать функцию потерь будем на основе целевых данных и выхода нейросети
            loss = criterion(net_out, labels)
            # делаем обратный ход
            loss.backward()
            # оптимизируем функцию потерь, запуская её по шагам полученным на backward этапе
            optimizer.step()
            # вывод информации
            print('Train Epoch: {} [{}/{} ({:.0f}%)]; Loss: {:.6f}'.format(
                epoch + 1, (batch_id + 1) * len(data), len(train_loader.dataset),
                100. * (batch_id + 1) / len(train_loader), loss.data))
            # len(train_loader.dataset)
            batch_count += 1
            final_loss += loss.data.item()
            # Отслеживание точности
            if (batch_id + 1) * len(data) == maxbatch_count * len(data):
                total = labels.size(0)
                _, predicted = torch.max(net_out.data, 1)
                correct += (predicted == labels).sum().item()
        # вносим текущее значение функции потерь
        losses_info.append(final_loss / batch_count)
        print("Average Loss={}".format(final_loss / batch_count))
        # вносим текущее значение точности распознавания
        acc_info.append(correct / total)


def test_nn(net, test_loader, device):
    # тестирование
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            # data = data.view(-1, 28 * 28)

            net_out = net(data)
            # Суммируем потери со всех партий
            test_loss += criterion(net_out, labels).data
            pred = net_out.data.max(1)[1]  # получаем индекс максимального значения
            # сравниваем с целевыми данными, если совпадает добавляем в correct
            correct += pred.eq(labels.data).sum()

    test_loss /= len(test_loader.dataset)
    test_acc = float(100. * correct / len(test_loader.dataset))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))
    return test_acc


# Сохраняет статистику в файл
def save_stats(accuracies, losses, common_time, save_file):
    with open(save_file, "w+") as file:
        file.write("Train:{}\nTest:{}\n".format(common_time[0], common_time[1]))
        epochs_count = len(accuracies)
        for i in range(0, epochs_count):
            file.write("{}:{}\n".format(accuracies[i], losses[i]))


# функции потерь на каждой эпохе
epoch_losses = list()
# список значений точности на каждой эпохе
acc_list = list()

model_path = r"models\CNN_EMNIST_BN_model"
is_model_exists = os.path.isfile(model_path)

DATASET = "EMNIST"
OPTIMIZER = "Adam"
NETWORK_TYPE = "CNN"

# Назначаем устройство на котором будет работать нейросеть, по возможности CUDA
dev = "cuda" if torch.cuda.is_available() else "cpu"
used_device = torch.device(dev)
print("Running on Device:{}".format(used_device))
# создаём модель
# передаём вычисления на нужное устройство gpu/cpu
net = Net(used_device).to(used_device)

# sys.setrecursionlimit(10000)
# print(net)


# проверяем есть ли сохранённая модель
if is_model_exists:
    net.load_state_dict(torch.load(model_path))
    net.eval()

# скорость обучения
learning_rate = 0.001
# Создаём оптимизатор- метод градиентного спуска, параметры learning_rate,momentum
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)
# optimizer = optim.Adam(net.parameters(), lr=learning_rate,betas=(0.2,0.01))
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# функция потерь - логарифмическая функция потерь (log cross entropy loss)
criterion = nn.NLLLoss()

# задаём остальные параметры
batch_size = 5000
learning_rate = 0.001
epochs = 20

# (train_loader,test_loader)
train_data, test_data = load_traindata(batch_size)
# a=5
# train_loader
# test_loader
# train_net(net,train[0],optimizer,criterion,device)
# test_nn(net,train[1],device)
start_time = time.time()

# обучение сети
train_net(net, train_data, optimizer, criterion, used_device, epoch_losses, acc_list)
train_time = time.time() - start_time
train_time_str = str(timedelta(seconds=round(train_time)))
print("Train time: %s secs (Wall clock time)" % timedelta(seconds=round(train_time)))
# тест результатов обучения сети
start_time = time.time()
avg_test_acc = test_nn(net, test_data, used_device)
print(avg_test_acc)
test_time = time.time() - start_time
test_time_str = str(timedelta(seconds=round(test_time)))
print("Test time: %s secs (Wall clock time)" % timedelta(seconds=round(test_time)))
# время тестирования
time = (train_time_str, test_time_str)
result_file = "{}_{}(op={},ep={},acc={:.3f})".format(NETWORK_TYPE,
                                                     DATASET,
                                                     OPTIMIZER,
                                                     epochs,
                                                     avg_test_acc)
# строим график обучения
graphs_shower.graphics_show_loss_acc(epoch_losses, acc_list, result_file + ".png")
# Сохраняем результаты в файл
save_stats(acc_list, epoch_losses, time, result_file + ".txt")

if not is_model_exists:
    torch.save(net.state_dict(), model_path)
    print("Model Saved!")
