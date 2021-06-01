# стандартные модули
import os.path
import time
from datetime import timedelta
from pathlib import Path

# импорт модулей pytorch
import torch
import torch.nn as nn

import torch.optim as optim
# from torch.autograd import Variable

import graphs_shower

from networks.simple_emnist_cnn import SimpleEmnistConvNet
from networks.simple_mnist_cnn import SimpleMnistConvNet
from networks.simple_emnist_ffn import SimpleEmnistFeedForward
from networks.simple_mnist_ffn import SimpleMnistFeedForward

from network_testing import test_net
from network_training import train_net
# from data_loaders import DataLoader
from datasets.emnist_loader import EmnistLoader
from datasets.mnist_loader import MnistLoader


# Сохраняет статистику в файл
def save_stats(accuracies, losses, common_time, save_file):
    """
    Save statistic to file

    :param accuracies:

    :param losses:

    :param common_time: time in format (train,test)

    :param save_file: filename for save file

    :return:
    """
    with open(save_file, "w+") as file:
        file.write("Train:{}\nTest:{}\n".format(common_time[0], common_time[1]))
        epochs_count = len(accuracies)
        for i in range(0, epochs_count):
            file.write("{}:{}\n".format(accuracies[i], losses[i]))


def execute_with_time_measure(func, *parameters):
    """

    :param func: function to measure

    :type func: function

    :param parameters: function parameters

    """
    start_time = time.time()
    func_result = func(*parameters)
    test_time = time.time() - start_time
    delta = timedelta(seconds=round(test_time))
    time_str = str(delta)
    print(f"{func.__name__}: {delta} secs (Wall clock time)")

    return func_result, time_str


# функции потерь на каждой эпохе
epoch_losses = list()
# список значений точности на каждой эпохе
acc_list = list()

NETWORK_TYPE = "CNN"

# Назначаем устройство на котором будет работать нейросеть, по возможности CUDA
dev = "cuda" if torch.cuda.is_available() else "cpu"
used_device = torch.device(dev)
print("Running on Device:{}".format(used_device))

MODELS = {
    "simple_ffn_mnist": SimpleMnistFeedForward,
    "simple_ffn_emnist": SimpleEmnistFeedForward,
    "simple_cnn_emnist": SimpleEmnistConvNet,
    "simple_cnn_mnist:": SimpleMnistConvNet,

}

# создаём модель
# передаём вычисления на нужное устройство gpu/cpu

print("Модели")
for index, model in enumerate(MODELS.keys()):
    print(f"{index}. {model}")
input_model = input("Выберите модель:")
chosen_model = (MODELS[input_model] if input_model in MODELS
                else MODELS["simple_ffn_mnist"])

print("Загружаем модель .....")
net_model = chosen_model(used_device).to(used_device)
# print(net_model)

# проверяем есть ли сохранённая модель
# model_path = r".\models\CNN_EMNIST_model"
model_path = r".\models\test"
is_model_exists = os.path.isfile(model_path)

if is_model_exists:
    net_model.load_state_dict(torch.load(model_path))
    net_model.eval()

# скорость обучения
learning_rate = 0.001
OPTIMIZERS = {
    "SGD": optim.SGD(net_model.parameters(), lr=learning_rate, momentum=0.9),
    "Adagrad": optim.Adagrad(net_model.parameters(), lr=learning_rate),
    "Adam": optim.Adam(net_model.parameters(), lr=learning_rate),
    "AdamBetas": optim.Adam(net_model.parameters(), lr=learning_rate, betas=(0.2, 0.01)),

}

print("Оптимизаторы")
for index, optimizer in enumerate(OPTIMIZERS.keys()):
    print(f"{index}. {optimizer}")
input_optimizer = input("Выберите оптимизатор:")
optimizer_name = input_optimizer if input_optimizer in OPTIMIZERS else "SGD"
optimizer = OPTIMIZERS[optimizer_name]


DATASETS = {
    "MNIST": MnistLoader,
    "EMNIST": EmnistLoader,
}

print("Датасеты")
for index, dataset in enumerate(DATASETS):
    print(f"{index}. {dataset}")
input_dataset = input("Выберите датасет:")

dataset_loader = (DATASETS[input_dataset] if input_dataset in DATASETS
                  else DATASETS["MNIST"])


str_path = input("Путь до датасета(если путь некоректный выбирается по умолчанию):")

dataset_path = (str_path if Path(str_path).exists()
                else None)

# For Future
CRITERIONS = {}

# функция потерь - логарифмическая функция потерь (negative log cross entropy loss)
criterion = nn.NLLLoss()

# задаём остальные параметры
# batch_size = 1000
batch_size_str = input("Введите размер батча(по умолчанию - 1000):")
batch_size = int(batch_size_str) if batch_size_str.isdigit() else 1000

# learning_rate = 0.001
learning_rate_str = input("Скорость обучения(по умолчанию - 10^-3):")
learning_rate = float(batch_size) if learning_rate_str.isdigit() else 0.001

epochs_str = input("Введите число эпох(по умолчанию - 20):")
epochs = int(epochs_str) if epochs_str.isdigit() else 20

# (train_loader,test_loader)
data_loader = dataset_loader(batch_size, dataset_path)
train_data, test_data = data_loader.dataset
# a=5
# train_loader
# test_loader
# train_net(net,train[0],optimizer,criterion,device)
# test_nn(net,train[1],device)

# обучение сети
avg_train_acc, train_time_str = execute_with_time_measure(
    train_net,
    net_model, train_data, optimizer,
    criterion, epochs, used_device, epoch_losses,
    acc_list, batch_size)
print(f"Точность при тренировке: {avg_train_acc}")

# тест сети
avg_test_acc, test_time_str = execute_with_time_measure(
    test_net,
    net_model,
    criterion,
    test_data,
    used_device
)
print(f"Точность при тестировании: {avg_test_acc}")

prepared_name = (f"{NETWORK_TYPE}_{dataset}"
                 f"(op={optimizer_name},ep={epochs},"
                 f"acc={avg_test_acc:.3f})")
save_filepath = "./results" / Path(prepared_name)

# строим график обучения
graphs_shower.graphics_show_loss_acc(epoch_losses, acc_list,
                                     str(save_filepath) + ".png")

time_info = (train_time_str, test_time_str)
# Сохраняем результаты в файл
save_stats(acc_list, epoch_losses, time_info,
           str(save_filepath) + ".txt")

if not is_model_exists:
    torch.save(net_model.state_dict(), model_path)
    print("Model Saved!")
