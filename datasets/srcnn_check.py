"""
Test file for checking dataset content and properties
"""

import torch

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from datasets.srcnn_loader import SrcnnLoader
from datasets.srcnn_emnist import SrcnnEmnist
# import os.path
#
# import cv2


BATCH_SIZE = 200
TEST_PATH = r"D:\EMNIST-CSV\emnist-balanced-test.csv"
TRAIN_PATH = r"D:\EMNIST-CSV\emnist-balanced-train.csv"

srcnn_loader = SrcnnLoader(BATCH_SIZE, TRAIN_PATH, TEST_PATH)
train_loader, _ = srcnn_loader.dataset


# Mean=44.6516, STD=84.9755
def show_statictics(train_loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, label in train_loader:
        # data = info
        # print(data)
        channels_sum += torch.mean(data)
        channels_squared_sum += torch.mean(data ** 2)
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print(f"Mean={mean:.4f}, STD={std:.4f}")


def iterate_dataset():
    for data, _ in SrcnnEmnist(TEST_PATH):
        print(data)


# iterate_dataset()
# show_statictics(train_loader)
test_loader = 0
# torch.utils.data.DataLoader(
# datasets.EMNIST('../data2', split="letters", train=False,
#     transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
#     ])),
#     batch_size=batch_size, shuffle=True,pin_memory=True)

print(f"Длина датасета: {len(train_loader.dataset)}")


train_loader, _ = srcnn_loader.dataset
dataiter = iter(train_loader)
images, labels = dataiter.next()

figure = plt.figure()
num_of_images = 100

image_numpy = images[0].numpy()

a = 5


def plot_show():
    for index in range(1, num_of_images + 1):
        plt.subplot(10, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

    plt.show()

plot_show()

# image = cv2.imread("./test_data/022.jpg")
# # cv2.imshow("Image", image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# image2 = image.copy()
#
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, threshold_image = cv2.threshold(image2, 150, 255, 0)
#
#
# def show_image(image, name_of_window):
#     cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
#     cv2.imshow(name_of_window, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
