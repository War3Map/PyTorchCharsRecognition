from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from base_loader import BaseLoader


class MnistLoader(BaseLoader):
    """
    Loader for MNIST dataset.
    """
    def __init__(self, batch_size, data_path='../dataMNIST'):
        """
        :param batch_size: batch size used in loader

        :param data_path: path to load dataset


        """
        super().__init__(name="MNIST",
                         data_path=data_path,
                         batch_size=batch_size,
                         need_resize=True)

    def load(self):
        """
        Load dataset
        :rtype: Tuple[,]
        :return: train and test dataset
        """
        transformations = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])

        train_loader = DataLoader(
            datasets.MNIST(self.data_path, train=True, download=True, transform=transformations),
            batch_size=self.batch_size, shuffle=True, pin_memory=True)

        labels_loader = DataLoader(
            datasets.MNIST(self.data_path, train=False, download=False, transform=transformations),
            batch_size=self.batch_size, shuffle=True, pin_memory=True)

        return train_loader, labels_loader





