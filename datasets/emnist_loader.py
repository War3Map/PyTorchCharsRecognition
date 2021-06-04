from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets.base_loader import BaseLoader


class EmnistLoader(BaseLoader):
    """
    Loader for EMNIST dataset.
    """
    def __init__(self, batch_size, data_path='../dataEMNIST'):
        """
        :param batch_size: batch size used in loader

        :param data_path: path to load dataset


        """
        super().__init__(name="EMNIST",
                         data_path=data_path,
                         batch_size=batch_size)

    def _load(self):
        """
        Load dataset
        :rtype: Tuple[,]
        :return: train and test dataset
        """
        transformations = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1722,), (0.3310,))])

        train_loader = DataLoader(
            datasets.EMNIST(self.data_path, split="byclass", train=True, download=True, transform=transformations),
            batch_size=self.batch_size, shuffle=True, pin_memory=True)

        test_loader = DataLoader(
            datasets.EMNIST(self.data_path, split="byclass", train=False, download=False,
                            transform=transformations),
            batch_size=self.batch_size, shuffle=True, pin_memory=True)

        dataset_test_len = len(test_loader.dataset)
        dataset_train_len = len(train_loader.dataset)
        print("Длина обучающего датасета {}\n Длина трениро"
              "вочного датасета\n".format(dataset_train_len, dataset_test_len))
        return train_loader, test_loader





