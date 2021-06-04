from torch.utils.data import DataLoader

from datasets.base_loader import BaseLoader
from datasets.srcnn_emnist import SrcnnEmnist


class SrcnnLoader(BaseLoader):
    """
    Loader for SRCNN network with emnist.
    """
    def __init__(self, batch_size, train_path, test_path):
        """
        :param batch_size: batch size used in loader




        """
        super().__init__(name="SRCNN",
                         batch_size=batch_size)
        self.train_path = train_path
        self.test_path = test_path

    def _load(self):
        """
        Load dataset
        :rtype: Tuple[,]
        :return: train and test dataset
        """

        train_loader = DataLoader(
            SrcnnEmnist(self.train_path, train=True),
            batch_size=self.batch_size, shuffle=True, pin_memory=True)

        test_loader = DataLoader(
            SrcnnEmnist(self.test_path, train=False),
            batch_size=self.batch_size, shuffle=True, pin_memory=True)

        return train_loader, test_loader
