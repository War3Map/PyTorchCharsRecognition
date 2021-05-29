class BaseLoader:
    """
    Loader for all datasets.
    """

    def __init__(self, name: str, batch_size, data_path='../data',
                 need_resize: bool = False):
        """

        :param name: Dataset name

        :param batch_size: batch size used in loader

        :param data_path: path to load dataset

        :param need_resize: is necessary to resize dataset


        """

        self.dataset_name = name
        self.data_path = data_path
        self.batch_size = batch_size
        self.need_resize = need_resize

    @property
    def dataset(self):
        """
        Returns dataset loaders
        :return: (train_loader, test_loader)
        """
        return self._load()

    def _load(self):
        """
        Load dataset
        :rtype: Tuple[,]
        :return: train and test dataset
        """

        return NotImplementedError
