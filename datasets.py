from statistics import mean
from math import ceil
import random

from typing import List, Any, Tuple
import torch
from sklearn.decomposition import PCA
import albumentations as A
import numpy as np
import torchvision


def get_samples(amount: int, dataset: Any) -> Any:
    """Get sample with normal distribution of a Dataset object

    Args:
        amount (int): how many elements from dataset to take
        dataset (Dataset): Dataset object

    Returns:
        Dataset: sample with normal distribution
    """
    dataset_idx = random.sample(range(0, len(dataset)), amount)
    return torch.utils.data.Subset(dataset, dataset_idx)


def get_mean(dataset: List[tuple]) -> tuple:
    """Mean of dataset image size

    Args:
        dataset (List[tuple]): Torch dataset with PIL image and label

    Returns:
        tuple: ceil of height and width
    """
    sizes = [elem[0].size for elem in dataset]
    height = mean([h[0] for h in sizes])
    width = mean([w[1] for w in sizes])
    return ceil(height), ceil(width)


def mean_sample(dataset: List[tuple], ratio: float = 0.75) -> tuple:
    """Apply mean of sample

    Args:
        dataset (List[tuple]): Torch sample mean

    Returns:
        tuple: ceil of height and width
    """
    samples = get_samples(ceil(len(dataset) * ratio), dataset)
    return get_mean(samples)


class Dataset:
    """
    Dataset class given a folder split into train and validation.
    Loads images and transforms into matrix.
    """

    def __init__(
        self,
        path_te: str,
        path_tr: str,
        transform_tr: A.Compose,
        transform_te: A.Compose,
        representation: str = "pca",
    ) -> None:
        self.path_tr = path_tr
        self.path_te = path_te
        self.transform_tr = transform_tr
        self.transform_te = transform_te
        self.__xtrain, self.__ytrain, self.__xtest, self.__ytest = Dataset.data_load(
            self
        )
        self.featurize(representation)

    def data_load(self) -> Tuple[np.ndarray, ...]:
        dataset_train = torchvision.datasets.ImageFolder(self.path_tr)
        dataset_test = torchvision.datasets.ImageFolder(self.path_tr)

        xtrain, ytrain = self.split(dataset_train)
        xtest, ytest = self.split(dataset_test, "test")
        return xtrain, ytrain, xtest, ytest

    def split(
        self, data: torchvision.datasets.VisionDataset, dset: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        xtrain = []
        ytrain = []

        for image, label in data:
            image = np.asarray(image)
            if dset == "train":
                image = self.transform_tr(image=image)["image"]
            else:
                image = self.transform_te(image=image)["image"]

            xtrain.append(image)
            ytrain.append(np.array(label))

        return np.dstack(xtrain).T, np.dstack(ytrain)

    def _pca_representation(self):
        print(self.__xtrain.squeeze())
        pca_tr = PCA(n_components=1)
        pca_te = PCA(n_components=1)

        pca_tr.fit(self.__xtrain)
        pca_te.fit(self.__xtest)

        self.__xtrain = pca_tr.components_
        self.__xtest = pca_te.components_

    def featurize(self, rep: str) -> None:
        if rep == "pca":
            self._pca_representation()

    @property
    def xtrain(self):
        return self.__xtrain

    @property
    def xtest(self):
        return self.__xtest

    @property
    def ytrain(self):
        return self.__ytrain

    @property
    def ytest(self):
        return self.__ytest
