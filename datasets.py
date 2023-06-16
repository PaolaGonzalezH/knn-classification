# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
from typing import List, Tuple
from dataclasses import dataclass
import logging
import numpy as np  # type: ignore
import torchvision  # type: ignore

logging.basicConfig(level=logging.INFO)


@dataclass
class KNNDataset:
    """Dataset of KNN algorithm"""

    xtrain: np.ndarray
    ytrain: np.ndarray
    xtest: np.ndarray
    ytest: np.ndarray


class Dataset:
    """
    Dataset class given a folder split into train and validation.
    Loads images and transforms into matrix.
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def __call__(self):
        logging.info("Creating dataset object.")
        # get all the dataset
        image_dataset = self.load_data()

        # return as numpy array
        return image_dataset

    def load_data(self) -> List[Tuple]:
        """Given a path: directory/label/item, returns list of items and labels

        Returns:
            List (List[PIL image], List[int])
        """
        images = torchvision.datasets.ImageFolder(self.data_path)
        return images
