# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
from dataclasses import dataclass
from typing import Any
import logging
import numpy as np
import torchvision  # type: ignore
import albumentations as A  # type: ignore

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

    def __init__(
        self, data_path: str, transforms: A.Compose, flatten: bool, histogram: bool
    ) -> None:
        self.data_path = data_path
        self.transforms = transforms
        self.flatten = flatten
        self.histogram = histogram

    def __call__(self):
        logging.info("Creating dataset object.")
        # get all the dataset
        image_list, label_list = self.load_data()

        # return as numpy array
        return np.stack(image_list), np.stack(label_list)

    def apply_augmentations(self, pil_image) -> Any:
        """Apply A.Compose to an PIL image.

        Args:
            pil_image (PIL): PIL image format

        Returns:
            image: PIL image with transformation
        """
        image = self.transforms(image=pil_image)["image"]
        return image

    def load_data(self):
        """Given a path: directory/label/item, returns list of items and labels

        Returns:
            Tuple (List[PIL image], List[int])
        """
        dataset = torchvision.datasets.ImageFolder(self.data_path)
        images = []
        labels = []
        if self.histogram is False:
            logging.warning(
                "Histogram representation not in use, need a different representation."
            )

        for img, label in dataset:
            img = np.array(img)
            img = self.apply_augmentations(img)

            if self.histogram:
                rchannel, gchannel, bchannel = img.T
                rchannel, _ = np.histogram(rchannel, bins=len(rchannel))
                gchannel, _ = np.histogram(gchannel, bins=len(gchannel))
                bchannel, _ = np.histogram(bchannel, bins=len(bchannel))
                img = np.array([rchannel, gchannel, bchannel])

            if self.flatten:
                images.append(img.ravel())
            labels.append(label)
        return images, labels

# processing.py
# hist(data: np.ndarray) -> np.ndarray:
#     if data.shape != (H, W, 3):
#           raise ShapeError(f'input should have shape () but it has {data.shape}')
