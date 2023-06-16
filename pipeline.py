# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
import warnings
import logging
import yaml
from yaml.loader import SafeLoader
import albumentations as A  # type: ignore

from knnclassifier import KNNClassifier
from datasets import KNNDataset, Dataset
from processing import get_representation, stack, green_change

warnings.simplefilter("ignore")

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """
    Main loop for extracting data and training model
    """

    # Open the file and load the file
    with open("config.yaml", encoding="utf-8") as file:
        configargs = yaml.load(file, Loader=SafeLoader)

    augm_tr = A.Compose([A.Resize(100, 100, 3), A.CenterCrop(50, 50)])
    augm_te = A.Compose([A.Resize(100, 100, 3), A.CenterCrop(50, 50)])

    train_data = Dataset(configargs["path_tr"])()
    val_data = Dataset(configargs["path_te"])()

    train_data = green_change(train_data, configargs["green_threshold"])
    val_data = green_change(val_data, configargs["green_threshold"])

    train_data, train_labels = get_representation(
        train_data, configargs["repr"], augm_tr, True
    )
    val_data, val_labels = get_representation(
        val_data, configargs["repr"], augm_te, True
    )
    # instantiate KNNDataset
    train_data = stack(train_data)
    val_data = stack(val_data)
    train_labels = stack(train_labels)
    val_labels = stack(val_labels)

    processed_dataset = KNNDataset(train_data, train_labels, val_data, val_labels)

    model = KNNClassifier(processed_dataset, k=configargs["k_neighbors"])
    model.train()
    model.evaluate()


if __name__ == "__main__":
    main()
