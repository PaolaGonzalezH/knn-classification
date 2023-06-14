# -*- coding: utf-8 -*-
import warnings
import logging
import yaml
from yaml.loader import SafeLoader
import albumentations as A

from KNNClassifier import KNNClassifier
from datasets import KNNDataset, Dataset

warnings.simplefilter("ignore")

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """
    Main loop for extracting data and training model
    """

    # Open the file and load the file
    with open("config.yaml", encoding="utf-8") as file:
        configargs = yaml.load(file, Loader=SafeLoader)

    augm_tr = A.Compose([A.Resize(100, 100, 3)])
    augm_te = A.Compose([A.Resize(100, 100, 3)])

    train_data, train_labels = Dataset(
        configargs["path_tr"], augm_tr, configargs["histogram"], configargs["flatten"]
    )()
    val_data, val_labels = Dataset(
        configargs["path_tr"],
        augm_te,
        configargs["histogram"],
        configargs["flatten"],
    )()

    # instantiate KNNDataset
    processed_dataset = KNNDataset(train_data, train_labels, val_data, val_labels)

    model = KNNClassifier(processed_dataset, k=configargs["k"])
    model.train()
    # predictions = np.asanyarray(model.predict())
    model.evaluate()


if __name__ == "__main__":
    main()
