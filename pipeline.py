import numpy as np
import albumentations as A
from sklearn.decomposition import PCA

from KNNClassifier import KNNClassifier
from datasets import KNNDataset, Dataset


def extrac_features_pca(data: np.ndarray) -> np.ndarray(float):
    """Tansfomation of vector space of images with PCA

    Args:
        data (np.ndarray): Images matrix representation.

    Returns:
        (np.ndarray): Data transformed with components.
    """
    pca = PCA(n_components=32)
    pca.fit(data)
    return pca.transform(data)


def main() -> None:
    """Main loop for extracting data and training model"""
    augm_tr = A.Compose(
        [
            A.Resize(256, 256, 3),
            A.Flip(p=0.5),
            A.Rotate(),
            # A.ToGray()
        ]
    )
    augm_te = A.Compose([A.Resize(256, 256, 3)])

    train_data, train_labels = Dataset(
        "~/Documents/block_one/17_flowers/train", augm_tr
    )()
    val_data, val_labels = Dataset(
        "~/Documents/block_one/17_flowers/validation", augm_te
    )()

    # call pca here
    train_component = extrac_features_pca(train_data)
    val_component = extrac_features_pca(val_data)

    # instantiate KNNDataset
    processed_dataset = KNNDataset(
        train_component, train_labels, val_component, val_labels
    )

    model = KNNClassifier(processed_dataset, k=17)
    model.train()
    predictions = np.asanyarray(model.predict())
    metric = model.evaluate()
    print(metric)


if __name__ == "__main__":
    main()
