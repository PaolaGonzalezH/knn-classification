import typing
import numpy as np
import albumentations as A
from sklearn.decomposition import PCA

from KNNClassifier import KNNClassifier
from datasets import KNNDataset, Dataset 


def extrac_features_pca(data: np.ndarray):
    pca = PCA(n_components=5)
    pca.fit(data)

    return pca.transform(data)



def main(): 

    augm_tr = A.Compose([
        A.Resize(256,256,3),
        A.Flip(p=0.5),
        A.Rotate(),
        #A.ToGray()
    ])
    augm_te = A.Compose([A.Resize(256,256,3)])


    train_data, train_labels = Dataset("~/Documents/block_one/17_flowers/train", augm_tr)()
    val_data, val_labels = Dataset("~/Documents/block_one/17_flowers/validation", augm_te)()

    # call pca here 
    train_component = extrac_features_pca(train_data)
    val_component = extrac_features_pca(val_data)
    
    # instantiate KNNDataset
    processed_dataset = KNNDataset(
        train_component,
        train_labels,
        val_component,
        val_labels
    )


    model = KNNClassifier(processed_dataset, 5)
    model.train()
    predictions = np.asanyarray(model.predict())
    metric = model.evaluate()
    print(predictions, metric)


if __name__ == "__main__":

    main()  