from dataclasses import dataclass

from typing import List, Any, Tuple
import torch
from sklearn.decomposition import PCA
import albumentations as A
import numpy as np
import torchvision
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class KNNDataset:
    xtrain: np.ndarray
    ytrain: np.array
    xtest: np.ndarray
    ytest: np.array
    

class Dataset:
    """
    Dataset class given a folder split into train and validation.
    Loads images and transforms into matrix.
    """

    def __init__(
        self,
        data_path: str,
        transforms: A.Compose,

    ) -> None:
        self.data_path = data_path
        self.transforms = transforms
        
    def __getitem__(self, idx): 
        pass 

    def __call__(self):
        
        # get all the dataset 
        image_list, label_list = self.load_data()
        num_instances = len(image_list)

        # return as numpy array
        image_list = np.stack(image_list)
        
        return image_list.reshape(num_instances,-1), np.stack(label_list)
    
    def apply_augmentations(self, pil_image):
        image = self.transforms(image=pil_image)["image"]
        return image 


    def load_data(self): 
        dataset = torchvision.datasets.ImageFolder(self.data_path)
        images = []
        labels = []
        for img, label in dataset:
            img = np.array(img)
            img = self.apply_augmentations(img)
            images.append(img.ravel())
            labels.append(label)
        return images, labels 