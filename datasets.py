import numpy as np
from statistics import mean
import dataclasses
import functools
import pathlib
import torchvision
from typing import List, Any 
from skimage.transform import resize
from PIL import Image

@dataclasses.dataclass
class VectorImage:
    vector: list
    label_index : int

#Missing: binarize image with threshold, maybe gray filter

pip install albumentations

import albumentations as A

img = ...

transformed_img = transforms('image'=image)

def apply_tranform_tr(img: np.ndarray) -> np.ndarray:
    transforms = A.Compose([A.Resize(), A.Normalize(...)])
    return transforms('image'=img)

#Data/train/...
dataset_train = torchvision.datasets.ImageFolder(tr_path)
dataset_test = torchvision.datasets.ImageFolder(te_path)

dataset[0]
PIL.Image..., int


class Dataset:
    def __init__(self, path, transformed = False):
        self.path = path
        self.transformed = transformed
        self.__xtrain, self.__xtest  = Dataset.data_load()

    def __len__(self):
        return len(list(pathlib.Path(self.path).rglob("*.jpg")))

    def data_load(self):
        xtrain = []
        xtest = []

        dataset = list(pathlib.Path(self.path).rglob("*.jpg"))
        _, class_to_idx = torchvision.datasets.folder.find_classes(dir=self.path)

        for inst in dataset:
            instancia = inst.parent.split('/')
            label = instancia[-1]

            instance = Dataset.img_to_matrix(str(inst),class_to_idx[label])

            if instancia[-2] == "train":
                xtrain.append(instance)
            else:
                xtest.append(instance)

        if self.transformed:
            xtrain,xtest = self.resize_datasets(xtrain,xtest)
        return xtrain, xtest

    def resize_dataset(self,xtrain,xtest):
        width, height = self.get_size(xtrain,xtest)
        xtrain_resized = []
        xtest_resized = []

        for data in xtrain:
            new_vector = resize(data.vector,(height,width))
            xtrain_resized.append(VectorImage( new_vector,data.label_index )) 

        for data in xtest:
            new_vector = resize(data.vector,(height,width))
            xtest_resized.append(VectorImage(data.idx, new_vector,data.label_index ))
        return xtrain_resized, xtest_resized 

    def img_to_matrix(self,filepath,index):
        vector = np.asarray(Image.open(filepath))
        return VectorImage(vector,index)


    def get_size(self,train, test):
        width = [w.size[0] for w in train+test]
        height = [h.size[1] for h in train+test]
        return mean(width), mean(height)

    def augment_data(self, ...):
        ...

    @property
    def xtrain(self):
        return self.__xtrain
    
    @property 
    def xtest(self):
        return self.__xtest
    