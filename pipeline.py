import albumentations as A

from KNNClassifier import KNNClassifier
from datasets import Dataset

augm_tr = A.Compose([
    A.Resize(256,256,3),
    A.Flip(p=0.5),
    A.Rotate(),
    #A.ToGray()
])
augm_te = A.Compose([A.Resize(256,256,3)])


data = Dataset("17_flowers/validate", "17_flowers/train", augm_tr, augm_te)
model = KNNClassifier(data, 5)
print(model.xtrain.shape, model.ytrain.shape)
model.train()
# model.predict()
