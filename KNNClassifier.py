import dataclasses
import functools
import pathlib
from typing import List, Any
from PIL import Image
from np.linalg import norm
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, data, k):
        self.classifier = KNeighborsClassifier(n_neighbors=k)
        self.__xtrain = [x.vector for x in data.xtrain]
        self.__ytrain = [y.label_index for y in data.xtrain]

        self.__xtest = [x.vector for x in data.xtest]
        self.__ytest = [y.label_index for y in data.xtest]

    def train(self) -> object:
        return self.classifier(self.neigh.fit(self.__xtrain, self.__ytrain))

    def predict(self) -> object:
        return self.classifier.predict(self.__xtest)

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
