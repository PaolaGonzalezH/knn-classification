from typing import List, Any
from PIL import Image
from numpy.linalg import norm
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, data: object, k :int) -> None:
        self.classifier = KNeighborsClassifier(n_neighbors=k)
        self.xtrain = data.xtrain
        self.ytrain = data.ytrain

        self.xtest = data.xtest
        self.ytest = data.ytest

    def train(self) -> object:
        return self.classifier(self.classifier.fit(self.xtrain, self.ytrain))

    def predict(self) -> object:
        return self.classifier.predict(self.xtest)

    def evaluate(self) -> float:
        return self.classifier.score(self.xtest, self.ytest)
