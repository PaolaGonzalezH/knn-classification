from typing import List
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    """K-Nearest Neighbors classifier model with train, predict and evaluation methods"""

    def __init__(self, data: object, k: int) -> None:
        self.classifier = KNeighborsClassifier(n_neighbors=k)
        self.xtrain = data.xtrain
        self.ytrain = data.ytrain

        self.xtest = data.xtest
        self.ytest = data.ytest

    def train(self) -> None:
        """Training KNN Classifier with matrices X and Y.
        Dimensions (N-images, M-features) and (N-images) respectively"""
        self.classifier.fit(self.xtrain, self.ytrain)

    def predict(self) -> List[int]:
        """Given test images represented in matrix (N-elements, M-features)
          and a fit model, predicts labels

        Returns:
            List[int]: List of label predictions of matrix (N-predictions)
        """
        return self.classifier.predict(self.xtest)

    def evaluate(self) -> float:
        """Given a fit model and labeled testing dataset, gets accuracy metric.

        Returns:
            float: accuracy of prediction
        """
        return self.classifier.score(self.xtest, self.ytest)
