from KNNClassifier import KNNClassifier
from datasets import *

def __init__(*args):
    path, transformed = args

    data = Dataset(path, transformed=True)
    model = KNNClassifier(data, 5)

    model.train()

    
    model.predict()