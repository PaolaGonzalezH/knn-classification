import random
import numpy as np
from typing import List, Any
from sklearn.decomposition import PCA
from math import ceil
from statistics import mean
from sklearn.manifold import TSNE
from scipy.spatial import distance
import torch
import umap


def extrac_features_pca(data: List[float]) -> Any:
    """Tansfomation of vector space of images with PCA

    Args:
        data (np.ndarray): Images matrix representation.

    Returns:
        (np.ndarray): Data transformed with components.
    """
    pca = PCA(n_components=32)
    pca.fit(data)
    return pca.transform(data)


def extrac_features_umap(data, components=2):
    um = umap.UMAP(n_components=components)
    data_fit = um.fit(data)
    data_umap = um.transform(data)
    return data_fit, data_umap


def cosine_distance(x1, x2):
    return distance(x1, x2)


def get_samples(amount: int, dataset: Any) -> Any:
    """Get sample with normal distribution of a Dataset object

    Args:
        amount (int): how many elements from dataset to take
        dataset (Dataset): Dataset object

    Returns:
        Dataset: sample with normal distribution
    """
    dataset_idx = random.sample(range(0, len(dataset)), amount)
    return torch.utils.data.Subset(dataset, dataset_idx)


def get_mean(dataset: List[tuple]) -> tuple:
    """Mean of dataset image size

    Args:
        dataset (List[tuple]): Torch dataset with PIL image and label

    Returns:
        tuple: ceil of height and width
    """
    sizes = [elem[0].size for elem in dataset]
    height = mean([h[0] for h in sizes])
    width = mean([w[1] for w in sizes])
    return ceil(height), ceil(width)


def mean_sample(dataset: List[tuple], ratio: float = 0.75) -> tuple:
    """Apply mean of sample

    Args:
        dataset (List[tuple]): Torch sample mean

    Returns:
        tuple: ceil of height and width
    """
    samples = get_samples(ceil(len(dataset) * ratio), dataset)
    return get_mean(samples)
