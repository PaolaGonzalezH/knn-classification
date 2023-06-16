# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
import random
from typing import List, Any
from math import ceil
from statistics import mean
import numpy as np  # type: ignore
from scipy.spatial import distance  # type: ignore
import torch  # type: ignore


def cosine_distance(xone: np.ndarray, xtwo: np.ndarray) -> float:
    """Cosine distance between two vectors u and v

    Args:
        xone (np.array): vector u
        xtwo (np.array): vector v

    Returns:
        (float): cosine distance
    """
    return distance.cosine(xone, xtwo)


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
