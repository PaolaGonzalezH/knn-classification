# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
from typing import List, Tuple, Any
import umap  # type: ignore
import numpy as np  # type: ignore
import albumentations as A  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from PIL import Image

from datasets import Dataset


def apply_transforms(pil_image: Any, transforms: A.Compose) -> Any:
    """Apply albumentations transforms to an image

    Args:
        pil_image (Image): PIL Image of instance
        transforms (A.Compose): List of transforms from albumentations to apply

    Returns:
        Image: PIL image transformed
    """
    image = transforms(image=pil_image)["image"]
    return image


def get_arrays(dataset: Dataset, transforms: A.Compose, flatten: bool) -> Tuple:
    """Convert images to arrays with transforms.

    Args:
        dataset (Dataset): Torchvision dataste
        transforms (A.Compose): List of transforms from albumentations to apply
        flatten (bool): shape of instance (nx1)

    Returns:
        Tuple: _description_
    """
    images = []
    labels = []

    for img, label in dataset:
        image = np.array(img)
        image = apply_transforms(image, transforms)

        if flatten is not True:
            images.append(image)
        else:
            images.append(image.ravel())
        labels.append(label)
    return images, labels


def get_representation(
    dataset: Dataset, representation: str, transforms: A.Compose, flatten: bool
) -> Tuple:
    """_summary_

    Args:
        dataset (Dataset):Torchvision Dataset with Tuple(image, label_idx)
        type (str): type of image representation
        transforms (A.Compose): Image transforms list from albumentations
        flatten (bool): Shape of each instance (nx1)

    Raises:
        NotImplementedError: Type of representation not supported.

    Returns:
        Tuple: Array of images, array of labels
    """
    if representation == "histogram":
        return hist_repr(dataset, transforms, flatten)
    if representation == "pca":
        return pca_repr(dataset, transforms, flatten)
    if representation == "umap":
        return umap_repr(dataset, 2, transforms, flatten)
    raise NotImplementedError(f"{representation} is not supported!")


def compare_arrays(
    rchannel: np.ndarray, gchannel: np.ndarray, bchannel: np.ndarray, threshold: int
) -> Tuple:
    """Setting to 0 only second channel with only green.

    Args:
        rchannel (np.ndarray): Red color channel of image
        gchannel (np.ndarray): Green color channel of image
        bchannel (np.ndarray): Blue color channel of image
        threshold (int): value from which we consider red and blue can mix with green to preserve green

    Returns:
        Tuple: Three channels with green modified.
    """
    green = []
    for rvec, bvec, gvec in zip(rchannel, bchannel, gchannel):
        mask = np.any([rvec > threshold, bvec > threshold], axis=0)
        gnew = gvec * mask
        green.append(np.array(gnew))
    return rchannel, np.array(green), gchannel


def green_change(dataset: Dataset, threshold: int) -> List:
    """Setting green channel to black at specific threshold

    Args:
        dataset (Dataset): Torchvision dataset
        threshold (int): float to account as green of black

    Returns:
        List: Images with green to black.
    """
    data = []
    for img, label in dataset:
        img = np.asarray(img)
        rchannel, gchannel, bchannel = img.T
        rchannel, gchannel, bchannel = compare_arrays(
            rchannel, gchannel, bchannel, threshold
        )
        data.append((img, label))
    return data


def hist_repr(dataset: Dataset, transforms: A.Compose, flatten: bool) -> Tuple:
    """Color histogram representation of images

    Args:
        dataset (Dataset): Torchvision dataset
        transforms (A.Compose): set of albumentations transforms to apply
        flatten (bool): shape of instance (nx1)

    Returns:
        Tuple: array of represented images, array of labels
    """
    images = []
    labels = []

    for img, label in dataset:
        img = np.array(img)
        img = apply_transforms(img, transforms)

        rchannel, gchannel, bchannel = img.T
        rchannel, _ = np.histogram(rchannel, bins=len(rchannel))
        gchannel, _ = np.histogram(gchannel, bins=len(gchannel))
        bchannel, _ = np.histogram(bchannel, bins=len(bchannel))
        img = np.array([rchannel, gchannel, bchannel])

        if flatten is not True:
            images.append(img)
        else:
            images.append(img.ravel())
        labels.append(label)
    return images, labels


def pca_repr(dataset: Dataset, transforms: A.Compose, flatten: bool) -> Tuple:
    """Tansfomation of vector space of images with PCA

    Args:
        data (np.ndarray): Images matrix representation.

    Returns:
        Tuple: Data transformed with components.
    """
    images, labels = get_arrays(dataset, transforms, flatten)

    pca = PCA(n_components=12)
    pca.fit(images)
    return pca.transform(images), labels


def stack(image_array: np.ndarray) -> Any:
    """Stack instances arrays

    Args:
        image_array (np.ndarray): Array with instances

    Returns:
        np.array: vector stack of images representation
    """
    return np.stack(image_array)


def umap_repr(
    dataset: Dataset, components: int, transforms: A.Compose, flatten: bool
) -> Tuple:
    """UMAP feature representation of images

    Args:
        dataset (Dataset): Torchvision Dataset
        components (int): Number of dimensions
        transforms (A.Compose): List of transforms form albumentations to apply
        flatten (bool): Shape of instance (nx1)

    Returns:
        Tuple: feature space, numerical UMAP representation of dataset, array of labels
    """
    data, labels = get_arrays(dataset, transforms, flatten)
    um_model = umap.UMAP(n_components=components)
    data_fit = um_model.fit(data)
    data_umap = um_model.transform(data)
    return data_fit, data_umap, labels
