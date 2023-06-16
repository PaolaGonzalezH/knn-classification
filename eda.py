# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
import warnings
from sklearn.cluster import KMeans  # type: ignore
import albumentations as A  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import torchvision  # type: ignore

from datasets import Dataset
from utils import mean_sample
from processing import extrac_features_umap

warnings.simplefilter("ignore")

augm_tr = A.Compose([A.Resize(256, 256, 3)])
augm_te = A.Compose([A.Resize(256, 256, 3)])

train_data, train_labels = Dataset("~/Documents/block_one/17_flowers/train")()
val_data, val_labels = Dataset("~/Documents/block_one/17_flowers/validation")()

# Kmeans
kmeans = KMeans(n_clusters=17, n_init=15, max_iter=500, random_state=0)
clusters = kmeans.fit_predict(train_data)
# Cluster centers
centroids = kmeans.cluster_centers_


_, train_umap, labels = extrac_features_umap(train_data, 2, augm_tr, True)
umap_df = pd.DataFrame(data=train_umap, columns=["umap comp. 1", "umap comp. 2"])


color = sns.color_palette("magma", as_cmap=True)

plt.figure(figsize=(8, 6))
plt.scatter(umap_df.iloc[:, 0], umap_df.iloc[:, 1], c=clusters, cmap=color, s=40)
plt.title("UMAP plot in 2D")
plt.xlabel("umap component 1")
plt.ylabel("umap component 2")
plt.savefig("umap_flowers.png")

print(len(train_data), len(val_data))

dataset = torchvision.datasets.ImageFolder("~/Documents/block_one/17_flowers/train")
print(mean_sample(dataset, 1))
