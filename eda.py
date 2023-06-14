from utils import mean_sample
from utils import extrac_features_umap
from datasets import KNNDataset, Dataset
from sklearn.cluster import KMeans
import albumentations as A
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import umap
import warnings

warnings.simplefilter("ignore")

augm_tr = A.Compose([A.Resize(256, 256, 3)])
augm_te = A.Compose([A.Resize(256, 256, 3)])

train_data, train_labels = Dataset(
    "~/Documents/block_one/17_flowers/train", augm_tr, flatten=True, histogram=True
)()
val_data, val_labels = Dataset(
    "~/Documents/block_one/17_flowers/validation", augm_te, flatten=True, histogram=True
)()

# Kmeans
kmeans = KMeans(n_clusters=17, n_init=15, max_iter=500, random_state=0)
clusters = kmeans.fit_predict(train_data)
# Cluster centers
centroids = kmeans.cluster_centers_


_, train_umap = extrac_features_umap(train_data)
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
