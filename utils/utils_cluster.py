# general imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from sklearn.metrics import silhouette_samples, silhouette_score

from fcmeans import FCM

import umap

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


def visualize_dimensionality_reduction(transformation, targets):
  # create a scatter plot of the t-SNE output
  plt.scatter(transformation[:, 0], transformation[:, 1], 
              c=plt.cm.tab10(np.array(targets).astype(int)))

  labels = np.unique(targets)

  # create a legend with the class labels and colors
  handles = [plt.scatter([],[], c=plt.cm.tab10(i), label=label) for i, label in enumerate(labels)]
  plt.legend(handles=handles, title='Classes')

  plt.show()


def kmeans_elbow(scaled_data, max_cluster:int = 15):
    _, axes = plt.subplots(figsize=(20,8))

    elbow = KElbowVisualizer(KMeans(), k= max_cluster, timings=False, locate_elbow=True, size=(1260,450))
    elbow.fit(scaled_data)

    axes.set_title("\nDistortion Score Elbow For KMeans Clustering\n",fontsize=25)
    axes.set_xlabel("\nK",fontsize=20)
    axes.set_ylabel("\nDistortion Score",fontsize=20)

    sns.despine(left=True, bottom=True)
    plt.show()


def cmeans_fpc(scaled_data, max_cluster:int = 10):
    fpcs = []
    for k in range(2, max_cluster+1):
        fcm = FCM(n_clusters=k)
        fcm.fit(scaled_data)
        membership_matrix = fcm.u
        max_membership = np.max(membership_matrix, axis=1)
        fpc = np.sum(max_membership**2) / len(scaled_data)
        fpcs.append(fpc)

    # Plot the elbow curve
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(2, max_cluster + 1), fpcs, marker='o')
    ax.set_title("\nFuzzy partition coefficient For Fuzzy C-means Clustering\n", fontsize=15)
    ax.set_xlabel("\nK", fontsize=12)
    ax.set_ylabel("\nFPC", fontsize=12)
    plt.show()

def plot_clusters(data):
    plt.subplots(figsize=(20, 8))
    p = sns.countplot(x=data, palette=["#F3AB60"], saturation=1, edgecolor="#1c1c1c", linewidth=3)
    p.axes.set_yscale("linear")
    p.axes.set_title("\nCustomer's Clusters\n", fontsize=30)
    p.axes.set_ylabel("Frequency", fontsize=20)
    p.axes.set_xlabel("\nCluster", fontsize=20)
    p.axes.set_xticklabels(p.get_xticklabels(), rotation=0, fontsize=15)  
    p.axes.set_yticklabels(p.get_yticks(), fontsize=15)  
    for container in p.containers:
        p.bar_label(container, label_type="center", padding=5, size=17, color="black", rotation=0,
                    bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "white",
                          "linewidth": 4, "alpha": 1})

    sns.despine(left=True, bottom=True)
    plt.show()


def plot_silhouette(data, cluster_solution, n_clusters:int = 7):
    # Get the cluster labels for each data point
    labels = cluster_solution.labels_

    # Calculate the silhouette coefficient for each data point
    silhouette_values = silhouette_samples(data, labels)

    # Calculate the overall silhouette score
    silhouette_avg = silhouette_score(data, labels)

    # Plot the silhouette plot
    fig, ax = plt.subplots()
    y_lower = 10
    for i in range(n_clusters):  # Iterate over each cluster
        cluster_silhouette_values = silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        cluster_size = cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            alpha=0.7
        )
        ax.text(
            -0.05,
            y_lower + 0.5 * cluster_size,
            str(i)
        )
        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])

    plt.show()


def umap_visual(data, cluster_data, n_neighbors:int = 15, min_dist:float = 0.3, seed:int =42):
    umap_object = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    umap_embedding = umap_object.fit_transform(data)
    visualize_dimensionality_reduction(umap_embedding, cluster_data)
    return umap_embedding


def hdbscan_ordered(data, cluster_name): 
    data = data.copy()
    # create a list of the cluster sizes, which will be in descending order
    cluster_size_list = [value for value in data[cluster_name].value_counts()]
    # reverse the list to have it on ascending order
    reversed_list = cluster_size_list[::-1]
    # store the column values for the hbscan cluster in 'cluster_values'
    cluster_values = data[cluster_name].copy()
    # iterate through the cluster size and the number of the cluster they should be in 
    for ordered_index, cluster_size in enumerate(reversed_list):
        # get the index of the actual cluster that has the cluster size in 'cluster_size'
        cluster_number = cluster_values.value_counts().loc[cluster_values.value_counts() == cluster_size].index[0]
        # for each element initially in cluster number 'cluster_number', replace it by the cluster with the corresponding ordered number
        for element in cluster_values:
            if element == cluster_number:
                data.loc[cluster_values == cluster_number, cluster_name] = ordered_index
    return data 