"""Source code to generate the DBSCAN scatter plots."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, preprocessing

eps = 0.29
seed = 0
lims = (-2.6, 2.6)
cmap = plt.cm.viridis
markers = ("o", "H", "D", "p")


def _get_cluster_colours(n_clusters):
    """Retrieve the set of colours used to create `n_clusters` scatters."""

    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=n_clusters - 1)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    return tuple(
        plt.matplotlib.colors.rgb2hex(mappable.to_rgba(i)[:-1])
        for i in range(n_clusters)
    )


def make_plot(dataset, true_labels, name):
    """Make the scatter plots for the inliers and outliers for a particular
    dataset."""

    _, ax = plt.subplots(dpi=300)

    n_clusters = len(set(true_labels))
    scaled = preprocessing.StandardScaler().fit_transform(dataset.copy())
    dbscan = cluster.DBSCAN(eps=eps).fit(scaled)

    labels = dbscan.labels_
    outlier_mask = labels == -1
    inliers = scaled[~outlier_mask, :]
    outliers = scaled[outlier_mask, :]

    inlier_true_labels = true_labels[~outlier_mask]
    inlier_labels = labels[~outlier_mask]
    n_clusters = len(set(inlier_labels))
    colours = _get_cluster_colours(n_clusters)

    for label in set(inlier_true_labels):
        true_mask = inlier_true_labels == label
        cluster_labels = inlier_labels[true_mask]
        xs = inliers[true_mask, 0]
        ys = inliers[true_mask, 1]

        ax.scatter(
            xs,
            ys,
            marker=markers[label],
            c=[colours[lab] for lab in cluster_labels],
            alpha=1.0,
            lw=0.5,
            s=60,
            ec="lightgray",
        )

    ax.scatter(
        outliers[:, 0], outliers[:, 1], marker="s", s=20, c="None", ec="k"
    )

    here = pathlib.Path(__file__).parent
    ax.set(aspect="equal", xticks=[], yticks=[], xlim=lims, ylim=lims)

    plt.tight_layout()
    plt.savefig(
        here / f"{name}.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.25,
    )


def main():
    """Create a plot for each dataset and write it to file."""

    here = pathlib.Path(__file__).parent
    data = here / "../data/"
    for name in ("moons", "ellipses", "spheres"):
        dataset = np.genfromtxt(data / name / "main.csv", delimiter=",")
        labels = np.genfromtxt(
            data / name / "labels.csv", delimiter=","
        ).astype(int)

        make_plot(dataset, labels, name)


if __name__ == "__main__":
    main()
