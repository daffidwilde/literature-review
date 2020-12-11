"""Source code to generate the hierarchical scatter plots."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, preprocessing

seed = 0
lims = (-2.6, 2.6)
linkage = "average"
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
    """Make the scatter plot."""

    _, ax = plt.subplots(dpi=300)

    n_clusters = len(set(true_labels))
    scaled = preprocessing.StandardScaler().fit_transform(dataset.copy())
    hierarchical = cluster.AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage
    ).fit(scaled)

    labels = hierarchical.labels_
    colours = _get_cluster_colours(n_clusters)

    for label in set(true_labels):

        mask = true_labels == label
        cluster_labels = labels[mask]
        cluster_colours = [colours[lab] for lab in cluster_labels]
        xs, ys = scaled[mask, 0], scaled[mask, 1]

        ax.scatter(
            xs,
            ys,
            marker=markers[label],
            c=cluster_colours,
            lw=0.5,
            s=60,
            ec="lightgray",
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
