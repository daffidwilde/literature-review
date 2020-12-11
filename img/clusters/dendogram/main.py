"""Source code to generate the hierarchical scatter plots."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from sklearn import cluster, preprocessing

linkage = "average"
default_colour = "#808080"
cmap = plt.cm.viridis


def _get_cluster_colours(n_clusters):
    """Retrieve the set of colours used to create `n_clusters` scatters."""

    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=n_clusters - 1)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    return tuple(
        plt.matplotlib.colors.rgb2hex(mappable.to_rgba(i)[:-1])
        for i in range(n_clusters)
    )


def _get_leaf_colours(data, criterion, n_clusters):
    """Identify the colours used for each point in a scatter of `data` into
    `n_clusters` parts with the colour map `cmap`."""

    cluster_colours = _get_cluster_colours(n_clusters)
    labels = (
        cluster.AgglomerativeClustering(
            n_clusters=n_clusters, linkage=criterion
        )
        .fit(data)
        .labels_
    )

    return {i: cluster_colours[label] for i, label in enumerate(labels)}


def _get_link_colours(linkage_matrix, leaf_colours):
    """Get the links to be coloured according to the leaves of the dendogram."""

    link_colours = {}
    for i, links in enumerate(linkage_matrix[:, :2].astype(int)):
        c1, c2 = (
            link_colours[link]
            if link > len(linkage_matrix)
            else leaf_colours[link]
            for link in links
        )
        link_colours[i + 1 + len(linkage_matrix)] = (
            c1 if c1 == c2 else default_colour
        )

    return link_colours


def make_plot(dataset, true_labels, name, **kwargs):
    """Create the linkage matrix for a dataset and then plot the dendrogram."""

    _, ax = plt.subplots(dpi=300)

    n_clusters = len(set(true_labels))
    scaled = preprocessing.StandardScaler().fit_transform(dataset.copy())
    leaf_colours = _get_leaf_colours(scaled, linkage, n_clusters)

    linkage_matrix = hierarchy.linkage(scaled, linkage)
    link_colours = _get_link_colours(linkage_matrix, leaf_colours)

    hierarchy.dendrogram(
        linkage_matrix,
        ax=ax,
        link_color_func=lambda x: link_colours[x],
        **kwargs,
    )

    ax.set(
        xticks=[],
        yticks=[],
    )

    ax.axis("off")

    here = pathlib.Path(__file__).parent
    ax.set(xticks=[], yticks=[])

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
