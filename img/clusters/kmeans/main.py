"""Source code to generate the k-means scatter plots."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sklearn import cluster, preprocessing

seed = 0
lims = (-2.6, 2.6)
cmap = plt.cm.viridis
alpha = 0.2
markers = ("o", "H", "D", "p")


def _get_cluster_colours(n_clusters):
    """Retrieve the set of colours used to create `n_clusters` scatters."""

    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=n_clusters - 1)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    return tuple(
        plt.matplotlib.colors.rgb2hex(mappable.to_rgba(i)[:-1])
        for i in range(n_clusters)
    )


def _get_voronoi_cells(centres):
    """Given a set of cluster centres, find the vertices and edges of their
    associated Voronoi cells.

    Adapted from https://nbviewer.jupyter.org/gist/pv/8037100
    """

    vor = spatial.Voronoi(centres)

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    radius = vor.points.ptp().max() * 10

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def make_plot(dataset, true_labels, name):
    """Make the scatter plot, the centre scatter plot and the Voronoi cells for
    a particular dataset."""

    _, ax = plt.subplots(dpi=300)

    n_clusters = len(set(true_labels))
    scaled = preprocessing.StandardScaler().fit_transform(dataset.copy())
    kmeans = cluster.KMeans(n_clusters, random_state=seed).fit(scaled)

    labels = kmeans.labels_
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

    centres = kmeans.cluster_centers_

    if n_clusters == 2:
        xs = np.linspace(-10, 10, 100)
        gradient = -np.diff(centres[:, 0]) / np.diff(centres[:, 1])
        midpoint = centres.mean(axis=0)
        intercept = midpoint[1] - gradient * midpoint[0]
        ys = gradient * xs + intercept

        for i, centre in enumerate(centres):
            extreme = max if centre[1] == max(centres[:, 1]) else min
            ax.fill_between(
                xs,
                ys,
                extreme(ys),
                fc=colours[i],
                ec="None",
                alpha=alpha,
                zorder=-2,
            )

    else:
        regions, vertices = _get_voronoi_cells(centres)
        for i, region in enumerate(regions):
            polygon = vertices[region]
            ax.fill(
                *zip(*polygon),
                fc=colours[i],
                alpha=alpha,
                zorder=-2,
            )

    edges = ["lightgray"] * (n_clusters - 1) + ["darkgray"]
    ax.scatter(
        centres[:, 0],
        centres[:, 1],
        marker="X",
        s=200,
        c=range(n_clusters),
        ec=edges,
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
