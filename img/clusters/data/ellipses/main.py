"""Source code to generate the synthetic ellipses dataset."""

import pathlib

import numpy as np
from sklearn import datasets

n_samples = 300
n_centres = 3
seed = 170
transformation = [[0.4, 0.8], [-0.6, 0.1]]


def main():
    """Create the dataset and write it to file."""

    here = pathlib.Path(__file__).parent

    data, labels = datasets.make_blobs(
        n_samples=n_samples, centers=n_centres, random_state=seed
    )
    data = np.dot(data, transformation)

    np.savetxt(here / "main.csv", data, delimiter=",")
    np.savetxt(here / "labels.csv", labels, delimiter=",")


if __name__ == "__main__":
    main()
