"""Source code to generate the synthetic moons dataset."""

import pathlib

import numpy as np
from sklearn import datasets

n_samples = 300
noise = 0.05
seed = 3


def main():
    """Create the dataset and write it to file."""

    here = pathlib.Path(__file__).parent

    data, labels = datasets.make_moons(
        n_samples=n_samples, noise=noise, random_state=seed
    )

    np.savetxt(here / "main.csv", data, delimiter=",")
    np.savetxt(here / "labels.csv", labels, delimiter=",")


if __name__ == "__main__":
    main()
