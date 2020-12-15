"""Main script for creating all the plots. Requires the datasets."""

import data
from kmeans import main as kmeans
from hierarchical import main as hierarchical
from dendogram import main as dendogram
from dbscan import main as dbscan


def main():
    """Create all the plots and write them to file."""

    kmeans.main()
    hierarchical.main()
    dendogram.main()
    dbscan.main()


if __name__ == "__main__":
    main()
