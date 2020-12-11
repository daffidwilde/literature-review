"""Source code to generate all the synthetic datasets."""

from moons import main as moons
from ellipses import main as ellipses
from spheres import main as spheres


def main():
    """Create the datasets and write them to file."""

    moons.main()
    ellipses.main()
    spheres.main()


if __name__ == "__main__":
    main()
