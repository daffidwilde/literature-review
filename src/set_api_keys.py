""" A short script to set the keys for the Springer and IEEE APIs in Arcas.

The documentation for Arcas has instructions on how to get API keys:
    https://arcas.readthedocs.io/en/latest/Guides/api_key.html

Once you have them and have stored them as described in `keys/README.md`, you
need only run this script from within the working environment, and your keys
will be set.
"""

import importlib
import os
import pathlib

import arcas


def _import_module_from_filename(filename):

    spec = importlib.util.spec_from_file_location("__main__", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    """ Copy the API key files over to the *current* Arcas install location. """

    install = pathlib.Path(arcas.__file__).parent
    for publisher in ("IEEE", "Springer"):

        origin = pathlib.Path(f"../keys/{publisher}.py")
        destination = install / publisher / "api_key.py"
        assert origin.exists(), FileNotFoundError(f"{origin} does not exist.")

        module = _import_module_from_filename(origin)
        assert hasattr(module, "api_key"), \
            f"{origin} does not have an `api_key` variable"
        assert isinstance(module.api_key, str), \
            "`api_key` variable must be a string"

        os.system(f"cp {origin} {destination}")


if __name__ == "__main__":
    main()
