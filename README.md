# pygendata

Synthetic data set generation tools for machine learning experiments.

## File overview

- ``generate.py``: the main data generation script
- ``output/``: empty directory for generated data sets

## Usage

All configuration options are currently defined and documented within ``generate.py`` (although it is intended that this will change to external configuration files in future versions). Once these are set:

``$ python3 generate.py``

There is also a ``Makefile`` with targets ``data`` to run ``generate.py`` as above, and ``clean`` to remove generated data sets from ``output/``. The latter should probably be used with care.

## Output

Under ``output/``, each generated data set has its own directory, with a naming convention based on its configuration. So for a data set named ``2_10_1000_r_0.5_004``, in order:

- number of clusters
- number of features
- number of samples
- cardinality (**u**niform or **r**andom)
- within-cluster standard deviation
- index ie. a counter, as we can generate multiple data sets for each configuration

For manageability, generated data sets are grouped into subdirectories based on number of clusters, ie. the current value from iterating ``OPTS_K``.

Each dataset folder contains:

- ``data.csv``: the data set itself
- ``labels.csv``: the class labels of the data points

Contents of ``output/`` are protected by a ``.gitignore`` file as it is not anticipated that users will commit them to this project on purpose.

## Requirements

- Python 3
- scikit-learn >= 0.20
- numpy


## Future work

- the ability to run from separate config files, eg. Yaml
- allow more flexible normalisation, eg. pluggable normalisation strategies


## Useful links

- [scikit-learn's ``make_blobs()`` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)


