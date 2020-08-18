"""
Generate synthetic datasets
"""

from concurrent import futures
import itertools
import os

import numpy as np
from sklearn.datasets import make_blobs


# Main configuration
OPTS_K = [2, 5, 10, 20]              # Number of clusters
OPTS_FEATS = [2, 10, 50, 100, 1000]  # Number of features
OPTS_SAMPS = [1000]                  # Number of data points
OPTS_CARD = ['u', 'r']               # Uniform or random cardinality
OPTS_STDEV = [0.5, 1, 1.5]           # Within-cluster standard deviation
N_EACH = 50                          # No. of sets per configuration
OUTPUT_DIR = './output/'             # Output directory
N_PROCESSES = None                   # Defaults to os.cpu_count()

NAME_SEPARATOR = '_'                 # Directory name separator

# Parameters for random cardinality clusters
MIN_CL_SIZE = 0.03                   # Minimum cluster size eg. 3%
WEIGHT_SHIFT = 0.005                 # Increase by until MIN_CL_SIZE is met

NORMALISE_DATA = True


def normalise(matrix):
    """Normalise data to have mean 0 and range 1"""

    matrix = np.array(matrix)

    treated = (matrix - matrix.mean(axis=0)) \
        / (matrix.max(axis=0) - matrix.min(axis=0))

    return treated


def _gen_dataset(no_clusters, no_feats, no_samps, card, stdev):
    """Generate individual dataset"""

    if card == 'r':
        weights = _gen_weights(no_clusters)
        sample_cts = np.ceil(weights * no_samps).astype(int)

        while sample_cts.sum() > no_samps:
            sample_cts[np.argmax(sample_cts)] -= 1

        centers = None
    else:
        sample_cts = no_samps
        centers = no_clusters

    return make_blobs(
        n_samples=sample_cts,
        centers=centers,
        n_features=no_feats,
        cluster_std=stdev)


def _gen_weights(num_clusters):
    """Generate the weightings for non-uniform clustering"""

    weights = np.random.random(num_clusters)
    weights /= weights.sum()

    # Brute-force ensure no cluster smaller than MIN_CL_SIZE
    while np.min(weights) < MIN_CL_SIZE:
        weights[np.argmin(weights)] += WEIGHT_SHIFT
        weights[np.argmax(weights)] -= WEIGHT_SHIFT

    return weights


def _gen_name(config, index):
    """Generate the unique name for a given dataset"""

    subdir = config[0]
    subdir = f"{subdir:02d}"

    return subdir + '/' + NAME_SEPARATOR.join(map(str, config)) + \
        NAME_SEPARATOR + f"{index:03d}"


def _save_to_disk(data, labels, config, index):
    """Save both files to disk in their own directory"""

    name = _gen_name(config, index)

    dirname = OUTPUT_DIR + name + '/'

    # Optionally normalise data
    if NORMALISE_DATA is True:
        data = normalise(data)

    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass  # no problem

    np.savetxt(dirname + 'data.csv', data, delimiter=',')
    np.savetxt(dirname + 'labels.csv', labels, fmt="%i", delimiter=',')


def _handler(config):
    """The callback for the Executor"""

    config = list(config)
    index = config.pop()   # counter for which of N_EACH this is

    # print("Beginning: ", config, '#', index)

    data, labels = _gen_dataset(*config)
    _save_to_disk(data, labels, config, index)

    print("Generated: ", config, '#', index)


def main():
    """Main method"""
    configs = itertools.product(OPTS_K, OPTS_FEATS, OPTS_SAMPS,
                                OPTS_CARD, OPTS_STDEV, range(0, N_EACH))

    with futures.ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        res = executor.map(_handler, configs)

    # Just calling this here is enough to display errors which may
    # have been hidden behind the Executor's parallel operation
    print(len(list(res)), 'data sets generated')


# Main ------------------------------------------------------------------------


if __name__ == '__main__':
    main()
