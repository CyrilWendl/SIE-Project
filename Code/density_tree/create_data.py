"""Functions to generate test data"""
import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def create_data(n_clusters, dimensions, covariance, npoints, minrange=1, maxrange=100, labelled=True,
                random_flip=False, nonlinearities=False):
    """
    Create Gaussian distributed point clusters, in n dimensions
    :param n_clusters: the number of clusters to create
    :param dimensions: the number of dimensions in which to create clusters points
    :param covariance: the covariance of a default cluster
    :param npoints: the number of points per cluster
    :param minrange: the minimum random cluster mean
    :param maxrange: the maximum random cluster mean
    :param labelled: whether to return the cluster labels or not
    :param random_flip: whether to randomly reverse the cluster covariance or not
    :param nonlinearities: whether to randomly transform clusters to nonlinear distributions
    """
    clusters = []
    for idx_c, c in enumerate(range(n_clusters)):
        make_nonlinear = False
        if nonlinearities:
            # set one covariance dimension to 1 to make a circular shape
            make_nonlinear = np.random.randint(0, 2)

        # cluster mean
        mean_c = []
        data_range = maxrange - minrange  # add some margin of data_range / 10
        for d in range(dimensions):
            mean_c.append(np.random.randint(int(minrange + data_range / 10), int(maxrange - data_range / 10)))

        cov_c = covariance.copy()

        # cluster covariance
        if random_flip:
            # randomly flip covariances (e.g., [10, 20] to [20, 10])
            if np.random.randint(0, 2):
                cov_c = np.fliplr(np.asarray([cov_c]))
                cov_c = cov_c.flatten()

        # if nonlinear, elongate covariance
        if make_nonlinear:
            dim = np.random.randint(0, 2)
            cov_c[dim] = 1  # set one dimension to 1
            cov_c[dim == 0] = cov_c[dim == 0] * 4  # elongate the other dimension more

        # reshape covariance
        cov_c = np.identity(dimensions) * cov_c

        # generate cluster points
        x, y = np.random.multivariate_normal(mean_c, cov_c, npoints).T

        # if nonlinear, curve points
        if make_nonlinear:
            distort_x = np.random.randint(0, 2)
            distort_y = np.random.randint(0, 2)
            if distort_x:  # random if done or not
                y_min = np.min(y, axis=0)
                y_max = np.max(y, axis=0)

                dy = (y - y_min) / (y_max - y_min)
                if np.random.randint(0, 2):  # random if done or not
                    x += gaussian(dy, .5, .25) * dy * 50
                else:
                    x -= gaussian(dy, .5, .25) * dy * 50
            if distort_x == 0 or distort_y:  # random if done or noT
                x_min = np.min(x, axis=0)
                x_max = np.max(x, axis=0)

                dx = (x - x_min) / (x_max - x_min)
                if np.random.randint(0, 2):  # random if done or not
                    y += gaussian(dx, .5, .25) * dx * 50
                else:
                    y -= gaussian(dx, .5, .25) * dx * 50

        # last, check we want to add the labels or not
        clusters.append(np.unique(list(zip(x, y, np.ones(len(x)) * (idx_c + 1))), axis=0))

    x = [c[:, 0] for c in clusters]
    y = [c[:, 1] for c in clusters]
    clusters = np.asarray(clusters)
    if labelled:
        label = [c[:, 2] for c in clusters]
        dataset = np.asarray([np.asarray(x).flatten(), np.asarray(y).flatten(), np.asarray(label).flatten()]).T
        return dataset, clusters
    else:
        dataset = np.asarray([np.asarray(x).flatten(), np.asarray(y).flatten()]).T
        return dataset


def data_to_clusters(dataset):
    """Helper function to get clusters from estimated labels"""
    clusters = []
    for val in np.unique(dataset[:, 2]):
        clusters.append(dataset[dataset[:, 2] == val])
    return clusters
