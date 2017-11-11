"""Functions to generate test data"""
from . import *

def createMeans(clusters, dimensions, minRange=1, maxRange=100):
    """Generate a random mean of n clusters in d dimensions."""
    mean = []
    for c in range(clusters):
        mean_c = []
        for d in range(dimensions):
            mean_c.append(random.randint(minRange, maxRange))
        mean.append(mean_c)
    return mean


def createCovs(clusters, dimensions, covariance):
    """Generate a covariance matrix for n clusters in d dimensions with covariance c."""
    covs = []
    for c in range(clusters):
        covs.append(np.identity(dimensions) * covariance)
    return covs


def createClusters(means, covs, npoints):
    """
    Generate `npoints` random points within two clusters characteristed by their `mean` and `diagonal covariance`
    # TODO generalize to return more than two clusters
    """
    x, y = [], []
    for i in range(len(means)):
        x1, y1 = np.random.multivariate_normal(means[i], covs[i], npoints).T
        x.append(np.array(x1))
        y.append(np.array(y1))

    return x, y


def createData(clusters, dimensions, covariance, npoints, minRange=1, maxRange=100):
    """Create Gaussian distributed points, in n dimensions"""
    means = createMeans(clusters, dimensions, minRange, maxRange)
    covs = createCovs(clusters, dimensions, covariance)

    x, y = createClusters(means, covs, npoints)

    # zip for having tuples (x,y), round and unique for having discrete coordinates (eliminating duplicate points)
    clusters = []
    for i in range(len(x)):
        clusters.append(np.unique(list(zip(x[i], y[i], np.ones(len(x[i])) * (i + 1))),
                                  axis=0))  # np.ones: label 1 for first cluster
    dataset = np.asarray(np.concatenate(clusters, axis=0))
    return dataset, clusters
    # connect unique points of cluster 1 and cluster 2

def data_to_clusters(dataset):
    '''Helper function to get clusters from estimated labels'''
    clusters = []
    for val in np.unique(dataset[:, 2]):
        clusters.append(dataset[dataset[:, 2] == val])
    return clusters