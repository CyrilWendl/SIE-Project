import numpy as np
import matplotlib.pyplot as plt
from math import log, e
import pylab

F = pylab.gcf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def createTwoClusters(mean1, mean2, cov1, cov2, npoints):
    x1, y1 = np.random.multivariate_normal(mean1, cov1, npoints).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, npoints).T
    return x1, y1, x2, y2


def plotData(x_c1, y_c1, x_c2, y_c2):
    plt.plot(x_c1, y_c1, 'x')
    plt.plot(x_c1, y_c2, 'o')
    plt.axis('equal')
    # DefaultSize = F.get_size_inches()
    # F.set_size_inches((DefaultSize[0] * 1.5, DefaultSize[1] * 1.8))
    # plt.savefig("/Users/cyrilwendl/Documents/EPFL/Projet SIE/SIE-Project/random_data.pdf", bbox_inches='tight')
    plt.show()


def entropy(labels, base=None):  # [1]
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

def differentialEntropy():
    pass
    """
    TODO implement: Gaussian entropy for continuous variables
    """

def split(index, value, dataset):  # [2]
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


mean1 = [0, 50]
cov1 = [[1, 0], [0, 10]]  # diagonal covariance

mean2 = [50, 0]
cov2 = [[20, 50], [50, 20]]  # diagonal covariance

# x1,y1,x2,y2=createTwoClusters(mean1,mean2,cov1,cov2,100)
# plotData(np.round(x1),np.round(y1),np.round(x2),np.round(y2))

# 1D labels
labels = [2, 1, 2, 2, 2, 1, 1, 1, 1]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
dataset = np.array([x, labels]).T  # organise data in rows (columns = variables)


# TODO Loop over all attributes
def loop_attribute(col_index):
    x_vals, entropy_vals = list(), list()

    for split_x in range(2, len(x) + 1):
        x_vals.append(split_x)

        left, right = split(col_index, split_x, dataset)
        left = np.asarray(left)
        right = np.asarray(right)

        left_labels = left[:, -1]  # last column = labels
        right_labels = right[:, -1]

        left_entropy = entropy(left_labels, base=2)
        right_entropy = entropy(right_labels, base=2)

        entropy_attr_split = left_entropy * len(left) / len(x) + right_entropy * len(right) / len(x)

        entropy_vals.append(entropy_attr_split)

    return entropy_vals, x_vals


entropy_vals, xs_vals = loop_attribute(col_index=0)
print(np.round(entropy_vals,2))
print(xs_vals)

# [1] https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
# [2] https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
