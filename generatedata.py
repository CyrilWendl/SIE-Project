import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log, e
import pylab

F = pylab.gcf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def createTwoClusters(mean1, mean2, cov1, cov2, npoints):
    x1, y1 = np.random.multivariate_normal(mean1, cov1, npoints).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, npoints).T
    return x1, y1, x2, y2


def plotData(cluster1,cluster2):
    plt.plot(cluster1[:,0], cluster1[:,1], 'x')
    plt.plot(cluster2[:,0], cluster2[:,1], 'o')
    plt.axis('equal')
    plt.grid()
    plt.savefig("/Users/cyrilwendl/Documents/EPFL/Projet SIE/SIE-Project/random_data.pdf", bbox_inches='tight')
    plt.show()


def entropy(labels, base=None):  # [1]
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

def differentialEntropy():
    pass
    # TODO implement: Gaussian entropy for continuous variables

def split(index, value, dataset):  # [2]
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def entropy_discrete(dataset, col_index):
    """
    Parameters
    ----------
    dataset :
        Input array with data and label in rows. The last column contains the labels.
    col_index :
        The index of the column for which the entropy should be computed.
    """
    x_vals, entropy_vals = list(), list()
    uniquevals=(np.unique(dataset[:,col_index]))
    for split_x in uniquevals[1:]:
        x_vals.append(split_x)

        # split
        left, right = split(col_index, split_x, dataset)
        left = np.asarray(left)
        right = np.asarray(right)

        # labels
        left_labels = left[:, -1]  # last column = labels
        right_labels = right[:, -1]

        # entropy
        left_entropy = entropy(left_labels, base=2)
        right_entropy = entropy(right_labels, base=2)

        # total entropy for attribute
        entropy_attr_split = left_entropy * len(left) / len(dataset) + right_entropy * len(right) / len(dataset)
        entropy_vals.append(entropy_attr_split)

    return entropy_vals, x_vals

mean1 = [18, 30]
cov1 = [[1, 0], [0, 1]]  # diagonal covariance
mean2 = [15, 40]
cov2 = [[2, 0], [0, 2]]  # diagonal covariance
x1,y1,x2,y2=createTwoClusters(mean1,mean2,cov1,cov2,30)

# zip for having tuples (x,y), round and unique for having discrete coordinates (eliminating duplicate points)
# TODO change to continuous
cluster1=np.unique(np.round(list(zip(x1,y1,np.ones(len(x1))))),axis=0) # np.ones: label 1 for first cluster
cluster2=np.unique(np.round(list(zip(x2,y2,np.ones(len(x2))*2))),axis=0) # np.ones*2: label 2 for second cluster

# connect unique points of cluster 1 and cluster 2
dataset=np.asarray(np.concatenate((cluster2,cluster1),axis=0))
plotData(cluster1,cluster2)
#print(dataset)

dfs=[]
for attribute in ["x",0],["y",1]:
    print(attribute[1])
    entropy_vals_attr, xs_vals_attr = \
        entropy_discrete(col_index=attribute[1], dataset=dataset)

    df=pd.DataFrame(entropy_vals_attr, xs_vals_attr)
    df.reset_index(inplace=True)
    df.columns=(["split_"+attribute[0],"Entropy"])
    print(df)
    dfs.append(df)

# [1] https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
# [2] https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python