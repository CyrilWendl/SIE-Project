import numpy as np

def entropy_gaussian(S, base=2):
    """
    Differential entropy of a d-variate Gaussian density
    :param S: dataset in R^(N*D)
    :param base: base of entropy
    :return: entropy
    """
    K = np.linalg.det(np.cov(S.T))
    d = np.shape(S)[1]
    
    entropy = np.multiply(np.power(2 * np.pi * np.exp(1), d), K)
    if entropy <= 0:
        return 0
    entropy = np.log(entropy) / (np.log(base) * 2)
    if np.isnan(entropy):
        entropy = 0
    return entropy


def get_ig_dim(dataset, dim, entropy_f = entropy_gaussian, base = 2):
    """for one dimension, get information gain"""
    ig_vals = []
    split_vals = []

    # loop over all possible cut values
    for split_val in np.linspace(dataset[2, dim],dataset[-2, dim],100):  # TODO remove 1:-2, find out why beginning and end cause crash
        # split values
        split_rand_l = dataset[dataset[:, dim] < split_val]
        split_rand_r = dataset[dataset[:, dim] >= split_val]
        
        # entropy
        entropy_l = entropy_f(split_rand_l, base=base)
        entropy_r = entropy_f(split_rand_r, base=base)
        entropy_tot = entropy_f(dataset, base=base)

        # information gain
        ig = entropy_tot - (entropy_l * len(split_rand_l) /
                            len(dataset) + entropy_r * len(split_rand_r) / len(dataset))
        # append split value and information gain
        split_vals.append(split_val)
        ig_vals.append(ig)

    return np.array(ig_vals), np.array(split_vals)

def get_best_split(dataset):
    """for a given dimension, get best split based on information gain"""

    # get all information gains on all dimensions
    ig_dims_vals, split_dims_vals = [], []
    for dim in range(np.shape(dataset)[1]): # loop all dimensions
        ig_vals, split_vals = get_ig_dim(dataset, dim)
        ig_dims_vals.append(ig_vals)
        split_dims_vals.append(split_vals)

    # select best information gain and dimension
    # TODO implement

    return ig_dims_vals, split_dims_vals