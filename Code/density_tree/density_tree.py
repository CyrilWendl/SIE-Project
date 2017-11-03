import numpy as np
import pandas as pd #TODO remove

def entropy_gaussian(S, base=2):
    """Differential entropy of a d-variate Gaussian density"""
    K = np.linalg.det(np.cov(S.T))
    d = np.shape(S)[1]
    # entropy = np.dot(np.power((2*np.pi*np.exp(1)),d),K) / 2
    entropy = np.multiply(np.power(2 * np.pi * np.exp(1), d), K)
    if entropy < 0:
        return 0
    entropy = np.log(entropy) / (np.log(base) * 2)
    return entropy


def information_gain(S, S_l, S_r, entropy_f=entropy_gaussian):
    """calculate information gain based on entropy after split"""
    # entropy
    entropy_l = entropy_f(S_l, base=2)
    entropy_r = entropy_f(S_r, base=2)
    entropy_tot = entropy_f(S, base=2)

    # information gain
    ig = entropy_tot - (entropy_l * len(S_l) /
                        len(S) + entropy_r * len(S_r) / len(S))

    return ig, entropy_l, entropy_r, entropy_tot


def get_ig_dim(dataset, dim):
    split_x = 12
    ig_max = -np.inf
    ig_vals = []
    split_vals = []
    for split_x in (dataset[2:-1, dim]):  # TODO remove 1:-2, find out why beginning and end cause crash
        split_vals.append(split_x)
        split_rand_l = dataset[dataset[:, dim] >= split_x]
        split_rand_r = dataset[dataset[:, dim] < split_x]

        entropy_gaussian(split_rand_r, base=2)

        ig, entropy_l, entropy_r, entropy_tot = information_gain(dataset, split_rand_l, split_rand_r)
        ig_vals.append(ig)
        if ig is not np.nan and ig > ig_max:
            ig_max = ig
    return np.array(ig_vals), np.array(split_vals)


def get_best_split(dataset, dim=0):
    ig_vals, split_vals = get_ig_dim(dataset, dim)
    # TODO remove Pandas
    df = pd.DataFrame(ig_vals, split_vals).reset_index()
    df.columns = (["possible splits (dim " + str(dim) + ")", "information gain"])
    display = df

    split_vals_opt = (split_vals[np.argmax(ig_vals)] + split_vals[np.argmax(ig_vals) - 1]) / 2
    split_vals_opt

    return split_vals_opt