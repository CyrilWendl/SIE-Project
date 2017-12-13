import numpy as np


def entropy(labels, base=np.e):  # [1]
    """
    Calculate the entropy for a set of labels.
    :param labels: an array of labels
    :param base: base of entropy, by default e
    :return: entropy
    """
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def entropy_gaussian(dataset, base=2):
    """
    Differential entropy of a d-variate Gaussian density
    :param dataset: dataset in R^(N*D)
    :param base: base of entropy
    :return: entropy
    """
    k = np.linalg.det(np.cov(dataset.T))
    d = dataset.shape[1]

    ent = np.multiply(np.power(2 * np.pi * np.exp(1), d), k)
    if ent <= 0:
        return 0
    ent = np.log(ent) / (np.log(base) * 2)
    if np.isnan(ent):
        ent = 0
    return ent


def split(dataset, index, split_value, get_entropy=False):  # [2]
    """
    split a dataset (columns: variables, rows: data) in two according to some column (index) value.
    :param dataset: input dataset
    :param index: index of dimension to split values on
    :param split_value: value of the dimension where the dataset is split
    :param get_entropy: optional indicator whether to return entropy
    :return: left and right split datasets
    """
    left, right = [], []
    for row in dataset:
        if row[index] < split_value:
            left.append(row)
        else:
            right.append(row)

    left = np.asarray(left)
    right = np.asarray(right)

    if get_entropy:
        e_left = entropy_gaussian(left)
        e_right = entropy_gaussian(right)
        return left, right, e_left, e_right
    else:
        return left, right


def print_rule(node):
    """Helper function to print the split decision of a given node"""
    rule_string = str(node.split_dimension) + "$<$" + str(np.round(node.split_value, 1))
    return rule_string


def print_density_tree_latex(node, tree_string):
    """print decision tree in a LaTeX syntax for visualizing the decision tree
    To be called as:
    tree_string = ""
    tree_string = printstuff(root,tree_string)
    """
    tree_string += "["

    tree_string += print_rule(node)
    print_rule(node)

    # check if node is leaf node
    if node.left is None:
        tree_string += "[ent:%.2f]" % node.left_entropy
    # check if node is leaf node
    if node.right is None:
        tree_string += "[ent:%.2f]" % node.right_entropy

    # iterate over node's children
    if node.left is not None:
        tree_string = print_density_tree_latex(node.left, tree_string)

    if node.right is not None:
        tree_string = print_density_tree_latex(node.right, tree_string)
    tree_string += "]"

    return tree_string


def print_decision_tree_latex(node, tree_string):
    """print decision tree in a LaTeX syntax for visualizing the decision tree
    To be called as:
    tree_string = ""
    tree_string = printstuff(root,tree_string)
    """
    tree_string += "["

    # check if node is split node
    if len(node.labels) > 1:
        tree_string += print_rule(node)
        print_rule(node)
    # check if node is leaf node
    if len(node.left_labels) == 1:
        tree_string += "[" + str(int(node.left_labels)) + "]"
    # checkif node is leaf node
    if len(node.right_labels) == 1:
        tree_string += "[" + str(int(node.right_labels)) + "]"

    # iterate over node's children
    if len(node.left_labels) > 1:
        tree_string = print_decision_tree_latex(node.left, tree_string)

    if len(node.right_labels) > 1:
        tree_string = print_decision_tree_latex(node.right, tree_string)
    tree_string += "]"

    return tree_string


def get_best_split(dataset, labelled=False):
    """
    for a given dimension, get best split based on information gain
    for labelled and unlabelled data

    :param dataset: dataset for which to find the best split
    :param labelled: indicator whether dataset contains labels or not

    :return dim_max: best split dimension
    :return val_dim_max: value at best split dimensions
    :return ig_dims_vals: information gains for all split values in all possible split dimensions
    :return split_dims_vals: split values corresponding to ig_dims_vals
    """

    # get all information gains on all dimensions
    ig_dims_vals, split_dims_vals = [], []

    if labelled:
        entropy_f = entropy
        dimensions = range(dataset.shape[1] - 1)

    else:
        entropy_f = entropy_gaussian
        dimensions = range(dataset.shape[1])

    for dim in dimensions:  # loop all dimensions
        ig_vals, split_vals = get_ig_dim(dataset, dim, entropy_f=entropy_f)
        ig_dims_vals.append(ig_vals)
        split_dims_vals.append(split_vals)

    # split dimension of maximum gain
    dim_max = int(np.argmax(np.max(ig_dims_vals, axis=1)))

    # maximum ig split indexes
    max_ind = np.where(np.equal(ig_dims_vals[dim_max], np.max(ig_dims_vals[dim_max])))
    # split value of maximum gain
    # get all maximum values and take the middle if there are several possible maximum values
    max_ind = int(np.floor(np.mean(max_ind)))
    val_dim_max = split_dims_vals[dim_max][max_ind]

    return dim_max, val_dim_max, ig_dims_vals, split_dims_vals


def get_ig_dim(dataset, dim, entropy_f=entropy_gaussian, base=2):
    """
    for one dimension, get information gain
    for labelled and unlabelled data

    :param dataset: dataset without labels (X)
    :param dim: dimension for which all cut values are to be calculated
    :param entropy_f: entropy function to be used (labelled / unlabelled)
    :param base: base to use for entropy calculation
    """
    ig_vals = []
    split_vals = []

    # loop over all possible cut values
    dataset_dim_min = np.min(dataset[:, dim])
    dataset_dim_max = np.max(dataset[:, dim])
    iter_set = np.linspace(dataset_dim_min, dataset_dim_max, 100)

    for split_val in iter_set:
        # split values
        split_l = dataset[dataset[:, dim] < split_val]
        split_r = dataset[dataset[:, dim] >= split_val]

        # entropy
        entropy_l = entropy_f(split_l, base=base)
        entropy_r = entropy_f(split_r, base=base)
        entropy_tot = entropy_f(dataset, base=base)

        # information gain
        ig = entropy_tot - (entropy_l * len(split_l) / len(dataset) + entropy_r * len(split_r) / len(dataset))

        # append split value and information gain
        split_vals.append(split_val)
        ig_vals.append(ig)

    return np.array(ig_vals), np.array(split_vals)
