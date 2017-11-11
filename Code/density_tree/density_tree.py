# %%writefile ./density_tree/density_tree.py

"""
unlabelled: working
"""
from . import *

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


"""
unlabelled: working
labelled: working
"""


def split(dataset, index, split_value, return_entropy=False):  # [2]
    """
    split a dataset (columns: variables, rows: data) in two according to some column (index) value.
    :param dataset: input dataset
    :param index: index of dimension to split values on
    :param split_value: value of the dimension where the dataset is split
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

    if return_entropy:
        e_left = entropy_gaussian(left)
        e_right = entropy_gaussian(right)
        return left, right, e_left, e_right
    else:
        return left, right


"""
labelled: ?
unlabelled: working
"""


def get_ig_dim(dataset, dim, entropy_f=entropy_gaussian, base=2, labelled=True):
    """
    for one dimension, get information gain
    :param dataset: dataset without labels (X)
    :param dim: dimension for which all cut values are to be calculated
    :param entropy_f: entropy function to be used (labelled / unlabelled)
    """
    ig_vals = []
    split_vals = []

    # loop over all possible cut values

    if labelled:
        iter_set = dataset[:, dim]  # e.g.: 3,5,1,2,6,10,4
    else:
        dataset_dim_min = np.min(dataset[:, dim])
        dataset_dim_max = np.max(dataset[:, dim])
        iter_set = np.linspace(dataset_dim_min, dataset_dim_max, 100)
        # iter_set = np.linspace(dataset[2, dim],dataset[-2, dim],100) # TODO remove 1:-2, find out why beginning and end cause crash

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


'''
unlabelled: working
labelled: working
'''


def get_best_split(dataset, labelled=False):
    '''for a given dimension, get best split based on information gain'''
    # get all information gains on all dimensions
    ig_dims_vals, split_dims_vals = [], []

    if labelled:
        entropy_f = entropy
        dimensions = range(np.shape(dataset)[1] - 1)

    else:
        entropy_f = entropy_gaussian
        dimensions = range(np.shape(dataset)[1])

    for dim in dimensions:  # loop all dimensions
        ig_vals, split_vals = get_ig_dim(dataset, dim, entropy_f=entropy_f, labelled=labelled)
        ig_dims_vals.append(ig_vals)
        split_dims_vals.append(split_vals)

    # split dimension of maximum gain
    dim_max = np.argmax(np.max(ig_dims_vals, axis=1))

    # split value of maximum gain
    # get all maximum values and take the middle if there are several possible maximum values
    # TODO get mean best split value, then find corresponding index

    # middle_max_ind = np.where(np.equal(ig_dims_vals[dim_max],np.max(ig_dims_vals[dim_max])))
    # middle_max_ind = int(np.floor(np.mean(middle_max_ind)))
    middle_max_ind = np.argmax(ig_dims_vals[dim_max])
    val_dim_max = split_dims_vals[dim_max][middle_max_ind]

    return dim_max, val_dim_max, ig_dims_vals, split_dims_vals


"""
unlabelled: working
"""


def create_decision_tree_unlabelled(dataset, dimensions, clusters, subsample=0, parentnode=None,
                                    side_label=None):
    """create decision tree be performing initial split,
    then recursively splitting until all labels are in unique bins
    init: flag for first iteration
    Principle:  create an initial split, save value, dimension, and entropies on node as well as on both split sides
    As long as total number of splits < nclusters - 1, perform another split on the side having the higher entropy
    Or, if there are parent nodes: perform a split on the side of the node that has the highest entropy on a side
    """
    # TODO modify such as to take as entry number of variables to create tree on, number of data subsamples etc.
    # split
    dim_max, val_dim_max, _, _ = get_best_split(dataset, labelled=False)
    left, right, e_left, e_right = split(dataset, dim_max, val_dim_max,
                                         return_entropy=True)  #  split along best dimension

    treenode = Node()  # initial node
    # save tree node
    treenode.split_dimension = dim_max
    treenode.split_value = val_dim_max
    treenode.dataset = dataset
    treenode.dataset_left = left
    treenode.dataset_right = right
    treenode.entropy = entropy_gaussian(dataset)
    treenode.left_entropy = e_left
    treenode.right_entropy = e_right

    # link parent node to new node.
    if parentnode is not None:
        treenode.parent = parentnode
        if side_label == 'left':
            parentnode.left = treenode
        elif side_label == 'right':
            parentnode.right = treenode

    clusters_left = clusters - 1
    if clusters_left > 1:
        # recursively continue splitting
        # continue splitting always splitting on worst side (highest entropy)
        # find node where left or right entropy is highest and left or right node is not split yet
        node_e, e, side = treenode.get_root().highest_entropy(dataset, 0, 'None')
        if side == 'left':
            dataset = node_e.dataset_left
            side_label = 'left'
        elif side == 'right':
            dataset = node_e.dataset_right
            side_label = 'right'

        create_decision_tree_unlabelled(dataset, dimensions, clusters=clusters_left, parentnode=node_e,
                                        side_label=side_label)  # iterate

    return treenode
