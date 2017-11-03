"""Functions for entropy and splitting"""

import numpy as np
import pandas as pd


def split(dataset, index, split_value):  # [2]
    """
    split a dataset (columns: variables, rows: data) in two according to some column (index) value.
    :param dataset: input dataset
    :param index: index of dimension to split values on
    :param split_value: value of the dimension where the dataset is split
    :return: left and right split datasets
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < split_value:
            left.append(row)
        else:
            right.append(row)
    return left, right


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

def entropy_discrete(dataset, dim):
    """
    calculate the entropy values for all cuts on one attribute (left<cut, right>=cut).
    :param dataset: Input array with data and label in rows. The last column contains the labels.
    :param dim: The index of the column for which the entropy should be computed.
    :return: entropy values, corresponding split values
    """
    x_vals, entropy_vals = list(), list()
    uniquevals = (np.unique(dataset[:, dim]))
    left_labels_unique, right_labels_unique = list(), list()
    
    # loop all possible split values
    for split_x in uniquevals[1:]:
        # append value
        x_vals.append(split_x)

        # split on value
        left, right = split(dataset, dim, split_x)
        left = np.asarray(left)
        right = np.asarray(right)

        # labels
        left_labels = left[:, -1]  # last column = labels
        right_labels = right[:, -1]

        # unique labels
        left_labels_unique.append(np.unique(left_labels))
        right_labels_unique.append(np.unique(right_labels))

        # entropy for split
        left_entropy = entropy(left_labels, base=2)
        right_entropy = entropy(right_labels, base=2)

        # total entropy for attribute
        # TODO change to information gain
        entropy_attr_split = left_entropy * len(left) / len(dataset) + right_entropy * len(right) / len(dataset)
        entropy_vals.append(entropy_attr_split)

    return entropy_vals, x_vals, left_labels_unique, right_labels_unique


def data_to_clusters(dataset):
    """Helper function to get clusters from estimated labels"""
    clusters = []
    for val in np.unique(dataset[:, 2]):
        clusters.append(dataset[dataset[:, 2] == val])
    return clusters


def get_unique_labels(labels):
    ul_side, ul_side_c = [], []

    for i in labels:
        ul_side_c.append(len(np.unique(i)))  # count of unique values on right side
        if len(np.unique(i)) == 1:  # if there is only one label, print the label
            ul_side.append(i)
        else:  # if there are several labels, print the number of the labels
            ul_side.append(i)
            # ul_side.append('several')
    return ul_side, ul_side_c


def calc_entropy_attribute(dataset):
    '''find the lowest entropy for a given attribute'''
    dfs = []
    entropy_attr = []
    x_attr = []

    for attribute_ind in range(np.shape(dataset, )[1] - 1): # loop over all attributes

        # get the entropy for all cuts
        entropy_vals_attr, xs_vals_attr, left_l_unique, right_l_unique = entropy_discrete(dataset,attribute_ind)
        # get number of unique labels (= cluster) on both sides
        ul_l, ul_l_c = get_unique_labels(left_l_unique)
        ul_r, ul_r_c = get_unique_labels(right_l_unique)

        # append values for all splits to dataframe
        x_attr.append(xs_vals_attr)
        entropy_attr.append(np.asarray(entropy_vals_attr))
        
        df = pd.DataFrame({'cut value': x_attr[attribute_ind],
                           'entropy': list(entropy_attr[attribute_ind]),
                           'left clusters': ul_l_c,
                           'right clusters': ul_r_c,
                           'left labels': ul_l,
                           'right labels': ul_r},
                          columns=['cut value', 'entropy', 'left clusters',
                                   'right clusters', 'left labels', 'right labels'])
        df.reset_index(inplace=True)
        dfs.append(df)
        
    return dfs


def get_best_attr(dfs):
    """get the attribute cutting which the information gain is highest"""
    min_e = np.infty

    for i in range(len(dfs)):  # loop all variables
        df = dfs[i]
        min_df_attr = df.loc[df['entropy'].argmin()]
        if min_df_attr['entropy'] < min_e:
            min_e = min_df_attr['entropy']
            min_df = pd.DataFrame(min_df_attr.drop("index")).transpose()
            min_df["dimension"] = i

    return min_df  # value of lowest entropy after possible cut, cut value, dimension


def next_split(left, right, results, root):
    """
    recursive method to split variables on dimension until all variables are contained in one subspace. G
    1. Get left (l) and right (r) based on split
    2. Check if labels unique
    3. Get entropies for all split values in ll rr
    4. Recurse -> 1.
    """
    for side in (left, right):  # loop both sides
        dt_node = Node() # decision tree node]
        dt_node.parent = root
        dt_node.labels = np.unique(np.asarray(side)[:, -1])  # get variables (in last column)
        
        if np.array_equal(left, side):
            dt_node.parent.left = dt_node
        elif np.array_equal(right, side):
            dt_node.parent.right = dt_node

        
        
        if len(dt_node.labels) > 1:  # if there are still more than one labels in a side
            side = np.asarray(side)
            dfs = calc_entropy_attribute(side)  # get entropies for all attributes within side
            min_df = get_best_attr(dfs)  # get best split value
            # dataframe
            results.append(min_df)
            
            dt_node.split_value = min_df["cut value"].values[0]
            dt_node.split_dimension = min_df["dimension"].values[0]
            left_new, right_new = split(side, 
                                        dt_node.split_dimension,
                                        dt_node.split_value) # get new left and right labels
            
            # save results for dataframe
            next_split(left_new, right_new, results, dt_node)  # split, recursion
            
def create_decision_tree(dimensions = 0, subsample = 0):
    """create decision tree be performing initial split,
    then recursively splitting until all labels are in unique bins
    """
    # TODO modify such as to take as entry number of variables to create tree on, number of data subsamples etc.
    
    root = Node() # initial node
    # initial split
    dfs = calc_entropy_attribute(dataset)
    min_df = get_best_attr(dfs)
    
    root.split_value = min_df["cut value"].values[0]
    root.split_dimension = min_df["dimension"].values[0]
    root.labels = np.unique(dataset[:,-1])
    
    left,right=split(dataset, 
                 min_df["dimension"].values[0], # dimension of min cut value
                 min_df["cut value"].values[0]) # min cut value
    results=[min_df]
    # recursively continue splitting
    next_split(left, right, results, root) # iterate
    return results, root