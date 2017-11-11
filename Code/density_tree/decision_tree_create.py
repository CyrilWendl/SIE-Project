#%%writefile ./density_tree/decision_tree_create.py
'''Functions for entropy and splitting with labelled data'''
from .density_tree import *

class Node:
    """
    constructor for new nodes in a decision tree or density tree.
    decision rule is the rule which splits labels in two groups labels_left and labels_right
    left_rule and right_rule are pointers to the rules that have to be used
    to further split labels_left and labels_right
    """

    def __init__(self):
        # data for node
        self.parent = None  # parent node
        self.split_value = None  # the split value
        self.split_dimension = None  # the split dimension

        # labelled data
        self.labels = None  # the labels contained at this split level

        # unlabelled data
        self.entropy = None  # entropy, for unlabelled nodes
        self.dataset = None
        self.dataset_left = None
        self.dataset_right = None

        # child nodes
        self.left = None  # node to the left, e.g., for value < split_value
        self.right = None

        # labelled data
        self.left_labels = None
        self.right_labels = None

        # unlabelled data
        self.left_entropy = None
        self.right_entropy = None

    def get_root(self):
        if self.parent != None:
            return self.parent.get_root()
        else:
            return self

    def has_children(self):
        """print data for node"""
        if (self.right != None) & (self.right != None):
            return True
        return False

    def depth(self):
        """get tree depth"""
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return max(left_depth, right_depth) + 1

    def highest_entropy(self, node, e, side):
        """get the node in tree which has the highest entropy,
        taking a root node as input.
        for every node, check the entropies left and right after splitting
        if the node is not split yet to one of the sides and the entropy on the unsplit side
        exceeds the  maximum entropy, return the node.
        """
        if self.left_entropy is not None and self.left is None:
            if self.left_entropy > e:
                node = self
                e = self.left_entropy
                side = 'left'

        if self.right_entropy is not None and self.right is None:
            if self.right_entropy > e:
                node = self
                e = self.right_entropy
                side = 'right'

        if self.left is not None:
            node_lower_l, e_lower_l, side_lower_l = self.left.highest_entropy(node, e, side)
        if self.right is not None:
            node_lower_r, e_lower_r, side_lower_r = self.right.highest_entropy(node, e, side)
        if self.left is not None:
            if e_lower_l > e:
                node, e, side = node_lower_l, e_lower_l, side_lower_l
        if self.right is not None:
            if e_lower_r > e:
                node, e, side = node_lower_r, e_lower_r, side_lower_r

        return node, e, side

    """traversal methods"""

    def traverse_inorder(self):
        if self.left is not None:
            self.left.traverse_inorder()
        self.__format__()
        if self.right is not None:
            self.right.traverse_inorder()

    def traverse_preorder(self):
        self.__format__()
        if self.left is not None:
            self.left.traverse_preorder()
        if self.right is not None:
            self.right.traverse_preorder()

    def traverse_postorder(self):
        if self.left is not None:
            self.left.traverse_preorder()
        if self.right is not None:
            self.right.traverse_preorder()
        self.__format__()
        raise NotImplementedError

    def __format__(self):
        # print("rule: " + self.decisionrule) # print a decision rule on one line as a string (e.g., `d(2) < 20`)
        print("labels: " + str(self.labels))
        if self.has_children():
            print("split dimension: " + str(self.split_dimension))
            print("split value: " + str(self.split_value))


def entropy(labels, base = np.e):  # [1]
    '''
    Calculate the entropy for a set of labels.
    :param labels: an array of labels
    :param base: base of entropy, by default e
    :return: entropy
    '''
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

def entropy_discrete(dataset, dim):
    '''
    calculate the entropy values for all cuts on one attribute (left<cut, right>=cut).
    :param dataset: Input array with data and label in rows. The last column contains the labels.
    :param dim: The index of the column for which the entropy should be computed.
    :return: entropy values, corresponding split values
    '''
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
        left_entropy = entropy(left_labels, base = 2)
        right_entropy = entropy(right_labels, base = 2)

        # total entropy for attribute
        # TODO change to information gain
        entropy_attr_split = left_entropy * len(left) / len(dataset) + right_entropy * len(right) / len(dataset)
        entropy_vals.append(entropy_attr_split)

    return entropy_vals, x_vals, left_labels_unique, right_labels_unique


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
    """find the lowest entropy for a given attribute"""
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

"""
labelled
"""
def create_decision_tree(dataset, parent_node = None, side_label = None, max_depth = np.infty):
    """
    create decision tree be performing initial split, then recursively splitting until all labels are in unique bins
    at the entry, we get a dataset with distinct labels that has to be split
    :param dataset: labelled dataset [X,y]
    :param max_depth: maximum depth of decision tree
    """
    # TODO add data and variable subsample for RF
    dim_max, val_dim_max, ig_dims_vals, split_dims_vals = get_best_split(dataset, labelled = True)
    
    # create binary tree node
    treenode = Node() 
    treenode.split_value = val_dim_max
    treenode.split_dimension = dim_max
    treenode.labels = np.unique(dataset[:,-1])
    if parent_node is not None:
        treenode.parent = parent_node
        if side_label == 'left':
            parent_node.left = treenode
        elif side_label == 'right':
            parent_node.right = treenode
    
    # recursively continue splitting
    left, right = split(dataset, dim_max, val_dim_max) # split along best split dimension
    treenode.left_labels = np.unique(left[:,-1])
    treenode.right_labels = np.unique(right[:,-1])
    
    # check if current tree depth > max tree depth
    current_tree_depth = treenode.get_root().depth()
    
    # continue splitting only if there are several distinct labels to a side and the maximum tree depth has not been reached yet.
    # TODO: maybe we should split where maximum IG can be achieved.
    if (len(np.unique(left[:,-1])) > 1) & (current_tree_depth < max_depth):
        create_decision_tree(left, parent_node = treenode, side_label = 'left', max_depth = max_depth)
    if (len(np.unique(right[:,-1])) > 1) & (current_tree_depth < max_depth):
        create_decision_tree(right, parent_node = treenode, side_label = 'right', max_depth = max_depth)
    return treenode