from .random_forest import *
from .decision_tree_traverse import descend_decision_tree_aux

def print_rule(node):
    """Helper function to print the split decision of a given node"""
    rule_string = str(node.split_dimension) + "$<$" + str(np.round(node.split_value, 1))
    return rule_string


def printstuff(node, tree_string):
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
    # check if node has left labels
    if len(node.left_labels) == 1:
        tree_string += "[" + str(int(node.left_labels)) + "]"
    # checkif node has right labels
    if len(node.right_labels) == 1:
        tree_string += "[" + str(int(node.right_labels)) + "]"

    # iterate over node's children
    if len(node.left_labels) > 1:
        tree_string = printstuff(node.left, tree_string)

    if len(node.right_labels) > 1:
        tree_string = printstuff(node.right, tree_string)
    tree_string += "]"

    return tree_string


def get_grid_labels(dataset, root, minRange, maxRange, density=100, rf=False):
    """get labels on a regular grid"""
    x_min, x_max = [minRange, maxRange]
    y_min, y_max = [minRange, maxRange]
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    dataset_grid = np.transpose([xx.ravel(), yy.ravel()])

    if rf:  # random forest
        dataset_grid_eval = random_forest_traverse(dataset_grid, root)
    else:  # decision tree
        dataset_grid_eval = descend_decision_tree_aux(dataset_grid, root)
    return dataset_grid_eval[:, -1]

