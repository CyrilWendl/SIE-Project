# Binary tree node to save binary tree nodes
class Node:
    """
    constructor for new nodes
    # decision rule is the rule which splits labels in two groups labels_left and labels_right
    # left_rule and right_rule are pointers to the rules that have to be used
    # to further split labels_left and labels_right
    """

    def __init__(self):
        # data for node
        self.parent = None  # parent node
        self.split_value = None  # the split value
        self.dimension = None  # the split dimension
        self.labels = None  # the labels contained at this split level

        # child nodes
        self.left = None  # node to the left, e.g., for value < split_value
        self.left_labels = None
        self.right = None
        self.right_labels = None

    def traverse_inorder(self):
        # TODO test
        if self.left is not None:
            self.left.traverse_inorder()
        self.__format__()
        if self.right is not None:
            self.right.traverse_inorder

    def traverse_preorder(self):
        # TODO test
        self.__format__()
        if self.left is not None:
            self.left.traverse_preorder()
        if self.right is not None:
            self.right.traverse_preorder()

    def traverse_postorder(self):
        # TODO implement
        raise NotImplementedError

    def __format__(self):
        # print("rule: " + self.decisionrule) # print a decision rule on one line as a string (e.g., `d(2) < 20`)
        print("labels: " + str(self.labels))
        print("dimension: " + str(self.dimension))
        print("split value: " + str(self.split_value))
        if self.left is not None:
            print("has left")
        if self.right is not None:
            print("has right")
