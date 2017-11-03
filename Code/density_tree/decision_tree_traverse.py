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
        self.labels = None  # the labels contained at this split level
        self.split_value = None  # the split value
        self.split_dimension = None  # the split dimension

        # child nodes
        self.left = None  # node to the left, e.g., for value < split_value
        self.left_labels = None
        self.right = None
        self.right_labels = None

    """print data for node"""

    def has_children(self):
        if (self.right != None) & (self.right != None):
            return True
        return False

    def __format__(self):
        # print("rule: " + self.decisionrule) # print a decision rule on one line as a string (e.g., `d(2) < 20`)
        print("labels: " + str(self.labels))
        if self.has_children():
            print("split dimension: " + str(self.split_dimension))
            print("split value: " + str(self.split_value))

    """tree traversal methods"""

    def traverse_inorder(self):
        if self.left is not None:
            self.left.traverse_inorder()
        self.__format__()
        if self.right is not None:
            self.right.traverse_inorder

    def traverse_preorder(self):
        self.__format__()
        if self.left is not None:
            self.left.traverse_preorder()
        if self.right is not None:
            self.right.traverse_preorder()

    def traverse_postorder(self):
        self.__format__()
        if self.left is not None:
            self.left.traverse_preorder()
        if self.right is not None:
            self.right.traverse_preorder()
        raise NotImplementedError