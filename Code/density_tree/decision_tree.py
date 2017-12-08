"""Decision Tree Data Structure"""

class Decision_Node:
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
        self.labels = None  # the labels contained at this split level

        # child nodes
        self.left = None  # node to the left, e.g., for value < split_value
        self.right = None
        self.left_labels = None
        self.right_labels = None

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
            print('\n left')
            self.left.traverse_inorder()
        self.__format__()
        if self.right is not None:
            print('\n right')
            self.right.traverse_inorder()

    def traverse_preorder(self):
        self.__format__()
        if self.left is not None:
            print('\n left')
            self.left.traverse_preorder()
        if self.right is not None:
            print('\n right')
            self.right.traverse_preorder()

    def traverse_postorder(self):
        if self.left is not None:
            self.left.traverse_preorder()
            print('\n left')
        if self.right is not None:
            self.right.traverse_preorder()
            print('\n right')
        self.__format__()
        raise NotImplementedError

    def __format__(self):
        # print("rule: " + self.decisionrule) # print a decision rule on one line as a string (e.g., `d(2) < 20`)
        print("labels: " + str(self.labels))
        print("entropy: %.2f " % self.entropy)
        if self.has_children():
            print("split dimension: " + str(self.split_dimension))
            print("split value: %.2f " % self.split_value)