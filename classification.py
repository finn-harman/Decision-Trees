##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the
# DecisionTreeClassifier 
##############################################################################

import numpy as np
from eval import *


class ParentNode(object):
    """
    Class that forms the node of a decision tree structure.

    Attributes
    ----------
    attributes (M x K NumPy array): Array of data sample attributes of
    current node.
    labels (M-dimensional NumPy array): Array of data sample labels of
    current node.
    b_value (int): chosen split value of current node.
    b_children (ChildrenData): Left and right children data based on the
    chosen splitting rule.
    b_information_gain (Float): the information gain based on the chosen
    splitting rule.
    is_terminal (boolean): Flag that indicates whether current node is a leaf
    node.
    tested_for_pruning (boolean): Flag that indicates whether current node
    has been tested for pruning.
    leaf (String): the class label of the current node is the node is terminal.
    left_child (ParentNode): Pointer to the child node which contains the
    data sample that satisfies
                            the splitting rule.
    right_child (ParentNode): Pointer to the child node which contains the
    data sample that does not
                            satisfy the splitting rule.

    Methods
    -------
    init__(att, labels, b_att, b_val, b_children, info_gain, is_terminal,
    tested_for_pruning)
        Initialises all attributes except for leaf, left_child, right_child
    set_leaf(leaf)
        Sets the leaf member of the current node to class label inputted as
        the argument leaf.
    set_left_child(left)
        Sets the left_child member to point to the ParentNode object with
        data samples that
        satisfy the splitting rule.
    set_right_child(right)
        Sets the right_child member to point to the ParentNode object with
        data samples that
        satisfy the splitting rule.

    """

    def __init__(self, att, labels, b_att, b_val, b_children, info_gain,
                 is_terminal, tested_for_pruning):
        self.attributes = att
        self.labels = labels
        self.b_attribute = b_att
        self.b_value = b_val
        self.b_children = b_children
        self.b_information_gain = info_gain
        self.is_terminal = is_terminal
        self.tested_for_pruning = tested_for_pruning

    def set_leaf(self, leaf):
        self.leaf = leaf

    def set_left_child(self, left):
        self.left_child = left

    def set_right_child(self, right):
        self.right_child = right


class ChildrenData(object):
    """
    Class that encapsulates the data attributes and labels that have been
    split in accordance
    with the splitting rule of the owning ParentNode object.

    Attributes
    ----------
    left_x (M x K NumPy Array): All data sample attributes that satisfy the
    splitting rule.
    left_y (M NumPy Array): All data sample class labels that satisfy the
    splitting rule.
    right_x (N x K NumPy Array): All data sample attributes that do not
    satisfy the splitting rule.
    left_y (N NumPy Array): All data sample class labels that do not satisfy
    the splitting rule.

     Methods
     -------
    _init__(self, left_x, left_y, right_x, right_y)
        Initialises all member attributes

    """

    def __init__(self, left_x, left_y, right_x, right_y):
        self.left_x = left_x
        self.left_y = left_y
        self.right_x = right_x
        self.right_y = right_y


def get_accuracy(decision_tree, validation_attributes, validation_labels):
    prediction = decision_tree.predict(validation_attributes)
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(prediction, validation_labels)
    accuracy = evaluator.accuracy(confusion)
    return accuracy


class DecisionTreeClassifier(object):
    """
    A decision tree classifier
    
    Attributes
    ----------
    is_trained (bool): Keeps track of whether the classifier has been trained
    root (ParentNode): the root node of the Decision Tree Structure
    
    Methods
    -------
    __init__()
        Sets is_trained to false.
    entropy(y)
        Calculates the information entropy of a given sample of labels y
    information_gain(parent,children)
        Calculates the information gain of a given parent dataset and it's
        children dataset.
    try_split(, attribute, value, x, y)
        Returns children data in accordance with the splitting rule defined
        by attribute and value.
    best_split(x,y)
        Returns a ParentNode with the most optimal split
    terminal_node_value(y)
        returns the most frequent label of the given label dataset y.
    split(node, max_depth = -1, depth = 0)
        Recursively splits a node until a leaf node is reached.
    print_tree_detailed(self, node, depth = 0)
        Prints entire decision tree structure (detailed mode)
    def print_tree_visual(self, node, depth = 0)
        Prints entire decision tree structure (visual mode).
    print_tree_visual_simple(self, node, depth = 0)
        Prints entire decision tree structure (simple visual mode).
    prune()
        Prunes through entire decision tree structure
    try_prune(self, node, baseline_accuracy)
        Prunes through node and all children of the node recursively.
    predict_instance(self, node, instance)
        Predicts the class label of a given data sample.
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    
    """

    def __init__(self):
        self.is_trained = False

    def entropy(self, y):

        """ Calculates entropy of the class label dataset y

        Parameters
        ----------
        y (NumPy array) An N-dimensional numpy array of class labels.

        Returns
        -------
        entropy_value (Float): the entropy of the class label dataset y
        """
        # Get list of unique labels in dataset and their respective frequencies
        unique_labels, unique_labels_freq = np.unique(y, return_counts=True)
        # Get proportions of each unique label in dataset
        label_proportion = unique_labels_freq / len(y)
        # Calculate entropy
        entropy_value = 0
        for label in range(len(unique_labels)):
            entropy_value += -label_proportion[label] * np.log2(
                label_proportion[label])
        return entropy_value

    def information_gain(self, parent, children):

        """ Calculates the information gain between the parent and children
        datasets

        Parameters
        ----------
        Parent (NumPy array): an N-Dimensional array of class labels.
        children (NumPy array): a K x N array of class labels that have been
        split K times based on a given
                                splitting rule
        Returns
        -------
        information_gain_value (Float): the information gain between the
        parent and children datasets.
        """
        # Calculate entropy of parent
        parent_entropy = self.entropy(parent)
        # Calculate total entropy of all children
        children_entropy = 0
        for child in range(len(children)):
            children_entropy += len(children[child]) / len(
                parent) * self.entropy(children[child])
        information_gain_value = parent_entropy - children_entropy
        return information_gain_value

    def try_split(self, attribute, value, x, y):
        """ Splits the datasets x and y in accordance with splitting rule
        defined by attribute and value.

        Parameters
        ----------
        attribute (Int): the index of the attribute where the value is located.
        value (Int): the threshold value by which the split takes place.
        x (NumPy array): M x N array of attributes.
        y (NumPy array): M-dimensional array of class labels.
        Returns
        -------
        -(ChildrenData): an object containing the arrays that have been split
        by value.
                        All 'left' arrays contain samples that have attribute
                        values less than value
                        All 'right' arrays contain samples that have
                        attribute values greater than
                        or equal to value
        """
        # Initialise child datasets as empty lists
        left_x, left_y, right_x, right_y = [], [], [], []
        # Add rows to each child dataset based on split rile
        for row in range(len(x)):
            # left contains all rows with attribute less than value
            if x[row, attribute] < value:
                left_x.append(x[row])
                left_y.append(y[row])
            # right contains all rows with attribute greater than or equal to
            # value
            else:
                right_x.append(x[row])
                right_y.append(y[row])
        return ChildrenData(np.array(left_x), np.array(left_y),
                            np.array(right_x), np.array(right_y))

    def ig_at_root_node(self, x, y):
        """ Finds all information gain values for every split

         Parameters
         ----------
         x (NumPy array): M x N array of attributes.
         y (NumPy array): M-dimensional array of class labels.
         Returns
         -------
         ig_array (NumPy array): an array of every information gain at the
         root node.
         """
        ig_array = []
        # For each attribute in the parent
        for attribute in range(len(x[0])):
            # Sort x and y by the attribute
            sorted_index_order = x[:, attribute].argsort()
            x = x[sorted_index_order]
            y = y[sorted_index_order]
            # Get a list of unique attribute values in x
            unique_values = np.unique(x[:, attribute])
            # For each unique attribute value (except fist one)
            for value in unique_values[1:]:
                children = self.try_split(attribute, value, x, y)
                children_labels = np.array([children.left_y, children.right_y])
                ig_array.append(self.information_gain(y, children_labels))

        ig_array = np.array(ig_array)
        return ig_array

    def best_split(self, x, y):
        """ Finds and returns the best split and the corresponding child node.

         Parameters
         ----------
         x (NumPy array): M x N array of attributes.
         y (NumPy array): M-dimensional array of class labels.

         Returns
         -------
         -(ParentNode): An object containing the data structures and
         attributes corresponding to
                        the the optimal splitting rule for datasets x and y.
         """

        # If all labels in y are the same, set terminal_node = true
        terminal_node = False
        if len(np.unique(y)) == 1:
            terminal_node = True
        if np.all(np.all(x == x[0, :], axis=0)):
            terminal_node = True
        # Else find the best split
        best_attribute, best_value, best_info_gain, best_children = -1, -1, \
                                                                    -1, -1
        # For each attribute in the parent
        for attribute in range(len(x[0])):
            # Sort x and y by the attribute
            sorted_index_order = x[:, attribute].argsort()
            x = x[sorted_index_order]
            y = y[sorted_index_order]
            # Get a list of unique attribute values in x
            unique_values = np.unique(x[:, attribute])
            # For each unique attribute value (except fist one)
            for value in unique_values[1:]:
                children = self.try_split(attribute, value, x, y)
                children_labels = np.array([children.left_y, children.right_y])
                info_gain = self.information_gain(y, children_labels)

                # if current split is better than current best split,
                # set current split to best split
                if info_gain > best_info_gain:
                    best_attribute, best_value, best_info_gain, best_children\
                        = attribute, value, info_gain, children

        return ParentNode(x, y, best_attribute, best_value, best_children,
                          best_info_gain, terminal_node, False)

    def terminal_node_value(self, y):

        """ Finds the most frequent class label of the given dataset y

         Parameters
         ----------
         y (NumPy array): M-dimensional array of class labels.
         Returns
         -------
         unique_label[index] (String): the most frequent label
         """
        unique_labels, unique_labels_freq = np.unique(y, return_counts=True)
        index = np.argmax(unique_labels_freq)
        return unique_labels[index]

    def split(self, node, max_depth=-1, depth=0):
        """ Recursively splits a node until a leaf node is reached.

         Parameters
         ----------
         node (ParentNode): the node to be split
         max_depth(int): the maximum depth whereupon the recursion should occur.
         depth(int): the current depth of recursion

         Returns
         -------
         nil
         """
        # Check depth and if too deep, call terminal node value and return
        if (depth >= max_depth) and (max_depth != -1):
            node.is_terminal = True
        # Check if node is terminal
        if node.is_terminal:
            node.set_leaf(self.terminal_node_value(node.labels))
            return
        # Find best split on left child
        node.set_left_child(
            self.best_split(node.b_children.left_x, node.b_children.left_y))
        # Recursively call split() to split left child further
        self.split(node.left_child, max_depth, depth + 1)
        # Likewise for right child
        node.set_right_child(
            self.best_split(node.b_children.right_x, node.b_children.right_y))
        self.split(node.right_child, max_depth, depth + 1)

    def print_tree_detailed(self, node, depth=0):
        """ Prints decision tree structure (detailed mode).

         Parameters
         ----------
         node (ParentNode): the node to start printing from.
         depth (int): the depth to start printing from.

         Returns
         -------
         nil
         """
        print("Depth: {}".format(depth))
        print("Parent dataset: \n {}".format(
            np.append(node.attributes, node.labels, axis=1)))
        if node.terminal:
            print("Node is terminal. Leaf with value: {}".format(node.leaf))
            print()
            return
        print("Split point: X{} < {}".format(node.b_attribute, node.b_value))
        print("Left child dataset: \n {}".format(
            np.append(node.b_children.left_x, node.b_children.left_y, axis=1)))
        print("Right Child dataset: \n {}".format(
            np.append(node.b_children.right_x, node.b_children.right_y,
                      axis=1)))
        print()
        self.print_tree_detailed(node.left_child, depth + 1)
        self.print_tree_detailed(node.right_child, depth + 1)

    def print_tree_visual(self, node, depth=0):
        """ Prints decision tree structure (visual mode).

         Parameters
         ----------
         node (ParentNode): the node to start printing from.
         depth (int): the depth to start printing from.

         Returns
         -------
         nil
         """
        attributes = np.array(
            ['x_box', 'y_box', 'width', 'height', 'onpix', 'x-bar', 'y-bar',
             'x2bar', 'y2bar',
             'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx'])
        unique_labels, unique_labels_freq = np.unique(node.labels,
                                                      return_counts=True)
        label_proportion = np.around(
            unique_labels_freq / sum(unique_labels_freq), decimals=2)

        if node.is_terminal:
            print("{}Leaf {}".format("         " * depth, node.leaf))
            return
        print("{}Depth {}. Distribution: {}{}. Split: {} < {}. IG: {}".format(
            "         " * depth, depth,
            unique_labels, label_proportion,
            attributes[node.b_attribute],
            node.b_value,
            np.around(node.b_information_gain,
                      decimals=2)))
        self.print_tree_visual(node.left_child, depth + 1)
        self.print_tree_visual(node.right_child, depth + 1)

    def print_tree_visual_simple(self, node, depth=0):
        """ Prints decision tree structure (simple visual mode).

         Parameters
         ----------
         node (ParentNode): the node to start printing from.
         depth (int): the depth to start printing from.

         Returns
         -------
         nil
         """
        unique_labels, unique_labels_freq = np.unique(node.labels,
                                                      return_counts=True)
        label_proportion = np.around(
            unique_labels_freq / sum(unique_labels_freq), decimals=2)

        if node.is_terminal:
            print("{}Leaf {}".format("         " * depth, node.leaf))
            return
        print("{}Depth {}. Distribution: {}{}. Split: X{} < {}. IG: {}".format(
            "         " * depth, depth,
            unique_labels, label_proportion,
            node.b_attribute, node.b_value,
            np.around(node.b_information_gain,
                      decimals=2)))
        self.print_tree_visual_simple(node.left_child, depth + 1)
        self.print_tree_visual_simple(node.right_child, depth + 1)

    def prune(self, validation_attributes, validation_labels):
        fully_pruned = False
        while not fully_pruned:
            # Calculate accuracy of baseline model
            baseline_accuracy = get_accuracy(self, validation_attributes,
                                             validation_labels)
            self.pruned = False
            self.try_prune(self.root, baseline_accuracy, validation_attributes,
                           validation_labels)
            if not self.pruned:
                fully_pruned = True
        return self

    def try_prune(self, node, baseline_accuracy, validation_attributes,
                  validation_labels):
        if self.pruned:
            return
        if node.is_terminal:
            return
        if node.tested_for_pruning:
            return
        if node.left_child.is_terminal and node.right_child.is_terminal:
            node.is_terminal = True
            node.leaf = self.terminal_node_value(node.labels)
            # calculate accuracy of new model
            pruned_accuracy = get_accuracy(self, validation_attributes,
                                           validation_labels)
            if pruned_accuracy >= baseline_accuracy:
                del node.left_child
                del node.right_child
                self.pruned = True
            else:
                node.tested_for_pruning = True
                node.is_terminal = False
                del node.leaf
            return
        self.try_prune(node.left_child, baseline_accuracy,
                       validation_attributes, validation_labels)
        self.try_prune(node.right_child, baseline_accuracy,
                       validation_attributes, validation_labels)

    def train(self, x, y):
        """ Constructs a decision tree classifier from data

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        # Set parameters for decision tree
        max_depth = 10

        # Train the decision tree
        self.root = self.best_split(x, y)
        self.split(self.root)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def predict_instance(self, node, instance):
        """ Predicts the class label of a given data sample instance by
        navigating through the
            decision tree structure.

         Parameters
         ----------
         node (ParentNode): the current node of the decision tree structure
         instance (NumPy array): M-dimensional array of attributes of the
         data sample.
         Returns
         -------
         node.left_child.leaf (String): the predicted class label.
         Otherwise will recursively call upon itself until a leaf node is
         reached.
         """

        # From node, check whether prediction should follow left or right child
        if int(instance[node.b_attribute]) < int(node.b_value):
            # If node is terminal, predict that value
            if node.left_child.is_terminal:
                return node.left_child.leaf
            else:
                return self.predict_instance(node.left_child, instance)
        else:
            if node.right_child.is_terminal:
                return node.right_child.leaf
            else:
                return self.predict_instance(node.right_child, instance)

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception(
                "Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        # #######################################################################
        # #                 ** TASK 2.2: COMPLETE THIS METHOD **
        # #######################################################################

        # For each row in test set, predict label
        for row in range(len(x)):
            predictions[row] = self.predict_instance(self.root, x[row])

        return predictions
