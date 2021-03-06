##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np


class Evaluator(object):
    """ Class to perform evaluation
    """

    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(np.concatenate((annotation, prediction)))

        confusion = np.zeros((len(class_labels), len(class_labels)),
                             dtype=np.int)

        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################

        # For each row in class_labels, run the following (A is given as
        # example)
        for row in range(len(class_labels)):
            # Get indices of observations in annotation where ground truth is A
            annotation_indices = np.where(annotation == class_labels[row])[0]
            # Get the predictions for each of these indices
            label_predictions = prediction[annotation_indices]
            # Get the indices in class_labels of these predictions
            class_label_indices = []
            for label in label_predictions:
                class_label_indices = np.append(class_label_indices,
                                                np.where(class_labels == label))
            if len(class_label_indices) != 0:
                class_label_indices = class_label_indices.astype(int)
            # For each index in class_label_indices, add 1 to corresponding
            # element of confusion matrix
            for label in class_label_indices:
                confusion[row][label] += 1

        return confusion

    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions
        
        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        # feel free to remove this
        accuracy = 0.0

        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################

        correct = wrong = 0

        for i in range(len(confusion)):
            for j in range(len(confusion[i])):
                if i == j:
                    correct += confusion[i][j]
                else:
                    wrong += confusion[i][j]

        if correct + wrong == 0:
            accuracy = np.nan
        else:
            accuracy = correct / (correct + wrong)

        return accuracy

    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.
        
        Also returns the macro-averaged precision across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.   
        """

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################

        for column in range(len(confusion)):
            correct = wrong = 0
            for row in range(len(confusion)):
                if row == column:
                    correct += confusion[row][column]
                else:
                    wrong += confusion[row][column]
            if correct + wrong == 0:
                p[column] = np.nan
            else:
                p[column] = correct / (correct + wrong)

        # You will also need to change this        
        macro_p = 0

        for i in range(len(p)):
            if not np.isnan(p[i]):
                macro_p += p[i]
        macro_p /= len(p)

        return p, macro_p

    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.
        
        Also returns the macro-averaged recall across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged recall score across C classes.   
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################

        for row in range(len(confusion)):
            correct = wrong = 0
            for column in range(len(confusion)):
                if row == column:
                    correct += confusion[row][column]
                else:
                    wrong += confusion[row][column]
            if correct + wrong == 0:
                r[row] = np.nan
            else:
                r[row] = correct / (correct + wrong)

        # You will also need to change this        
        macro_r = 0

        for i in range(len(r)):
            if not np.isnan(r[i]):
                macro_r += r[i]
        macro_r /= len(r)

        return r, macro_r

    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.
        
        Also returns the macro-averaged f1-score across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged f1 score across C classes.   
        """

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion),))

        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################

        precision = self.precision(confusion)[0]
        recall = self.recall(confusion)[0]

        for label in range(len(f)):
            if np.isnan(precision[label]):
                f[label] = np.nan
            elif np.isnan(recall[label]):
                f[label] = np.nan
            else:
                f[label] = 2 * (precision[label] * recall[label]) / (
                            precision[label] + recall[label])

        # You will also need to change this        
        macro_f = 0

        for i in range(len(f)):
            if not np.isnan(f[i]):
                macro_f += f[i]
        macro_f /= len(f)

        return f, macro_f
