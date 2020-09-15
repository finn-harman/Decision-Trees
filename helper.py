import numpy as np
from classification import *
from eval import *


def k_random_subsets(x, y, k):
    """ Creates k random subsets from dataset

    Parameters
    ----------
    x : np.array
        Attributes of dataset
    y : np.array
        Labels of dataset
    k : int
        Number of folds

    Returns
    -------
    np.array[]
        An array of attribute arrays corresponding to each fold

    np.array[]
        An array of label arrays corresponding to each fold
    """
    if k > len(y):
        raise Exception(
            "Cannot split a dataset into more folds than it has rows.")
    if k < 2:
        raise Exception("Cannot split a dataset into fewer than 2 fold.")
    # Randomly shuffle dataset
    y = [[i] for i in y]
    z = np.append(x, y, axis=1)
    np.random.seed(0)
    np.random.shuffle(z)
    x = z[:, :-1]
    y = z[:, -1]
    # Create k equally sized subsets from the randomly sorted dataset
    subset_size = int(len(y) / k)
    remainder = len(y) - (subset_size * k)
    folds_x = list()
    folds_y = list()
    start = 0
    end = subset_size
    for i in range(k):
        fold_x = list(x[start:end])
        fold_y = list(y[start:end])
        folds_x.append(fold_x)
        folds_y.append(fold_y)
        start += subset_size
        end += subset_size

    for i in range(remainder):
        folds_x[i].append(x[-i])
        folds_y[i].append(y[-i])

    folds_x = np.array(folds_x).astype(np.int)
    folds_y = np.array(folds_y)
    return folds_x, folds_y


def k_fold_cross_validation(k_folds):
    """ Performs k-fold cross-validation

    Parameters
    ----------
    k_folds : np.array[]
        All attribute and label information for all folds

    Returns
    -------
    Classifiers and their predictions, as well as their accuracies
    """
    folds_x = k_folds[0]
    folds_y = k_folds[1]

    b_accuracy = 0
    b_classifier = 0
    predictions = list()
    classifiers = list()
    accuracies = list()

    for i in range(len(folds_y)):
        test_x = np.array(folds_x[i])
        test_y = folds_y[i]
        train_x = np.concatenate(np.delete(folds_x, i, 0))
        train_y = np.concatenate(np.delete(folds_y, i, 0))

        classifier = DecisionTreeClassifier()
        classifier = classifier.train(train_x, train_y)
        fold_predictions = classifier.predict(test_x)

        evaluator = Evaluator()
        confusion = evaluator.confusion_matrix(fold_predictions, test_y)
        accuracy = evaluator.accuracy(confusion)

        if accuracy > b_accuracy:
            b_accuracy = accuracy
            b_classifier = classifier

        predictions.append(fold_predictions)
        classifiers.append(classifier)
        accuracies.append(accuracy)

    predictions = np.array(predictions)
    return predictions, classifiers, accuracies, b_accuracy, b_classifier


def combined_predictions(classifiers, test_attributes):
    """ Combines classifiers to create aggregated classifier

    Parameters
    ----------
    classifiers : DecisionTreeClassifier[]
        All classifiers to be aggregated

    test_attributes : np.array[]
        Test data for comparing classifiers' predictions

    Returns
    -------
    np.array[]
        Combined predictions
    """
    all_predictions = list()
    for model in classifiers:
        predictions = model.predict(test_attributes)
        all_predictions.append(predictions)
    comb_predictions = all_predictions[0]
    for i in range(len(all_predictions[0])):
        predicted_labels = list()
        for fold in all_predictions:
            predicted_labels.append(fold[i])
        predicted_labels = np.array(predicted_labels)
        labels, freq = np.unique(predicted_labels, return_counts=True)
        comb_predictions[i] = labels[np.argmax(freq)]
    return comb_predictions
