from dataset import *
from helper import *
from classification import *
from eval import *
import numpy as np

# Global Variables
dataDirectory = 'data/'

# Import data via Dataset class
print("Loading the training dataset...");
simple1 = Dataset(directory=dataDirectory, name='simple1')
simple2 = Dataset(directory=dataDirectory, name='simple2')
test = Dataset(directory=dataDirectory, name='test')
toy = Dataset(directory=dataDirectory, name='toy')
train_full = Dataset(directory=dataDirectory, name='train_full')
train_noisy = Dataset(directory=dataDirectory, name='train_noisy')
train_sub = Dataset(directory=dataDirectory, name='train_sub')
validation = Dataset(directory=dataDirectory, name='validation')
toy2 = Dataset(directory=dataDirectory, name='toy2')
toy3 = Dataset(directory=dataDirectory, name='toy3')
print()

print("Training the decision tree...")
x = train_full.attributes
y = train_full.labels

validation_attributes = validation.attributes
validation_labels = validation.labels

test_attributes = test.attributes
test_labels = test.labels

classifier = DecisionTreeClassifier()
classifier = classifier.train(x, y)
classifier.print_tree_visual_simple(classifier.root)

print("Predicting...")
predictions = classifier.predict(test_attributes)
print("Input: \n{}".format(test_attributes))
print("Predictions: \n{}".format(predictions))
print()

print("Evaluating...")
evaluator = Evaluator()
confusion = evaluator.confusion_matrix(predictions, test_labels)
print("Confusion matrix: \n{}".format(confusion))
accuracy = evaluator.accuracy(confusion)
print("Accuracy: {}".format(accuracy))
precision = evaluator.precision(confusion)
print("Precision: {}".format(precision))
recall = evaluator.recall(confusion)
print("Recall: {}".format(recall))
f1_score = evaluator.f1_score(confusion)
print("F1 Score: {}".format(f1_score))
print()

print("k-Fold Cross-Validation...")
k_folds = k_random_subsets(x, y, 10)
k_fold_results = k_fold_cross_validation(k_folds)
k_fold_predictions = k_fold_results[0]
k_fold_classifiers = k_fold_results[1]
k_fold_accuracies = k_fold_results[2]
k_fold_best_accuracy = k_fold_results[3]
k_fold_best_classifier = k_fold_results[4]
print("K-Fold Accuracies: \n{}".format(k_fold_accuracies))
print("K-Fold Best Accuracy: \n{}".format(k_fold_best_accuracy))

print("Predicting using best classifier from k-Fold Cross-Validation...")
predictions = k_fold_best_classifier.predict(test_attributes)
print("Input: \n{}".format(test_attributes))
print("Predictions: \n{}".format(predictions))
print()

print("Evaluating predictions of best model from k_fold Cross-Validation...")
evaluator = Evaluator()
confusion = evaluator.confusion_matrix(predictions, test_labels)
print("Confusion matrix: \n{}".format(confusion))
accuracy = evaluator.accuracy(confusion)
print("Accuracy: {}".format(accuracy))
precision = evaluator.precision(confusion)
print("Precision: {}".format(precision))
recall = evaluator.recall(confusion)
print("Recall: {}".format(recall))
f1_score = evaluator.f1_score(confusion)
print("F1 Score: {}".format(f1_score))
print()

print("Combining predictions of all Cross-Validated classifiers...")
print("Predicting using combined classifer...")
combined_pred = combined_predictions(k_fold_classifiers, test_attributes)
print("Evaluating predictions of combined classifier...")
evaluator = Evaluator()
confusion = evaluator.confusion_matrix(combined_pred, test_labels)
print("Confusion matrix: \n{}".format(confusion))
accuracy = evaluator.accuracy(confusion)
print("Accuracy: {}".format(accuracy))
precision = evaluator.precision(confusion)
print("Precision: {}".format(precision))
recall = evaluator.recall(confusion)
print("Recall: {}".format(recall))
f1_score = evaluator.f1_score(confusion)
print("F1 Score: {}".format(f1_score))
print()

print("Pruning...")
classifier = classifier.prune(validation_attributes, validation_labels)
print()
print("Final pruned tree:")
classifier.print_tree_visual_simple(classifier.root)

print("Predicting Pruned Tree...")
predictions = classifier.predict(test_attributes)
print("Input: \n{}".format(test_attributes))
print("Predictions: \n{}".format(predictions))
print()

print("Evaluating Pruned Tree...")
confusion = evaluator.confusion_matrix(predictions, test_labels)
accuracy = evaluator.accuracy(confusion)
print("Accuracy: {}".format(accuracy))
precision = evaluator.precision(confusion)
print("Precision: {}".format(precision))
recall = evaluator.recall(confusion)
print("Recall: {}".format(recall))
f1_score = evaluator.f1_score(confusion)
print("F1 Score: {}".format(f1_score))
print()
