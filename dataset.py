import numpy as np
from classification import *
import matplotlib.pyplot as plt
import matplotlib as mtp


class Dataset:

    def __init__(self, directory, name):
        self.name = name
        self.full_data = np.char.strip(
            np.loadtxt(directory + name + '.txt', dtype=str, delimiter=','))
        self.set_attributes()
        self.set_labels()
        self.rownum = len(self.attributes)
        self.num_attributes = len(self.attributes[0])
        self.unique_labels, self.unique_counts = np.unique(self.labels,
                                                           return_counts=True)
        self.num_unique_labels = len(self.unique_labels)
        self.unique_label_freq = self.unique_counts / self.rownum * 100
        self.min_attribute_values = np.amin(self.attributes, axis=0)
        self.max_attribute_values = np.amax(self.attributes, axis=0)
        self.range_attribute_values = self.max_attribute_values - \
                                      self.min_attribute_values
        self.sort_data()

    def set_attributes(self):
        """ sets attribute data to attributes
        """
        self.attributes = self.full_data[:, 0:-1].astype(np.int)

    def set_labels(self):
        """ sets label data to labels
        """
        self.labels = self.full_data[:, -1]

    def info(self):
        """ prints useful information of the dataset
        """
        print("{} has:".format(self.name))
        print("   {} rows".format(self.rownum))
        print("   {} unique class labels: {}".format(self.num_unique_labels,
                                                     self.unique_labels))
        print("      with corresponding frequencies: {}".format(
            self.unique_label_freq))
        print("   {} attributes".format(self.num_attributes))
        print("      with minimum values: {}".format(self.min_attribute_values))
        print("      with maximum values: {}".format(self.max_attribute_values))
        print("      with range values: {}".format(self.range_attribute_values))
        print()

    def sort_data(self):
        """ sorts the full data set, the labels, and attributes in attribute
        order.
        """
        # Create a tuple of datasets in reverse column order
        full_ordered_indx = tuple(
            np.ndarray.transpose(self.attributes[:, ::-1]))

        # Use lexsort to order data in accordance with above tuple
        self.full_ordered_data = self.full_data[np.lexsort(full_ordered_indx)]
        self.full_ordered_labels = self.full_ordered_data[:, -1]
        self.full_ordered_attributes = self.full_ordered_data[:, 0:-1].astype(
            np.int)


def calc_diff_between_datasets(data_full, data_noisy):
    """Calculates the percentage difference between the labels of the full
    and noisy training set

        Parameters
        ----------
        data_full (NumPy array): M x N array of the full training set.
        data_noisy (NumPy array): M x N array of the noisy training set.

        Returns
        -------
        diff_percent(Float): the percentage differences of the labels in both
        datasets.
    """
    full_labels = data_full.full_ordered_labels
    noisy_labels = data_noisy.full_ordered_labels

    diff_percent = np.sum((full_labels != noisy_labels)) / noisy_labels.shape[
        0] * 100
    return diff_percent


def bar_plot_labels_diff(data_full, data_noisy):
    """Plots the differences in labels between the full and noisy training sets.

        Parameters
        ----------
        data_full (NumPy array): M x N array of the full training set.
        data_noisy (NumPy array): M x N array of the noisy training set.
    """
    full_labels_freq = data_full.unique_counts
    noisy_labels_freq = data_noisy.unique_counts
    perc_diff = (full_labels_freq - noisy_labels_freq)
    X = np.arange(full_labels_freq.shape[0])

    mtp.rc('font', family='times new roman')
    plt.bar(X, perc_diff)
    plt.ylabel("Frequency difference[%]")
    plt.xlabel("Class Labels[-]")
    plt.title("Difference in labels between train_full and train_noisy")
    plt.xticks(X, data_full.unique_labels)
    plt.axhline(y=0, color='k', lineWidth=0.5)
    plt.show()


def bar_plot_labels(data_full, data_sub):
    """Plots the frequency of the full and sub training data labels

        Parameters
        ----------
        data_full (NumPy array): M x N array of the full training set.
        data_sub (NumPy array): K x N array of the noisy training set.
    """
    full_labels_freq = data_full.unique_label_freq
    sub_labels_freq = data_sub.unique_label_freq
    X = np.arange(full_labels_freq.shape[0])

    mtp.rc('font', family='times new roman')
    plt.bar(X + 0.0, full_labels_freq, color='b', width=0.25)
    plt.bar(X + 0.25, sub_labels_freq, color='g', width=0.25)
    plt.ylabel("Frequency difference[%]")
    plt.xlabel("Class Labels[-]")
    plt.title("Frequency of labels between train_full and train_sub")
    plt.legend(["Full data", "Sub data"])
    plt.xticks(X, data_full.unique_labels)
    plt.show()


def bar_plot_labels_of_training(data_full, data_noisy, data_sub):
    """Plots the frequency of the full, noisy, and sub data labels

        Parameters
        ----------
        data_full (NumPy array): M x N array of the full training set.
        data_sub (NumPy array): K x N array of the noisy training set.
    """
    full_labels_freq = data_full.unique_label_freq
    noisy_labels_freq = data_noisy.unique_label_freq
    sub_labels_freq = data_sub.unique_label_freq
    X = np.arange(full_labels_freq.shape[0])

    mtp.rc('font', family='times new roman')
    plt.bar(X + 0.0, full_labels_freq, color='b', width=0.25)
    plt.bar(X + 0.25, noisy_labels_freq, color='g', width=0.25)
    plt.bar(X + 0.50, sub_labels_freq, color='r', width=0.25)
    plt.ylabel("Frequency [%]")
    plt.xlabel("Class Labels[-]")
    plt.title(
        "Frequency of labels between train_full, train_noisy and validation")
    plt.legend(["Full data", "Noisy data", "Validation data"])
    plt.xticks(X, data_noisy.unique_labels)
    plt.show()


def plot_ig_of_root_node(data):
    """Plots the information gain of every split at the root node

        Parameters
        ----------
        data (NumPy array): M x N array of the full training set's attributes
    """
    dtc = DecisionTreeClassifier()
    ig_array = dtc.ig_at_root_node(data.attributes, data.labels)

    X = np.arange(ig_array.shape[0])

    mtp.rc('font', family='times new roman')
    plt.bar(X, ig_array)
    plt.ylabel("Information Gain")
    plt.xlabel("Splits")
    plt.title("Information Gains per split")
    plt.show()
