Decision Trees

### Introduction

I have implemented a decision tree from scratch in Python 3 to classify a set of black-and-white pixel images into one of several letters in the English alphabet

### Data

The ``data/`` directory contains the datasets.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``classification.py``

	* Contains the skeleton code for the ``DecisionTreeClassifier`` class.


- ``eval.py``

	* Contains the skeleton code for the ``Evaluator`` class.


- ``example_main.py``

	* Contains an example of how the evaluation script uses the classes
and invoke the methods defined in ``classification.py`` and ``eval.py``.



