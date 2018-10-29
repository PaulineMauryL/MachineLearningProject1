# EPFL Machine Learning - Project 1

Machine Learning algorithm using Ridge regression and selected features to find the Higg's Boson in Kaggle competition: classify whether a data was an event of the Higg's Boson or not. 


### Prerequisites

Download [training and testing set](https://www.kaggle.com/c/epfml18-higgs) from Kaggle. 


## Project files

### run.py
**Input:** training and testing data in the same folder

**Output:** prediction of the classification of the test set
* Load the data and divide it in 4 categories. 
* Pre-process each category.

Compute the weight according to the selected features.
Classify the test set.

### implementation.py
Implementation of the methods seen in class
 - least_squares_GD(y, tx, initial_w, max_iters, gamma)
 - least_squares_SGD(y, tx, initial_w, max_iters, gamma)
 - least_squares(y, tx)
 - ridge_regression(y, tx, lambda_)
 - logistic regression(y, tx, initial_w, max_iters, gamma)
 - reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
all methods return (w, loss): the last weight vector and the corresponding loss 
function

### preprocessing.py
Implements the functions used by run.py to preprocess the data
 - split_categories: - split the dataset according to the jet number value (PRI_jet_num)
                     - for each category, remove the features containing '-999' values
 - processing: pre-process the input, calls build_data and standardize
 - build_data: method for feature engineering, builds data


### proj1_helpers.py

Methods provided by the [course](https://mlo.epfl.ch/page-157255-en-html/) for project 1.


## Authors

* Jérôme Savary -*MT_MA1*- 

* Audrey Jordan -*MT_MA3*-

* Pauline Maury Laribière -*MT_MA3*- [PaulineMauryL](https://github.com/PaulineMauryL)


## Acknowledgments

* [Physics background](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf)

* [Feature Engineering](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)

* [Advice for applying Machine Learning](http://cs229.stanford.edu/materials/ML-advice.pdf)

* [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)

* [Machine Learning CS-433 course from M. Jaggi](https://mlo.epfl.ch/page-157255-en-html/)
