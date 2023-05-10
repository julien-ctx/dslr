# DSLR: Logistic Regression To Classify Individuals In Hogwarts Houses | 42

*This project is part of the 42 cursus AI field.*

It consists in classifying individuals according to their characteristics in the different Harry Potter houses like the Sorting Hat does.

To do so, we use **logistic regression** using **One-vs-All** approach with **sigmoid function**.

## Usage

Install the required libraries with `pip3 install -r requirements.txt`.

- To *train the model*, go to `src/logistic_regression/` folder and use `python3 logreg_train.py [dataset_train.csv] [Default/Stochastic]`
- To make a prediction for a dataset, go to go to `src/logistic_regression/` folder and use `python3 logreg_train.py [dataset_test.csv] [weights.csv]`

CSV files used by the model are stored in `assets` folder.

## Model and techniques

In this project, we have 4 classes: *Hufflepuff, Gryffindor, Ravenclaw, Slytherin*. We use `dataset_train.csv` for training and `dataset_test` for validation.

### Data preprocessing

Thanks to **histograms, pair plots and scatter plots** that you can find in `src/logistic_regression/visualization` folder, we decided to remove irrelevant features from the dataset (similar distribution features, etc).

After that, we improved the model by **converting features** such as dates of birth into age, and best hand into binary values.

### Logistic Regression

Usually, logistic regression models use **softmax function** in order to determine probabilities of a sample to belong to a specific class.

In this project, we are using **sigmoid function**, and therefore we train the model **once for each class**. After that, we select the class with the **biggest probability** and make our prediction accordingly.

**Gradient descent** is used to reduce the loss, with `α = 0.001` and `epochs = 10000`. 1000 epochs is usually enough to retrieve accurate training weights.

# Bonuses

A few more features have been implemented to enhance the model performance, or to visualize the data:
- Stochastic Gradient Descent (SGD).

## Authors

- [@julien-ctx](https://github.com/julien-ctx)
- [@mgkgng](https://github.com/mgkgng)
