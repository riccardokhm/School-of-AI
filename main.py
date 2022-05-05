# Repeated k-Fold Cross Validation

import numpy as np
from numpy import mean
import pandas as pd
import tensorflow as tf
import datetime
import shap
from matplotlib import pyplot
from sklearn.model_selection import RepeatedKFold, cross_val_score, LeaveOneOut
from sklearn.linear_model import SGDClassifier


def get_dataset(file):
    # Loading dataset and verifying the nature of the CSV file.
    df = pd.read_csv(file)
    print(f"Original dataset of shape  {df.shape} made of {df.size} elements, including labels")

    # Preparing the dataset for the cross-validation
    df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
    df[df.select_dtypes(np.int64).columns] = df.select_dtypes(np.int64).astype(np.int8)
    x = np.array(df[['HA', 'EDA', 'PA', 'EDA_Task', 'RR', 'Rula']])
    y = np.array(df[['Y']])
    return x, np.ravel(y)


def evaluate_model(i, n, shapley=False):
    if not shapley:
        print(f"Applying K-Fold Cross Validation {n} times with {i} folds and computing predicted values with a Logistic "
          f"Regression model")
    rcv = RepeatedKFold(n_splits=i, n_repeats=n, random_state=1)
    model = SGDClassifier(max_iter=100, shuffle=True)
    start_time = datetime.datetime.now()
    scores = cross_val_score(model, X, Y, scoring='accuracy', cv=rcv, n_jobs=-1)
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).seconds
    if shapley:
        print('Assessing features influence on the predicted values using Shapley algorithm with the best k value')
        model.fit(X, Y)
        x_train = shap.utils.sample(X, nsamples=50)
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(x_train)
        shap.summary_plot(shap_values, x_train, plot_type="bar")
    return scores.mean(), scores.min(), scores.max(), scores.std(), elapsed_time


# Create dataset and initialize parameters
print("Using TensorFlow version", tf.__version__)
dataset_path = "Dataset_Raw.csv"
print('Creating the dataset from collected data')
X, Y = get_dataset(dataset_path)
r = 9
accuracies = {}

# Calculate the ideal test condition
loo = LeaveOneOut()
sgdc = SGDClassifier()
ideal = mean(cross_val_score(sgdc, X, Y, scoring='accuracy', cv=loo, n_jobs=-1))
print('Ideal accuracy with the LeaveOneOut cross validation algorithm: %.3f' % ideal)

# Sensitivity analysis for K
folds = range(2, 11)
means, mins, maxs = list(), list(), list()

for f in folds:
    # retrieve values for each k attempt of mean, max and min and append to relative lists on evaluating the model.
    k_mean, k_min, k_max, k_std, training_time = evaluate_model(f, r)
    means.append(k_mean)
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

    # report performance
    print(f'Elapsed time: {training_time} s')
    print('Average accuracy: %.3f,  Standard deviation: %.3f , Maximum accuracy reached: %.3f , Minimum accuracy'
          ' reached:' '%.3f \n' % (k_mean, k_std, k_max, k_min))
    accuracies[f] = k_mean

# Plotting sensitivity analysis result for k
pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
pyplot.plot(folds, [ideal for _ in range(len(folds))], color='r')
pyplot.show()

# Assessing features influence on the predicted values using Shapley algorithm with the best k value.
k_best = max(accuracies, key=accuracies.get)
k_best_mean, k_best_min, k_best_max, k_best_std, k_best_time = evaluate_model(k_best, r, shapley=True)
