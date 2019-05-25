import sys
import matplotlib
import sklearn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import scipy
import pandas as pd
import numpy as np

print 'Loading data ...'
dataset = pd.read_csv('creditcard.csv')
print 'Done Loading!'
#dataset = dataset.sample(frac = 0.1, random_state = 1)
FRAUD = dataset[dataset['Class'] == 1]
VALID = dataset[dataset['Class'] == 0]

outlier_fraction = len(FRAUD) * 1.0 / len(VALID)

# correlation_matrix = dataset.corr()

columns = dataset.columns.tolist()
columns = [col for col in columns if col not in ['Class']]

target = 'Class'

X_data = dataset[columns]
Y_data = dataset[target]


random_state = 1
classifiers = {
    "Isolation Forest" : IsolationForest(
        max_samples = len(X_data),
        contamination = outlier_fraction,
        random_state = random_state
    ),
    "Local Outlier Factor" : LocalOutlierFactor(
        n_neighbors = 20,
        contamination = outlier_fraction
    )
}

outliers = len(FRAUD)
for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == 'Local Outlier Factor':
        Y_prediction = clf.fit_predict(X_data)
        scores_predict = clf.negative_outlier_factor_
    else:
        clf.fit(X_data)
        scores_predict = clf.decision_function(X_data)
        Y_prediction = clf.predict(X_data)

Y_prediction[Y_prediction == 1] = 0
Y_prediction[Y_prediction == -1] = 1

error_count = (Y_prediction != Y_data).sum()


print clf_name, error_count
print accuracy_score(Y_data, Y_prediction)

print classification_report(Y_data, Y_prediction)
