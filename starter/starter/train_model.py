# Script to train machine learning model.


# Add the necessary imports for the starter code.
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.model import train_model, compute_model_metrics, inference, slice
from ml.data import process_data

file_dir = os.path.dirname(__file__)


# Add code to load in the data.
data = pd.read_csv("../data/census.csv")


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", 
    training=False, encoder=encoder, lb=lb
)


# Train and save a model.
clf = train_model(X_train, y_train)
pickle.dump(clf, open(os.path.join(file_dir, '../model/clf.pkl'), 'wb'))
pickle.dump(encoder, open(os.path.join(file_dir, '../model/encoder.pkl'), 'wb'))
pickle.dump(lb, open(os.path.join(file_dir, '../model/lb.pkl'), 'wb'))

feature_importances = pd.DataFrame(list(zip(data.columns, clf.feature_importances_)), columns =['Feature', 'Importance'])
feature_importances.to_csv(os.path.join(file_dir, '../model/feature_importances.csv'))


# Predict
y_pred = inference(clf, X_test)


# Metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(f'Precision = {precision},\nRecall = {recall},\nFBeta = {fbeta}')


# Metrics in slices of the categorical features
metrics = slice(test, 'salary', clf, cat_features, encoder, lb)
metrics.to_csv(os.path.join(file_dir, '../model/metrics_slice.csv'))
