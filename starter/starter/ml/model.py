from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os

from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = GradientBoostingClassifier()

    # Hyperparameter Optimization
    parameters = {
        "n_estimators":[5, 20, 30, 50],
        "max_depth":[3, 5],
        "learning_rate":[0.01, 0.1, 0.2]
    }

    # Run the grid search
    grid = GridSearchCV(clf, parameters)
    grid = grid.fit(X_train, y_train)

    # Set the clf to the best combination of parameters
    print(f'The best parameters are: {grid.best_params_}')
    clf = grid.best_estimator_

    # Train the model using the training sets 
    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : GradientBoostingClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    y_pred = model.predict(X)

    return y_pred


def slice(test,
	  target,
          model,
          categ_col,
          encoder,
          lb
):
    """
    Output the performance of the model on slices of the data

    args:
        - test (dataframe): dataframe of teste split
        - target (str): class label 
        - model (ml.model): trained machine learning model
        - categ_col (list): list of categorical columns
        - encoder (OneHotEncoder): One Hot Encoder
        - lb (LabelBinarizer): label binarizer
    returns:
        - metrics (DataFrame): dataframe with the calculated metrics


    """

    rows_list = list()
    for col in categ_col:
        for category in test[col].unique():
            row = {}
            tmp_df = test[test[col]==category]

            X, y, _, _ = process_data(
                X=tmp_df,
                categorical_features=categ_col,
                label=target,
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = inference(model, X)
            precision, recall, f_one = compute_model_metrics(y, preds)

            row['col'] = col
            row['category'] = category
            row['precision'] = precision
            row['recall'] = recall
            row['f1'] = f_one

            rows_list.append(row)

    metrics = pd.DataFrame(rows_list, columns=["col", "category", "precision", "recall", "f1"])

    return metrics 
