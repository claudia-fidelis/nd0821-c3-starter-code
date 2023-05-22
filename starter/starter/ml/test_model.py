from ml.model import train_model, compute_model_metrics, inference
import os
import sys
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)


@pytest.fixture()
def data():
    """Fixture: generate a random 2-class classification problem data
    """
    X, y = make_classification(n_samples=100)
    return X, y


def test_train_model(data):
    """Test train_model
    """
    X, y = data
    model = train_model(X, y)
    assert type(model) == GradientBoostingClassifier


def test_compute_model_metrics():
    """Test compute_model_metrics
    """
    y, preds = [1, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # precision = 0.75, recall = 0.75, fbeta = 0.75
    assert abs(precision - 0.75) < 0.01 and abs(recall - 0.75) < 0.01 and abs(fbeta - 0.75) < 0.01


def test_inference():
    """Test inference
    """    
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size=10)
    model = train_model(X, y)
    pred = inference(model, X)
    # Check if pred.shape is similar to y.shape
    assert y.shape == pred.shape
