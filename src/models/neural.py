"""Multilayer Perceptron (MLP) neural network regressor wrapper."""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.models.base import BaseModel


class MLPModel(BaseModel):
    """Multilayer Perceptron regressor wrapper."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(name="MultilayerPerceptron", params=params)
        default_params = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
            "n_iter_no_change": 20,
        }
        if self.params:
            default_params.update(self.params)
        self.params = default_params
        self.model = MLPRegressor(**self.params)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "MLPModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return np.asarray(self.model.predict(X))
