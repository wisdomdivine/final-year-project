"""Multiple Linear Regression (MLR) baseline model."""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.base import BaseModel


class MultipleLinearRegressionModel(BaseModel):
    """Multiple Linear Regression baseline wrapper."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(name="MultipleLinearRegression", params=params)
        self.model = LinearRegression(**self.params)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "MultipleLinearRegressionModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return np.asarray(self.model.predict(X))

