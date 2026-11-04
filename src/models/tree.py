"""Random Forest Regressor wrapper."""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest regressor model wrapper."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(name="RandomForest", params=params)
        default_params = {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        if self.params:
            default_params.update(self.params)
        self.params = default_params
        self.model = RandomForestRegressor(**self.params)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "RandomForestModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return np.asarray(self.model.predict(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to get feature importances.")
        return self.model.feature_importances_
