"""CatBoost Regressor wrapper."""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost gradient boosting regressor wrapper."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(name="CatBoost", params=params)
        default_params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": False,
        }
        if self.params:
            default_params.update(self.params)
        self.params = default_params
        self.model = CatBoostRegressor(**self.params)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        eval_set: Optional[Tuple[Any, Any]] = None,
    ) -> "CatBoostModel":
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
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
        return self.model.get_feature_importance()
