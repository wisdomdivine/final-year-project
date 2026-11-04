"""Base interface and abstract wrapper for all regression models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for pore pressure models."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.model: Any = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "BaseModel":
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate predictions for input feature matrix."""
        pass

    def save(self, filepath: Union[str, Path]) -> str:
        """Serialize model artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return str(path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseModel":
        """Load serialized model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        return joblib.load(path)
