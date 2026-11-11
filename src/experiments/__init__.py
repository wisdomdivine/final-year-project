"""Controlled experimental comparisons for pore-pressure prediction.

Experiments:
- Experiment A: Depth-aware models (incorporating TVD_MSL)
- Experiment B: Physics-informed features (Sv, Ph, effective stress, DT/DT_nct)
- Experiment C: Formation-aware models (independent stratigraphy and compartment baffles)
"""

from src.experiments.features import (
    build_experiment_a_features,
    build_experiment_b_features,
    build_experiment_c_features,
)
from src.experiments.pipeline import run_all_experiments

__all__ = [
    "build_experiment_a_features",
    "build_experiment_b_features",
    "build_experiment_c_features",
    "run_all_experiments",
]
