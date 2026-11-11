"""Evaluation and residual diagnostics modules."""

from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.residuals import compute_residuals, analyze_residuals
from src.evaluation.pipeline import (
    prepare_training_data,
    train_all_models,
    evaluate_all_models,
    generate_diagnostic_figures,
    run_training_pipeline,
)

__all__ = [
    "compute_regression_metrics",
    "compute_residuals",
    "analyze_residuals",
    "prepare_training_data",
    "train_all_models",
    "evaluate_all_models",
    "generate_diagnostic_figures",
    "run_training_pipeline",
]


