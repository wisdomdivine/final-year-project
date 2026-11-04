"""Physics-based geomechanical and petrophysical modules."""

from src.physics.trajectory import interpolate_tvd_from_survey
from src.physics.overburden import (
    integrate_overburden_stress,
    calculate_overburden_gradient,
)
from src.physics.hydrostatic import calculate_hydrostatic_pressure
from src.physics.nct import fit_normal_compaction_trend, predict_normal_transit_time
from src.physics.eaton import calculate_eaton_pore_pressure

__all__ = [
    "interpolate_tvd_from_survey",
    "integrate_overburden_stress",
    "calculate_overburden_gradient",
    "calculate_hydrostatic_pressure",
    "fit_normal_compaction_trend",
    "predict_normal_transit_time",
    "calculate_eaton_pore_pressure",
]

