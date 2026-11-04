"""Three-stage signal/HIC/risk chain must use one underlying acceleration signal."""
import numpy as np
from analysis_utils import calibrated_demo_acceleration_csv, parse_acceleration_csv, trajectory_frame
from injury_risk import nhtsa_skull_fracture_risk_pct


def test_calibrated_signals_reach_target_hic():
    b, r = 623.0, 496.5
    result = parse_acceleration_csv(calibrated_demo_acceleration_csv(b, r))
    assert np.isclose(result.baseline_hic15, b, atol=1e-7)
    assert np.isclose(result.rl_hic15, r, atol=1e-7)


def test_risk_directly_matches_same_hic_signal():
    result = parse_acceleration_csv(calibrated_demo_acceleration_csv(269.7, 243.0))
    frame = trajectory_frame(result)
    assert len(frame) == len(result.t_ms)
    for side in ('baseline', 'rl'):
        hic = frame[f'{side}_running_hic15'].to_numpy()
        risk = frame[f'{side}_skull_fracture_risk_pct'].to_numpy()
        assert (np.diff(hic) >= -1e-10).all()
        assert (np.diff(risk) >= -1e-10).all()
        assert np.allclose(risk,nhtsa_skull_fracture_risk_pct(hic))
    assert frame.baseline_skull_fracture_risk_pct.iloc[0] == 0
    assert frame.rl_skull_fracture_risk_pct.iloc[0] == 0


def test_example_csv_is_parseable_and_positive():
    result = parse_acceleration_csv(calibrated_demo_acceleration_csv(232.9, 218.5))
    assert result.baseline_hic15 > 0
    assert result.rl_hic15 > 0
