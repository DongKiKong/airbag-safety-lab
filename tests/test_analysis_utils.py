import numpy as np
import pandas as pd
import pytest
from io import BytesIO
from analysis_utils import running_hic15, parse_acceleration_csv


def test_constant_10g_for_20ms():
    t = np.arange(0.0, 21.0)
    actual = running_hic15(t, np.full_like(t, 10.0))
    assert actual[-1] == pytest.approx(0.015 * 10**2.5, rel=1e-10)
    assert np.all(np.diff(actual) >= 0)


def test_shorter_interval_is_allowed():
    t = np.array([0.0, 5.0, 10.0])
    running = running_hic15(t, np.full_like(t, 5.0))
    assert running[-1] == pytest.approx(0.010 * 5**2.5)


def test_csv_complete_report():
    df = pd.DataFrame({"t_ms": np.arange(0,31), "rl_head_acc_g": 4, "baseline_head_acc_g": 8})
    result = parse_acceleration_csv(df.to_csv(index=False).encode())
    assert result.rl_hic15 < result.baseline_hic15
    assert result.reduction_pct == pytest.approx((1-(4/8)**2.5)*100)


def test_reject_invalid_acceleration_and_time():
    with pytest.raises(ValueError):
        running_hic15(np.array([0,0,1]), np.ones(3))
    with pytest.raises(ValueError):
        running_hic15(np.array([0,1,2]), np.array([1,-1,1]))
