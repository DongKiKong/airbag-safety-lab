import numpy as np
import pytest

from injury_risk import nhtsa_skull_fracture_risk_pct, reference_risk_rows


@pytest.mark.parametrize('hic, percent', [
    (100, 0.27), (200, 2.46), (400, 12.55), (600, 25.17),
    (700, 31.31), (800, 37.09), (1000, 47.37), (1400, 62.99),
])
def test_nhtsa_reference_curve_matches_report(hic, percent):
    assert nhtsa_skull_fracture_risk_pct(hic) == pytest.approx(percent, abs=0.015)


def test_nhtsa_risk_vectorized_monotone_and_zero():
    x = np.linspace(0, 3400, 1000)
    y = nhtsa_skull_fracture_risk_pct(x)
    assert y.shape == x.shape
    assert y[0] == 0
    assert np.all(np.diff(y) >= 0)
    assert 0 <= y.min() <= y.max() <= 100


def test_nhtsa_invalid_input():
    for value in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            nhtsa_skull_fracture_risk_pct(value)


def test_reference_data_are_all_formula_derived():
    for row in reference_risk_rows():
        hic = row['HIC15']
        percent = float(row['AIS 2+ 두개골 골절 추정 위험'].rstrip('%'))
        assert percent == pytest.approx(nhtsa_skull_fracture_risk_pct(hic), abs=0.06)


