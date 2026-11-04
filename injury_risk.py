"""NHTSA (1999) Chapter 2, Figure 2-3 skull-fracture risk from HIC.

Source: https://www.nhtsa.gov/sites/nhtsa.gov/files/rev_criteria.pdf
Equation on printed page 2-10: P(AIS>=2 skull fracture) = Phi((ln HIC - 6.96352)/0.84664).
This is an adult-reference statistical model, not a personal injury diagnosis.
"""
from __future__ import annotations

import math
import numpy as np

NHTSA_SOURCE = "https://www.nhtsa.gov/sites/nhtsa.gov/files/rev_criteria.pdf"
MEAN_LN_HIC = 6.96352
SD_LN_HIC = 0.84664
ADULT_REFERENCE_HIC15 = 700.0
# Original human-surrogate data ranged from HIC 175 to 3400 (NHTSA 1999, printed p. 2-9).
SOURCE_DATA_MIN_HIC = 175.0
SOURCE_DATA_MAX_HIC = 3400.0


def nhtsa_skull_fracture_risk_pct(hic: float | np.ndarray) -> float | np.ndarray:
    """Return estimated P(AIS2+ skull fracture) (%) as a monotone function of HIC.

    For HIC below 175, this evaluates an extrapolated fitted curve; warn in the UI.
    Zero HIC maps to zero probability by continuity. Arrays accepted for charting.
    """
    values = np.asarray(hic, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("HIC 값은 음수가 아닌 유한한 숫자여야 합니다.")
    safe = np.maximum(values, np.finfo(float).tiny)
    z = (np.log(safe) - MEAN_LN_HIC) / (SD_LN_HIC * math.sqrt(2.0))
    # math.erf is in Python's standard library; numpy does not expose np.erf.
    flat = z.ravel()
    probabilities = np.fromiter((50.0 * (1.0 + math.erf(float(x))) for x in flat), dtype=float, count=flat.size)
    probabilities = probabilities.reshape(values.shape)
    probabilities = np.where(values == 0, 0.0, probabilities)
    if values.ndim == 0:
        return float(probabilities)
    return probabilities


def reference_risk_rows() -> list[dict[str, object]]:
    """Reference values all computed from the source formula, not guessed bands."""
    points = [100, 200, 300, 400, 500, 600, 700, 800, 1000, 1400]
    return [{"HIC15": point,
             "AIS 2+ 두개골 골절 추정 위험": f"{nhtsa_skull_fracture_risk_pct(point):.1f}%",
             "비고": "자료범위 외 추정" if point < SOURCE_DATA_MIN_HIC else ("성인 참고 기준 700" if point == 700 else "")}
            for point in points]
