"""Pure numeric functions for reproducible HIC15 calculations; no UI dependencies."""
from __future__ import annotations

import io
from typing import NamedTuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("t_ms", "rl_head_acc_g", "baseline_head_acc_g")


class Result(NamedTuple):
    t_ms: np.ndarray
    rl_g: np.ndarray
    baseline_g: np.ndarray
    rl_running: np.ndarray
    baseline_running: np.ndarray
    rl_hic15: float
    baseline_hic15: float
    reduction_pct: float | None


def running_hic15(t_ms: np.ndarray, acc_g: np.ndarray) -> np.ndarray:
    """Maximize (Δt) * mean(a)^2.5 for all sampled endpoints with Δt <= 15 ms.

    Δt is seconds, a is head resultant acceleration in multiples of g.
    Samples are linearly interpolated by trapezoidal numerical integration.
    The continuous-time theoretical maximum may lie between sampled endpoints.
    """
    t = np.asarray(t_ms, dtype=float) / 1000.0
    a = np.asarray(acc_g, dtype=float)
    if t.ndim != 1 or len(t) != len(a) or len(t) < 2:
        raise ValueError("시간과 가속도 배열에는 동일한 길이의 샘플이 2개 이상 있어야 합니다.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(a)):
        raise ValueError("NaN 또는 무한대 값이 포함되어 있습니다.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t_ms는 중복 없이 엄격하게 증가해야 합니다.")
    if np.any(a < 0):
        raise ValueError("합성가속도(g)는 음수가 될 수 없습니다. x/y/z 성분은 합성가속도로 변환하세요.")
    if len(t) > 20000:
        raise ValueError("20,000개 이하의 시간 샘플만 지원합니다.")

    area = np.r_[0.0, np.cumsum(0.5 * (a[:-1] + a[1:]) * np.diff(t))]
    max_so_far = 0.0
    running = np.zeros(len(t))
    left = 0
    for j in range(1, len(t)):
        while left < j and (t[j] - t[left] > 0.015 + 1e-12):
            left += 1
        # Include all window sizes up to 15 ms, not only exactly 15 ms.
        i = np.arange(left, j)
        if len(i):
            duration = t[j] - t[i]
            mean_a = (area[j] - area[i]) / duration
            candidates = duration * np.power(np.maximum(mean_a, 0.0), 2.5)
            max_so_far = max(max_so_far, float(np.max(candidates)))
        running[j] = max_so_far
    return running


def parse_acceleration_csv(csv_bytes: bytes) -> Result:
    if len(csv_bytes) > 5_000_000:
        raise ValueError("CSV는 5 MB 이하여야 합니다.")
    try:
        frame = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"CSV를 읽지 못했습니다: {exc}") from exc
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing))
    try:
        arrays = [pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float) for col in REQUIRED_COLUMNS]
        t, rl, baseline = arrays
        rl_running = running_hic15(t, rl)
        baseline_running = running_hic15(t, baseline)
    except (ValueError, TypeError) as exc:
        raise ValueError(str(exc)) from exc
    b, r = float(baseline_running[-1]), float(rl_running[-1])
    return Result(t, rl, baseline, rl_running, baseline_running, r, b, ((b-r)/b*100 if b > 0 else None))


def sample_csv() -> bytes:
    """Clearly labelled synthetic signal example for checking CSV import only."""
    t = np.arange(0.0, 150.1, 1.0)
    baseline = 48 * np.exp(-0.5 * ((t-65) / 11)**2)
    rl = 30 * np.exp(-0.5 * ((t-70) / 13)**2)
    frame = pd.DataFrame({"t_ms": t, "rl_head_acc_g": rl, "baseline_head_acc_g": baseline})
    return frame.to_csv(index=False).encode("utf-8-sig")


def calibrated_demo_acceleration_csv(baseline_hic: float, rl_hic: float) -> bytes:
    """ILLUSTRATION ONLY: artificial Gaussian pulses calibrated to final HIC15 means.

    Deliberately not a reconstructed/observed head-acceleration history.
    HIC is homogeneous of order 2.5 in acceleration. Scaling by
    (target / initial_HIC)**0.4 makes the *computed* HIC reach the displayed
    illustrative target, without claiming physical simulation realism.
    """
    b_target, r_target = float(baseline_hic), float(rl_hic)
    if not (np.isfinite(b_target) and np.isfinite(r_target)) or b_target < 0 or r_target < 0:
        raise ValueError("예시 HIC15 목표값이 올바르지 않습니다.")
    t = np.arange(0.0, 121.0, 1.0)  # ms; for responsive demo charts
    # Different shape/timing to illustrate that the policies may change trajectories.
    b = (np.exp(-.5*((t-58)/10.5)**2) + .23*np.exp(-.5*((t-80)/7.0)**2))
    r = (.85*np.exp(-.5*((t-62)/11.4)**2) + .13*np.exp(-.5*((t-83)/8.5)**2))
    def scale(signal: np.ndarray, target: float) -> np.ndarray:
        h = float(running_hic15(t,signal)[-1])
        return signal*(target/h)**0.4 if target>0 else np.zeros_like(signal)
    df = pd.DataFrame({'t_ms':t,
                       'rl_head_acc_g':scale(r,r_target),
                       'baseline_head_acc_g':scale(b,b_target)})
    return df.to_csv(index=False).encode('utf-8-sig')


def trajectory_frame(result: Result) -> pd.DataFrame:
    """All three stages at each sampled time, risk based on RUNNING max HIC.

    The calculated probability is an NHTSA fitted risk-curve ordinate, not
    the probability that a skull fractures *at this instant*.
    """
    from injury_risk import nhtsa_skull_fracture_risk_pct
    return pd.DataFrame({
        't_ms':result.t_ms,
        'baseline_head_acc_g':result.baseline_g,
        'rl_head_acc_g':result.rl_g,
        'baseline_running_hic15':result.baseline_running,
        'rl_running_hic15':result.rl_running,
        'baseline_skull_fracture_risk_pct':nhtsa_skull_fracture_risk_pct(result.baseline_running),
        'rl_skull_fracture_risk_pct':nhtsa_skull_fracture_risk_pct(result.rl_running),
    })
