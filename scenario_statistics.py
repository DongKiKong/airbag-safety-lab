"""Transparent example-data generation and valid paired inference.

Crucial provenance distinctions:
  * historical sample summary does not supply any case-level pairs/variances;
  * 100/140 km/h example data are explicitly synthetic, not evidence of RL performance;
  * paired tests are permitted only when individual paired HIC values exist.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from scipy import stats

from injury_risk import nhtsa_skull_fracture_risk_pct

SPEEDS = (60, 100, 140)
MASSES = (60, 90)
DIRECTIONS = ("전방", "사선")
SPEED_FACTORS = {100: 2.44, 140: 3.50}  # Presentation-only assumptions, NOT physics laws.


@dataclass(frozen=True)
class Aggregate:
    baseline: float
    rl: float
    label: str
    source_detail: str
    has_paired_raw: bool = False


def historical_aggregate(measured: dict, direction: str, mass: int) -> Aggregate:
    if direction == "사선":
        a = measured["scenario_1"]
        return Aggregate(float(a["baseline"]["mean"]), float(a["rl"]["mean"]),
                         "원본 시뮬레이션 집계", "사선 50쌍 · 체중별 미분리 · 개별 충돌속도 상이")
    group = measured["scenario_2"]["groups"][f"{mass}kg"]
    return Aggregate(float(group["baseline"]["mean"]), float(group["rl"]["mean"]),
                     "원본 시뮬레이션 집계", f"전방 · {mass}kg 집계 · 원래 충돌조건 80 km/h")


def scenario_aggregate(measured: dict, direction: str, mass: int, speed: int) -> Aggregate:
    if direction not in DIRECTIONS or mass not in MASSES or speed not in SPEEDS:
        raise ValueError("충돌 방향·체중·속도 선택값이 유효하지 않습니다.")
    historical = historical_aggregate(measured, direction, mass)
    if speed == 60:
        return historical
    factor = SPEED_FACTORS[speed]
    return Aggregate(round(historical.baseline * factor, 1), round(historical.rl * factor, 1),
                     "가상 데이터셋 · 속도 가정", f"{speed} km/h 가상 배율 ×{factor:g} · 물리 검증 없음")


def generate_synthetic_pairs(measured: dict, direction: str, mass: int,
                             speed: int, difference_sd: float = 90., n: int = 50) -> pd.DataFrame:
    """Seeded synthetic paired runs with assumed mean levels and sample spread.

    Exact target means are obtained by centering random samples; the variance is
    deliberately a tunable *assumption*, not estimated from unavailable raw data.
    """
    if speed == 60:
        raise ValueError("원본 60 슬롯은 실험별 쌍 데이터 없이 생성값을 실측값으로 대체할 수 없습니다.")
    if n < 3 or not 10 <= difference_sd <= 500:
        raise ValueError("n >= 3 및 유효한 표준편차 가정이 필요합니다.")
    target = scenario_aggregate(measured, direction, mass, speed)
    seed_data = f"{speed}|{mass}|{direction}|{difference_sd:g}|{n}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_data).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    # Bounded, centered draws keep high-spread examples in physical HIC>=0 range.
    # The assumed samples are not an estimate of an observed distribution.
    common = rng.uniform(-1, 1, size=n)
    common -= common.mean()
    common = common / common.std(ddof=1) * (difference_sd * 0.08)
    diff_variation = rng.normal(0, 1, size=n)
    diff_variation = np.clip(diff_variation, -1.25, 1.25)
    diff_variation -= diff_variation.mean()
    diff_variation = diff_variation / diff_variation.std(ddof=1) * difference_sd
    # Control the paired difference directly: baseline-RL has assumed s.d.
    baselines = target.baseline + common
    rl = baselines - (target.baseline - target.rl) - diff_variation
    if min(baselines.min(), rl.min()) <= 0:
        # Synthetic HIC must remain in its nonnegative physical support.
        # Boundedness is checked instead of silently clipping (which shifts means).
        raise ValueError("변동성 가정이 너무 커서 음수 HIC15가 생성됩니다. 편차를 낮추세요.")
    return pd.DataFrame({"pair_id": np.arange(1, n + 1),
                         "baseline_hic15": np.round(baselines, 5),
                         "rl_hic15": np.round(rl, 5)})


def paired_t_test(df: pd.DataFrame, *, risk: bool = False) -> dict[str, float | int | bool | str]:
    """Paired t-test, 95% two-sided CI and one-sided superiority test.

    Positive difference = baseline larger than RL = HIC/risk improvement.
    """
    needed = {"baseline_hic15", "rl_hic15"}
    if not needed.issubset(df.columns):
        raise ValueError("짝지어진 baseline_hic15, rl_hic15 두 열이 필요합니다.")
    b = pd.to_numeric(df["baseline_hic15"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(df["rl_hic15"], errors="coerce").to_numpy(dtype=float)
    if b.size != r.size or b.size < 3 or not np.isfinite(b).all() or not np.isfinite(r).all():
        raise ValueError("세 쌍 이상의 유효한 실험값이 필요합니다.")
    if (b < 0).any() or (r < 0).any():
        raise ValueError("HIC15는 음수일 수 없습니다.")
    if risk:
        b = np.asarray(nhtsa_skull_fracture_risk_pct(b))
        r = np.asarray(nhtsa_skull_fracture_risk_pct(r))
    differences = b - r
    n = len(differences)
    mean_diff = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    if sd < 1e-12:
        raise ValueError("모든 차이가 같아 t검정의 차이 표준편차가 0입니다.")
    se = sd / np.sqrt(n)
    t_stat = mean_diff / se
    p_two = float(stats.t.sf(abs(t_stat), n - 1) * 2.0)
    p_one = float(stats.t.sf(t_stat, n - 1))  # alternative: mean baseline-RL > 0
    quantile = float(stats.t.ppf(0.975, n - 1))
    ci_low, ci_high = mean_diff - quantile * se, mean_diff + quantile * se
    improved = bool(p_one < 0.05 and mean_diff > 0)
    return {
        "n": n, "baseline_mean": float(np.mean(b)), "rl_mean": float(np.mean(r)),
        "mean_difference": mean_diff, "sd_difference": sd, "t": float(t_stat),
        "df": n - 1, "p_two": p_two, "p_one": p_one, "ci_low": ci_low,
        "ci_high": ci_high, "cohens_dz": float(mean_diff / sd),
        "improvement_supported": improved,
    }


def read_matched_csv(source) -> pd.DataFrame:
    """Load one-row-per-pair real HIC data; optional fields select the case.

    columns: speed_kmh, mass_kg, direction, baseline_hic15, rl_hic15, pair_id
    direction can be Korean or English. One row is one matched *condition*.
    """
    df = pd.read_csv(source, encoding="utf-8-sig")
    required = {"speed_kmh", "mass_kg", "direction", "baseline_hic15", "rl_hic15"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("누락된 CSV 열: " + ", ".join(sorted(missing)))
    for col in ("speed_kmh", "mass_kg", "baseline_hic15", "rl_hic15"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["direction"] = df["direction"].astype(str).str.strip().replace(
        {"front": "전방", "frontal": "전방", "oblique": "사선", "전방충돌": "전방", "사선충돌": "사선"})
    if df[list(required)].isna().any().any():
        raise ValueError("CSV에 숫자가 아닌 값이나 공백이 있습니다.")
    if (~df["direction"].isin(DIRECTIONS)).any():
        raise ValueError("direction은 전방/front 또는 사선/oblique여야 합니다.")
    if (df[["baseline_hic15", "rl_hic15"]] < 0).any().any():
        raise ValueError("음수 HIC15가 입력되었습니다.")
    if "pair_id" in df and df.duplicated(["speed_kmh", "mass_kg", "direction", "pair_id"]).any():
        raise ValueError("동일 시나리오 내 pair_id가 중복되었습니다.")
    return df


def filter_pairs(df: pd.DataFrame, speed: int, mass: int, direction: str) -> pd.DataFrame:
    return df.loc[(df["speed_kmh"] == speed) & (df["mass_kg"] == mass) &
                  (df["direction"] == direction), ["baseline_hic15", "rl_hic15"]].reset_index(drop=True)
