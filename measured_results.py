"""User-reported simulation aggregates and transparent HIC-to-risk transforms."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from injury_risk import nhtsa_skull_fracture_risk_pct


def load_measured(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def compute_comparison(baseline_mean: float, rl_mean: float) -> dict[str, float]:
    """Transforms aggregated mean HIC, not an empirical mean of injury probabilities."""
    if baseline_mean <= 0 or rl_mean < 0:
        raise ValueError("HIC15 평균은 기본값 > 0, 정책값 >= 0이어야 합니다.")
    p_baseline = float(nhtsa_skull_fracture_risk_pct(baseline_mean))
    p_rl = float(nhtsa_skull_fracture_risk_pct(rl_mean))
    return {
        "baseline_mean": baseline_mean, "rl_mean": rl_mean,
        "delta_hic": rl_mean - baseline_mean,
        "reduction_pct": (baseline_mean - rl_mean) / baseline_mean * 100.0,
        "baseline_risk_pct": p_baseline, "rl_risk_pct": p_rl,
        "risk_delta_pp": p_rl - p_baseline,
    }


def summarize_measured(data: dict[str, Any]) -> pd.DataFrame:
    s1 = data["scenario_1"]
    rows = [("사선충돌 · 50쌍", s1["baseline"]["mean"], s1["rl"]["mean"])]
    for key, item in data["scenario_2"]["groups"].items():
        rows.append((f"전방충돌 · {key}", item["baseline"]["mean"], item["rl"]["mean"]))
    results=[]
    for name, b, rl in rows:
        c=compute_comparison(float(b),float(rl))
        results.append({"조건":name,"기존 HIC15 평균":b,"RL HIC15 평균":rl,
                        "HIC15 감소율 (%)":round(c["reduction_pct"],3),
                        "기존 평균 HIC 대응 위험 (%)":round(c["baseline_risk_pct"],4),
                        "RL 평균 HIC 대응 위험 (%)":round(c["rl_risk_pct"],4),
                        "위험 변화 (%p)":round(c["risk_delta_pp"],4)})
    return pd.DataFrame(results)
