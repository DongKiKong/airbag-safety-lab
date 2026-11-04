"""Metadata consistency checks requiring no Streamlit installation."""
import json
from pathlib import Path


def test_video_manifest_paths():
    root = Path(__file__).resolve().parents[1]
    obj = json.loads((root / "data/scenario_manifest.json").read_text(encoding="utf-8"))
    assert set(obj) == {"front", "oblique"}
    for info in obj.values():
        keys = ["rl_video", "baseline_video", "comparison_video"]
        if "swapped_comparison_video" in info:
            keys += ["swapped_comparison_video"]
        for key in keys:
            filename = root / "assets" / info[key]
            assert filename.exists() and filename.stat().st_size > 1000, filename
