"""Integration checks for 360-degree selection and combined result sources."""
from pathlib import Path
import sys
import pandas as pd
import pytest
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from collision_view import collision_class,collision_figure
from injury_risk import nhtsa_skull_fracture_risk_pct
from scenario_statistics import paired_t_test

@pytest.mark.parametrize('angle,label',[(0,'전방'),(18,'전방'),(45,'사선'),(70,'측면'),
                                        (90,'측면'),(180,'후방/측후방'),(270,'측면'),(330,'사선')])
def test_angle_categories(angle,label):
    assert collision_class(angle)==label

@pytest.mark.parametrize('angle',[0,45,90,180,270,355])
def test_dial_has_two_cars_and_playable_frames(angle):
    f=collision_figure(angle)
    assert len(f.frames)==13
    assert len(f.data)>=13
    assert all(len(frame.traces)==6 for frame in f.frames)

@pytest.mark.parametrize('key,means,expected',[
    ('Oblique',(623.0,496.5),0.028367),
    ('Light 60kg',(575.0,470.0),0.056627),
    ('Heavy 90kg',(685.0,530.0),0.024290),
])
def test_preserved_high_variance_generated_dataset(key,means,expected):
    csv=ROOT/'data'/'demo_pairs_100kmh_high_variance.csv'
    df=pd.read_csv(csv)
    df=df.loc[df.scenario_key.eq(key)]
    assert len(df)==100
    assert df.rule_HIC15.mean()==pytest.approx(means[0],abs=1e-5)
    assert df.RL_HIC15.mean()==pytest.approx(means[1],abs=1e-5)
    paired=df.rename(columns={'rule_HIC15':'baseline_hic15','RL_HIC15':'rl_hic15'})
    assert paired_t_test(paired)['p_one']==pytest.approx(expected,abs=2e-6)
    assert paired_t_test(paired,risk=True)['p_one']>0


def test_risk_monotonic():
    assert 0<nhtsa_skull_fracture_risk_pct(200)<nhtsa_skull_fracture_risk_pct(700)<100


def test_uploaded_assets_complete():
    videos=list((ROOT/'assets').rglob('*.mp4'))
    assert len(videos)==7
    assert all(p.stat().st_size>200_000 for p in videos)


def test_entrypoint_and_batch_line_endings():
    source=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'collision_figure(st.session_state.impact_angle)' in source
    assert 'st.slider' in source and 'box_plot' in source and 'risk_curve' in source
    bat=(ROOT/'run_dashboard.bat').read_bytes()
    assert b'\r\n' in bat and bat.replace(b'\r\n',b'').find(b'\n')==-1
