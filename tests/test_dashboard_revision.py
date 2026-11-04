from pathlib import Path
import json
import pytest

from measured_results import load_measured, compute_comparison, summarize_measured

ROOT=Path(__file__).resolve().parents[1]
DATA=load_measured(ROOT/'data/measured_scenarios.json')

@pytest.mark.parametrize('condition,b,rl,expected',[
    ('사선충돌',245.6,224.7,8.510),
    ('60kg',232.9,218.5,6.183),
    ('90kg',269.7,243.0,9.900),
])
def test_three_reliable_aggregates(condition,b,rl,expected):
    c=compute_comparison(b,rl)
    assert c['reduction_pct']==pytest.approx(expected,abs=.001)
    assert c['baseline_risk_pct']>c['rl_risk_pct']
    assert c['risk_delta_pp']<0


def test_oblique_source_summary_and_worst_case_regression():
    obj=DATA['scenario_1']
    assert obj['sample_size']==50
    assert obj['baseline']=={'mean':245.6,'median':245.0,'min':222.0,'max':273.0}
    assert obj['rl']=={'mean':224.7,'median':223.0,'min':200.0,'max':278.0}
    assert obj['rl']['max']>obj['baseline']['max']


def test_weight_actions_and_hic():
    groups=DATA['scenario_2']['groups']
    assert groups['60kg']['deployment_timing_mean_ms']==9.8
    assert groups['90kg']['deployment_timing_mean_ms']==20.3
    assert groups['60kg']['deployment_pressure_mean_kpa']==115.8
    assert groups['90kg']['deployment_pressure_mean_kpa']==184.5
    assert groups['60kg']['rl']['mean']==218.5
    assert groups['90kg']['baseline']['median']==271.0


def test_reference_transform_is_not_empirical_average_risk():
    report=summarize_measured(DATA)
    assert len(report)==3
    assert set(report['조건'])=={'사선충돌 · 50쌍','전방충돌 · 60kg','전방충돌 · 90kg'}
    for item in report['HIC15 감소율 (%)']:
        assert 0<item<12


def test_old_unused_sections_removed_and_time_series_not_synthetic():
    text=(ROOT/'app.py').read_text(encoding='utf-8')
    for removed in ('에어백 정책 결정 · 영상 관찰', '원본 파일 매핑 및 해석 범위',
                    '머리 가속도 CSV 연결 방법', '충돌 후 시점 선택 (ms)', 'sample_csv('):
        assert removed not in text
    assert 'risk_curve' in text
    assert 'statistical_panel' in text


def test_uploaded_source_screenshots_packaged():
    for x in (DATA['scenario_1']['source_image'],DATA['scenario_2']['source_image']):
        assert (ROOT/'docs'/x).stat().st_size>100_000
