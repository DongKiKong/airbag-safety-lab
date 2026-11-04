import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from measured_results import load_measured
from scenario_statistics import (SPEEDS, MASSES, DIRECTIONS, scenario_aggregate,
                                 generate_synthetic_pairs, paired_t_test,
                                 read_matched_csv, filter_pairs)

M = load_measured(Path(__file__).resolve().parents[1] / 'data/measured_scenarios.json')

@pytest.mark.parametrize('direction,mass,base,rl',[
    ('사선',60,245.6,224.7),('사선',90,245.6,224.7),
    ('전방',60,232.9,218.5),('전방',90,269.7,243.0)
])
def test_anchor_measured_numbers(direction,mass,base,rl):
    a = scenario_aggregate(M,direction,mass,60)
    assert a.baseline == base and a.rl == rl

@pytest.mark.parametrize('s', [100,140])
@pytest.mark.parametrize('m', MASSES)
@pytest.mark.parametrize('d', DIRECTIONS)
def test_synthetic_complete_and_deterministic(d,m,s):
    a = generate_synthetic_pairs(M,d,m,s)
    b = generate_synthetic_pairs(M,d,m,s)
    assert a.equals(b)
    assert len(a) == 50
    target = scenario_aggregate(M,d,m,s)
    assert a.baseline_hic15.mean() == pytest.approx(target.baseline,abs=1e-4)
    assert a.rl_hic15.mean() == pytest.approx(target.rl,abs=1e-4)
    assert (a[['baseline_hic15','rl_hic15']]>=0).all().all()
    assert paired_t_test(a)['n']==50
    assert paired_t_test(a,risk=True)['n']==50

@pytest.mark.parametrize('s', [100,140])
@pytest.mark.parametrize('m', MASSES)
@pytest.mark.parametrize('d', DIRECTIONS)
def test_large_spread_is_still_valid(d,m,s):
    df=generate_synthetic_pairs(M,d,m,s,difference_sd=260)
    assert (df[['baseline_hic15','rl_hic15']]>0).all().all()


def test_no_fake_pairs_for_real_aggregate():
    with pytest.raises(ValueError): generate_synthetic_pairs(M,'사선',60,60)


def test_paired_ci_and_pvalue_not_imply_equality():
    data=pd.DataFrame({'baseline_hic15':[600,580,620,590,610],
                       'rl_hic15':[560,570,565,555,585]})
    out=paired_t_test(data)
    assert out['mean_difference']>0
    assert out['ci_low']<out['mean_difference']<out['ci_high']
    assert out['p_one']<out['p_two']


def test_csv_pair_filter_and_invalid():
    obj=pd.DataFrame([{'speed_kmh':100,'mass_kg':90,'direction':'oblique',
                       'pair_id':x,'baseline_hic15':600+x,'rl_hic15':555+x}
                      for x in range(4)])
    raw=obj.to_csv(index=False).encode('utf-8-sig')
    got=read_matched_csv(io.BytesIO(raw))
    assert len(filter_pairs(got,100,90,'사선'))==4
    assert len(filter_pairs(got,140,90,'사선'))==0
    obj.loc[1,'pair_id']=0
    with pytest.raises(ValueError):read_matched_csv(io.BytesIO(obj.to_csv(index=False).encode()))


def test_source_distinction_and_unwanted_sections_removed():
    app=(Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8')
    for banned in ('원본 영상 각각 재생','시뮬레이션을 새로 돌리는 입력 화면',
                   '최악 사례 유의','계산 유의: 각 집단','시간별 Running HIC15 분석',
                   '실험 수치 원본 슬라이드 보기'):
        assert banned not in app
    assert '가상' in app and '실제 p값을 계산할 수 없습니다' in app
