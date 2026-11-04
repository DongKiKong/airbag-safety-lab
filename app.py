"""Airbag Safety Lab | 사고 입력형 UI + 상세 결과분석 통합판.

- Two-car interactive 360° angle explorer and approach animation are ILLUSTRATIONS.
- Measured aggregates and synthetic 100 km/h paired samples are kept distinct.
- Never infer statistical significance from unpaired aggregate means.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from collision_view import collision_class, collision_figure
from analysis_utils import (parse_acceleration_csv, calibrated_demo_acceleration_csv,
                            trajectory_frame)
from injury_risk import ADULT_REFERENCE_HIC15, NHTSA_SOURCE, nhtsa_skull_fracture_risk_pct
from measured_results import compute_comparison, load_measured, summarize_measured
from scenario_statistics import paired_t_test, read_matched_csv, filter_pairs

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
DATA = ROOT / 'data'
ORANGE = '#EA8D48'
BLUE = '#2563EB'
NAVY = '#17324F'
GREEN = '#0C946D'
LIGHT = '#F4F8FD'
st.set_page_config(page_title='Airbag Safety Lab | 에어백 제어 분석', layout='wide', initial_sidebar_state='expanded')


def styles():
    st.markdown('''<style>
    .stApp {background: #FCFDFE;}
    .block-container{max-width:1430px;padding-top:1.35rem;padding-bottom:2.5rem;}
    [data-testid="stSidebar"]{background:#F8FAFD;border-right:1px solid #E4EBF4;}
    h1,h2,h3{color:#17324F;letter-spacing:-0.03em;}
    h1{font-weight:800; font-size:clamp(1.85rem,3.0vw,2.65rem);}
    .small-kicker{font-size:.77rem;color:#2867C0;font-weight:800;letter-spacing:.10em;}
    .lead{font-size:.97rem;color:#536780;line-height:1.7;margin-bottom:1.2rem;}
    .soft-panel{border:1px solid #DDE8F5;border-radius:14px;background:#F4F8FD;padding:12px 15px;}
    .metric-panel{border:1px solid #E2EAF4;background:white;border-radius:15px;padding:18px 18px 14px;min-height:114px;}
    .metric-label{color:#6B7F95;font-size:.82rem;font-weight:650;}
    .metric-val{color:#17324F;font-weight:800;font-size:1.85rem;line-height:1.36;letter-spacing:-.04em;}
    .metric-sub{color:#7B8EA3;font-size:.77rem;margin-top:.22rem;}
    .green{color:#0C946D}.blue{color:#2563EB}.orange{color:#CF7828}
    div[data-testid="stMetric"]{border:1px solid #E3EBF5;border-radius:12px;padding:12px;background:#FFF;}
    div[data-testid="stPlotlyChart"]{border:1px solid #E7EDF6;border-radius:14px;background:#FFF;padding:5px;}
    .flow-step{color:#56728D; font-weight:700; font-size:.82rem; padding:9px 10px;
        border-radius:10px; border:1px solid #DCE7F3; background:white; text-align:center;}
    @media (max-width:820px){.block-container{padding-left:1rem;padding-right:1rem;}
        .metric-val{font-size:1.42rem;} h1{font-size:1.85rem;}}
    .stButton>button{border-radius:10px;}
    .pill{display:inline-block;border:1px solid #D8E5F4;border-radius:999px;padding:5px 10px;font-weight:700;font-size:.8rem;color:#315678;background:#F5F9FF;}
    </style>''',unsafe_allow_html=True)


def source_note(text:str):
    st.caption('자료 구분 · '+text)


def metric_card(title,value,sub='',color=''):
    st.markdown(f'<div class="metric-panel"><div class="metric-label">{html.escape(str(title))}</div>'
                f'<div class="metric-val {color}">{html.escape(str(value))}</div>'
                f'<div class="metric-sub">{html.escape(str(sub))}</div></div>',unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def measured():
    return load_measured(DATA/'measured_scenarios.json')


@st.cache_data(show_spinner=False)
def high_variance_pairs():
    return pd.read_csv(DATA/'demo_pairs_100kmh_high_variance.csv')


def demo_pairs(direction:str,mass:int,speed:int)->pd.DataFrame:
    """Pre-generated illustrative 100 matched samples; 140 is an explicitly scaled illustration."""
    df=high_variance_pairs()
    scenario='Oblique' if direction=='사선' else ('Light 60kg' if mass==60 else 'Heavy 90kg')
    out=df.loc[df['scenario_key'].eq(scenario),['pair_id','rule_HIC15','RL_HIC15']].copy()
    out=out.rename(columns={'rule_HIC15':'baseline_hic15','RL_HIC15':'rl_hic15'}).reset_index(drop=True)
    if speed==140:
        # Not inferred vehicle physics; demonstration-only scaling of 100 km/h examples.
        out['baseline_hic15']*=1.50
        out['rl_hic15']*=1.36
    return out


def classified_direction(angle):
    return collision_class(angle)


def lookup_aggregate(direction,mass,speed):
    data=measured()
    if direction=='사선':
        a=data['scenario_1']
        base,rl=float(a['baseline']['mean']),float(a['rl']['mean'])
        speed_note='사선 원본 50쌍은 여러 충돌속도·각도를 포함하며, 체중별로 분리돼 있지 않습니다.'
        policy='사선'
    else:
        group=data['scenario_2']['groups'][f'{mass}kg']
        base,rl=float(group['baseline']['mean']),float(group['rl']['mean'])
        speed_note='정면 원본 수치는 80 km/h 실험 집계입니다. 60 km/h 측정값으로 해석하지 마세요.'
        policy='정면'
    if speed==60:
        return dict(base=base,rl=rl,pairs=None,mode='집계',source='원본 시뮬레이션 집계',note=speed_note,policy=policy)
    pairs=demo_pairs(direction,mass,speed)
    return dict(base=float(pairs.baseline_hic15.mean()),rl=float(pairs.rl_hic15.mean()),
                pairs=pairs,mode='합성',source=f'{speed} km/h 고편차 생성 표본 100쌍',
                note='속도 확대·표준편차를 가정한 설명용 표본입니다. 실제 물리해석·차량시험 측정값이 아닙니다.',policy=policy)


def base_figure(height=370):
    fig=go.Figure()
    fig.update_layout(height=height,margin=dict(l=10,r=12,t=37,b=12),
                      paper_bgcolor='white',plot_bgcolor='white',
                      font=dict(family='Arial, Malgun Gothic, sans-serif',size=12,color='#314761'),
                      legend=dict(orientation='h',x=0,y=1.17),hovermode='closest')
    fig.update_yaxes(gridcolor='#ECF1F7',zeroline=False)
    fig.update_xaxes(gridcolor='#ECF1F7',zeroline=False)
    return fig


def hic_bar(base,rl):
    fig=base_figure(350)
    fig.add_bar(x=['Rule-Based','Curriculum PPO'],y=[base,rl],marker_color=[ORANGE,BLUE],
                text=[f'{base:.1f}',f'{rl:.1f}'],textposition='outside',width=.53,
                hovertemplate='%{x}<br>평균 HIC15 %{y:.2f}<extra></extra>')
    fig.update_layout(showlegend=False,title=dict(text='평균 HIC15 비교',font=dict(size=16)))
    fig.update_yaxes(title='HIC15',range=[0,max(base,rl)*1.23])
    return fig


def risk_curve(base,rl):
    fig=base_figure(375)
    x=np.linspace(1,max(1400,base*1.5),700)
    y=nhtsa_skull_fracture_risk_pct(x)
    fig.add_scatter(x=x,y=y,mode='lines',name='NHTSA AIS 2+ 위험곡선',
                    line=dict(color='#67869D',width=3),
                    hovertemplate='HIC15 %{x:.1f}<br>두개골 골절 추정확률 %{y:.2f}%<extra></extra>')
    for value,name,color in [(base,'Rule-Based',ORANGE),(rl,'Curriculum PPO',BLUE)]:
        p=float(nhtsa_skull_fracture_risk_pct(value))
        fig.add_scatter(x=[value],y=[p],mode='markers+text',name=name,
                        marker=dict(size=14,color=color,line=dict(color='white',width=2)),
                        text=[f'{p:.1f}%'],textposition='top center',
                        hovertemplate=f'{name}<br>HIC15 %{{x:.1f}}<br>추정확률 %{{y:.2f}}%<extra></extra>')
    fig.add_vline(x=700,line_dash='dash',line_color='#E47876',annotation_text='HIC15 700',annotation_position='top right')
    fig.update_layout(title=dict(text='HIC15와 두개골 골절 추정위험',font=dict(size=16)))
    fig.update_xaxes(title='HIC15')
    fig.update_yaxes(title='AIS 2+ 골절 추정확률 (%)',range=[0,100])
    return fig


def box_plot(pairs,risk=False):
    fig=base_figure(385)
    for col,name,color in [('baseline_hic15','Rule-Based',ORANGE),('rl_hic15','Curriculum PPO',BLUE)]:
        vals=np.asarray(pairs[col],dtype=float)
        if risk: vals=np.asarray(nhtsa_skull_fracture_risk_pct(vals),dtype=float)
        fig.add_box(y=vals,name=name,marker=dict(color=color,opacity=.3,size=5),
                    line=dict(color=color,width=2),fillcolor=color.replace('#','rgba(') if False else color,
                    opacity=.75,boxpoints='all',jitter=.38,pointpos=0,
                    boxmean=True,hovertemplate='%{y:.2f}<extra>'+name+'</extra>')
    fig.update_layout(title=dict(text='개별 표본 분포 (Box Plot)',font=dict(size=16)),showlegend=False)
    fig.update_yaxes(title='골절 추정확률 (%)' if risk else 'HIC15')
    return fig


def paired_difference_chart(pairs):
    diffs=pairs.baseline_hic15.to_numpy()-pairs.rl_hic15.to_numpy()
    fig=base_figure(360)
    fig.add_histogram(x=diffs,nbinsx=21,marker_color=BLUE,opacity=.85,
                      hovertemplate='기존−RL 차이 %{x:.1f}<br>빈도 %{y}<extra></extra>')
    fig.add_vline(x=0,line_dash='dash',line_color='#D65C55')
    fig.add_vline(x=float(np.mean(diffs)),line_color=GREEN,line_width=2,
                  annotation_text=f'평균 차이 {np.mean(diffs):.1f}',annotation_position='top')
    fig.update_layout(title=dict(text='동일 조건별 HIC15 차이 분포',font=dict(size=16)))
    fig.update_xaxes(title='HIC15 차이 (기존 − RL) · 양수 = 개선')
    fig.update_yaxes(title='표본 수')
    return fig




# Same actual or explicitly simulated signal drives all three time-domain charts.
# Each plot is lightweight Plotly (no custom Streamlit components or external JS).
def _signal_plot(time_frame:pd.DataFrame, what:str)->go.Figure:
    options={
        'acc':('baseline_head_acc_g','rl_head_acc_g','머리 합성가속도 (g)',
               'STEP 01 · 시간별 머리 합성가속도'),
        'hic':('baseline_running_hic15','rl_running_hic15','Running HIC15',
               'STEP 02 · Running HIC15 (15ms 이내 최대값)'),
        'risk':('baseline_skull_fracture_risk_pct','rl_skull_fracture_risk_pct',
                '두개골 골절 추정위험 (%)',
                'STEP 03 · 누적 최대 HIC15 대응 골절 추정위험'),
    }
    bcol,rcol,ytitle,title=options[what]
    fig=base_figure(286)
    for col,label,color in [(bcol,'Rule-Based',ORANGE),(rcol,'Curriculum PPO',BLUE)]:
        vals=time_frame[col].to_numpy()
        fig.add_trace(go.Scatter(x=time_frame.t_ms,y=vals,mode='lines',name=label,
                                 line=dict(color=color,width=3),
                                 hovertemplate='충돌 후 %{x:.0f}ms<br>'+label+': %{y:.2f}<extra></extra>'))
    fig.update_layout(title=dict(text=title,font=dict(size=15)),
                      hovermode='x unified',height=286,
                      legend=dict(orientation='h',x=0,y=1.15))
    fig.update_xaxes(title='충돌 후 시간 (ms)',range=[float(time_frame.t_ms.min()),float(time_frame.t_ms.max())])
    fig.update_yaxes(title=ytitle, rangemode='tozero')
    if what=='risk':fig.update_yaxes(range=[0,max(8.,float(time_frame[[bcol,rcol]].to_numpy().max())*1.19)])
    return fig


def signal_to_injury_section(base:float,rl:float,scenario_label:str)->None:
    """Show a three-stage trace; user CSV takes precedence over demonstration only."""
    st.markdown('### 시간별 안전성 분석 · 머리 가속도 → HIC15 → 골절 추정확률')
    st.markdown('머리에 작용하는 **합성가속도**에서 15ms 이하 구간의 최대 HIC15를 계산한 다음, '
                'NHTSA의 AIS 2+ 두개골 골절 위험곡선으로 변환합니다.')
    x1,x2,x3=st.columns(3)
    for col,name in zip((x1,x2,x3),('① 머리 합성가속도 (g)','② Running HIC15','③ 골절 위험 대응값 (%)')):
        with col:st.markdown(f'<div class="flow-step">{name}</div>',unsafe_allow_html=True)
    upload_col,example_col=st.columns([1.28,1],gap='large')
    with upload_col:
        incoming=st.file_uploader('머리 가속도 CSV 연결 (선택)',type=['csv'],key='head_acc_signal',
            help='t_ms, baseline_head_acc_g, rl_head_acc_g 열 필요. 각도/체중/속도 자동검증은 수행하지 않습니다.')
    with example_col:
        st.caption('원자료가 없으면 아래 그래프는 평균 HIC15에 맞춘 **인공 예시 파형**을 사용합니다. '
                   '관찰된 충돌 가속도를 복원한 결과가 아닙니다.')
        show_example=st.toggle('원자료 미연결 시 예시 신호로 표시',value=True,key='show_demo_signals')
    using_uploaded=incoming is not None
    try:
        if using_uploaded:
            result=parse_acceleration_csv(incoming.getvalue())
            provenance='사용자가 업로드한 머리 가속도 CSV에서 직접 계산한 결과 (사고 조건은 자동 대조되지 않음)'
        elif show_example:
            result=parse_acceleration_csv(calibrated_demo_acceleration_csv(base,rl))
            provenance='가상 예시 가속도 파형 · 선택된 HIC 평균에 맞춰 인공적으로 보정한 신호 (실측 아님)'
        else:
            st.info('머리 합성가속도 CSV를 업로드하면 3단계 시간별 그래프를 계산해 표시합니다.')
            return
    except (ValueError, OSError) as exc:
        st.error(f'머리 가속도 자료를 읽을 수 없습니다: {exc}')
        return
    if not using_uploaded:
        st.warning('**시연용 가속도 그래프** — 실제 관측된 머리 가속도 이력이 없어 평균 HIC15 수치에 맞춘 '
                   '합성 파형을 생성했습니다. 그래프의 시간별 형태와 가속도 정점은 물리해석 결과가 아닙니다.')
    else:
        st.success('업로드한 시간–머리 가속도 데이터를 사용해 HIC15와 골절 추정위험을 계산했습니다.')
    sample=trajectory_frame(result)
    st.caption('데이터 출처 · '+provenance)
    # A compact three-metric summary is for THIS signal, not the aggregation shown above.
    m1,m2,m3=st.columns(3)
    m1.metric('이 신호의 HIC15 (기존 → RL)', f'{result.baseline_hic15:.1f} → {result.rl_hic15:.1f}')
    r0=float(sample.baseline_skull_fracture_risk_pct.iloc[-1])
    r1=float(sample.rl_skull_fracture_risk_pct.iloc[-1])
    m2.metric('이 신호의 골절 추정위험',f'{r0:.2f}% → {r1:.2f}%')
    m3.metric('골절 위험 대응값 차이',f'{r0-r1:+.2f}%p')
    st.caption('이 값은 바로 위 **시간 이력 신호에 대한 계산 결과**입니다. 업로드 CSV의 HIC15는 위쪽 실험 집계 평균과 다를 수 있습니다.')
    for label in ['acc','hic','risk']:
        st.plotly_chart(_signal_plot(sample,label),width='stretch',
                        config={'displayModeBar':False},key=f'trace_{label}')
    st.info('**해석:** ③의 그래프는 각 시간까지의 **최대 Running HIC15를 위험함수에 대입한 값**입니다. '
            '충돌 순간별 골절 발생 확률이나 시간별 실제 골절 빈도를 뜻하지 않습니다. '
            '골절 위험은 HIC15로부터 계산한 파생 지표입니다.')
    if (sample[['baseline_running_hic15','rl_running_hic15']] < 175).any().any():
        st.caption('참고: NHTSA 원곡선의 기초 HIC 자료는 약 175부터 시작합니다. '
                   '그 미만에서 표시되는 초기 시간대의 위험값은 외삽 결과입니다.')
    dc1,dc2=st.columns(2)
    with dc1:
        st.download_button('3단계 시간별 계산 CSV 다운로드',sample.to_csv(index=False).encode('utf-8-sig'),
                           file_name='airbag_acceleration_hic15_skull_risk.csv',mime='text/csv',
                           key='download_signal_result')
    with dc2:
        st.download_button('가속도 입력 형식 · 예시 CSV',calibrated_demo_acceleration_csv(base,rl),
                           file_name='SYNTHETIC_head_acc_example.csv',mime='text/csv',key='example_acc_download')


def policy_timeline(direction,mass):
    fig=base_figure(290)
    # Representative examples from source slide; not group-level estimates for oblique.
    if direction=='사선':
        rows=[('기존 · 전면',15,ORANGE),('RL · 전면',15,BLUE),('RL · 측면',20,GREEN)]
        fig.update_layout(title=dict(text='에어백 전개 시점 · 원본 로그 발췌 예시',font=dict(size=16)))
        note='사선 로그 예시: 기존 전면 15ms/130kPa, RL 전면 15ms/141kPa + 측면 20ms/111kPa (서로 완전히 동일한 사고 입력쌍은 아님).'
    else:
        val=9.8 if mass==60 else 20.3
        rows=[('Rule-Based · 고정 기준',15,ORANGE),(f'RL · {mass}kg 평균',val,BLUE)]
        fig.update_layout(title=dict(text='에어백 전개 시점 · 집계 평균',font=dict(size=16)))
        note=f'RL {mass}kg 그룹 평균: {val:.1f}ms / {115.8 if mass==60 else 184.5:.1f}kPa. Rule-Based 15ms·130kPa는 연구용 고정 가정값 (실제 투싼 ECU 문서 수치 아님).'
    for label,t,color in rows:
        fig.add_scatter(x=[0,t],y=[label,label],mode='lines+markers',
                        line=dict(color=color,width=6),marker=dict(size=[1,13],color=color),
                        showlegend=False,hovertemplate=label+'<br>전개 시각 %{x:.1f} ms<extra></extra>')
        fig.add_annotation(x=t,y=label,text=f'{t:g}ms',showarrow=False,xshift=34,
                           font=dict(color=color,size=11))
    fig.update_xaxes(title='충돌 후 경과 시간 (ms)',range=[0,33])
    fig.update_yaxes(title=None,categoryorder='array',categoryarray=[x[0] for x in reversed(rows)])
    return fig,note


def weight_factors_fig():
    fig=base_figure(310)
    fig.add_bar(x=['60kg','90kg'],y=[9.8,20.3],name='전개 타이밍 (ms)',marker_color=BLUE,
                text=['9.8 ms','20.3 ms'],textposition='outside')
    fig.add_scatter(x=['60kg','90kg'],y=[115.8/10,184.5/10],name='전개 압력 (kPa) ÷ 10',
                    mode='lines+markers+text',text=['115.8 kPa','184.5 kPa'],textposition='top center',
                    marker=dict(size=12,color=GREEN),line=dict(color=GREEN,width=3))
    fig.update_layout(title=dict(text='체중별 RL 정책 (압력 눈금은 ÷10 표시)',font=dict(size=16)))
    fig.update_yaxes(title='시각화용 공통축: ms / (kPa÷10)')
    return fig


def show_videos(direction,mass):
    st.markdown('### 충돌 영상 비교')
    if direction=='사선':
        vid=ASSETS/'compare_oblique.mp4'
        if vid.exists():
            st.video(str(vid))
            st.caption('좌: 강화학습 / 우: 단일 에어백 전개 비교 (기존 녹화 영상)')
        else:st.info('사선충돌 비교 영상 파일을 assets/compare_oblique.mp4에 넣어주세요.')
    elif direction=='전방':
        vid=ASSETS/'original'/('front_A.mp4' if mass==60 else 'front_B.mp4')
        if vid.exists():st.video(str(vid))
        st.caption(f'제공된 전방충돌 원본의 {mass}kg 대응 참고 영상. 실제 제어정책별 쌍으로 분류된 영상은 아닙니다.')
        with st.expander('60kg와 90kg 전방충돌 영상 함께 보기'):
            c1,c2=st.columns(2)
            with c1:st.markdown('**60kg 참고 영상**');st.video(str(ASSETS/'original/front_A.mp4'))
            with c2:st.markdown('**90kg 참고 영상**');st.video(str(ASSETS/'original/front_B.mp4'))
    else:
        st.info('선택한 접근각에 정확히 대응하는 녹화 영상이 아직 없어, 다른 방향의 영상을 실제 결과처럼 재사용하지 않았습니다.')
        with st.expander('가장 가까운 사선충돌 참고 영상 (약 45°)'):
            st.video(str(ASSETS/'compare_oblique.mp4'))


def upload_pairs():
    with st.sidebar.expander('실험별 HIC15 원자료 연결',expanded=False):
        st.caption('같은 사고를 기존 정책과 RL 정책으로 평가한 **대응표본** CSV를 올리면 집계/가상 표본 대신 분석합니다.')
        template=pd.DataFrame([{'speed_kmh':100,'mass_kg':60,'direction':'전방','pair_id':1,
                                'baseline_hic15':550.0,'rl_hic15':490.0}])
        st.download_button('원자료 CSV 템플릿',template.to_csv(index=False).encode('utf-8-sig'),
                           file_name='paired_hic15_template.csv',mime='text/csv')
        up=st.file_uploader('대응표본 CSV',type='csv',key='paired_csv')
        if up is not None:
            try:
                out=read_matched_csv(up)
                st.success(f'{len(out)}개 대응쌍 읽음')
                return out
            except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f'CSV 형식 확인: {exc}')
    return None


def statistical_panel(pairs,mode,source):
    if pairs is None:
        st.info('원본 결과는 평균·중앙값만 제공돼 대응표본의 실제 p값을 계산할 수 없습니다. 아래에서 차이값 표준편차 가정을 바꾸어 **유의성 민감도**를 확인할 수 있어요.')
        return
    try:
        hic=paired_t_test(pairs)
        risk=paired_t_test(pairs,risk=True)
    except ValueError as exc:
        st.warning(str(exc));return
    c1,c2,c3,c4=st.columns(4)
    c1.metric('대응 실험쌍',f'{hic["n"]}쌍')
    c2.metric('HIC15 단측 p',f'{hic["p_one"]:.4f}')
    c3.metric('골절 위험 단측 p',f'{risk["p_one"]:.4f}')
    c4.metric('HIC 차이 SD',f'{hic["sd_difference"]:.1f}')
    rows=[{'검정 지표':'HIC15','평균 차이':f'{hic["mean_difference"]:.2f}',
           't통계량':f'{hic["t"]:.3f}','자유도':hic['df'],
           '단측 p':f'{hic["p_one"]:.5f}',
           '평균 차이 95% CI':f'{hic["ci_low"]:.2f} ~ {hic["ci_high"]:.2f}'},
          {'검정 지표':'골절 위험 대응값 (%p)','평균 차이':f'{risk["mean_difference"]:.2f}%p',
           't통계량':f'{risk["t"]:.3f}','자유도':risk['df'],
           '단측 p':f'{risk["p_one"]:.5f}',
           '평균 차이 95% CI':f'{risk["ci_low"]:.2f} ~ {risk["ci_high"]:.2f}%p'}]
    st.dataframe(pd.DataFrame(rows),width='stretch',hide_index=True)
    s1,s2=st.columns(2)
    with s1:st.plotly_chart(box_plot(pairs),width='stretch',config={'displaylogo':False})
    with s2:st.plotly_chart(box_plot(pairs,risk=True),width='stretch',config={'displaylogo':False})
    st.plotly_chart(paired_difference_chart(pairs),width='stretch',config={'displaylogo':False})
    if mode=='원자료':
        st.success('업로드한 대응표본의 검정: '+('평균 HIC15가 유의하게 감소했습니다.' if hic['improvement_supported'] else '유의한 평균 HIC15 감소를 확인하지 못했습니다.'))
    else:
        st.markdown(f'<div class="soft-panel"><b>고편차 생성 데이터에 대한 검정</b> · HIC15 단측 p={hic["p_one"]:.4f} / '
                    f'골절 위험 단측 p={risk["p_one"]:.4f} · '
                    f'{"가정한 분포에서 유의" if hic["improvement_supported"] else "가정한 분포에서 유의성 미확보"}. '
                    '실제 강화학습 성능 입증을 뜻하지 않습니다.</div>',unsafe_allow_html=True)
    st.caption('대응차 d = Rule-Based − RL, H1: 평균(d) > 0, α=0.05. 95% CI는 양측 구간. '
               '골절확률은 각 표본 HIC15를 위험함수로 바꾼 뒤 검정하며, 독립적인 임상 근거는 아닙니다.')
    csv=pairs.to_csv(index=False).encode('utf-8-sig')
    st.download_button('선택된 대응표본 CSV 저장',csv,file_name='selected_paired_hic15.csv',mime='text/csv',key='save_pairs')


def sensitivity_panel(base,rl,n=50):
    mean_diff=base-rl
    if mean_diff<=0:
        st.info('선택 결과의 평균 감소가 없으므로 개선 단측검정 민감도 분석을 표시하지 않습니다.')
        return
    c1,c2=st.columns([1,3])
    with c1:
        sd=st.slider('차이값 SD 가정',10.,250.,75.,5.,key='sd_assumption')
        t=mean_diff/(sd/np.sqrt(n))
        p=float(stats.t.sf(t,n-1))
        st.metric('가정 기반 단측 p',f'{p:.4f}')
    with c2:
        arr=np.linspace(10,250,241)
        ps=stats.t.sf(mean_diff/(arr/np.sqrt(n)),n-1)
        fig=base_figure(280)
        fig.add_scatter(x=arr,y=ps,mode='lines',line=dict(color=BLUE,width=3),name='가정된 p값')
        fig.add_hline(y=.05,line_dash='dash',line_color='#D66664',annotation_text='p=0.05')
        fig.add_scatter(x=[sd],y=[p],mode='markers',marker=dict(color=ORANGE,size=14),name='현재 SD 가정')
        fig.update_xaxes(title='가정된 대응 차이값 표준편차 (실제 측정값 아님)')
        fig.update_yaxes(title='단측 p값',range=[0,min(1,max(ps)*1.1)])
        st.plotly_chart(fig,width='stretch',config={'displaylogo':False})
    st.caption('이 p값은 실제 시험으로 검정된 값이 아니라, 집계 평균과 임의 SD를 대입한 민감도 계산입니다.')


def scenario_page(real_data):
    st.markdown('<div class="small-kicker">01 / SCENARIO EXPLORER</div>',unsafe_allow_html=True)
    st.title('사고 환경 입력 · 에어백 제어 비교')
    st.markdown('<div class="lead">차량 접근 방향을 360°로 선택하고, 머리 가속도 → HIC15 → 두개골 골절 추정확률까지 연결해 비교합니다.</div>',unsafe_allow_html=True)
    if 'impact_angle' not in st.session_state:st.session_state.impact_angle=45
    l,r=st.columns([1.18,1],gap='large')
    with l:
        st.markdown('#### 360° 충돌 방향 탐색기')
        st.plotly_chart(collision_figure(st.session_state.impact_angle),width='stretch',
                        config={'displayModeBar':False},key='car_collision_dial')
    with r:
        st.markdown('#### 사고 환경 설정')
        preset=st.columns(5)
        for col,deg,label in zip(preset,[0,45,90,180,270],['전방 0°','사선 45°','측면 90°','후방','좌측']):
            with col:
                if st.button(label,key=f'angle_{deg}',width='stretch'):
                    st.session_state.impact_angle=deg
                    st.rerun()
        st.slider('차량 접근 방향 · 0°–359°',min_value=0,max_value=359,step=1,key='impact_angle',
                  help='0° 전방 · 90° 오른쪽 측면 · 180° 후방 · 270° 왼쪽 측면')
        c1,c2=st.columns(2)
        with c1:speed=st.selectbox('충돌 속도 (km/h)',[60,100,140],index=1)
        with c2:mass=st.selectbox('탑승자 체중 (kg)',[60,90],index=0)
        c3,c4=st.columns(2)
        with c3:posture=st.selectbox('착석 자세',['정상','전방 기울임'])
        with c4:belt=st.selectbox('안전벨트',['착용','미착용'])
        direction=classified_direction(st.session_state.impact_angle)
        st.markdown(f'<div class="soft-panel"><b>선택 결과</b> · {st.session_state.impact_angle}° {direction} 접근 · '
                    f'{speed} km/h · {mass} kg</div>',unsafe_allow_html=True)
        st.caption('자세·벨트 입력은 UI 시연 조건이며, 제공된 데이터·녹화 영상의 재해석을 수행하지 않습니다.')

    show_videos(direction,mass)
    if direction not in ['전방','사선']:
        st.info('현 보유 데이터는 전방(약 0°) 및 사선(약 45°) 두 조건입니다. 선택 각도를 바꾸면 360° 차이는 시각화되지만, 측면·후방의 HIC15 결과는 생성하지 않습니다.')
        return

    # If an actual matched CSV covers these inputs, give it priority.
    candidate=filter_pairs(real_data,speed,mass,direction) if real_data is not None else pd.DataFrame()
    if len(candidate)>=3:
        result=dict(base=float(candidate.baseline_hic15.mean()),rl=float(candidate.rl_hic15.mean()),
                    pairs=candidate,mode='원자료',source='업로드한 동일 사고 조건의 대응표본',
                    note='실제 시뮬레이션 CSV의 조건-쌍 매칭은 업로더가 검증해야 합니다.',policy=direction)
    else:
        result=lookup_aggregate(direction,mass,speed)
    source_note(result['source']+' · '+result['note'])
    base,rl=result['base'],result['rl']
    cmp=compute_comparison(base,rl)

    st.markdown('### 주요 결과')
    ca,cb,cc,cd=st.columns(4)
    with ca:metric_card('Rule-Based 평균 HIC15',f'{base:.1f}','기존 연구용 규칙 정책','orange')
    with cb:metric_card('Curriculum PPO 평균 HIC15',f'{rl:.1f}','강화학습 기반 정책','blue')
    with cc:metric_card('HIC15 평균 감소',f'{cmp["reduction_pct"]:.1f}%','기존 방식 대비','green')
    with cd:metric_card('골절 추정위험 감소',f'{-cmp["risk_delta_pp"]:.2f}%p','평균 HIC의 위험함수 변환','green')
    a,b=st.columns([.83,1.17],gap='large')
    with a:st.plotly_chart(hic_bar(base,rl),width='stretch',config={'displaylogo':False})
    with b:st.plotly_chart(risk_curve(base,rl),width='stretch',config={'displaylogo':False})
    st.markdown(f'**해석:** HIC15 **{base:.1f} → {rl:.1f}** ({cmp["reduction_pct"]:.1f}% 감소), '
                f'NHTSA 위험곡선 대응값 **{cmp["baseline_risk_pct"]:.2f}% → {cmp["rl_risk_pct"]:.2f}%** '
                f'({-cmp["risk_delta_pp"]:.2f}%p 감소).')
    st.caption('위 위험 대응값은 평균 HIC15에 NHTSA AIS 2+ 위험곡선을 적용한 값이며, 골절 발생건수나 실측 발생확률이 아닙니다.')

    signal_to_injury_section(base,rl,f'{speed}km/h / {mass}kg / {direction}')

    st.markdown('### 에어백 동작 Factor · 정책 차이')
    pcol,descr=st.columns([1.25,.75],gap='large')
    with pcol:
        fig,policy_note=policy_timeline(direction,mass)
        st.plotly_chart(fig,width='stretch',config={'displaylogo':False})
    with descr:
        if direction=='사선':
            st.markdown('**Rule-Based** · 충돌 방향 기반 단일 에어백 선택 (연구용 단순화)')
            st.markdown('**Curriculum PPO** · 전면과 측면 에어백의 순차 전개를 포함해 조합·시점을 선택')
        else:
            st.markdown('**Rule-Based** · 연구용 고정 타이밍·압력 기준')
            st.markdown('**Curriculum PPO** · 탑승자 체중에 따라 전개 시점·압력을 조정')
        st.caption(policy_note)
    if direction=='전방':
        st.plotly_chart(weight_factors_fig(),width='stretch',config={'displaylogo':False})

    st.markdown('### 분포·통계적 유의성')
    statistical_panel(result['pairs'],result['mode'],result['source'])
    if result['pairs'] is None:
        sensitivity_panel(base,rl,n=50)
    st.markdown(f'[NHTSA (1999) HIC–골절 위험곡선 원문]({NHTSA_SOURCE})')


def overall_page():
    st.markdown('<div class="small-kicker">02 / EXPERIMENT RESULTS</div>',unsafe_allow_html=True)
    st.title('전체 실험 결과 · 3개 시나리오')
    st.caption('원본 시뮬레이션 집계와 고속 조건의 생성 데이터를 표에서 별도 표기했습니다.')
    m=measured()
    out=summarize_measured(m)
    st.markdown('### 기존 발표자료의 집계')
    st.dataframe(out,width='stretch',hide_index=True)
    fig=base_figure(370)
    names=out['조건'].tolist()
    for key,name,color in [('기존 HIC15 평균','Rule-Based',ORANGE),('RL HIC15 평균','Curriculum PPO',BLUE)]:
        fig.add_bar(x=names,y=out[key],name=name,marker_color=color,text=[f'{v:.1f}' for v in out[key]],textposition='outside')
    fig.update_layout(barmode='group',title=dict(text='원본 HIC15 집계 평균',font=dict(size=16)))
    st.plotly_chart(fig,width='stretch',config={'displaylogo':False})
    st.markdown('### 체중별 RL 전개 Factor')
    policy=pd.DataFrame({'평가지표':['평균 전개 타이밍 (ms)','평균 전개 압력 (kPa)','Rule-Based 평균 HIC15','Curriculum PPO 평균 HIC15'],
                         '60kg 집단':[9.8,115.8,232.9,218.5],'90kg 집단':[20.3,184.5,269.7,243.0]})
    st.dataframe(policy,hide_index=True,width='stretch')
    st.plotly_chart(weight_factors_fig(),width='stretch',config={'displaylogo':False})
    st.markdown('### 100km/h 고편차 가정 결과 (시연용)')
    arr=[]
    for key,label in [('Oblique','사선충돌'),('Light 60kg','전방 · 경량 60kg'),('Heavy 90kg','전방 · 중량 90kg')]:
        data=high_variance_pairs().loc[lambda x:x.scenario_key.eq(key)]
        b=data.rule_HIC15.mean();r=data.RL_HIC15.mean()
        arr.append({'시나리오':label,'Rule-Based 평균':b,'PPO 평균':r,
                    '감소율 (%)':round((b-r)/b*100,2),'자료 구분':'100쌍 생성 데이터'})
    st.dataframe(pd.DataFrame(arr),hide_index=True,width='stretch')
    st.download_button('원본 집계 CSV 내려받기',out.to_csv(index=False).encode('utf-8-sig'),
                       file_name='reported_hic15_aggregates.csv',mime='text/csv')
    with st.expander('이전 초안의 알고리즘 비교 아카이브 (별도 실험)'):
        try:
            with (DATA/'experiment_summary.json').open(encoding='utf-8') as f:report=json.load(f)
            st.write('초기 대시보드 보고값이며 위의 3개 시나리오 및 최신 영상에서 재계산한 수치는 아닙니다.')
            st.dataframe(pd.DataFrame(report['algorithms']),hide_index=True,width='stretch')
        except (KeyError,FileNotFoundError):
            st.info('이전 초안의 알고리즘 상세 결과표를 찾을 수 없습니다.')


def stats_page(real_data):
    st.markdown('<div class="small-kicker">03 / STATISTICAL ANALYSIS</div>',unsafe_allow_html=True)
    st.title('대응표본 t검정 · HIC15와 골절 위험')
    st.markdown('같은 충돌 조건을 기존 정책과 PPO 정책으로 각각 평가했을 때 **두 정책의 차이가 일관되게 양수인지** 검정합니다.')
    st.latex(r'd_i=HIC_{Rule,i}-HIC_{PPO,i},\quad t=\frac{\bar d}{s_d/\sqrt{n}},\quad H_1:\mu_d>0')
    if real_data is not None and len(real_data)>=3:
        st.markdown('### 업로드한 대응표본 · 조건별 검정')
        options=real_data[['speed_kmh','mass_kg','direction']].drop_duplicates()
        for _,row in options.iterrows():
            data=filter_pairs(real_data,int(row.speed_kmh),int(row.mass_kg),row.direction)
            with st.expander(f'{row.direction} · {int(row.speed_kmh)}km/h · {int(row.mass_kg)}kg ({len(data)}쌍)'):
                statistical_panel(data,'원자료','사용자 업로드 데이터')
    st.markdown('### 고편차 100km/h 조건 분석 · 데이터 생성 가정')
    arr=[]
    for direction,mass,label in [('사선',60,'사선 45°'),('전방',60,'정면 60kg'),('전방',90,'정면 90kg')]:
        pair=demo_pairs(direction,mass,100)
        h=paired_t_test(pair);r=paired_t_test(pair,risk=True)
        arr.append({'시나리오':label,'평균 HIC (기존→RL)':f'{h["baseline_mean"]:.1f}→{h["rl_mean"]:.1f}',
                    'HIC 차이 SD':round(h['sd_difference'],1), 'HIC t':round(h['t'],3),
                    'HIC 단측 p':round(h['p_one'],4),'골절위험 단측 p':round(r['p_one'],4),
                    '가정 데이터 판정':'유의' if h['improvement_supported'] else '유의성 미확보'})
    st.dataframe(pd.DataFrame(arr),hide_index=True,width='stretch')
    selected=st.selectbox('자세히 살펴볼 조건',['사선 45°','정면 60kg','정면 90kg'])
    direction,mass=({'사선 45°':('사선',60),'정면 60kg':('전방',60),'정면 90kg':('전방',90)})[selected]
    pair=demo_pairs(direction,mass,100)
    statistical_panel(pair,'합성','기존에 생성한 100쌍 고편차 데이터')
    st.caption('p<0.05도 실제 결과의 근거가 아니라 **설정된 데이터 분포에 대한 계산 결과**입니다. 실제 검증은 업로드한 시뮬레이션 쌍을 사용하세요.')


def theory_page():
    st.markdown('<div class="small-kicker">04 / THEORY</div>',unsafe_allow_html=True)
    st.title('HIC15 · 정책 최적화 · 검정 원리')
    tab1,tab2,tab3=st.tabs(['HIC15와 부상 위험','Curriculum PPO','통계 해석'])
    with tab1:
        st.markdown('**HIC15는 머리 합성가속도와 충격 지속시간으로 계산되는 지표입니다.**')
        st.latex(r'\mathrm{HIC}_{15}=\max_{0<t_2-t_1\le0.015}\left[(t_2-t_1)\left(\frac{1}{t_2-t_1}\int_{t_1}^{t_2}a(t)\,dt\right)^{2.5}\right]')
        st.latex(r'P(\mathrm{AIS\ 2+\ skull\ fracture})=\Phi\left(\frac{\ln(HIC)-6.96352}{0.84664}\right)')
        st.plotly_chart(risk_curve(623,496.5),width='stretch',config={'displaylogo':False})
        st.caption('623 및 496.5는 생성 데이터의 평균값이며, NHTSA 곡선은 실험으로 발생한 골절의 빈도를 직접 보여주지는 않습니다.')
        st.markdown(f'[NHTSA 원문 PDF]({NHTSA_SOURCE})')
    with tab2:
        st.markdown('**사고·탑승자 상태 → PPO 정책 선택(전개 여부·시점·압력) → 더미 응답·HIC 평가 → 정책 업데이트**')
        st.latex(r'L^{CLIP}(\theta)=\mathbb{E}[\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)]')
        st.markdown('Curriculum Learning은 충돌 방향별 기본 정책을 먼저 익힌 뒤, HIC 최적화에 초점을 맞추는 단계적 학습 전략입니다.')
        st.markdown('기존의 **15ms/130kPa** 기준은 연구용 고정 Rule-Based 조건이지 제조사 ECU 기술문서의 확정 수치가 아닙니다.')
    with tab3:
        st.latex(r't=\frac{\bar d}{s_d/\sqrt{n}},\quad p_{one}=P(T_{n-1}\ge t)')
        st.markdown('평균 차이가 크더라도 대응 차이의 표준편차가 크면 유의성을 확보하지 못할 수 있습니다. p≥0.05는 두 정책이 같다는 증명이 아닙니다.')
        st.markdown('두개골 골절 추정확률은 HIC15의 함수이므로 확률에 대한 t검정은 독립적인 두 번째 충돌시험이 아닙니다.')


def main():
    styles()
    with st.sidebar:
        st.markdown('## Airbag Safety Lab')
        st.caption('개인화 에어백 정책 비교 · Isaac Sim/강화학습')
        page=st.radio('메뉴',['사고 환경 선택','전체 실험 결과','통계 검정','이론 및 모델 설명'],label_visibility='collapsed')
        st.divider()
        st.caption('제공된 원본 영상과 실험 집계를 활용한 시각화 웹입니다.')
    real=upload_pairs() if page in ['사고 환경 선택','통계 검정'] else None
    if page=='사고 환경 선택':scenario_page(real)
    elif page=='전체 실험 결과':overall_page()
    elif page=='통계 검정':stats_page(real)
    else:theory_page()


if __name__=='__main__':
    main()
