"""Lightweight, dependency-free-on-assets Plotly car approach visualization.
Angle is a UI-controlled geometric illustration, NOT an impact simulation.
0° = frontal, 90° = from the right, 180° = rear, 270° = from left.
"""
from __future__ import annotations
import math
import numpy as np
import plotly.graph_objects as go

INK = '#142D4C'
BLUE = '#2563EB'
PALE = '#C4D7F1'
ORANGE = '#E98B45'


def collision_class(angle: int) -> str:
    a = min(angle % 360, 360 - (angle % 360))
    if a <= 20:
        return '전방'
    if a <= 65:
        return '사선'
    if a <= 120:
        return '측면'
    return '후방/측후방'


def _rotate(x, y, cx, cy, heading):
    t = math.radians(heading)
    c, s = math.cos(t), math.sin(t)
    return cx + c * x + s * y, cy - s * x + c * y


def _poly_trace(cx, cy, heading, xy, fill, stroke, name, width=1.3):
    pts = [_rotate(x, y, cx, cy, heading) for x, y in xy]
    return go.Scatter(
        x=[p[0] for p in pts] + [pts[0][0]],
        y=[p[1] for p in pts] + [pts[0][1]],
        mode='lines', fill='toself',
        line=dict(color=stroke, width=width), fillcolor=fill,
        showlegend=False, hoverinfo='skip', name=name,
    )


def _car(cx, cy, heading, body, label):
    """Car viewed from above. Local positive y is its forward direction."""
    return [
        _poly_trace(cx, cy, heading,
                    [(-.51, -.98),(.51,-.98),(.58,-.71),(.58,.67),(.49,1.0),(-.49,1.0),(-.58,.67),(-.58,-.71)],
                    body, INK, label, 2),
        _poly_trace(cx, cy, heading,
                    [(-.43,-.26),(.43,-.26),(.44,.47),(-.44,.47)],
                    '#D9ECFA', '#51758F', label),
        _poly_trace(cx, cy, heading,
                    [(-.43,.50),(.43,.50),(.39,.72),(-.39,.72)],
                    '#92C7EB', '#51758F', label),
        _poly_trace(cx, cy, heading,
                    [(-.43,-.72),(.43,-.72),(.42,-.57),(-.42,-.57)],
                    '#52708C', '#52708C', label),
        _poly_trace(cx, cy, heading,
                    [(-.45,.91),(.45,.91),(.45,.98),(-.45,.98)],
                    '#F8E0A7', '#F8E0A7', label),
    ]


def collision_figure(angle: int) -> go.Figure:
    """Return a Plotly 360° two-car interactive approach diagram.

    Angle is driven by an actual Streamlit slider; Plotly play button animates
    a schematic approach without claiming physics-based vehicle motion.
    """
    rad = math.radians(angle % 360)
    xdir, ydir = math.sin(rad), math.cos(rad)
    car_heading = (angle + 180) % 360
    circle = np.linspace(0, 2*np.pi, 241)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=3.25*np.cos(circle), y=3.25*np.sin(circle),
                             mode='lines', line=dict(color='#D9E5F3',width=1.7,dash='dot'),
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[0,3.65*xdir],y=[0,3.65*ydir],mode='lines',
                             line=dict(color=ORANGE,width=3,dash='dot'),
                             hoverinfo='skip',showlegend=False))
    for t, tag in [(0,'0° FRONT'),(45,'45°'),(90,'90° RIGHT'),(180,'180° REAR'),(270,'270° LEFT'),(315,'315°')]:
        a = math.radians(t)
        fig.add_annotation(x=4.0*math.sin(a),y=4.0*math.cos(a),text=tag,showarrow=False,
                           font=dict(size=10,color='#71839B'))
    for trace in _car(0,0,0, '#A9BBD0', 'Occupant vehicle'):
        fig.add_trace(trace)

    moving_indices = list(range(len(fig.data), len(fig.data)+5))
    for trace in _car(3.65*xdir,3.65*ydir,car_heading, BLUE, 'Approaching vehicle'):
        fig.add_trace(trace)
    fig.add_trace(go.Scatter(x=[3.65*xdir],y=[3.65*ydir],mode='markers',
                             marker=dict(color=ORANGE,size=9),showlegend=False,
                             hovertemplate=f'접근 방향: {angle%360}°<extra></extra>'))
    moving_indices.append(len(fig.data)-1)

    frames=[]
    for index, dist in enumerate(np.linspace(3.65,1.55,13)):
        traces=_car(dist*xdir,dist*ydir,car_heading,BLUE,'Approaching vehicle')
        traces.append(go.Scatter(x=[dist*xdir], y=[dist*ydir],mode='markers',
                                 marker=dict(color=ORANGE,size=9), showlegend=False,
                                 hovertemplate=f'충돌 방향: {angle%360}°<extra></extra>'))
        frames.append(go.Frame(name=str(index),data=traces,traces=moving_indices))
    fig.frames=frames
    fig.add_annotation(x=-4.4,y=-4.15,text='차량 운동은 각도 선택을 돕는 도식 (실제 충돌해석 아님)',
                       showarrow=False,xanchor='left',font=dict(size=10,color='#74869C'))
    fig.update_layout(
        height=435, autosize=True, margin=dict(l=4,r=4,t=10,b=8),
        xaxis=dict(range=[-4.7,4.7],visible=False,scaleanchor='y',scaleratio=1),
        yaxis=dict(range=[-4.6,4.55],visible=False),
        plot_bgcolor='#FBFDFF',paper_bgcolor='#FBFDFF',
        updatemenus=[dict(type='buttons',showactive=False,x=.02,y=.06,xanchor='left',
                          buttons=[dict(label='▶ 충돌 접근 보기',method='animate',
                                        args=[None,dict(frame=dict(duration=85,redraw=True),
                                                        transition=dict(duration=0),fromcurrent=False,mode='immediate')]),
                                   dict(label='↻ 처음으로',method='animate',
                                        args=[['0'],dict(frame=dict(duration=0,redraw=True),mode='immediate')])],
                          bgcolor='#EBF3FE',bordercolor='#D7E4F8',font=dict(color=INK,size=11))]
    )
    return fig
