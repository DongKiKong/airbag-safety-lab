from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_deployment_pins():
 s=(ROOT/'requirements.txt').read_text()
 assert 'streamlit==1.61.1' in s
 assert 'starlette==1.4.1' in s
 assert 'plotly==6.5.2' in s

def test_static_fallback_is_complete():
 html=(ROOT/'docs/index.html').read_text()
 for text in ['360°','머리 합성가속도','Running HIC15','골절','pairsUpload','statsPage','riskCurve']:
  assert text in html
 for file in ['compare_oblique.mp4','front_A.mp4','front_B.mp4']:
  assert (ROOT/'docs/assets'/file).stat().st_size>100_000
  assert f'assets/{file}' in html
 assert '<script src=' not in html.lower()
 assert 'unpkg.com' not in html and 'cdn.jsdelivr' not in html

def test_code_no_old_width_parameter():
 assert 'use_container_width=' not in (ROOT/'app.py').read_text()
