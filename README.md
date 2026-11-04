# Airbag Safety Lab — 배포 오류 대응 안정화판 (2026-11-04)

사고 환경 입력(360° 차량 충돌 방향)과 기존 영상, HIC15, NHTSA AIS 2+ 두개골 골절 추정위험, 3단계 시간별 분석, 대응표본 t검정을 제공하는 프로젝트입니다.

## 1. 이번 오류의 원인 및 수정사항

사용자가 보내준 Streamlit Cloud 로그는 **Python 3.14.10 / Streamlit 1.59.2 / Starlette 1.10.0** 환경에서 JavaScript 청크 요청이 실패하면서 다음과 같은 ASGI 예외가 발생했음을 보여줍니다.

```
RuntimeError: Expected ASGI message 'http.response.start', but got 'http.response.body'.
```

이 오류는 Python의 HIC 계산식과 관계없으며, Streamlit/Starlette의 GZip 미들웨어에 관련된 서버 측 오류로 의심됩니다. 공식 이슈 https://github.com/streamlit/streamlit/issues/16341 도 확인했습니다. 다만 동일 근본 원인이라고 100% 단정할 수는 없습니다.

**안정화 조치**

- `requirements.txt`에서 Streamlit `1.61.1`, Starlette `1.4.1`, Plotly `6.5.2`로 고정. 공식 이슈에서 Streamlit `1.61.1`과 Starlette `1.4.1`이 호환성 패치로 안내됨.
- 이전 `use_container_width=True` 인자 모두 `width='stretch'`로 교체.
- 이전 입력 UI와 영상·3단계 분석·NHTSA 위험곡선·고편차 통계 예시는 유지.
- Streamlit Cloud가 다시 불안정해지더라도 영상과 결과를 열어볼 수 있도록 **GitHub Pages용 순수 HTML 백업**(`docs/index.html`)을 추가. 외부 JS/프레임워크 CDN 없이 동작하도록 제작.
- Python **3.12**는 Streamlit Community Cloud의 **새 배포 Advanced settings**에서 직접 선택해야 함. `requirements.txt`, `.python-version`, `runtime.txt`만으로 Python 버전 변경 불가.

> 코드 및 계산 테스트는 수행했지만, 이 환경에서는 Streamlit 패키지 설치 및 실제 브라우저 접속이 제한되어 **Streamlit Cloud 실제 재배포 성공은 확인하지 못했습니다**. 실제 서비스 응답은 재배포 직후 확인해야 합니다.

## 2. 먼저 할 일: 서버 의존성만 교체 (가장 간단)

현재 GitHub 저장소 `DongKiKong/airbag-safety-lab`를 유지하면서 바로 시험하려면 **기존 `requirements.txt` 내용을 아래처럼 바꾸고 Commit changes** 하세요.

```txt
streamlit==1.61.1
starlette==1.4.1
plotly==6.5.2
numpy>=1.26,<3.0
pandas>=2.1,<3.0
scipy>=1.12,<2.0
```

Streamlit Cloud가 새 패키지를 설치하면 **시크릿 창**에서 테스트해 보세요. 예전 페이지 캐시로 인해 오래된 JS 청크를 요청하는 문제를 피하기 위한 조치입니다.

그런데도 실패한다면 다음 절차로 재배포:

1. 기존 앱 주소를 기록한 뒤 Streamlit Community Cloud **My apps → 앱 메뉴 → Delete** (GitHub 리포지토리는 삭제하지 않음).
2. `Create app → Deploy a public app from GitHub`.
3. Repository: `DongKiKong/airbag-safety-lab` / Branch: `main` / Main file path: `app.py`.
4. **Advanced settings → Python 3.12** 선택 후 Deploy.
5. 새 주소(또는 이전 주소 재사용)를 열어 입력 버튼·차트·동영상을 점검.

공식 문서: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python

## 3. GitHub Pages 백업 배포 (Streamlit 서버가 계속 오류일 때 권장)

`docs/index.html`은 **Python/Streamlit 서버 없이** 360° 충돌 선택, 영상 3개, 주요 결과, 머리 가속도 → Running HIC15 → NHTSA 골절 위험 그래프, 가정 데이터 t검정, CSV 입력을 지원합니다.

아래 파일/폴더가 GitHub `main` 브랜치의 저장소 루트에 있어야 합니다.

- `docs/index.html`
- `docs/assets/compare_oblique.mp4`
- `docs/assets/front_A.mp4`
- `docs/assets/front_B.mp4`

기존 저장소의 `assets/`, `data/`, `app.py`는 **삭제하지 마세요**. `docs/assets`는 GitHub Pages에서 영상을 재생하기 위한 별도 사본입니다. GitHub Pages에서 `/docs`를 사이트 루트로 지정하면 저장소 상위의 `/assets`에 직접 접근하지 못하기 때문에 별도 복사했습니다.

**설정**: GitHub 저장소 → `Settings` → `Pages` → `Build and deployment` → `Deploy from a branch` → `main` / `/docs` → `Save`.

공개될 주소 형식은 `https://dongkikong.github.io/airbag-safety-lab/` 입니다(설정 및 게시가 완료된 뒤에만 접속 가능). 정적 웹은 `~/+/static/js/...`와 같은 Streamlit 전용 JS 파일을 요청하지 않습니다. 단, 해당 GitHub Pages 링크도 여기서 실접속 검증한 것은 아닙니다.

## 4. 데이터 해석 시 주의

- 사선충돌 원본 집계: Rule-Based 245.6 / PPO 224.7. 50쌍 보고 수치, **속도 혼합 / 체중별 미분리**.
- 정면 60kg 집계: 232.9 / 218.5. **원래 80km/h** 자료.
- 정면 90kg 집계: 269.7 / 243.0. **원래 80km/h** 자료.
- 100km/h 고편차 100쌍과 140km/h 예시는 **기존에 생성한 가상 데이터**, 실제 충돌해석 결과가 아님.
- 머리 합성가속도 CSV가 없다면 시간별 3단계 그래프는 **평균 HIC15에 맞춘 시연용 가상 파형**으로 생성됨. 실제 시간별 충돌 거동으로 주장하면 안 됨.
- 골절 추정확률은 NHTSA 성인 참조 AIS 2+ 두개골 골절 위험곡선 변환값. 실제 골절 발생률 또는 실험 독립 관찰치가 아님.
- 골절 위험 t검정은 HIC에서 변환된 추정확률의 paired test로, HIC 검정과 독립적인 물리 검증이 아님.

NHTSA 위험식: `P(AIS2+ skull fracture)=Φ((ln(HIC)−6.96352)/0.84664)`

## 5. 사용 방법 및 검사

로컬 실행(Windows):

```bat
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

정적 HTML은 `docs/index.html`을 더블클릭해서 미리 볼 수도 있습니다.

검사:

```bat
py -m pytest tests -q
```

이번 파일 기준 자동 Python 기능 테스트 **71개 통과**, 전체 Streamlit 페이지 코드를 호출하는 대체 UI 스모크 테스트 통과, JavaScript `node --check` 문법 검사 및 데이터·차트 초기화 스모크 테스트 통과, 영상 파일 `ffprobe`로 H.264/yuv420p 확인. **실제 배포 브라우저에서의 확인은 별도로 필요**합니다.
