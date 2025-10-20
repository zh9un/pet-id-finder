# Pet-ID Finder

AI 기반 유실동물 유사 개체 검색 시스템

## 프로젝트 개요

Pet-ID Finder는 YOLO와 CLIP을 결합한 ML 파이프라인을 통해 유실동물을 찾는 시스템입니다.

**경기데이터드림 공공 API**에서 실제 유기동물 보호 현황 데이터를 수집하여, 사용자가 유실동물 사진을 업로드하면 AI가 자동으로 분석하여 데이터베이스에서 가장 유사한 동물을 찾아줍니다.

## 핵심 기능

- **공공 API 연동**: 경기데이터드림에서 실제 유기동물 데이터 수집
- **이미지 업로드**: 목격자/보호자 역할별 UI 제공
- **YOLO 객체 탐지**: 동물 영역 자동 추출 (개/고양이 구분)
- **CLIP 특징 벡터**: 512차원 임베딩 생성
- **코사인 유사도 검색**: DB 전체 비교 후 순위 산출
- **목격 정보 관리**: 발견 장소, 시간 정보 저장 및 표시

## 기술 스택

- **Backend**: Flask 3.0.0
- **AI Models**: YOLOv8n + CLIP (OpenAI)
- **Database**: SQLite3
- **ML**: PyTorch, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript

## 시스템 아키텍처

```
┌──────────────────────────┐
│  경기데이터드림 API       │
│  (공공 유기동물 데이터)   │
└───────┬──────────────────┘
        │
        v
┌───────────────────────────┐
│  api_register.py          │
│  - API 데이터 수집 (500개) │
│  - 이미지 다운로드         │
│  - ML 분석 및 DB 저장     │
└───────┬───────────────────┘
        │
        v
┌─────────────┐         ┌─────────────────────┐
│   Client    │  <----> │   Flask Server      │
│  (Browser)  │         │     (app.py)        │
└─────────────┘         └──────┬──────────────┘
                               │
                               v
                    ┌─────────────────────────────────┐
                    │     ML Pipeline                 │
                    │   (ml_pipeline.py)              │
                    │                                 │
                    │  Step 1: YOLO 객체 탐지         │
                    │  Step 2: 이미지 Crop            │
                    │  Step 3: CLIP 특징 추출         │
                    │  Output: 512차원 벡터           │
                    └──────┬──────────────────────────┘
                           │
                           v
                    ┌─────────────────────┐
                    │   SQLite Database   │
                    │     (pets.db)       │
                    └─────────────────────┘
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd pet-id-finder

# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 수집

API에서 실제 유기동물 데이터를 수집합니다:

```bash
# 경기데이터드림 API에서 500개 데이터 수집
python api_register.py
```

**소요 시간**: 약 10-15분

**수집 내용**:
- 실제 유기동물 사진 (개/고양이만 필터링)
- 발견 장소 정보
- 발견 날짜 정보
- YOLO 검증 후 DB 저장

### 3. 실행

```bash
# Flask 서버 시작
python app.py
```

브라우저에서 `http://localhost:5000` 접속

## 사용 방법

### 역할 선택

메인 페이지에서 역할을 선택합니다:

1. **보호자**: 잃어버린 반려동물을 찾고 싶은 경우
2. **목격자**: 유기동물을 발견하여 신고하는 경우

### 보호자 모드 (검색)

1. "보호자입니다" 버튼 클릭
2. 찾고 싶은 반려동물 사진 업로드
3. "검색하기" 버튼 클릭
4. 유사도 순위로 정렬된 결과 확인
5. 발견 장소 및 시간 정보 확인

### 목격자 모드 (신고)

1. "목격자입니다" 버튼 클릭
2. 발견한 유기동물 사진 업로드
3. "신고하기" 버튼 클릭
4. AI가 자동으로 분석하여 DB에 저장

## 프로젝트 구조

```
pet-id-finder/
├── app.py                  # Flask 서버
├── ml_pipeline.py          # ML 파이프라인 (YOLO + CLIP)
├── api_register.py         # API 데이터 수집 스크립트 ⭐ NEW
├── requirements.txt        # 패키지 의존성
├── README.md              # 프로젝트 설명서
├── .gitignore             # Git 제외 파일
├── static/
│   ├── uploads/           # 사용자 업로드 이미지
│   └── api_images/        # API에서 수집한 이미지 ⭐ NEW
├── templates/
│   ├── index.html         # 메인 페이지 (역할 선택)
│   ├── search_page.html   # 보호자 페이지 (검색)
│   ├── report_page.html   # 목격자 페이지 (신고)
│   └── results.html       # 검색 결과 페이지
├── models/
│   ├── yolov8n.pt         # YOLO 모델 (자동 다운로드)
│   └── clip-vit-base-patch32/  # CLIP 모델 (자동 다운로드)
└── pets.db                # SQLite 데이터베이스
```

## 핵심 알고리즘

### ML Pipeline

1. **YOLO 객체 탐지**: 이미지에서 'dog' 또는 'cat' 탐지
2. **이미지 Crop**: 탐지된 영역만 추출
3. **CLIP 특징 추출**: 512차원 벡터 생성
4. **코사인 유사도**: DB 벡터와 비교하여 순위 산출

```python
# 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(query_embedding, db_embedding)[0][0]
```

## 모델 변경

기본적으로 CLIP 모델을 사용하지만, DINO로 변경 가능합니다.

### DINO로 전환

`app.py` 파일에서 모델 타입 변경:

```python
MODEL_TYPE = 'dino'  # 'clip' -> 'dino'
```

### 모델 테스트

```bash
# DINO 테스트
python tests/test_dino.py

# CLIP 테스트
python tests/test_clip.py
```

## 기술적 특징

### 1. 공공 데이터 활용

- **경기데이터드림 API** 연동
- 실제 유기동물 보호 현황 500건 수집
- 이미지 다운로드 및 자동 ML 분석
- 발견 장소/시간 메타데이터 저장

### 2. ML 파이프라인

- **YOLO**: 객체 탐지 및 개/고양이 구분
- **CLIP**: 512차원 특징 벡터 추출
- 교수님 강의의 파이프라인 개념 적용
- GPU 자동 감지 및 최적화

### 3. 데이터베이스 설계

- SQLite 사용
- 특징 벡터를 JSON 형식으로 저장
- 목격 정보 (location, sighted_at) 포함
- 동물 종류별 필터링 지원

### 4. 견고한 에러 처리

- API 요청 재시도 로직 (최대 3회)
- 동물 미탐지 시 자동 스킵
- 이미지 다운로드 실패 처리
- 진행 상황 및 통계 출력

## 데이터 출처

본 프로젝트는 **경기데이터드림**의 공공 데이터를 활용합니다:

- **데이터명**: 유기동물 보호 현황
- **제공기관**: 경기도
- **API URL**: https://openapi.gg.go.kr/AbdmAnimalProtect
- **활용 방식**: 실제 유기동물 사진, 발견 장소, 발견 날짜 수집

## 향후 개선 방향

1. **FAISS 벡터 DB**: 대용량 데이터 고속 검색
2. **Re-ID 모델**: 외형 변화 대응 (나이, 털 상태 등)
3. **지도 연동**: 발견 위치 시각화 (카카오맵/네이버맵)
4. **실시간 알림**: 유사 동물 발견 시 SMS/이메일 알림
5. **전국 확대**: 서울시, 인천시 등 타 지역 API 연동
6. **모바일 앱**: React Native 개발


