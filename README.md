# Pet-ID Finder

AI 기반 유실동물 유사 개체 검색 시스템

## 프로젝트 개요

Pet-ID Finder는 YOLO와 CLIP을 결합한 ML 파이프라인을 통해 유실동물을 찾는 시스템입니다.
사용자가 유실동물 사진을 업로드하면 AI가 자동으로 분석하여 데이터베이스에서 가장 유사한 동물을 찾아줍니다.

### 핵심 기능
- 이미지 업로드 및 AI 자동 분석
- YOLO 객체 탐지로 동물 영역 추출
- CLIP 특징 벡터 생성
- 코사인 유사도 기반 검색
- 유사도 순위 시각화

### 기술 스택
- **Backend**: Flask 3.0.0
- **AI Models**: YOLOv8n + CLIP (OpenAI)
- **Database**: SQLite3
- **ML**: PyTorch, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript

## 시스템 아키텍처

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │
       v
┌─────────────────────┐
│   Flask Server      │
│     (app.py)        │
└──────┬──────────────┘
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

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 실행

```bash
# Flask 서버 시작
python app.py
```

브라우저에서 `http://localhost:5000` 접속

## 사용 방법

### 1. 유실동물 등록
1. "유실동물 등록" 섹션에서 이미지 선택
2. "등록하기" 버튼 클릭
3. AI가 자동으로 분석하여 DB에 저장

### 2. 유사 동물 검색
1. "유사 동물 검색" 섹션에서 이미지 선택
2. "검색하기" 버튼 클릭
3. 유사도 순위로 정렬된 결과 확인

## 프로젝트 구조

```
pet-id-finder/
├── app.py                  # Flask 서버
├── ml_pipeline.py          # ML 파이프라인
├── requirements.txt        # 패키지 의존성
├── README.md              # 프로젝트 설명서
├── .gitignore             # Git 제외 파일
├── static/
│   └── uploads/           # 업로드 이미지 저장
├── templates/
│   ├── index.html         # 메인 페이지
│   └── search.html        # 검색 결과 페이지
├── tests/
│   ├── test_dino.py       # DINO 모델 테스트
│   └── test_clip.py       # CLIP 모델 테스트
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

### 1. ML 파이프라인
- YOLO로 전처리, CLIP으로 특징 추출
- 교수님 강의의 파이프라인 개념 적용

### 2. 데이터베이스
- SQLite 사용
- 특징 벡터를 JSON 형식으로 저장

### 3. 완벽한 에러 처리
- 동물 미탐지 처리
- 파일 검증
- 재시도 로직

## 향후 개선 방향

1. **FAISS 벡터 DB**: 대용량 데이터 고속 검색
2. **Re-ID 모델**: 외형 변화 대응
3. **위치 정보**: 발견 위치 지도 표시
4. **실시간 알림**: 유사 동물 발견 시 알림
5. **모바일 앱**: React Native 개발

## 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.
