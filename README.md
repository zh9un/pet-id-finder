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
- **품종 분류 (Fine-tuned)**: ResNet18 기반 Transfer Learning으로 9개 품종 분류 (87.65% 정확도)
- **품종별 필터링**: 같은 품종만 검색하는 옵션 제공
- **코사인 유사도 검색**: DB 전체 비교 후 순위 산출
- **목격 정보 관리**: 발견 장소, 시간 정보 저장 및 표시
- **Grad-CAM 시각화**: AI 품종 판단 근거를 히트맵으로 표시 (설명 가능한 AI)

## 기술 스택

- **Backend**: Flask 3.0.0
- **AI Models**:
  - YOLOv8n (객체 탐지)
  - CLIP (특징 추출)
  - ResNet18 Fine-tuned (품종 분류, 87.65% 정확도)
- **Database**: SQLite3
- **ML**: PyTorch, scikit-learn, OpenCV
- **XAI**: Grad-CAM (설명 가능한 AI)
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
                    │  Step 4: 품종 분류              │
                    │    (Fine-tuned ResNet18)        │
                    │  Output: 512차원 벡터 + 품종    │
                    └──────┬──────────────────────────┘
                           │
                           v
                    ┌─────────────────────┐
                    │   SQLite Database   │
                    │     (pets.db)       │
                    └─────────────────────┘
                           │
                           v
                    ┌─────────────────────┐
                    │   Grad-CAM          │
                    │  (gradcam.py)       │
                    │  - AI 분석 근거     │
                    │  - 시각화 히트맵    │
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

### 3. 품종 분류 모델 학습 (선택, 가산점)

Stanford Dogs Dataset을 활용하여 품종 분류 모델을 학습합니다:

**3-1. Stanford Dogs Dataset 다운로드**

[Kaggle Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)에서 다운로드하여 `datasets/stanford_dogs/images/` 폴더에 압축 해제

**3-2. 품종 데이터 추출**

```bash
# 9개 품종만 추출 (총 1,616장)
python extract_breeds.py
```

**3-3. 모델 학습**

```bash
# ResNet18 Transfer Learning (CPU 기준 약 2시간 소요)
python train_breed_classifier.py
```

**학습 결과**:
- Best Validation Accuracy: **87.65%**
- 생성 파일:
  - `models/breed_classifier_best.pth` (최고 성능 모델)
  - `models/breed_classifier_final.pth` (최종 모델)
  - `models/breed_labels.json` (레이블 매핑)
  - `models/training_history.json` (학습 히스토리)

**학습 품종 (9개)**:
- Chihuahua (치와와)
- Maltese (말티즈)
- Shih-Tzu (시츄)
- Yorkshire Terrier (요크셔테리어)
- Golden Retriever (골든 리트리버)
- Pomeranian (포메라니안)
- Toy Poodle (토이 푸들)
- Miniature Poodle (미니어처 푸들)
- Standard Poodle (스탠다드 푸들)

### 4. 기존 데이터 품종 재분석 (선택)

API로 수집한 기존 데이터에 품종 정보를 추가합니다:

```bash
python update_breeds.py
```

**소요 시간**: 데이터 개수에 따라 3-5분

**수행 작업**:
- 품종 정보가 없는 모든 데이터 조회
- Fine-tuned ResNet18 또는 CLIP Zero-shot으로 품종 분류
- 데이터베이스 업데이트

### 5. 실행

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
3. (선택) "같은 품종만 검색" 체크박스 선택
4. "검색하기" 버튼 클릭
5. 유사도 순위로 정렬된 결과 확인
6. 발견 장소, 시간, 품종 정보 확인
7. **"AI 분석 근거 보기" 버튼 클릭하여 Grad-CAM 시각화 확인**

### 목격자 모드 (신고)

1. "목격자입니다" 버튼 클릭
2. 발견한 유기동물 사진 업로드
3. "신고하기" 버튼 클릭
4. AI가 자동으로 분석하여 DB에 저장

### Grad-CAM 시각화 (설명 가능한 AI)

검색 결과 화면에서 각 동물 카드의 **"AI 분석 근거 보기"** 버튼을 클릭하면:

- AI가 품종을 판단할 때 **중요하게 본 영역**을 히트맵으로 표시
- **빨간색 영역**: AI가 품종 분류 시 가장 집중한 부분 (얼굴, 털 패턴 등)
- Fine-tuned ResNet18 모델의 의사결정 과정을 투명하게 공개
- 사용자에게 AI의 판단 근거 제공으로 신뢰성 향상

## 프로젝트 구조

```
pet-id-finder/
├── app.py                       # Flask 서버 (Grad-CAM 라우트 포함)
├── ml_pipeline.py               # ML 파이프라인 (YOLO + CLIP + Fine-tuned 품종 분류)
├── gradcam.py                   # Grad-CAM 구현 (설명 가능한 AI)
├── train_breed_classifier.py   # ResNet18 품종 분류 학습 스크립트
├── extract_breeds.py            # Stanford Dogs Dataset 품종 추출
├── api_register.py              # API 데이터 수집 스크립트
├── update_breeds.py             # 기존 데이터 품종 재분석 스크립트
├── requirements.txt             # 패키지 의존성
├── README.md                    # 프로젝트 설명서
├── .gitignore                   # Git 제외 파일
├── static/
│   ├── uploads/                 # 사용자 업로드 이미지
│   └── api_images/              # API에서 수집한 이미지
├── templates/
│   ├── index.html               # 메인 페이지 (역할 선택)
│   ├── search_page.html         # 보호자 페이지 (검색, 품종 필터)
│   ├── report_page.html         # 목격자 페이지 (신고)
│   └── results.html             # 검색 결과 페이지 (Grad-CAM 버튼 포함)
├── models/
│   ├── yolov8n.pt               # YOLO 모델 (자동 다운로드)
│   ├── clip-vit-base-patch32/   # CLIP 모델 (자동 다운로드)
│   ├── breed_classifier_best.pth       # Fine-tuned ResNet18 (87.65% 정확도)
│   ├── breed_classifier_final.pth      # Fine-tuned ResNet18 (최종)
│   ├── breed_labels.json        # 품종 레이블 매핑 (9개 클래스)
│   └── training_history.json    # 학습 히스토리
├── datasets/
│   ├── stanford_dogs/           # Stanford Dogs Dataset (원본)
│   └── dog_breeds_9/            # 추출된 9개 품종 (1,616장)
└── pets.db                      # SQLite 데이터베이스
```

## 핵심 알고리즘

### ML Pipeline

1. **YOLO 객체 탐지**: 이미지에서 'dog' 또는 'cat' 탐지
2. **이미지 Crop**: 탐지된 영역만 추출
3. **CLIP 특징 추출**: 512차원 벡터 생성
4. **품종 분류 (Fine-tuned ResNet18)**: Transfer Learning으로 학습한 9개 품종 분류 (87.65% 정확도)
5. **코사인 유사도**: DB 벡터와 비교하여 순위 산출

```python
# 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(query_embedding, db_embedding)[0][0]
```

### 품종 분류 (Fine-tuned ResNet18)

Stanford Dogs Dataset 9개 품종으로 학습한 ResNet18 모델 사용:

```python
# ResNet18 Transfer Learning
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 9)  # 9개 클래스

# 품종 예측
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    breed = breed_labels[predicted_class]
```

**학습 결과**:
- Best Validation Accuracy: **87.65%**
- Epochs: 10
- 학습 시간: 약 2시간 (CPU)

### Grad-CAM 시각화 (설명 가능한 AI)

AI가 품종을 판단할 때 이미지의 어느 부분을 중요하게 보는지 시각화:

```python
# Grad-CAM 생성
gradcam = GradCAM(model, target_layer=model.layer4)
cam = gradcam.generate_cam(input_tensor, predicted_class)

# 히트맵 오버레이
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
result = heatmap * 0.4 + original_image * 0.6
```

**Grad-CAM 특징**:
- **빨간색 영역**: AI가 가장 집중한 부분 (얼굴, 털 패턴 등)
- **파란색 영역**: AI가 덜 집중한 부분
- ResNet18의 마지막 convolutional layer (layer4) 활용
- 사용자에게 AI 의사결정 과정 투명하게 공개

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
- **ResNet18 Fine-tuned**: 9개 품종 분류 (87.65% 정확도)
- 교수님 강의의 파이프라인 개념 적용
- GPU 자동 감지 및 최적화
- Transfer Learning을 통한 효율적 학습

### 2-1. 품종 분류 모델 Fine-tuning (가산점)

- **Stanford Dogs Dataset** 활용 (1,616장)
- **ResNet18 Transfer Learning**:
  - Pre-trained ImageNet 가중치 사용
  - 마지막 FC layer만 9개 클래스로 교체
  - Data Augmentation (Random Crop, Flip, Color Jitter)
- **학습 성과**:
  - Best Validation Accuracy: **87.65%**
  - Epochs: 10
  - CPU 학습 (약 2시간)
- **CLIP Zero-shot Fallback**: Fine-tuned 모델이 없을 경우 자동 대체

### 2-2. Grad-CAM 시각화 (설명 가능한 AI)

- **Gradient-weighted Class Activation Mapping** 구현
- AI 품종 판단 근거를 히트맵으로 표시
- 사용자에게 AI 의사결정 과정 투명하게 공개
- Flask API 엔드포인트: `/gradcam/<pet_id>`
- 실시간 히트맵 생성 (약 5-10초)

### 3. 데이터베이스 설계

- SQLite 사용
- 특징 벡터를 JSON 형식으로 저장
- 목격 정보 (location, sighted_at) 포함
- 품종 정보 (breed) 저장
- 동물 종류별, 품종별 필터링 지원

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

## 가산점 근거

본 프로젝트는 다음과 같은 **추가 학습 모델 및 고급 기법**을 구현하여 가산점을 받을 수 있습니다:

### 1. Fine-tuned 품종 분류 모델 (추가 학습 모델)

**구현 내용**:
- Stanford Dogs Dataset 9개 품종 (1,616장)으로 ResNet18 Transfer Learning 수행
- Pre-trained ImageNet 가중치를 활용하여 효율적 학습
- Data Augmentation 적용 (Random Crop, Flip, Color Jitter)

**학습 결과**:
- Best Validation Accuracy: **87.65%**
- Epochs: 10 (CPU 기준 약 2시간)
- 학습 가중치 파일: `models/breed_classifier_best.pth` (44.7MB)

**실제 학습 증명**:
```
[Epoch 1/10] Val Acc: 79.32%
[Epoch 2/10] Val Acc: 82.10%
[Epoch 3/10] Val Acc: 85.80%
[Epoch 4/10] Val Acc: 86.42%
[Epoch 5/10] Val Acc: 84.88%
[Epoch 6/10] Val Acc: 87.65% ← Best Model
[Epoch 7/10] Val Acc: 86.73%
[Epoch 8/10] Val Acc: 87.35%
[Epoch 9/10] Val Acc: 87.04%
[Epoch 10/10] Val Acc: 86.42%
```

**가산점 해당 사항**:
- ✅ 추가 데이터셋 활용 (Stanford Dogs Dataset)
- ✅ 실제 학습 수행 (학습 가중치 파일 존재)
- ✅ 높은 정확도 달성 (87.65%)
- ✅ 실제 서비스에 통합 (ml_pipeline.py)

### 2. Grad-CAM 시각화 (설명 가능한 AI, XAI)

**구현 내용**:
- Gradient-weighted Class Activation Mapping 구현
- AI가 품종 판단 시 중요하게 본 이미지 영역을 히트맵으로 시각화
- ResNet18 layer4의 gradient 정보 활용

**기술적 특징**:
- Forward Hook & Backward Hook을 통한 activation 및 gradient 추출
- Global Average Pooling으로 가중치 계산
- 히트맵 오버레이 (JET 컬러맵 사용)

**사용자 경험**:
- 검색 결과 화면에서 "AI 분석 근거 보기" 버튼 클릭
- 실시간 Grad-CAM 생성 (약 5-10초)
- AI의 의사결정 과정을 투명하게 공개

**가산점 해당 사항**:
- ✅ 설명 가능한 AI (Explainable AI) 구현
- ✅ 사용자에게 AI 판단 근거 제공
- ✅ 신뢰성 향상 및 투명성 확보

### 3. 코드 파일 증명

**학습 관련 파일**:
- `train_breed_classifier.py`: 품종 분류 학습 스크립트
- `extract_breeds.py`: Stanford Dogs Dataset 품종 추출
- `models/breed_classifier_best.pth`: 학습된 모델 가중치 (87.65% 정확도)
- `models/breed_labels.json`: 9개 클래스 레이블 매핑
- `models/training_history.json`: 전체 학습 히스토리

**Grad-CAM 관련 파일**:
- `gradcam.py`: Grad-CAM 구현 모듈
- `app.py` (line 243-320): Flask Grad-CAM API 엔드포인트
- `templates/results.html` (line 275, 293-341): Grad-CAM UI 및 JavaScript

## 향후 개선 방향

1. **FAISS 벡터 DB**: 대용량 데이터 고속 검색
2. **Re-ID 모델**: 외형 변화 대응 (나이, 털 상태 등)
3. **지도 연동**: 발견 위치 시각화 (카카오맵/네이버맵)
4. **실시간 알림**: 유사 동물 발견 시 SMS/이메일 알림
5. **전국 확대**: 서울시, 인천시 등 타 지역 API 연동
6. **모바일 앱**: React Native 개발


