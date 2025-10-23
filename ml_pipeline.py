"""
ML Pipeline: YOLO + DINO/CLIP 파이프라인

교수님 강의의 ML 파이프라인 개념 적용:
1. YOLO 객체 탐지로 동물 영역 추출
2. DINO/CLIP으로 특징 벡터 생성
3. 유사도 검색을 위한 표준화된 출력
"""

import torch
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms


class ImageAnalyzer:
    """
    이미지 분석 파이프라인 클래스

    Step 1: YOLO로 객체 탐지
    Step 2: DINO/CLIP으로 특징 추출
    """

    def __init__(self, model_type='clip', use_finetuned_breed=True):
        """
        Args:
            model_type (str): 'dino' 또는 'clip'
            use_finetuned_breed (bool): Fine-tuned ResNet18 품종 분류 사용 여부
        """
        print(f"[ImageAnalyzer] 초기화 시작 (모델: {model_type.upper()})")

        # 모델 저장 폴더 생성
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)

        # GPU/CPU 자동 선택
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   디바이스: {self.device}")

        # YOLO 모델 로드 (로컬 캐시 사용)
        yolo_path = os.path.join(self.models_dir, 'yolov8n.pt')
        print("   YOLO 모델 로딩...")
        self.yolo_model = YOLO(yolo_path if os.path.exists(yolo_path) else 'yolov8n.pt')
        print("   YOLO 로딩 완료")

        # 특징 추출 모델 로드
        self.model_type = model_type.lower()

        if self.model_type == 'dino':
            self._load_dino()
        elif self.model_type == 'clip':
            self._load_clip()
        else:
            raise ValueError(f"지원하지 않는 모델: {model_type}. 'dino' 또는 'clip'을 사용하세요.")

        # 품종 분류 모델 초기화
        self._init_breed_classifier(use_finetuned_breed)

        print("[ImageAnalyzer] 초기화 완료\n")

    def _load_dino(self):
        """DINO 모델 로드"""
        print("   DINO 모델 로딩...")
        self.feature_model = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vits8',
            trust_repo=True,
            verbose=False
        )
        self.feature_model = self.feature_model.to(self.device)
        self.feature_model.eval()

        # DINO 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.feature_processor = None
        print("   DINO 로딩 완료")

    def _load_clip(self):
        """CLIP 모델 로드"""
        print("   CLIP 모델 로딩...")
        from transformers import CLIPModel, CLIPProcessor

        clip_model_path = os.path.join(self.models_dir, 'clip-vit-base-patch32')

        # 로컬에 모델이 있으면 로컬에서 로드, 없으면 다운로드 후 저장
        if os.path.exists(clip_model_path):
            print("   로컬 캐시에서 CLIP 모델 로드 중...")
            self.feature_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True)
            self.feature_processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)
        else:
            print("   CLIP 모델 다운로드 중 (최초 1회만)...")
            self.feature_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.feature_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # 로컬에 저장
            self.feature_model.save_pretrained(clip_model_path)
            self.feature_processor.save_pretrained(clip_model_path)
            print(f"   모델 저장 완료: {clip_model_path}")

        self.feature_model.eval()
        self.transform = None
        print("   CLIP 로딩 완료")

    def _init_breed_classifier(self, use_finetuned_breed):
        """품종 분류 모델 초기화"""
        import json
        from torchvision import models
        import torch.nn as nn

        self.use_finetuned_breed = use_finetuned_breed
        self.breed_classifier = None
        self.breed_labels = None
        self.breed_transform = None

        # Fine-tuned 모델 사용 시도
        if use_finetuned_breed:
            model_path = os.path.join(self.models_dir, 'breed_classifier_best.pth')
            labels_path = os.path.join(self.models_dir, 'breed_labels.json')

            if os.path.exists(model_path) and os.path.exists(labels_path):
                print("   Fine-tuned ResNet18 품종 분류 모델 로딩...")

                # 레이블 로드
                with open(labels_path, 'r', encoding='utf-8') as f:
                    breed_to_idx = json.load(f)

                self.breed_labels = {v: k for k, v in breed_to_idx.items()}
                num_classes = len(breed_to_idx)

                # 모델 로드
                self.breed_classifier = models.resnet18(pretrained=False)
                self.breed_classifier.fc = nn.Linear(self.breed_classifier.fc.in_features, num_classes)
                self.breed_classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                self.breed_classifier = self.breed_classifier.to(self.device)
                self.breed_classifier.eval()

                # 전처리 변환
                self.breed_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                print(f"   Fine-tuned 모델 로드 완료 ({num_classes}개 클래스)")
            else:
                print(f"   Fine-tuned 모델 없음 - CLIP Zero-shot으로 대체")
                self.use_finetuned_breed = False

        # CLIP Zero-shot 사용 (fallback)
        if not self.use_finetuned_breed:
            # 주요 개 품종 목록
            self.dog_breeds = [
                "Poodle", "Golden Retriever", "Labrador Retriever", "Bulldog", "German Shepherd",
                "Beagle", "Rottweiler", "Yorkshire Terrier", "Dachshund", "Boxer",
                "Shih Tzu", "Chihuahua", "Pomeranian", "Husky", "Corgi",
                "Maltese", "Cocker Spaniel", "Border Collie", "Samoyed", "Jindo"
            ]

            # 주요 고양이 품종 목록
            self.cat_breeds = [
                "Persian", "Maine Coon", "Siamese", "Ragdoll", "British Shorthair",
                "Bengal", "Abyssinian", "Birman", "Sphynx", "Russian Blue",
                "Scottish Fold", "American Shorthair", "Norwegian Forest Cat", "Korat", "Turkish Angora"
            ]

            print("   품종 분류: CLIP Zero-shot")

    def process_and_extract_features(self, image_path):
        """
        이미지 처리 및 특징 추출 (전체 파이프라인)

        Args:
            image_path (str): 이미지 파일 경로

        Returns:
            tuple: (특징 벡터, 동물 종류, 품종) 또는 (None, None, None) (동물 미탐지)
                - features (numpy.ndarray): 특징 벡터 (shape: (1, dim))
                - animal_type (str): 'dog' 또는 'cat'
                - breed (str): 품종 이름 (예: "Poodle", "Persian")
        """
        try:
            # Step 1: 이미지 로드
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)

            # Step 2: YOLO 객체 탐지
            results = self.yolo_model(img_np, verbose=False)

            # Step 3: 동물 필터링
            animal_detected = False
            animal_type = None
            cropped_img = None

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]
                    confidence = float(box.conf[0])

                    # COCO 데이터셋: dog(16), cat(15)
                    if class_name in ['dog', 'cat'] and confidence > 0.3:
                        animal_detected = True
                        animal_type = class_name  # 'dog' 또는 'cat' 저장

                        # Bounding box 좌표 추출
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # 이미지 crop
                        cropped_img = img.crop((x1, y1, x2, y2))
                        break

                if animal_detected:
                    break

            # 동물 미탐지
            if not animal_detected or cropped_img is None:
                return None, None, None

            # Step 4: 특징 벡터 추출
            if self.model_type == 'dino':
                features = self._extract_dino_features(cropped_img)
            else:
                features = self._extract_clip_features(cropped_img)

            # Step 5: 품종 분류
            breed = self.classify_breed(cropped_img, animal_type)

            return features, animal_type, breed

        except Exception as e:
            print(f"[ERROR] 이미지 처리 실패: {str(e)}")
            return None, None, None

    def _extract_dino_features(self, img):
        """DINO 특징 추출"""
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_model(img_tensor)

        return features.cpu().numpy()

    def _extract_clip_features(self, img):
        """CLIP 특징 추출"""
        inputs = self.feature_processor(images=img, return_tensors="pt")

        with torch.no_grad():
            features = self.feature_model.get_image_features(**inputs)

        return features.cpu().numpy()

    def classify_breed(self, img, animal_type):
        """
        품종 분류 (Fine-tuned ResNet18 또는 CLIP Zero-shot)

        Args:
            img (PIL.Image): 동물 이미지 (크롭된 이미지)
            animal_type (str): 'dog' 또는 'cat'

        Returns:
            str: 품종 이름 (예: "Poodle", "Persian")
        """
        try:
            # Fine-tuned 모델 사용
            if self.use_finetuned_breed and self.breed_classifier is not None:
                # 이미지 전처리
                input_tensor = self.breed_transform(img).unsqueeze(0).to(self.device)

                # 품종 예측
                with torch.no_grad():
                    output = self.breed_classifier(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    predicted_idx = output.argmax(dim=1).item()
                    confidence = probs[0, predicted_idx].item()

                # 품종 이름 조회
                breed_folder = self.breed_labels[predicted_idx]

                # 품종 이름 정리 (n02113624-toy_poodle -> Toy Poodle)
                breed_name = breed_folder.split('-', 1)[1].replace('_', ' ').title()

                print(f"   품종 분류 (Fine-tuned): {breed_name} (신뢰도: {confidence:.2f})")
                return breed_name

            # CLIP Zero-shot Classification (fallback)
            elif self.model_type == 'clip':
                # 품종 목록 선택
                if animal_type == 'dog':
                    breeds = self.dog_breeds
                elif animal_type == 'cat':
                    breeds = self.cat_breeds
                else:
                    return "Unknown"

                # 텍스트 프롬프트 생성
                text_inputs = [f"a photo of a {breed}" for breed in breeds]

                # 이미지와 텍스트 임베딩 계산
                inputs = self.feature_processor(
                    text=text_inputs,
                    images=img,
                    return_tensors="pt",
                    padding=True
                )

                with torch.no_grad():
                    outputs = self.feature_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # 가장 높은 확률의 품종 선택
                max_prob_idx = probs.argmax().item()
                breed = breeds[max_prob_idx]
                confidence = probs[0, max_prob_idx].item()

                print(f"   품종 분류 (CLIP Zero-shot): {breed} (신뢰도: {confidence:.2f})")
                return breed

            else:
                return "Unknown (Model not available)"

        except Exception as e:
            print(f"[ERROR] 품종 분류 실패: {str(e)}")
            return "Unknown"


if __name__ == "__main__":
    """
    테스트 코드

    실행 방법:
    1. CLIP 테스트: python ml_pipeline.py
    2. DINO 테스트: python ml_pipeline.py dino
    """
    import sys
    import os

    # 모델 타입 선택
    model_type = 'clip'
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'dino':
        model_type = 'dino'

    print("=" * 60)
    print(f"ML Pipeline 테스트 ({model_type.upper()} 모델)")
    print("=" * 60 + "\n")

    # ImageAnalyzer 초기화
    analyzer = ImageAnalyzer(model_type=model_type)

    # 테스트 이미지 다운로드
    print("[테스트] 샘플 이미지 다운로드 중...")
    import requests
    from io import BytesIO

    test_url = "https://images.dog.ceo/breeds/terrier-yorkshire/n02094433_1024.jpg"

    try:
        response = requests.get(test_url, timeout=10)
        response.raise_for_status()

        # 임시 저장
        test_path = "static/uploads/test_dog.jpg"
        os.makedirs("static/uploads", exist_ok=True)

        with open(test_path, 'wb') as f:
            f.write(response.content)

        print(f"   이미지 저장: {test_path}\n")

        # 특징 추출 테스트
        print("[테스트] 특징 벡터 추출 중...")
        features = analyzer.process_and_extract_features(test_path)

        if features is not None:
            print("\n[SUCCESS] 특징 추출 성공!")
            print(f"   Shape: {features.shape}")
            print(f"   Type: {type(features).__name__}")
            print(f"   차원: {features.shape[1]}차원")
            print(f"   값 범위: [{features.min():.4f}, {features.max():.4f}]")
            print(f"   평균: {features.mean():.4f}")
            print(f"   샘플 (처음 5개): {features[0, :5]}")

            print("\n" + "=" * 60)
            print("[RESULT] ML Pipeline 테스트 성공!")
            print(f"   YOLO + {model_type.upper()} 파이프라인이 정상 작동합니다.")
            print("=" * 60)
        else:
            print("\n[ERROR] 동물이 탐지되지 않았습니다.")
            print("   다른 이미지로 테스트해보세요.")

    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {str(e)}")
        print("   인터넷 연결을 확인하세요.")
