"""
CLIP 모델 백업 테스트 스크립트
목적: DINO 실패 시 즉시 전환 가능한 대체 솔루션
성공 기준: 특징 벡터가 정상 출력되면 성공
"""

import torch
import time
import sys
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from transformers import CLIPProcessor, CLIPModel


def print_progress(message, steps=3, delay=0.3):
    """진행 상황 애니메이션"""
    for i in range(steps):
        sys.stdout.write(f"\r{message}{'.' * (i + 1)}")
        sys.stdout.flush()
        time.sleep(delay)
    print()


def load_test_image():
    """
    테스트 이미지 로드 (인터넷 실패 시 로컬 백업)
    Returns:
        PIL.Image: RGB 이미지
    """
    try:
        test_image_url = "https://images.dog.ceo/breeds/terrier-yorkshire/n02094433_1024.jpg"
        print("   인터넷에서 테스트 이미지 다운로드 시도...")
        response = requests.get(test_image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"   [SUCCESS] 다운로드 성공 (크기: {img.size})")
        return img

    except Exception as e:
        print(f"   [WARNING] 인터넷 다운로드 실패: {str(e)[:50]}...")
        print("   로컬 백업 이미지 생성 중...")

        dummy_img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(dummy_img)
        print(f"   [SUCCESS] 백업 이미지 생성 성공 (크기: {img.size})")
        return img


def load_clip_model(max_retries=3):
    """
    CLIP 모델 로드 (재시도 로직 포함)
    Args:
        max_retries: 최대 재시도 횟수
    Returns:
        tuple: (model, processor, success)
    """
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"   재시도 {attempt}/{max_retries}...")

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            model.eval()
            return model, processor, True

        except Exception as e:
            print(f"   [WARNING] 시도 {attempt} 실패: {str(e)[:80]}...")

            if attempt < max_retries:
                wait_time = 5
                print(f"   {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                print(f"   [ERROR] 모든 시도 실패")
                return None, None, False

    return None, None, False


def extract_features(model, processor, img):
    """
    이미지에서 특징 벡터 추출
    Args:
        model: CLIP 모델
        processor: CLIP processor
        img: PIL Image
    Returns:
        numpy.ndarray: 특징 벡터
    """
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    return features.cpu().numpy()


def test_clip_model():
    """
    CLIP 모델 전체 테스트 함수
    Returns:
        bool: 성공 시 True, 실패 시 False
    """
    print("=" * 60)
    print("        CLIP 모델 테스트 시작")
    print("=" * 60)

    try:
        # Step 1: CLIP 모델 로드
        print("\n[1/3] CLIP 모델 로딩 중", end='')
        print_progress("", steps=3, delay=0.5)

        start_time = time.time()
        model, processor, success = load_clip_model()

        if not success:
            raise Exception("CLIP 모델 로딩 실패")

        load_time = time.time() - start_time
        print(f"[SUCCESS] CLIP 모델 로딩 성공 (소요 시간: {load_time:.2f}초)")

        # Step 2: 테스트 이미지 로드
        print("\n[2/3] 테스트 이미지 준비 중...")
        img = load_test_image()

        # Step 3: 특징 추출
        print("\n[3/3] 특징 벡터 추출 중", end='')
        print_progress("", steps=2, delay=0.3)

        features_np = extract_features(model, processor, img)

        print(f"[SUCCESS] 특징 벡터 추출 성공")
        print(f"   - Shape: {features_np.shape}")
        print(f"   - Type: {type(features_np).__name__}")
        print(f"   - 차원: {features_np.shape[1]}차원")
        print(f"   - 값 범위: [{features_np.min():.4f}, {features_np.max():.4f}]")
        print(f"   - 평균: {features_np.mean():.4f}")
        print(f"   - 샘플 (처음 5개): {features_np[0, :5]}")

        # Step 4: 성공 판정
        print("\n" + "=" * 60)

        expected_dim = 512
        actual_dim = features_np.shape[1]

        if features_np.shape[0] == 1 and actual_dim > 0:
            print("[SUCCESS] CLIP 모델 테스트 성공!")
            print(f"   특징 벡터: {actual_dim}차원")

            if actual_dim == expected_dim:
                print("   [PASS] 예상 차원(512)과 일치")
            else:
                print(f"   [WARNING] 예상 차원({expected_dim})과 다름, 하지만 사용 가능")

            print("\n[비교] DINO vs CLIP")
            print("   - DINO: 384차원, 로딩 느림, 더 정밀한 특징")
            print("   - CLIP: 512차원, 로딩 빠름, 안정적")

            print("\n결론: 이 모델을 메인 프로젝트에서 사용할 수 있습니다.")
            print("=" * 60)
            print("\n다음 단계: ML Pipeline 구현 (ml_pipeline.py)")
            return True
        else:
            print("[ERROR] 예상치 못한 출력 형태:")
            print(f"   Shape: {features_np.shape}")
            print("   벡터 형태가 올바르지 않습니다.")
            print("=" * 60)
            return False

    except Exception as e:
        print("\n" + "=" * 60)
        print("[ERROR] CLIP 모델 테스트 실패")
        print(f"\n에러 메시지:")
        print(f"   {str(e)}")
        print("\n권장 조치:")
        print("   1. 인터넷 연결 확인")
        print("   2. transformers 패키지 확인: pip list | grep transformers")
        print("   3. 캐시 삭제 후 재시도: rm -rf ~/.cache/huggingface")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = test_clip_model()

    print("\n" + "=" * 60)
    if success:
        print("[RESULT] CLIP 모델 검증 완료 - Plan B로 진행 가능")
    else:
        print("[RESULT] 모든 모델 실패 - 환경 점검 필요")
    print("=" * 60)
