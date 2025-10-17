"""
DINO 모델 테스트 스크립트
목적: DINO 모델의 작동 여부를 48시간 내에 검증
성공 기준: 특징 벡터가 정상 출력되면 성공
"""

import torch
import time
import sys
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from torchvision import transforms


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
    # 시도 1: 인터넷에서 다운로드
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

        # 시도 2: 로컬 더미 이미지 생성
        try:
            # 224x224 RGB 테스트 이미지
            dummy_img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(dummy_img)
            print(f"   [SUCCESS] 백업 이미지 생성 성공 (크기: {img.size})")
            return img

        except Exception as e2:
            raise Exception(f"이미지 로드 완전 실패: {str(e2)}")


def load_dino_model(device, max_retries=3):
    """
    DINO 모델 로드 (재시도 로직 포함)
    Args:
        device: torch.device
        max_retries: 최대 재시도 횟수
    Returns:
        tuple: (model, success)
    """
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"   재시도 {attempt}/{max_retries}...")

            model = torch.hub.load(
                'facebookresearch/dino:main',
                'dino_vits8',
                trust_repo=True,
                verbose=False
            )
            model = model.to(device)
            model.eval()
            return model, True

        except Exception as e:
            print(f"   [WARNING] 시도 {attempt} 실패: {str(e)[:80]}...")

            if attempt < max_retries:
                wait_time = 5
                print(f"   {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                print(f"   [ERROR] 모든 시도 실패")
                return None, False

    return None, False


def extract_features(model, img, device):
    """
    이미지에서 특징 벡터 추출
    Args:
        model: DINO 모델
        img: PIL Image
        device: torch.device
    Returns:
        numpy.ndarray: 특징 벡터
    """
    # DINO 표준 전처리
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)

    return features.cpu().numpy()


def test_dino_model():
    """
    DINO 모델 전체 테스트 함수
    Returns:
        bool: 성공 시 True, 실패 시 False
    """
    print("=" * 60)
    print("        DINO 모델 테스트 시작")
    print("=" * 60)

    try:
        # Step 1: 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[1/4] 디바이스 설정: {device}")
        if device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        # Step 2: DINO 모델 로드
        print("\n[2/4] DINO 모델 로딩 중", end='')
        print_progress("", steps=3, delay=0.5)

        start_time = time.time()
        model, success = load_dino_model(device)

        if not success:
            raise Exception("DINO 모델 로딩 실패")

        load_time = time.time() - start_time
        print(f"[SUCCESS] DINO 모델 로딩 성공 (소요 시간: {load_time:.2f}초)")

        # Step 3: 테스트 이미지 로드
        print("\n[3/4] 테스트 이미지 준비 중...")
        img = load_test_image()

        # Step 4: 특징 추출
        print("\n[4/4] 특징 벡터 추출 중", end='')
        print_progress("", steps=2, delay=0.3)

        features_np = extract_features(model, img, device)

        print(f"[SUCCESS] 특징 벡터 추출 성공")
        print(f"   - Shape: {features_np.shape}")
        print(f"   - Type: {type(features_np).__name__}")
        print(f"   - 차원: {features_np.shape[1]}차원")
        print(f"   - 값 범위: [{features_np.min():.4f}, {features_np.max():.4f}]")
        print(f"   - 평균: {features_np.mean():.4f}")
        print(f"   - 샘플 (처음 5개): {features_np[0, :5]}")

        # Step 5: 성공 판정
        print("\n" + "=" * 60)

        expected_dim = 384
        actual_dim = features_np.shape[1]

        # 기본 검증: 1개 샘플, 양수 차원
        if features_np.shape[0] == 1 and actual_dim > 0:
            print("[SUCCESS] DINO 모델 테스트 성공!")
            print(f"   특징 벡터: {actual_dim}차원")

            if actual_dim == expected_dim:
                print("   [PASS] 예상 차원(384)과 일치")
            else:
                print(f"   [WARNING] 예상 차원({expected_dim})과 다름, 하지만 사용 가능")

            print("   이 모델을 메인 프로젝트에서 사용할 수 있습니다.")
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
        print("[ERROR] DINO 모델 테스트 실패")
        print(f"\n에러 메시지:")
        print(f"   {str(e)}")
        print("\n권장 조치:")
        print("   1. 인터넷 연결 확인 (GitHub, 이미지 다운로드)")
        print("   2. PyTorch 버전 확인: pip list | grep torch")
        print("   3. CLIP 백업 모델로 전환: python test_clip.py")
        print("=" * 60)
        print("\n다음 단계: CLIP 백업 모델 테스트 (python test_clip.py)")
        return False


if __name__ == "__main__":
    success = test_dino_model()

    # 최종 결과 요약
    print("\n" + "=" * 60)
    if success:
        print("[RESULT] DINO 모델 검증 완료 - Plan A 진행 가능")
    else:
        print("[RESULT] DINO 실패 - Plan B(CLIP) 전환 필요")
    print("=" * 60)
