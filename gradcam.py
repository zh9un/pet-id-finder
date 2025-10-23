"""
Grad-CAM (Gradient-weighted Class Activation Mapping) 구현

AI가 품종 판단 시 이미지의 어느 부분을 중요하게 보는지 시각화
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


class GradCAM:
    """Grad-CAM 시각화 클래스"""

    def __init__(self, model, target_layer):
        """
        Args:
            model: ResNet 모델
            target_layer: 시각화할 레이어 (예: model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook 등록
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Forward/Backward hook 등록"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Hook 등록
        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def generate_cam(self, input_image, target_class):
        """
        Grad-CAM 생성

        Args:
            input_image (torch.Tensor): 입력 이미지 (1, 3, 224, 224)
            target_class (int): 타겟 클래스 인덱스

        Returns:
            numpy.ndarray: Grad-CAM heatmap (224, 224)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_image)

        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # Grad-CAM 계산
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global Average Pooling
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize(self, original_image, cam, alpha=0.4):
        """
        Grad-CAM을 원본 이미지에 오버레이

        Args:
            original_image (PIL.Image): 원본 이미지
            cam (numpy.ndarray): Grad-CAM heatmap
            alpha (float): 오버레이 투명도

        Returns:
            PIL.Image: 시각화된 이미지
        """
        # 원본 이미지를 numpy 배열로 변환
        img = np.array(original_image)
        h, w = img.shape[:2]

        # CAM을 원본 이미지 크기로 리사이즈
        cam_resized = cv2.resize(cam, (w, h))

        # Heatmap 생성 (JET 컬러맵)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 오버레이
        overlayed = heatmap * alpha + img * (1 - alpha)
        overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)

        return Image.fromarray(overlayed)

    def __del__(self):
        """Hook 제거"""
        for handle in self.hook_handles:
            handle.remove()


def generate_gradcam_for_pet(model, image_path, target_class, device='cpu'):
    """
    반려동물 이미지에 대한 Grad-CAM 생성 (편의 함수)

    Args:
        model: 학습된 ResNet 모델
        image_path (str): 이미지 경로
        target_class (int): 예측된 품종 클래스
        device (str): 'cpu' 또는 'cuda'

    Returns:
        PIL.Image: Grad-CAM 시각화 이미지
    """
    # 이미지 로드
    original_image = Image.open(image_path).convert('RGB')

    # 전처리
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Grad-CAM 생성
    gradcam = GradCAM(model, target_layer=model.layer4)
    cam = gradcam.generate_cam(input_tensor, target_class)

    # 시각화를 위해 224x224로 리사이즈한 원본 이미지 생성
    resized_image = original_image.resize((224, 224))
    result = gradcam.visualize(resized_image, cam)

    return result


if __name__ == "__main__":
    """
    테스트 코드

    실행 방법:
    python gradcam.py
    """
    import os
    import json
    from torchvision import models
    import torch.nn as nn

    print("=" * 60)
    print("   Grad-CAM 테스트")
    print("=" * 60)

    # 학습된 모델 로드
    model_path = "models/breed_classifier_best.pth"
    labels_path = "models/breed_labels.json"

    if not os.path.exists(model_path):
        print(f"\n[ERROR] 모델 파일이 없습니다: {model_path}")
        print("먼저 train_breed_classifier.py를 실행하여 모델을 학습하세요.")
        exit(1)

    if not os.path.exists(labels_path):
        print(f"\n[ERROR] 레이블 파일이 없습니다: {labels_path}")
        exit(1)

    # 레이블 로드
    with open(labels_path, 'r', encoding='utf-8') as f:
        breed_to_idx = json.load(f)

    num_classes = len(breed_to_idx)

    # 모델 로드
    print(f"\n[1/4] 모델 로드 중... ({num_classes}개 클래스)")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("   완료!")

    # 테스트 이미지 선택
    test_image = "datasets/dog_breeds_9/n02113624-toy_poodle/n02113624_1000.jpg"

    if not os.path.exists(test_image):
        # 첫 번째 품종의 첫 번째 이미지 사용
        breed_folders = os.listdir("datasets/dog_breeds_9")
        breed_path = os.path.join("datasets/dog_breeds_9", breed_folders[0])
        images = [f for f in os.listdir(breed_path) if f.endswith('.jpg')]
        test_image = os.path.join(breed_path, images[0])

    print(f"\n[2/4] 테스트 이미지: {test_image}")

    # 품종 예측
    print("\n[3/4] 품종 예측 중...")
    img = Image.open(test_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    # 품종 이름 찾기
    idx_to_breed = {v: k for k, v in breed_to_idx.items()}
    predicted_breed = idx_to_breed[predicted_class]

    print(f"   예측 품종: {predicted_breed}")

    # Grad-CAM 생성
    print("\n[4/4] Grad-CAM 생성 중...")
    result_image = generate_gradcam_for_pet(
        model, test_image, predicted_class, device='cpu'
    )

    # 결과 저장
    output_path = "static/uploads/gradcam_test.jpg"
    os.makedirs("static/uploads", exist_ok=True)
    result_image.save(output_path)

    print(f"   완료!")
    print(f"\n저장 위치: {output_path}")
    print("\n" + "=" * 60)
    print("   Grad-CAM 테스트 성공!")
    print("=" * 60)
