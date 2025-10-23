"""
품종 분류 모델 학습 스크립트

ResNet18 기반 Transfer Learning
- 9개 품종 분류
- CPU 학습 (예상 시간: 1-2시간)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import json
import time


class DogBreedDataset(Dataset):
    """강아지 품종 데이터셋"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 이미지 로드
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def load_dataset(data_dir):
    """데이터셋 로드 및 레이블 생성"""
    print("\n" + "=" * 60)
    print("   데이터셋 로드")
    print("=" * 60)

    image_paths = []
    labels = []
    breed_to_idx = {}

    breed_folders = sorted(os.listdir(data_dir))

    for idx, breed_folder in enumerate(breed_folders):
        breed_path = os.path.join(data_dir, breed_folder)

        if not os.path.isdir(breed_path):
            continue

        breed_to_idx[breed_folder] = idx

        # 이미지 파일 수집
        images = [f for f in os.listdir(breed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in images:
            img_path = os.path.join(breed_path, img_name)
            image_paths.append(img_path)
            labels.append(idx)

        print(f"[{idx+1}/{len(breed_folders)}] {breed_folder}: {len(images)}장")

    print("\n" + "-" * 60)
    print(f"총 {len(breed_folders)}개 품종, {len(image_paths)}장의 이미지")
    print("=" * 60)

    return image_paths, labels, breed_to_idx


def create_data_loaders(image_paths, labels, batch_size=32, test_size=0.2):
    """데이터 로더 생성"""
    print("\n데이터 분할: Train/Val = 80%/20%")

    # Train/Val 분할
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )

    print(f"- Train: {len(train_paths)}장")
    print(f"- Val: {len(val_paths)}장")

    # 데이터 증강 (Training용)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터 증강 없음 (Validation용)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 생성
    train_dataset = DogBreedDataset(train_paths, train_labels, train_transform)
    val_dataset = DogBreedDataset(val_paths, val_labels, val_transform)

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def create_model(num_classes):
    """ResNet18 모델 생성 (Transfer Learning)"""
    print("\n" + "=" * 60)
    print("   ResNet18 모델 로드 (Transfer Learning)")
    print("=" * 60)

    # Pre-trained ResNet18 로드
    model = models.resnet18(pretrained=True)

    # 마지막 FC layer만 교체 (9개 클래스)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    print(f"- 입력: 224x224 RGB 이미지")
    print(f"- 출력: {num_classes}개 클래스")
    print(f"- 마지막 FC layer 교체: {num_features} -> {num_classes}")
    print("=" * 60)

    return model


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """모델 학습"""
    print("\n" + "=" * 60)
    print("   모델 학습 시작")
    print("=" * 60)
    print(f"- Epochs: {num_epochs}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Device: {device}")
    print(f"- Batch Size: {train_loader.batch_size}")
    print("=" * 60 + "\n")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ===== Training Phase =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # 진행 상황 출력 (10 batch마다)
            if (batch_idx + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f}", end='\r')

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # ===== Validation Phase =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # 기록 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        # 결과 출력
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/breed_classifier_best.pth')
            print(f"   -> Best model saved (Val Acc: {val_acc:.2f}%)")

    print("\n" + "=" * 60)
    print(f"   학습 완료! Best Val Acc: {best_val_acc:.2f}%")
    print("=" * 60)

    return model, history


def save_label_mapping(breed_to_idx, filename='models/breed_labels.json'):
    """레이블 매핑 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(breed_to_idx, f, indent=2, ensure_ascii=False)
    print(f"\n레이블 매핑 저장: {filename}")


if __name__ == "__main__":
    # ===== 설정 =====
    DATA_DIR = "datasets/dog_breeds_9"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001

    # GPU/CPU 자동 선택
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("   품종 분류 모델 학습 (ResNet18 Transfer Learning)")
    print("=" * 60)
    print(f"디바이스: {device}")
    print(f"데이터 경로: {DATA_DIR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("=" * 60)

    # 모델 저장 폴더 생성
    os.makedirs('models', exist_ok=True)

    # 1. 데이터셋 로드
    image_paths, labels, breed_to_idx = load_dataset(DATA_DIR)
    num_classes = len(breed_to_idx)

    # 2. 데이터 로더 생성
    train_loader, val_loader = create_data_loaders(
        image_paths, labels, batch_size=BATCH_SIZE
    )

    # 3. 모델 생성
    model = create_model(num_classes)

    # 4. 학습
    start_time = time.time()
    model, history = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )
    total_time = time.time() - start_time

    # 5. 최종 모델 저장
    torch.save(model.state_dict(), 'models/breed_classifier_final.pth')
    print(f"\n최종 모델 저장: models/breed_classifier_final.pth")

    # 6. 레이블 매핑 저장
    save_label_mapping(breed_to_idx)

    # 7. 학습 히스토리 저장
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"학습 히스토리 저장: models/training_history.json")

    print("\n" + "=" * 60)
    print("   모든 작업 완료!")
    print("=" * 60)
    print(f"총 소요 시간: {total_time/60:.1f}분")
    print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print("\n생성된 파일:")
    print("- models/breed_classifier_best.pth (최고 성능 모델)")
    print("- models/breed_classifier_final.pth (최종 모델)")
    print("- models/breed_labels.json (레이블 매핑)")
    print("- models/training_history.json (학습 히스토리)")
    print("=" * 60)
