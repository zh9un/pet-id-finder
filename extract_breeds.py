"""
Stanford Dogs Dataset에서 필요한 9개 품종만 추출
"""

import os
import shutil

# 경로 설정
dataset_path = "datasets/stanford_dogs/images"
output_path = "datasets/dog_breeds_9"

# 필요한 9개 품종 (소문자로 검색)
target_breeds = [
    "poodle",
    "yorkshire",
    "maltese",
    "chihuahua",
    "corgi",
    "bichon",
    "shih",
    "golden",
    "pomeranian"
]

def find_breed_folders():
    """images 폴더에서 필요한 품종 폴더 찾기"""
    print("=" * 60)
    print("Stanford Dogs Dataset - 품종 폴더 검색")
    print("=" * 60)

    all_folders = os.listdir(dataset_path)
    print(f"전체 품종 수: {len(all_folders)}\n")

    found_breeds = {}

    for folder in all_folders:
        folder_lower = folder.lower()
        for breed in target_breeds:
            if breed in folder_lower:
                found_breeds[breed] = folder
                image_count = len(os.listdir(os.path.join(dataset_path, folder)))
                print(f"[OK] 발견: {folder} ({image_count}장)")
                break

    print("\n" + "=" * 60)
    print(f"찾은 품종: {len(found_breeds)} / {len(target_breeds)}")

    # 못 찾은 품종 표시
    missing = set(target_breeds) - set(found_breeds.keys())
    if missing:
        print(f"누락된 품종: {', '.join(missing)}")

    print("=" * 60)

    return found_breeds

def copy_breed_data(found_breeds):
    """선택된 품종 데이터만 복사"""
    print("\n데이터 복사를 시작합니다...")

    # 출력 폴더 생성
    os.makedirs(output_path, exist_ok=True)

    for breed_key, folder_name in found_breeds.items():
        src = os.path.join(dataset_path, folder_name)
        dst = os.path.join(output_path, folder_name)

        print(f"복사 중: {folder_name}...")
        shutil.copytree(src, dst)
        print(f"  완료: {len(os.listdir(dst))}장")

    print("\n✅ 모든 데이터 복사 완료!")
    print(f"저장 위치: {output_path}")

if __name__ == "__main__":
    # 1. 품종 폴더 찾기
    found_breeds = find_breed_folders()

    # 2. 확인 및 복사
    if len(found_breeds) >= 7:  # 최소 7개 이상이면 진행
        print("\n데이터를 복사하시겠습니까? (y/n)")
        # 자동으로 yes
        copy_breed_data(found_breeds)
    else:
        print("\n[ERROR] 품종이 부족합니다. 데이터셋을 확인해주세요.")
