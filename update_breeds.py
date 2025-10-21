"""
기존 데이터의 품종 정보 업데이트 스크립트

실행 방법: python update_breeds.py
"""

import sqlite3
import os
from PIL import Image
from ml_pipeline import ImageAnalyzer

def update_breeds():
    """데이터베이스의 모든 pets에 대해 품종 정보 업데이트"""

    print("=" * 60)
    print("   품종 정보 업데이트 스크립트")
    print("=" * 60)

    # ML Pipeline 초기화
    print("\n[1/4] ML Pipeline 초기화 중...")
    analyzer = ImageAnalyzer(model_type='clip')
    print("✓ ML Pipeline 초기화 완료\n")

    # 데이터베이스 연결
    print("[2/4] 데이터베이스 연결 중...")
    conn = sqlite3.connect('pets.db')
    c = conn.cursor()

    # breed가 NULL인 레코드 조회
    c.execute('SELECT id, image_path, animal_type FROM pets WHERE breed IS NULL OR breed = ""')
    pets_to_update = c.fetchall()

    total = len(pets_to_update)
    print(f"✓ 업데이트 대상: {total}개\n")

    if total == 0:
        print("업데이트할 데이터가 없습니다.")
        conn.close()
        return

    # 각 레코드 처리
    print("[3/4] 품종 분석 중...")
    print("-" * 60)

    success_count = 0
    error_count = 0

    for idx, (pet_id, image_path, animal_type) in enumerate(pets_to_update, 1):
        try:
            # 진행 상황 출력
            print(f"[{idx}/{total}] ID {pet_id} 처리 중... ", end='')

            # 이미지 파일 존재 확인
            if not os.path.exists(image_path):
                print(f"❌ 파일 없음: {image_path}")
                error_count += 1
                continue

            # 이미지 로드 및 품종 분류
            img = Image.open(image_path).convert('RGB')

            # YOLO로 동물 영역 찾기
            import numpy as np
            img_np = np.array(img)
            results = analyzer.yolo_model(img_np, verbose=False)

            cropped_img = None
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]
                    confidence = float(box.conf[0])

                    if class_name in ['dog', 'cat'] and confidence > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cropped_img = img.crop((x1, y1, x2, y2))
                        break

                if cropped_img:
                    break

            if cropped_img is None:
                print("❌ 동물 미탐지")
                error_count += 1
                continue

            # 품종 분류
            breed = analyzer.classify_breed(cropped_img, animal_type)

            # 데이터베이스 업데이트
            c.execute('UPDATE pets SET breed = ? WHERE id = ?', (breed, pet_id))
            conn.commit()

            print(f"✓ {breed}")
            success_count += 1

        except Exception as e:
            print(f"❌ 오류: {str(e)}")
            error_count += 1

    conn.close()

    # 결과 출력
    print("-" * 60)
    print(f"\n[4/4] 완료!")
    print(f"✓ 성공: {success_count}개")
    print(f"✗ 실패: {error_count}개")
    print(f"총 처리: {success_count + error_count}개 / {total}개")
    print("\n" + "=" * 60)
    print("   품종 정보 업데이트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    update_breeds()
