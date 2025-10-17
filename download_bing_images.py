"""
Bing Image Downloader를 이용한 데모 이미지 자동 수집
- 키워드별 이미지 다운로드 후 demo_images/ 단일 폴더에 저장
- 총 목표: ~50장
"""

import os
import shutil
from bing_image_downloader import downloader

# 출력 폴더 설정
OUTPUT_FOLDER = 'demo_images'  # 단일 폴더에 모두 저장

# 키워드 및 다운로드 개수 설정
KEYWORDS = [
    ("길고양이 코리안숏헤어", 15),
    ("유기견 믹스견 보호소", 15),
    ("유기동물 보호소 개", 15),
    ("길고양이 검은고양이", 10),
]


def download_images():
    """Bing에서 이미지 다운로드 후 단일 폴더로 정리"""

    print("\n" + "=" * 60)
    print("  Bing Image Downloader - 데모 이미지 자동 수집")
    print("=" * 60)

    total_target = sum([count for _, count in KEYWORDS])
    print(f"  총 목표: {total_target}장")
    print("=" * 60 + "\n")

    # 메인 폴더 생성
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 임시 다운로드 폴더
    temp_folder = 'temp_downloads'

    collected_count = 0
    image_counter = 1

    for keyword, limit in KEYWORDS:
        print(f"\n[키워드: {keyword}] 다운로드 중... (목표: {limit}장)")

        try:
            # Bing에서 임시 폴더로 다운로드
            downloader.download(
                keyword,
                limit=limit,
                output_dir=temp_folder,
                adult_filter_off=True,
                force_replace=False,
                timeout=15,
                verbose=False
            )

            # 다운로드된 파일들을 메인 폴더로 이동
            keyword_folder = os.path.join(temp_folder, keyword)

            if os.path.exists(keyword_folder):
                files = [f for f in os.listdir(keyword_folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]

                moved = 0
                for file in files[:limit]:  # 최대 limit개까지만
                    src = os.path.join(keyword_folder, file)
                    ext = os.path.splitext(file)[1]
                    dst = os.path.join(OUTPUT_FOLDER, f"pet_{image_counter:03d}{ext}")

                    try:
                        shutil.move(src, dst)
                        moved += 1
                        image_counter += 1
                    except Exception as e:
                        print(f"  [경고] 파일 이동 실패: {file} - {e}")

                collected_count += moved
                print(f"  [OK] {moved}장 수집 완료")
            else:
                print(f"  [FAIL] 다운로드 실패 (폴더 없음)")

        except Exception as e:
            print(f"  [ERROR] 오류 발생: {str(e)}")

    # 임시 폴더 정리
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    print("\n" + "=" * 60)
    print(f"  다운로드 완료!")
    print("=" * 60)
    print(f"  총 수집: {collected_count}장")
    print(f"  저장 위치: {OUTPUT_FOLDER}/")
    print("=" * 60 + "\n")

    if collected_count > 0:
        print("[다음 단계]")
        print("  python batch_register.py를 실행하여 DB에 등록하세요.")
        print("=" * 60 + "\n")
    else:
        print("[경고] 이미지가 수집되지 않았습니다.")
        print("  인터넷 연결을 확인하거나 키워드를 변경해보세요.")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    download_images()
