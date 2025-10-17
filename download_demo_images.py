"""
포인핸드 사이트 이미지 자동 수집 스크립트 (원본 화질)

3그룹 전략으로 120장의 시연용 이미지를 자동 다운로드:
- 그룹 1: 순종 그룹 50장
- 그룹 2: 믹스견/묘 그룹 50장
- 그룹 3: 고난이도 그룹 20장 (수동 선별)
"""

import os
import requests
import time
from bs4 import BeautifulSoup
import urllib.parse

# 설정
OUTPUT_FOLDER = 'demo_images'
TOTAL_TARGET = 120
GROUP1_TARGET = 50  # 순종
GROUP2_TARGET = 50  # 믹스
GROUP3_TARGET = 20  # 고난이도 (그룹2에서 수동 선별)

# 순종 품종별 타겟
PUREBRED_TARGETS = {
    '포메라니안': 10,
    '푸들': 10,
    '치와와': 8,
    '말티즈': 8,
    '시츄': 7,
    '요크셔테리어': 7
}

# 포인핸드 기본 URL
BASE_URL = "https://www.pawinhand.kr"
ADOPTION_URL = f"{BASE_URL}/adoption/dog"


def create_output_folders():
    """출력 폴더 생성"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "group1_purebred"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "group2_mixed"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "group3_difficult"), exist_ok=True)
    print(f"[폴더 생성] '{OUTPUT_FOLDER}' 및 하위 3개 그룹 폴더 생성 완료")


def download_image(url, filepath):
    """이미지 다운로드"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            f.write(response.content)

        return True
    except Exception as e:
        print(f" [다운로드 실패: {e}]", end='')
        return False


def get_detail_page_image(detail_url):
    """상세 페이지에서 원본 이미지 URL 추출"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(detail_url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')

        # 원본 이미지 찾기 (여러 선택자 시도)
        img_selectors = [
            'div.animal-image img',
            'div.detail-image img',
            'img.main-image',
            'div.photo img'
        ]

        for selector in img_selectors:
            img_tag = soup.select_one(selector)
            if img_tag and img_tag.get('src'):
                img_url = img_tag['src']
                # 상대 URL을 절대 URL로 변환
                return urllib.parse.urljoin(BASE_URL, img_url)

        return None

    except Exception as e:
        return None


def fetch_pawinhand_list(breed=None, page=1):
    """
    포인핸드 목록 페이지에서 동물 정보 가져오기

    Returns:
        list: [(detail_url, thumbnail_url), ...] 형태의 리스트
    """
    try:
        params = {'page': page}
        if breed:
            params['search'] = breed

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(ADOPTION_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')

        # 동물 카드 찾기
        animal_cards = soup.select('div.animal-card, div.card, a.animal-item')

        results = []

        for card in animal_cards:
            # 상세 페이지 링크 찾기
            link_tag = card.find('a') if card.name != 'a' else card
            if not link_tag or not link_tag.get('href'):
                continue

            detail_url = urllib.parse.urljoin(BASE_URL, link_tag['href'])

            # 썸네일 이미지 (fallback용)
            img_tag = card.find('img')
            thumbnail_url = urllib.parse.urljoin(BASE_URL, img_tag['src']) if img_tag else None

            results.append((detail_url, thumbnail_url))

        return results

    except Exception as e:
        print(f"  [목록 스크래핑 실패] {e}")
        return []


def download_group(group_name, folder_name, target_count, breed=None):
    """지정된 그룹의 이미지를 목표 개수만큼 다운로드"""
    print("\n" + "=" * 60)
    print(f"[{group_name}] 다운로드 시작 ({target_count}장 목표)")
    if breed:
        print(f"  품종: {breed}")
    print("=" * 60)

    downloaded_count = 0
    page = 1
    max_pages = 10  # 최대 페이지 수 제한

    while downloaded_count < target_count and page <= max_pages:
        print(f"\n  페이지 {page} 스크래핑 중...")

        animal_list = fetch_pawinhand_list(breed=breed, page=page)

        if not animal_list:
            print("  더 이상 데이터가 없습니다.")
            break

        for detail_url, thumbnail_url in animal_list:
            if downloaded_count >= target_count:
                break

            # 파일명 생성
            filename = f"{folder_name}_{downloaded_count + 1:03d}.jpg"
            filepath = os.path.join(OUTPUT_FOLDER, folder_name, filename)

            print(f"  [{downloaded_count + 1}/{target_count}] {filename}", end='')

            # 1. 상세 페이지에서 원본 이미지 URL 추출 시도
            original_url = get_detail_page_image(detail_url)

            # 2. 원본을 못 찾으면 썸네일 사용
            image_url = original_url if original_url else thumbnail_url

            if not image_url:
                print(" [이미지 URL 없음]")
                continue

            # 3. 다운로드
            if download_image(image_url, filepath):
                print(" ✓")
                downloaded_count += 1
            else:
                print(" ✗")

            time.sleep(1.0)  # 서버 부하 방지 (원본 추출하므로 딜레이 증가)

        page += 1

    print(f"\n[{group_name} 완료] 총 {downloaded_count}장 다운로드")
    return downloaded_count


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("  포인핸드 시연용 이미지 자동 수집 스크립트")
    print("  (원본 화질 버전)")
    print("=" * 60)
    print(f"  목표: {TOTAL_TARGET}장")
    print(f"  - 그룹 1 (순종): {GROUP1_TARGET}장")
    print(f"  - 그룹 2 (믹스): {GROUP2_TARGET}장")
    print(f"  - 그룹 3 (고난이도): 그룹2에서 수동 선별")
    print("=" * 60)

    # 폴더 생성
    create_output_folders()

    # 시작 시간
    start_time = time.time()

    total_downloaded = 0

    # 그룹 1: 순종 다운로드
    print("\n" + "=" * 60)
    print("[그룹 1] 순종 그룹 다운로드")
    print("=" * 60)

    for breed, count in PUREBRED_TARGETS.items():
        downloaded = download_group(
            group_name=f"그룹1-{breed}",
            folder_name="group1_purebred",
            target_count=count,
            breed=breed
        )
        total_downloaded += downloaded

    # 그룹 2: 믹스견 다운로드
    downloaded = download_group(
        group_name="그룹2-믹스견",
        folder_name="group2_mixed",
        target_count=GROUP2_TARGET,
        breed=None  # 품종 지정 없이 검색
    )
    total_downloaded += downloaded

    # 소요 시간
    elapsed_time = time.time() - start_time

    # 결과 출력
    print("\n" + "=" * 60)
    print("  전체 다운로드 완료!")
    print("=" * 60)
    print(f"  총 다운로드: {total_downloaded}장")
    print(f"  소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")
    print(f"  저장 위치: {OUTPUT_FOLDER}/")
    print("=" * 60)

    # 그룹 3 안내
    print("\n[그룹 3 안내]")
    print("  'group2_mixed' 폴더에서 다음 조건의 사진 20장을 선별하여")
    print("  'group3_difficult' 폴더로 수동 이동해주세요:")
    print("  - 털이 심하게 엉킨 사진")
    print("  - 어두운 환경에서 촬영된 사진")
    print("  - 일부 미용되거나 상태가 안 좋은 사진")
    print("=" * 60)

    if total_downloaded < (GROUP1_TARGET + GROUP2_TARGET):
        print("\n[참고] 목표 개수에 미달했습니다.")
        print("수동으로 추가 이미지를 다운로드하거나,")
        print("스크립트를 다시 실행하여 부족한 만큼 채워주세요.")


if __name__ == "__main__":
    main()
