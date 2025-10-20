"""
경기데이터드림 API 연동 스크립트

실제 유기동물 보호 현황 데이터를 API에서 가져와
ML 분석 후 DB에 등록합니다.

API: 경기데이터드림 - 유기동물 보호 현황
"""

import os
import sqlite3
import json
import requests
import time
from datetime import datetime
from ml_pipeline import ImageAnalyzer

# ============================================================
# API 설정
# ============================================================
API_KEY = '34496c8897e74ffcb2ba95465a2b2ac2'
API_URL = 'https://openapi.gg.go.kr/AbdmAnimalProtect'
PAGE_SIZE = 50  # 한 페이지당 가져올 데이터 수
MAX_PAGES = 10  # 최대 페이지 수 (50 x 10 = 500개)

# 동물 종류 코드 (SPECIES_NM 필드)
SPECIES_DOG = '000114'
SPECIES_CAT_1 = '000072'
SPECIES_CAT_2 = '000200'

# ============================================================
# 로컬 설정
# ============================================================
IMAGE_FOLDER = 'static/api_images'
DB_PATH = 'pets.db'
MODEL_TYPE = 'clip'  # app.py와 동일한 모델 사용

# 재시도 설정
MAX_RETRIES = 3
RETRY_DELAY = 2  # 초


def init_db_with_location():
    """데이터베이스 초기화 (목격 정보 컬럼 포함)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 테이블이 없으면 생성
    c.execute('''
        CREATE TABLE IF NOT EXISTS pets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            embedding TEXT NOT NULL,
            animal_type TEXT,
            location TEXT,
            sighted_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("[Database] 데이터베이스 초기화 완료")


def fetch_api_data(page_index):
    """
    API에서 데이터 가져오기

    Args:
        page_index (int): 페이지 번호 (1부터 시작)

    Returns:
        list: API 응답 데이터 리스트 또는 None (실패 시)
    """
    params = {
        'KEY': API_KEY,
        'Type': 'json',
        'pIndex': page_index,
        'pSize': PAGE_SIZE
    }

    try:
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # API 응답 구조: AbdmAnimalProtect -> head, row
        if 'AbdmAnimalProtect' in data and len(data['AbdmAnimalProtect']) > 1:
            # row가 두 번째 요소에 있음
            rows = data['AbdmAnimalProtect'][1].get('row', [])
            return rows
        else:
            print(f"[API Warning] 페이지 {page_index}: 데이터 없음")
            return []

    except requests.exceptions.RequestException as e:
        print(f"[API ERROR] 페이지 {page_index} 요청 실패: {str(e)}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"[API ERROR] 페이지 {page_index} 응답 파싱 실패: {str(e)}")
        return None


def download_image(image_url, save_path):
    """
    이미지 다운로드 (재시도 로직 포함)

    Args:
        image_url (str): 이미지 URL
        save_path (str): 저장 경로

    Returns:
        bool: 성공 여부
    """
    if not image_url or image_url == '':
        return False

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(image_url, timeout=15, stream=True)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    재시도 중... ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    다운로드 실패: {str(e)}")
                return False

    return False


def parse_date(date_str):
    """
    YYYYMMDD 형식 문자열을 datetime 객체로 변환

    Args:
        date_str (str): YYYYMMDD 형식 날짜

    Returns:
        datetime: 변환된 datetime 객체 또는 None
    """
    if not date_str or len(date_str) != 8:
        return None

    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        return None


def register_from_api():
    """
    API에서 유기동물 데이터를 가져와 DB에 등록하는 메인 함수
    """
    print("=" * 60)
    print("  경기데이터드림 API 연동 스크립트")
    print("=" * 60)

    # 1. 이미지 폴더 생성
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    print(f"\n[1/5] 이미지 저장 폴더 생성: {IMAGE_FOLDER}")

    # 2. 데이터베이스 초기화
    print("\n[2/5] 데이터베이스 초기화 중...")
    init_db_with_location()

    # 3. ML Pipeline 초기화
    print(f"\n[3/5] ML Pipeline 초기화 중 (모델: {MODEL_TYPE.upper()})...")
    analyzer = ImageAnalyzer(model_type=MODEL_TYPE)

    # 4. 데이터베이스 연결
    print("\n[4/5] 데이터베이스 연결 완료")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 5. API 데이터 수집 및 등록
    print(f"\n[5/5] API에서 데이터 수집 중 (최대 {PAGE_SIZE * MAX_PAGES}개)...")
    print("=" * 60)

    total_fetched = 0  # API에서 가져온 총 개수
    total_filtered = 0  # 개/고양이 필터링 후 개수
    success_count = 0  # ML 탐지 성공 및 DB 등록 성공
    fail_count = 0  # 다운로드 실패 또는 ML 탐지 실패

    for page in range(1, MAX_PAGES + 1):
        print(f"\n[페이지 {page}/{MAX_PAGES}] 데이터 가져오는 중...")

        rows = fetch_api_data(page)

        if rows is None:
            print(f"[ERROR] 페이지 {page} API 요청 실패, 중단합니다.")
            break

        if not rows:
            print(f"[INFO] 페이지 {page}에 더 이상 데이터가 없습니다. 종료합니다.")
            break

        total_fetched += len(rows)
        print(f"  가져온 데이터: {len(rows)}개")

        # 각 항목 처리
        for row in rows:
            # 동물 종류 필터링 (개/고양이만)
            species_code = row.get('SPECIES_NM', '')
            if species_code not in [SPECIES_DOG, SPECIES_CAT_1, SPECIES_CAT_2]:
                continue

            total_filtered += 1

            # 동물 타입 결정
            animal_type = 'dog' if species_code == SPECIES_DOG else 'cat'
            animal_name_kr = "강아지" if animal_type == 'dog' else "고양이"

            # 유기번호 (파일명으로 사용)
            abandonment_no = row.get('ABDM_IDNTFY_NO', '') or row.get('SIGUN_CD', 'unknown')

            # 이미지 URL
            image_url = row.get('IMAGE_COURS', '') or row.get('THUMB_IMAGE_COURS', '')
            if not image_url:
                print(f"  건너뜀: {abandonment_no} - 이미지 URL 없음")
                fail_count += 1
                continue

            # 발견 장소
            location = (
                row.get('DISCVRY_PLC_INFO', '') or
                row.get('REFINE_ROADNM_ADDR', '') or
                row.get('REFINE_LOTNO_ADDR', '') or
                '장소 정보 없음'
            )

            # 발견 날짜
            recept_date = row.get('RECEPT_DE', '')
            sighted_at = parse_date(recept_date)
            if sighted_at is None:
                sighted_at = datetime.now()  # 날짜 파싱 실패 시 현재 시각 사용

            # 파일명 생성 (확장자는 jpg로 통일)
            safe_filename = f"{abandonment_no.replace('/', '_')}.jpg"
            filepath = os.path.join(IMAGE_FOLDER, safe_filename)

            print(f"\n  처리 중: {abandonment_no} ({animal_name_kr})")

            # 이미지 다운로드
            if not download_image(image_url, filepath):
                fail_count += 1
                continue

            # ML Pipeline으로 특징 벡터 추출
            features, detected_type = analyzer.process_and_extract_features(filepath)

            if features is None or detected_type is None:
                print(f"    -> YOLO 동물 미탐지, 이미지 삭제")
                if os.path.exists(filepath):
                    os.remove(filepath)
                fail_count += 1
                continue

            # YOLO가 탐지한 동물 타입과 API의 종류가 일치하는지 확인 (선택사항)
            if detected_type != animal_type:
                print(f"    -> 경고: API는 {animal_name_kr}, YOLO는 {detected_type} 탐지")
                print(f"    -> YOLO 결과를 우선하여 {detected_type}로 등록")
                animal_type = detected_type  # YOLO 결과를 신뢰

            # numpy 배열을 JSON 문자열로 변환
            embedding_json = json.dumps(features.tolist())

            # DB에 저장
            c.execute('''
                INSERT INTO pets (image_path, embedding, animal_type, location, sighted_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (filepath, embedding_json, animal_type, location, sighted_at))

            print(f"    -> 등록 성공!")
            print(f"       종류: {animal_type}")
            print(f"       장소: {location[:50]}...")
            print(f"       시간: {sighted_at.strftime('%Y-%m-%d')}")
            success_count += 1

        # 페이지 간 딜레이 (API 서버 부담 감소)
        if page < MAX_PAGES:
            time.sleep(1)

    conn.commit()
    conn.close()

    # 최종 결과 출력
    print("\n" + "=" * 60)
    print("  API 연동 완료!")
    print("=" * 60)
    print(f"  API에서 가져온 데이터: {total_fetched}개")
    print(f"  개/고양이 필터링 후: {total_filtered}개")
    print(f"  YOLO 탐지 및 DB 등록 성공: {success_count}개")
    print(f"  실패/건너뜀: {fail_count}개")
    print(f"  '{DB_PATH}'에 모든 데이터가 저장되었습니다.")
    print("=" * 60)

    if success_count > 0:
        print("\n[다음 단계]")
        print("  이제 'python app.py'를 실행하여 웹 서버를 시작하세요.")
        print("  브라우저에서 http://localhost:5000 으로 접속하면")
        print(f"  {success_count}개의 실제 유기동물 데이터로 검색 시연을 할 수 있습니다.")


if __name__ == "__main__":
    register_from_api()
