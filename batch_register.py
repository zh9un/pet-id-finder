"""
시연용 이미지 일괄 등록 스크립트

demo_images/ 폴더의 모든 이미지를 DB에 자동으로 등록하고,
가상 목격 정보 (장소, 시간)를 생성합니다.
"""

import os
import sqlite3
import json
import random
from datetime import datetime, timedelta
from ml_pipeline import ImageAnalyzer

# 설정
IMAGE_FOLDER = 'static/demo_images'
DB_PATH = 'pets.db'
MODEL_TYPE = 'clip'  # app.py와 동일한 모델 사용

# 가상 목격 장소 (서울 주요 지역)
VIRTUAL_LOCATIONS = [
    "강남역 10번 출구 인근",
    "홍대입구역 9번 출구 근처",
    "성수동 카페거리",
    "연남동 골목길",
    "혜화역 마로니에 공원 앞",
    "이태원역 2번 출구",
    "건대입구역 5번 출구",
    "신촌역 주변 상가",
    "명동역 8번 출구",
    "잠실역 롯데월드 앞",
    "고속터미널역 지하상가",
    "선릉역 삼성전자 빌딩 앞",
    "시청역 서울광장",
    "종로3가역 탑골공원",
    "여의도역 한강공원",
    "상수역 경의선숲길",
    "합정역 메세나폴리스",
    "신사역 가로수길",
    "압구정역 로데오거리",
    "삼성역 코엑스몰 근처"
]


def get_random_past_datetime():
    """최근 30일 내의 랜덤한 날짜와 시간을 반환"""
    days_ago = random.randint(0, 30)
    hours_ago = random.randint(0, 23)
    minutes_ago = random.randint(0, 59)

    random_date = datetime.now() - timedelta(
        days=days_ago,
        hours=hours_ago,
        minutes=minutes_ago
    )

    return random_date


def init_db_with_location():
    """데이터베이스 초기화 (목격 정보 컬럼 포함)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 기존 테이블 삭제 후 재생성 (스키마 변경)
    c.execute('DROP TABLE IF EXISTS pets')

    c.execute('''
        CREATE TABLE pets (
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
    print("[Database] 데이터베이스 초기화 완료 (목격 정보 스키마 적용)")


def batch_register_pets():
    """
    지정된 폴더의 모든 이미지를 '가상 목격 정보'와 함께 DB에 자동으로 등록
    """
    print("=" * 60)
    print("  시연용 데이터 자동 등록 스크립트")
    print("=" * 60)

    # 1. 이미지 폴더 존재 확인
    if not os.path.exists(IMAGE_FOLDER):
        print(f"[ERROR] '{IMAGE_FOLDER}' 폴더를 찾을 수 없습니다.")
        print("        먼저 download_demo_images.py를 실행하여 이미지를 수집해주세요.")
        return

    # 2. 데이터베이스 초기화
    print("\n[1/4] 데이터베이스 초기화 중...")
    init_db_with_location()

    # 3. ML Pipeline 초기화
    print(f"\n[2/4] ML Pipeline 초기화 중 (모델: {MODEL_TYPE.upper()})...")
    analyzer = ImageAnalyzer(model_type=MODEL_TYPE)

    # 4. 데이터베이스 연결
    print("\n[3/4] 데이터베이스 연결 완료")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 5. 이미지 처리 및 등록
    print(f"\n[4/4] '{IMAGE_FOLDER}' 폴더의 이미지들을 등록합니다...")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    # 모든 하위 폴더 탐색
    for root, dirs, files in os.walk(IMAGE_FOLDER):
        for filename in files:
            # 이미지 파일만 처리
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                continue

            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath, IMAGE_FOLDER)

            print(f"\n  처리 중: {relative_path}")

            # ML Pipeline으로 특징 벡터 및 동물 종류 추출
            features, animal_type = analyzer.process_and_extract_features(filepath)

            if features is not None:
                # numpy 배열을 JSON 문자열로 변환
                embedding_json = json.dumps(features.tolist())

                # 가상 목격 정보 생성
                location = random.choice(VIRTUAL_LOCATIONS)
                sighted_time = get_random_past_datetime()

                # DB에 저장
                c.execute('''
                    INSERT INTO pets (image_path, embedding, animal_type, location, sighted_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (filepath, embedding_json, animal_type, location, sighted_time))

                animal_name = "강아지" if animal_type == "dog" else "고양이"
                print(f"    -> 등록 성공!")
                print(f"       종류: {animal_name}")
                print(f"       장소: {location}")
                print(f"       시간: {sighted_time.strftime('%Y-%m-%d %H:%M')}")
                success_count += 1
            else:
                print(f"    -> 동물 미탐지, 건너뜀")
                fail_count += 1

    conn.commit()
    conn.close()

    # 결과 출력
    print("\n" + "=" * 60)
    print("  자동 등록 완료!")
    print("=" * 60)
    print(f"  성공: {success_count}개")
    print(f"  실패/건너뜀: {fail_count}개")
    print(f"  '{DB_PATH}'에 모든 데이터가 준비되었습니다.")
    print("=" * 60)

    if success_count > 0:
        print("\n[다음 단계]")
        print("  이제 'python app.py'를 실행하여 웹 서버를 시작하세요.")
        print("  브라우저에서 http://localhost:5000 으로 접속하면")
        print(f"  {success_count}개의 목격 데이터로 검색 시연을 할 수 있습니다.")


if __name__ == "__main__":
    batch_register_pets()
