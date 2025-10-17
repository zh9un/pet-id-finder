"""
Flask 웹 서버: Pet-ID Finder

유실동물 유사 개체 검색 시스템
- 등록: 이미지 업로드 -> YOLO + DINO/CLIP -> DB 저장
- 검색: 이미지 업로드 -> 코사인 유사도 계산 -> 순위 표시
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import sqlite3
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ml_pipeline import ImageAnalyzer

app = Flask(__name__)

# 설정
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ML Pipeline 초기화 (CLIP 사용, DINO로 변경 가능)
MODEL_TYPE = 'clip'  # 'dino' 또는 'clip'
print(f"\n[Flask] ML Pipeline 초기화 중 (모델: {MODEL_TYPE.upper()})...")
analyzer = ImageAnalyzer(model_type=MODEL_TYPE)
print("[Flask] ML Pipeline 초기화 완료\n")


def init_db():
    """데이터베이스 초기화"""
    conn = sqlite3.connect('pets.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS pets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            embedding TEXT NOT NULL,
            animal_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("[Database] 데이터베이스 초기화 완료")


def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    """메인 페이지"""
    if request.method == 'POST':
        # 등록 요청
        if 'image' in request.files:
            return register_pet()

        # 검색 요청
        elif 'search_image' in request.files:
            return search_pet()

        else:
            return jsonify({'error': '파일이 첨부되지 않았습니다.'}), 400

    # GET 요청
    return render_template('index.html')


def register_pet():
    """유실동물 등록 (DB 저장)"""
    file = request.files['image']

    # 파일 검증
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

    try:
        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 중복 파일명 처리
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filepath):
            filename = f"{base}_{counter}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            counter += 1

        file.save(filepath)

        # ML Pipeline: 특징 추출
        features = analyzer.process_and_extract_features(filepath)

        if features is None:
            # 동물 미탐지
            os.remove(filepath)
            return jsonify({
                'error': '이미지에서 동물(개/고양이)을 탐지하지 못했습니다.',
                'detail': 'YOLO 모델이 동물을 인식하지 못했습니다. 다른 이미지를 사용해주세요.'
            }), 400

        # numpy -> JSON 변환
        embedding_json = json.dumps(features.tolist())

        # DB 저장
        conn = sqlite3.connect('pets.db')
        c = conn.cursor()

        c.execute('''
            INSERT INTO pets (image_path, embedding, animal_type)
            VALUES (?, ?, ?)
        ''', (filepath, embedding_json, 'unknown'))

        conn.commit()
        pet_id = c.lastrowid
        conn.close()

        print(f"[Register] 등록 성공 - ID: {pet_id}, 파일: {filename}")

        return jsonify({
            'success': True,
            'message': '동물 정보가 성공적으로 등록되었습니다!',
            'pet_id': pet_id,
            'image_path': filepath,
            'feature_dim': features.shape[1]
        }), 200

    except Exception as e:
        print(f"[Register ERROR] {str(e)}")
        return jsonify({'error': f'등록 중 오류 발생: {str(e)}'}), 500


def search_pet():
    """유실동물 검색 (코사인 유사도)"""
    file = request.files['search_image']

    # 파일 검증
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

    try:
        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"search_{filename}")

        # 중복 파일명 처리
        base, ext = os.path.splitext(filepath)
        counter = 1
        while os.path.exists(filepath):
            filepath = f"{base}_{counter}{ext}"
            counter += 1

        file.save(filepath)

        # ML Pipeline: 특징 추출
        query_features = analyzer.process_and_extract_features(filepath)

        if query_features is None:
            os.remove(filepath)
            return jsonify({
                'error': '검색 이미지에서 동물(개/고양이)을 탐지하지 못했습니다.',
                'detail': 'YOLO 모델이 동물을 인식하지 못했습니다.'
            }), 400

        # DB에서 모든 동물 데이터 로드
        conn = sqlite3.connect('pets.db')
        c = conn.cursor()

        c.execute('SELECT id, image_path, embedding FROM pets')
        all_pets = c.fetchall()
        conn.close()

        if len(all_pets) == 0:
            return jsonify({
                'error': '등록된 동물이 없습니다.',
                'detail': '먼저 동물을 등록해주세요.'
            }), 404

        # 코사인 유사도 계산
        results = []

        for pet_id, img_path, embedding_json in all_pets:
            db_embedding = np.array(json.loads(embedding_json))

            # 유사도 계산
            similarity = cosine_similarity(query_features, db_embedding)[0][0]

            results.append({
                'id': pet_id,
                'image_path': img_path,
                'similarity': float(similarity),
                'similarity_percent': f"{similarity * 100:.2f}%"
            })

        # 유사도 내림차순 정렬
        results.sort(key=lambda x: x['similarity'], reverse=True)

        print(f"[Search] 검색 완료 - 결과: {len(results)}개")

        return render_template('search.html',
                               query_image=filepath,
                               results=results,
                               total_count=len(results))

    except Exception as e:
        print(f"[Search ERROR] {str(e)}")
        return jsonify({'error': f'검색 중 오류 발생: {str(e)}'}), 500


@app.route('/stats')
def stats():
    """통계 API (선택)"""
    conn = sqlite3.connect('pets.db')
    c = conn.cursor()

    c.execute('SELECT COUNT(*) FROM pets')
    total_count = c.fetchone()[0]

    conn.close()

    return jsonify({
        'total_pets': total_count,
        'model_type': MODEL_TYPE.upper()
    })


if __name__ == '__main__':
    # 업로드 폴더 생성
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # 데이터베이스 초기화
    init_db()

    print("\n" + "=" * 60)
    print("   Pet-ID Finder 서버 시작")
    print("=" * 60)
    print(f"   URL: http://localhost:5000")
    print(f"   모델: YOLO + {MODEL_TYPE.upper()}")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
