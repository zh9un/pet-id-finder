"""
Flask 웹 서버: Pet-ID Finder (최종 수정 버전)
- DB 스키마에 location, sighted_at 추가
- 검색 시 해당 정보 포함하여 반환
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import sqlite3
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ml_pipeline import ImageAnalyzer
from datetime import datetime

app = Flask(__name__)

# 설정
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ML 파이프라인 초기화
MODEL_TYPE = 'clip'
print(f"\n[Flask] ML Pipeline 초기화 중 (모델: {MODEL_TYPE.upper()})...")
analyzer = ImageAnalyzer(model_type=MODEL_TYPE)
print("[Flask] ML Pipeline 초기화 완료\n")


def init_db():
    """데이터베이스 초기화 (location, sighted_at 컬럼 추가)"""
    conn = sqlite3.connect('pets.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS pets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            embedding TEXT NOT NULL,
            location TEXT,
            sighted_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("[Database] 데이터베이스 초기화 완료")


def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """메인 페이지 - 역할 선택"""
    return render_template('index.html')


@app.route('/search')
def search_page():
    """보호자 전용 페이지 (검색)"""
    return render_template('search_page.html')


@app.route('/report')
def report_page():
    """목격자 전용 페이지 (신고)"""
    return render_template('report_page.html')


@app.route('/register_pet', methods=['POST'])
def register_pet():
    """유실동물 신고 처리 (목격자) - AJAX 요청"""
    if 'image' not in request.files:
        return jsonify({'error': '파일이 첨부되지 않았습니다.'}), 400

    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '파일이 선택되지 않았거나 허용되지 않는 형식입니다.'}), 400

    try:
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

        features = analyzer.process_and_extract_features(filepath)

        if features is None:
            os.remove(filepath)
            return jsonify({'error': '이미지에서 동물(개/고양이)을 탐지하지 못했습니다.'}), 400

        embedding_json = json.dumps(features.tolist())

        # 수동 등록 시 기본 정보
        location = "수동 등록"
        sighted_at = datetime.now()

        conn = sqlite3.connect('pets.db')
        c = conn.cursor()
        c.execute(
            'INSERT INTO pets (image_path, embedding, location, sighted_at) VALUES (?, ?, ?, ?)',
            (filepath, embedding_json, location, sighted_at)
        )
        conn.commit()
        pet_id = c.lastrowid
        conn.close()

        print(f"[Register] 등록 성공 - ID: {pet_id}, 파일: {filename}")
        return jsonify({'message': f'동물 정보가 성공적으로 등록되었습니다 (ID: {pet_id})'}), 200

    except Exception as e:
        print(f"[Register ERROR] {str(e)}")
        return jsonify({'error': f'등록 중 오류 발생: {str(e)}'}), 500


@app.route('/search_pet', methods=['POST'])
def search_pet():
    """유실동물 검색 처리 (보호자) - Form 제출"""
    if 'search_image' not in request.files:
        return "파일이 없습니다.", 400

    file = request.files['search_image']

    if file.filename == '' or not allowed_file(file.filename):
        return "파일이 없거나 잘못된 형식입니다.", 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"search_{filename}")

        # 중복 파일명 처리
        base, ext = os.path.splitext(filepath)
        counter = 1
        while os.path.exists(filepath):
            filepath = f"{base}_{counter}{ext}"
            counter += 1

        file.save(filepath)

        query_features = analyzer.process_and_extract_features(filepath)

        if query_features is None:
            os.remove(filepath)
            return "검색 이미지에서 동물을 찾을 수 없습니다.", 400

        conn = sqlite3.connect('pets.db')
        conn.row_factory = sqlite3.Row  # Row 팩토리 사용
        c = conn.cursor()

        c.execute('SELECT id, image_path, embedding, location, sighted_at FROM pets')
        all_pets = c.fetchall()
        conn.close()

        if not all_pets:
            return render_template('results.html', query_image=filepath, results=[], total_count=0)

        results = []
        for pet in all_pets:
            db_embedding = np.array(json.loads(pet['embedding']))
            similarity = cosine_similarity(query_features, db_embedding)[0][0]

            results.append({
                'id': pet['id'],
                'image_path': pet['image_path'],
                'location': pet['location'] if pet['location'] else '장소 정보 없음',
                'sighted_at': pet['sighted_at'] if pet['sighted_at'] else '시간 정보 없음',
                'similarity': float(similarity),
                'similarity_percent': f"{similarity * 100:.2f}%"
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        print(f"[Search] 검색 완료 - 결과: {len(results)}개")

        return render_template('results.html', query_image=filepath, results=results, total_count=len(results))

    except Exception as e:
        print(f"[Search ERROR] {str(e)}")
        return f"검색 중 오류 발생: {str(e)}", 500


@app.route('/stats')
def stats():
    """통계 API"""
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
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    init_db()

    print("\n" + "=" * 60)
    print("   Pet-ID Finder 서버 시작")
    print("=" * 60)
    print(f"   URL: http://localhost:5000")
    print(f"   모델: YOLO + {MODEL_TYPE.upper()}")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
