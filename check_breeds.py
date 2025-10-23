"""
Stanford Dogs Dataset 품종 목록 확인
"""

# TensorFlow Datasets를 사용하여 품종 목록 확인
try:
    import tensorflow_datasets as tfds

    # 데이터셋 정보 로드 (다운로드 없이)
    ds_info = tfds.builder('stanford_dogs').info

    # 품종 목록 출력
    breeds = ds_info.features['label'].names

    print("=" * 60)
    print(f"Stanford Dogs Dataset - 총 {len(breeds)}개 품종")
    print("=" * 60)

    # 진도견 찾기
    jindo_found = False
    for i, breed in enumerate(breeds):
        print(f"{i+1:3d}. {breed}")
        if 'jindo' in breed.lower() or 'korean' in breed.lower():
            print(f"     ★★★ 진도견 발견! ★★★")
            jindo_found = True

    print("=" * 60)
    if jindo_found:
        print("✅ 진도견(Korean Jindo)이 포함되어 있습니다!")
    else:
        print("❌ 진도견(Korean Jindo)이 없습니다.")
    print("=" * 60)

except ImportError:
    print("TensorFlow Datasets가 설치되지 않았습니다.")
    print("다른 방법으로 확인하겠습니다...")

    # 대체 방법: 알려진 Stanford Dogs 품종 목록
    print("\n=== Stanford Dogs Dataset 알려진 품종 목록 (일부) ===")
    print("Poodle, Yorkshire Terrier, Maltese, Chihuahua, Welsh Corgi,")
    print("Bichon Frise, Shih Tzu, Golden Retriever, Pomeranian...")
    print("\n진도견 포함 여부는 공식 문서를 확인해야 합니다.")
