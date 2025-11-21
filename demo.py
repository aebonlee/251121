"""
YOLO11 객체 검출 데모
간단한 사용 예제
"""

from yolo_detector import YOLODetector
import os

def demo():
    # 검출기 생성
    detector = YOLODetector()
    
    # 샘플 이미지 경로 (테스트용)
    # 실제 사용시 이미지 파일 경로를 지정하세요
    sample_images = [
        "sample1.jpg",  # 이미지 파일 경로
        "sample2.png",  # 다른 이미지 파일
    ]
    
    print("YOLO11 객체 검출 데모 시작")
    print("-" * 50)
    
    for image_path in sample_images:
        if os.path.exists(image_path):
            print(f"\n처리 중: {image_path}")
            
            # 자동 도형 선택 (객체마다 다른 도형)
            detector.detect_and_label(
                image_path=image_path,
                shape_type='auto'  # rectangle, circle, polygon 자동 선택
            )
        else:
            print(f"파일을 찾을 수 없음: {image_path}")
    
    print("\n" + "=" * 50)
    print("데모 완료!")
    print("\n사용법:")
    print("python yolo_detector.py -i [이미지경로] -s [도형타입]")
    print("예시: python yolo_detector.py -i image.jpg -s auto")

if __name__ == "__main__":
    demo()