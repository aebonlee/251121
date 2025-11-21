"""
YOLO11 검출기 테스트 스크립트
샘플 이미지로 객체 검출 테스트
"""

from yolo_detector import YOLODetector
import urllib.request
import os

def download_sample_image():
    """테스트용 샘플 이미지 다운로드"""
    url = "https://raw.githubusercontent.com/ultralytics/assets/main/bus.jpg"
    filename = "test_image.jpg"
    
    if not os.path.exists(filename):
        print("샘플 이미지 다운로드 중...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"샘플 이미지 다운로드 완료: {filename}")
        except:
            print("샘플 이미지 다운로드 실패")
            return None
    return filename

def test_detection():
    """객체 검출 테스트"""
    # 샘플 이미지 다운로드
    image_path = download_sample_image()
    
    if image_path and os.path.exists(image_path):
        print("\n=== YOLO11 객체 검출 테스트 시작 ===\n")
        
        # 검출기 생성
        detector = YOLODetector()
        
        # 1. 자동 도형 선택 모드 테스트
        print("1. 자동 도형 선택 모드 (각 객체마다 다른 도형)")
        detector.detect_and_label(
            image_path=image_path,
            output_path="output_auto.jpg",
            shape_type='auto'
        )
        
        print("\n" + "-"*50 + "\n")
        
        # 2. 사각형 모드 테스트
        print("2. 사각형 모드")
        detector.detect_and_label(
            image_path=image_path,
            output_path="output_rectangle.jpg",
            shape_type='rectangle'
        )
        
        print("\n" + "-"*50 + "\n")
        
        # 3. 원 모드 테스트
        print("3. 원 모드")
        detector.detect_and_label(
            image_path=image_path,
            output_path="output_circle.jpg",
            shape_type='circle'
        )
        
        print("\n" + "-"*50 + "\n")
        
        # 4. 다각형 모드 테스트
        print("4. 다각형 모드")
        detector.detect_and_label(
            image_path=image_path,
            output_path="output_polygon.jpg",
            shape_type='polygon'
        )
        
        print("\n=== 테스트 완료 ===")
        print("\n생성된 파일:")
        print("  - output_auto.jpg (자동 도형 선택)")
        print("  - output_rectangle.jpg (사각형)")
        print("  - output_circle.jpg (원)")
        print("  - output_polygon.jpg (다각형)")
        
    else:
        print("테스트 이미지를 찾을 수 없습니다.")

if __name__ == "__main__":
    test_detection()