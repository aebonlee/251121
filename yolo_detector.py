import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.patches as patches

class YOLODetector:
    def __init__(self, model_path='yolo11n.pt'):
        """
        YOLO11 객체 검출기 초기화
        
        Args:
            model_path: YOLO 모델 경로 (기본값: yolo11n.pt)
        """
        self.model = YOLO(model_path)
        self.colors = {}
        
    def get_random_color(self, class_id):
        """각 클래스에 대해 랜덤 색상 생성"""
        if class_id not in self.colors:
            self.colors[class_id] = (random.random(), random.random(), random.random())
        return self.colors[class_id]
    
    def detect_and_label(self, image_path, output_path=None, shape_type='auto'):
        """
        이미지에서 객체 검출 및 라벨링
        
        Args:
            image_path: 입력 이미지 경로
            output_path: 출력 이미지 경로 (기본값: input_labeled.jpg)
            shape_type: 표시할 도형 타입 ('rectangle', 'circle', 'polygon', 'auto')
                       'auto'인 경우 객체 인덱스에 따라 자동 선택
        """
        if not os.path.exists(image_path):
            print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return
        
        # 이미지 읽기
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 객체 검출 수행
        results = self.model(image_path)
        
        # matplotlib 설정
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        ax.axis('off')
        
        # 검출된 객체들 처리
        for idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 클래스 정보
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{result.names[cls]} {conf:.2f}"
                    
                    # 색상 선택
                    color = self.get_random_color(cls)
                    
                    # shape_type이 'auto'인 경우 인덱스에 따라 도형 선택
                    if shape_type == 'auto':
                        shape_idx = i % 3
                        current_shape = ['rectangle', 'circle', 'polygon'][shape_idx]
                    else:
                        current_shape = shape_type
                    
                    # 도형 그리기
                    if current_shape == 'rectangle':
                        # 사각형
                        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor=color, 
                                       facecolor='none', alpha=0.8)
                        ax.add_patch(rect)
                        
                    elif current_shape == 'circle':
                        # 원 (바운딩 박스의 중심과 반지름 계산)
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        radius = min(x2-x1, y2-y1) / 2
                        circle = Circle((center_x, center_y), radius, 
                                      linewidth=2, edgecolor=color, 
                                      facecolor='none', alpha=0.8)
                        ax.add_patch(circle)
                        
                    elif current_shape == 'polygon':
                        # 8각형 (바운딩 박스 기반)
                        width = x2 - x1
                        height = y2 - y1
                        offset_x = width * 0.2
                        offset_y = height * 0.2
                        
                        points = [
                            [x1 + offset_x, y1],
                            [x2 - offset_x, y1],
                            [x2, y1 + offset_y],
                            [x2, y2 - offset_y],
                            [x2 - offset_x, y2],
                            [x1 + offset_x, y2],
                            [x1, y2 - offset_y],
                            [x1, y1 + offset_y]
                        ]
                        polygon = Polygon(points, closed=True, 
                                        linewidth=2, edgecolor=color, 
                                        facecolor='none', alpha=0.8)
                        ax.add_patch(polygon)
                    
                    # 라벨 텍스트 추가
                    ax.text(x1, y1-5, label, fontsize=10, color='white',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor=color, alpha=0.7))
        
        # 결과 저장
        if output_path is None:
            base_name = Path(image_path).stem
            output_path = f"{base_name}_labeled.jpg"
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150, pad_inches=0.1)
        plt.show()
        
        print(f"라벨링된 이미지가 저장되었습니다: {output_path}")
        print(f"검출된 객체 수: {sum(len(r.boxes) if r.boxes is not None else 0 for r in results)}")
        
        # 검출된 객체 정보 출력
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    print(f"  - {result.names[cls]}: {conf:.2%} 신뢰도")

def main():
    parser = argparse.ArgumentParser(description='YOLO11 객체 검출 및 라벨링 프로그램')
    parser.add_argument('--image', '-i', type=str, required=True, 
                       help='입력 이미지 파일 경로')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='출력 이미지 파일 경로')
    parser.add_argument('--model', '-m', type=str, default='yolo11n.pt',
                       help='YOLO 모델 파일 경로 (기본값: yolo11n.pt)')
    parser.add_argument('--shape', '-s', type=str, default='auto',
                       choices=['rectangle', 'circle', 'polygon', 'auto'],
                       help='라벨링 도형 타입 (기본값: auto - 자동 선택)')
    
    args = parser.parse_args()
    
    # 검출기 생성 및 실행
    detector = YOLODetector(model_path=args.model)
    detector.detect_and_label(args.image, args.output, args.shape)

if __name__ == "__main__":
    main()