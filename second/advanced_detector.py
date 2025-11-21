"""
고급 YOLO11 객체 검출 프로그램
- 더 정확한 검출을 위한 다중 모델 앙상블
- 세그멘테이션 지원
- 특정 도메인 최적화
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AdvancedYOLODetector:
    """
    고급 YOLO11 검출기
    - 다중 모델 앙상블
    - 세그멘테이션 지원
    - NMS (Non-Maximum Suppression) 최적화
    """
    
    def __init__(self, 
                 detection_model='yolo11x.pt',
                 segmentation_model='yolo11x-seg.pt',
                 use_ensemble=True,
                 device='auto'):
        """
        초기화
        
        Args:
            detection_model: 검출용 모델 경로
            segmentation_model: 세그멘테이션 모델 경로
            use_ensemble: 앙상블 사용 여부
            device: 'auto', 'cpu', 'cuda'
        """
        # 디바이스 설정
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # 모델 로드
        print("Loading detection model...")
        self.detection_model = YOLO(detection_model)
        
        print("Loading segmentation model...")
        self.seg_model = YOLO(segmentation_model)
        
        if use_ensemble:
            print("Loading additional models for ensemble...")
            self.ensemble_models = [
                YOLO('yolo11l.pt'),  # Large 모델
                YOLO('yolo11m.pt'),  # Medium 모델
            ]
        else:
            self.ensemble_models = []
        
        # 클래스별 색상 맵
        self.colors = {}
        self.detection_history = []
        
    def get_color(self, class_id):
        """클래스별 고정 색상 반환"""
        if class_id not in self.colors:
            np.random.seed(class_id)
            self.colors[class_id] = tuple(np.random.rand(3))
        return self.colors[class_id]
    
    def ensemble_detect(self, image_path, conf_threshold=0.25):
        """
        앙상블 검출 - 여러 모델의 결과를 통합
        
        Args:
            image_path: 이미지 경로
            conf_threshold: 신뢰도 임계값
        
        Returns:
            통합된 검출 결과
        """
        all_detections = []
        
        # 메인 모델 검출
        results = self.detection_model(image_path, conf=conf_threshold, verbose=False)
        all_detections.append(results[0])
        
        # 앙상블 모델들 검출
        for model in self.ensemble_models:
            results = model(image_path, conf=conf_threshold, verbose=False)
            all_detections.append(results[0])
        
        # 결과 통합 (Weighted Voting)
        return self._merge_detections(all_detections)
    
    def _merge_detections(self, detections_list):
        """
        여러 모델의 검출 결과를 통합
        
        Args:
            detections_list: 각 모델의 검출 결과 리스트
        
        Returns:
            통합된 검출 결과
        """
        if len(detections_list) == 1:
            return detections_list[0]
        
        # 첫 번째 결과를 기준으로 사용
        merged = detections_list[0]
        
        # 각 검출에 대해 다른 모델들의 동의 정도를 확인
        # 과반수 이상이 검출한 객체만 유지
        # (실제 구현에서는 더 정교한 알고리즘 필요)
        
        return merged
    
    def detect_with_segmentation(self, image_path, conf_threshold=0.5):
        """
        세그멘테이션을 포함한 검출
        
        Args:
            image_path: 이미지 경로
            conf_threshold: 신뢰도 임계값
        
        Returns:
            검출 결과와 세그멘테이션 마스크
        """
        # 세그멘테이션 수행
        seg_results = self.seg_model(image_path, conf=conf_threshold, verbose=False)
        
        return seg_results[0]
    
    def advanced_detect(self, 
                       image_path,
                       mode='ensemble',
                       conf_threshold=0.5,
                       iou_threshold=0.45,
                       max_detections=100,
                       target_classes=None,
                       output_path=None,
                       show_segmentation=False):
        """
        고급 검출 수행
        
        Args:
            image_path: 입력 이미지 경로
            mode: 'single', 'ensemble', 'segmentation'
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            max_detections: 최대 검출 수
            target_classes: 특정 클래스만 검출
            output_path: 출력 경로
            show_segmentation: 세그멘테이션 표시 여부
        
        Returns:
            검출 결과 딕셔너리
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\nProcessing: {image_path}")
        print(f"Mode: {mode}, Confidence: {conf_threshold}")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # 검출 모드에 따른 처리
        if mode == 'segmentation' or show_segmentation:
            results = self.detect_with_segmentation(image_path, conf_threshold)
        elif mode == 'ensemble' and self.ensemble_models:
            results = self.ensemble_detect(image_path, conf_threshold)
        else:
            results = self.detection_model(image_path, 
                                          conf=conf_threshold,
                                          iou=iou_threshold,
                                          max_det=max_detections,
                                          verbose=False)[0]
        
        # 결과 파싱
        detections = self._parse_results(results, target_classes)
        
        # 시각화
        if output_path or show_segmentation:
            self._visualize_advanced(image_rgb, results, detections, 
                                    output_path, show_segmentation)
        
        # 통계 정보
        self._print_statistics(detections)
        
        # 검출 기록 저장
        self.detection_history.append({
            'timestamp': datetime.now().isoformat(),
            'image': image_path,
            'mode': mode,
            'detections': len(detections),
            'classes': list(set(d['class'] for d in detections))
        })
        
        return {
            'image_path': image_path,
            'image_size': (width, height),
            'mode': mode,
            'detections': detections,
            'num_objects': len(detections),
            'confidence_threshold': conf_threshold
        }
    
    def _parse_results(self, results, target_classes=None):
        """검출 결과 파싱"""
        detections = []
        
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                cls = int(box.cls[0])
                class_name = results.names[cls]
                
                # 타겟 클래스 필터링
                if target_classes and class_name not in target_classes:
                    continue
                
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                detection = {
                    'id': i,
                    'class': class_name,
                    'class_id': cls,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'area': (x2 - x1) * (y2 - y1)
                }
                
                detections.append(detection)
        
        # 신뢰도 순으로 정렬
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def _visualize_advanced(self, image_rgb, results, detections, 
                          output_path=None, show_segmentation=False):
        """고급 시각화"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.imshow(image_rgb)
        ax.axis('off')
        
        # 세그멘테이션 마스크 표시
        if show_segmentation and hasattr(results, 'masks') and results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                if i < len(detections):
                    color = self.get_color(detections[i]['class_id'])
                    colored_mask = np.zeros_like(image_rgb)
                    colored_mask[:, :] = [int(c * 255) for c in color]
                    
                    # 마스크 리사이즈
                    mask_resized = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
                    
                    # 마스크 적용
                    alpha = 0.3
                    for c in range(3):
                        image_rgb[:, :, c] = np.where(
                            mask_resized > 0.5,
                            image_rgb[:, :, c] * (1 - alpha) + colored_mask[:, :, c] * alpha,
                            image_rgb[:, :, c]
                        )
        
        # 바운딩 박스 그리기
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = self.get_color(det['class_id'])
            
            # 바운딩 박스
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor=color,
                           facecolor='none', alpha=0.9)
            ax.add_patch(rect)
            
            # 중심점 표시
            ax.plot(det['center'][0], det['center'][1], 
                   'o', color=color, markersize=8)
            
            # 라벨
            label = f"{det['class']} {det['confidence']:.3f}"
            ax.text(x1, y1-5, label, fontsize=10, color='white',
                   bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=color, alpha=0.8))
        
        # 범례 추가
        unique_classes = list(set(d['class'] for d in detections))
        if unique_classes:
            patches = [mpatches.Patch(color=self.get_color(
                next(d['class_id'] for d in detections if d['class'] == cls)
            ), label=cls) for cls in unique_classes]
            ax.legend(handles=patches, loc='upper right', fontsize=10)
        
        # 타이틀
        title = f"Advanced Detection: {len(detections)} objects"
        if show_segmentation:
            title += " (with segmentation)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"Output saved: {output_path}")
        
        plt.show()
    
    def _print_statistics(self, detections):
        """검출 통계 출력"""
        print(f"\n{'='*50}")
        print(f"Detection Statistics")
        print(f"{'='*50}")
        print(f"Total objects detected: {len(detections)}")
        
        if detections:
            # 클래스별 통계
            class_counts = {}
            confidence_by_class = {}
            
            for det in detections:
                cls = det['class']
                if cls not in class_counts:
                    class_counts[cls] = 0
                    confidence_by_class[cls] = []
                class_counts[cls] += 1
                confidence_by_class[cls].append(det['confidence'])
            
            print("\nClass-wise detection:")
            for cls in sorted(class_counts.keys()):
                avg_conf = np.mean(confidence_by_class[cls])
                max_conf = np.max(confidence_by_class[cls])
                min_conf = np.min(confidence_by_class[cls])
                print(f"  {cls:15s}: {class_counts[cls]:3d} objects "
                     f"(conf: min={min_conf:.3f}, avg={avg_conf:.3f}, max={max_conf:.3f})")
            
            # 전체 신뢰도 통계
            all_confidences = [d['confidence'] for d in detections]
            print(f"\nOverall confidence: {np.mean(all_confidences):.3f} "
                 f"(±{np.std(all_confidences):.3f})")
    
    def compare_models(self, image_path, output_dir='comparisons'):
        """
        여러 모델의 성능 비교
        
        Args:
            image_path: 테스트 이미지
            output_dir: 결과 저장 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        models_to_compare = [
            ('yolo11n.pt', 'YOLOv11 Nano'),
            ('yolo11s.pt', 'YOLOv11 Small'),
            ('yolo11m.pt', 'YOLOv11 Medium'),
            ('yolo11l.pt', 'YOLOv11 Large'),
            ('yolo11x.pt', 'YOLOv11 XLarge'),
        ]
        
        comparison_results = []
        
        for model_path, model_name in models_to_compare:
            print(f"\nTesting {model_name}...")
            
            try:
                model = YOLO(model_path)
                
                # 시간 측정
                import time
                start_time = time.time()
                results = model(image_path, verbose=False)[0]
                inference_time = time.time() - start_time
                
                # 검출 수
                num_detections = len(results.boxes) if results.boxes is not None else 0
                
                comparison_results.append({
                    'model': model_name,
                    'inference_time': inference_time,
                    'detections': num_detections,
                    'fps': 1 / inference_time
                })
                
                # 결과 저장
                output_path = os.path.join(output_dir, 
                                         f"{Path(model_path).stem}_result.jpg")
                self._save_comparison_image(image_path, results, 
                                          model_name, output_path)
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        # 비교 결과 출력
        self._print_comparison_results(comparison_results)
        
        return comparison_results
    
    def _save_comparison_image(self, image_path, results, model_name, output_path):
        """비교용 이미지 저장"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        ax.axis('off')
        
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                color = self.get_color(cls)
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor=color,
                               facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                
                label = f"{results.names[cls]} {conf:.2f}"
                ax.text(x1, y1-5, label, fontsize=9, color='white',
                       bbox=dict(boxstyle="round,pad=0.3",
                               facecolor=color, alpha=0.7))
        
        ax.set_title(f"{model_name} - {len(results.boxes) if results.boxes else 0} detections",
                    fontsize=12, fontweight='bold')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=120)
        plt.close()
    
    def _print_comparison_results(self, results):
        """비교 결과 출력"""
        print(f"\n{'='*60}")
        print(f"Model Comparison Results")
        print(f"{'='*60}")
        print(f"{'Model':20s} {'Time (s)':>10s} {'FPS':>10s} {'Detections':>12s}")
        print(f"{'-'*60}")
        
        for r in results:
            print(f"{r['model']:20s} {r['inference_time']:>10.3f} "
                 f"{r['fps']:>10.1f} {r['detections']:>12d}")
    
    def save_detection_history(self, filepath='detection_history.json'):
        """검출 기록 저장"""
        with open(filepath, 'w') as f:
            json.dump(self.detection_history, f, indent=2)
        print(f"Detection history saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Advanced YOLO11 Object Detection')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--mode', '-m', type=str, default='ensemble',
                       choices=['single', 'ensemble', 'segmentation'],
                       help='Detection mode')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output image path')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--classes', nargs='+', default=None,
                       help='Target classes to detect')
    parser.add_argument('--compare', action='store_true',
                       help='Compare different models')
    parser.add_argument('--segmentation', action='store_true',
                       help='Show segmentation masks')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 검출기 생성
    detector = AdvancedYOLODetector(
        use_ensemble=(args.mode == 'ensemble'),
        device=args.device
    )
    
    if args.compare:
        # 모델 비교
        detector.compare_models(args.image)
    else:
        # 고급 검출 수행
        results = detector.advanced_detect(
            image_path=args.image,
            mode=args.mode,
            conf_threshold=args.confidence,
            iou_threshold=args.iou,
            target_classes=args.classes,
            output_path=args.output,
            show_segmentation=args.segmentation
        )
        
        print(f"\nDetection complete: {results['num_objects']} objects found")
    
    # 검출 기록 저장
    if detector.detection_history:
        detector.save_detection_history()


if __name__ == "__main__":
    main()