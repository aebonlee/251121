"""
YOLO11 커스텀 데이터셋 학습 및 파인튜닝 시스템
더 정확한 객체 검출을 위한 맞춤형 학습 도구
"""

import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CustomDatasetPreparer:
    """
    커스텀 데이터셋 준비 및 전처리 클래스
    YOLO 형식으로 데이터 변환 및 구성
    """
    
    def __init__(self, dataset_name="custom_dataset"):
        """
        초기화
        
        Args:
            dataset_name: 데이터셋 이름
        """
        self.dataset_name = dataset_name
        self.base_path = Path(f"datasets/{dataset_name}")
        self.annotations = []
        self.class_names = []
        self.image_paths = []
        
        # 디렉토리 구조 생성
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """YOLO 학습용 디렉토리 구조 생성"""
        dirs = [
            self.base_path / "images" / "train",
            self.base_path / "images" / "val",
            self.base_path / "images" / "test",
            self.base_path / "labels" / "train",
            self.base_path / "labels" / "val",
            self.base_path / "labels" / "test",
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 데이터셋 디렉토리 구조 생성 완료: {self.base_path}")
    
    def add_custom_classes(self, class_names: List[str]):
        """
        커스텀 클래스 추가
        
        Args:
            class_names: 클래스 이름 리스트
        """
        self.class_names = class_names
        print(f"✅ {len(class_names)}개의 커스텀 클래스 추가:")
        for i, name in enumerate(class_names):
            print(f"   {i}: {name}")
    
    def prepare_annotation_data(self, images_dir: str, annotations_file: str,
                               format_type="coco"):
        """
        어노테이션 데이터 준비
        
        Args:
            images_dir: 이미지 디렉토리
            annotations_file: 어노테이션 파일 경로
            format_type: 어노테이션 형식 (coco, pascal_voc, yolo)
        """
        if format_type == "coco":
            self._convert_from_coco(annotations_file)
        elif format_type == "pascal_voc":
            self._convert_from_pascal_voc(annotations_file)
        else:
            print(f"지원하지 않는 형식: {format_type}")
            return
        
        # 이미지 복사
        self._copy_images(images_dir)
        
    def _convert_from_coco(self, coco_file):
        """COCO 형식에서 YOLO 형식으로 변환"""
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        # 카테고리 매핑
        category_map = {}
        for cat in coco_data['categories']:
            category_map[cat['id']] = cat['name']
        
        # 이미지별 어노테이션 그룹화
        img_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        # YOLO 형식으로 변환
        for img_info in coco_data['images']:
            img_id = img_info['id']
            img_width = img_info['width']
            img_height = img_info['height']
            
            if img_id in img_annotations:
                yolo_annotations = []
                for ann in img_annotations[img_id]:
                    # COCO bbox: [x, y, width, height]
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2]/2) / img_width
                    y_center = (bbox[1] + bbox[3]/2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    # YOLO format: class_id x_center y_center width height
                    class_id = ann['category_id'] - 1  # 0-indexed
                    yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
                
                self.annotations.append({
                    'image': img_info['file_name'],
                    'labels': yolo_annotations
                })
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        데이터셋 분할
        
        Args:
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
        """
        total = len(self.annotations)
        indices = list(range(total))
        np.random.shuffle(indices)
        
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:]
        }
        
        print(f"\n📊 데이터셋 분할:")
        print(f"   Train: {len(splits['train'])} images")
        print(f"   Val: {len(splits['val'])} images")
        print(f"   Test: {len(splits['test'])} images")
        
        return splits
    
    def create_yaml_config(self):
        """YOLO 학습용 YAML 설정 파일 생성"""
        config = {
            'path': str(self.base_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.base_path / f"{self.dataset_name}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📝 YAML 설정 파일 생성: {yaml_path}")
        return yaml_path


class YOLOFineTuner:
    """
    YOLO11 파인튜닝 클래스
    기존 모델을 커스텀 데이터셋으로 재학습
    """
    
    def __init__(self, base_model="yolo11n.pt", device="auto"):
        """
        초기화
        
        Args:
            base_model: 기본 모델 경로
            device: 학습 디바이스 (auto, cpu, cuda)
        """
        self.base_model = base_model
        
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.training_history = []
        
        print(f"🚀 파인튜닝 준비")
        print(f"   Base model: {base_model}")
        print(f"   Device: {self.device}")
        
    def configure_training(self, epochs=100, batch_size=16, imgsz=640,
                         patience=50, learning_rate=0.01):
        """
        학습 설정 구성
        
        Args:
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            imgsz: 입력 이미지 크기
            patience: Early stopping patience
            learning_rate: 학습률
        """
        self.training_config = {
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'patience': patience,
            'lr0': learning_rate,
            'lrf': 0.01,  # 최종 학습률
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj': 1.0,
            'obj_pw': 1.0,
            'iou_t': 0.20,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        print(f"⚙️ 학습 설정 완료:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Image size: {imgsz}")
        print(f"   Learning rate: {learning_rate}")
        
    def train(self, data_yaml, project_name="custom_training",
             pretrained=True, resume=False):
        """
        모델 학습
        
        Args:
            data_yaml: 데이터셋 설정 파일
            project_name: 프로젝트 이름
            pretrained: 사전학습 가중치 사용 여부
            resume: 이전 학습 재개 여부
        """
        print(f"\n🎯 학습 시작: {project_name}")
        
        # 모델 로드
        self.model = YOLO(self.base_model)
        
        # 학습 실행
        results = self.model.train(
            data=data_yaml,
            project=f"runs/{project_name}",
            name=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            exist_ok=True,
            pretrained=pretrained,
            device=self.device,
            **self.training_config
        )
        
        # 학습 기록 저장
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'model': self.base_model,
            'epochs': self.training_config['epochs'],
            'results': results
        })
        
        print(f"✅ 학습 완료!")
        
        return results
    
    def validate(self, data_yaml):
        """
        모델 검증
        
        Args:
            data_yaml: 데이터셋 설정 파일
        """
        if self.model is None:
            print("❌ 학습된 모델이 없습니다.")
            return
        
        print("\n🔍 모델 검증 시작...")
        
        metrics = self.model.val(
            data=data_yaml,
            device=self.device,
            batch=self.training_config['batch'],
            imgsz=self.training_config['imgsz']
        )
        
        print(f"\n📊 검증 결과:")
        print(f"   mAP@0.5: {metrics.box.map50:.4f}")
        print(f"   mAP@0.5-0.95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_model(self, format='onnx', output_path=None):
        """
        모델 내보내기
        
        Args:
            format: 내보낼 형식 (onnx, torchscript, tflite, etc.)
            output_path: 저장 경로
        """
        if self.model is None:
            print("❌ 학습된 모델이 없습니다.")
            return
        
        print(f"\n💾 모델 내보내기: {format}")
        
        exported_model = self.model.export(
            format=format,
            device=self.device,
            imgsz=self.training_config['imgsz']
        )
        
        if output_path:
            shutil.move(exported_model, output_path)
            print(f"✅ 모델 저장: {output_path}")
        
        return exported_model


class CustomObjectDetector:
    """
    파인튜닝된 모델을 사용한 커스텀 객체 검출기
    더 정확한 검출을 위한 고급 후처리 포함
    """
    
    def __init__(self, model_path, class_names, confidence_threshold=0.5):
        """
        초기화
        
        Args:
            model_path: 파인튜닝된 모델 경로
            class_names: 클래스 이름 리스트
            confidence_threshold: 기본 신뢰도 임계값
        """
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.conf_threshold = confidence_threshold
        
        # 클래스별 신뢰도 임계값 (더 정확한 검출을 위해)
        self.class_thresholds = {name: confidence_threshold 
                                for name in class_names}
        
        # 후처리 설정
        self.nms_threshold = 0.45
        self.max_detections = 100
        
        print(f"🎯 커스텀 검출기 초기화 완료")
        print(f"   모델: {model_path}")
        print(f"   클래스 수: {len(class_names)}")
        
    def set_class_threshold(self, class_name, threshold):
        """특정 클래스의 신뢰도 임계값 설정"""
        if class_name in self.class_names:
            self.class_thresholds[class_name] = threshold
            print(f"✅ {class_name} 클래스 임계값 설정: {threshold}")
        else:
            print(f"❌ 알 수 없는 클래스: {class_name}")
    
    def detect(self, image_path, apply_enhancement=True):
        """
        향상된 객체 검출
        
        Args:
            image_path: 이미지 경로
            apply_enhancement: 이미지 향상 적용 여부
        
        Returns:
            검출 결과 딕셔너리
        """
        # 이미지 전처리
        if apply_enhancement:
            image = self._enhance_image(image_path)
        else:
            image = cv2.imread(image_path)
        
        # 검출 수행
        results = self.model(
            image,
            conf=min(self.class_thresholds.values()),
            iou=self.nms_threshold,
            max_det=self.max_detections,
            verbose=False
        )
        
        # 후처리
        processed_results = self._postprocess(results[0])
        
        # 시각화
        self._visualize_detections(image, processed_results)
        
        return processed_results
    
    def _enhance_image(self, image_path):
        """
        이미지 품질 향상
        
        Args:
            image_path: 이미지 경로
        
        Returns:
            향상된 이미지
        """
        image = cv2.imread(image_path)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 노이즈 제거
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _postprocess(self, results):
        """
        검출 결과 후처리
        
        Args:
            results: YOLO 검출 결과
        
        Returns:
            처리된 검출 결과
        """
        processed = []
        
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else 'unknown'
                confidence = float(box.conf[0])
                
                # 클래스별 임계값 확인
                if confidence >= self.class_thresholds.get(cls_name, self.conf_threshold):
                    # 추가 검증 (예: 최소 크기 확인)
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 10 and height > 10:  # 최소 크기 필터
                        processed.append({
                            'class': cls_name,
                            'confidence': confidence,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'area': width * height,
                            'aspect_ratio': width / height if height > 0 else 0
                        })
        
        # 중복 제거 (추가 NMS)
        processed = self._remove_duplicates(processed)
        
        return processed
    
    def _remove_duplicates(self, detections, iou_threshold=0.5):
        """
        중복 검출 제거
        
        Args:
            detections: 검출 결과 리스트
            iou_threshold: IoU 임계값
        
        Returns:
            중복이 제거된 검출 결과
        """
        if len(detections) <= 1:
            return detections
        
        # 신뢰도 순으로 정렬
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        for det in detections:
            is_duplicate = False
            
            for kept_det in kept:
                if det['class'] == kept_det['class']:
                    iou = self._calculate_iou(det['bbox'], kept_det['bbox'])
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                kept.append(det)
        
        return kept
    
    def _calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _visualize_detections(self, image, detections):
        """검출 결과 시각화"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        ax.axis('off')
        
        # 클래스별 색상
        colors = plt.cm.hsv(np.linspace(0, 1, len(self.class_names)))
        color_map = {name: colors[i] for i, name in enumerate(self.class_names)}
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = color_map.get(det['class'], 'red')
            
            # 바운딩 박스
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor=color,
                                facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # 라벨
            label = f"{det['class']}: {det['confidence']:.3f}"
            ax.text(x1, y1-5, label, fontsize=10,
                   color='white', backgroundcolor=color[:3],
                   bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=color[:3], alpha=0.7))
        
        plt.title(f"Custom Detection: {len(detections)} objects")
        plt.tight_layout()
        plt.show()


class AutoFineTuningPipeline:
    """
    자동 파인튜닝 파이프라인
    데이터 준비부터 학습, 평가까지 자동화
    """
    
    def __init__(self, project_name="auto_finetuning"):
        """
        초기화
        
        Args:
            project_name: 프로젝트 이름
        """
        self.project_name = project_name
        self.dataset_preparer = None
        self.fine_tuner = None
        self.detector = None
        self.results = {}
        
        print(f"🤖 자동 파인튜닝 파이프라인 초기화: {project_name}")
    
    def prepare_dataset(self, images_dir, annotations_file, 
                       class_names, format_type="coco"):
        """
        데이터셋 준비
        
        Args:
            images_dir: 이미지 디렉토리
            annotations_file: 어노테이션 파일
            class_names: 클래스 이름 리스트
            format_type: 어노테이션 형식
        """
        print("\n📊 데이터셋 준비 중...")
        
        self.dataset_preparer = CustomDatasetPreparer(self.project_name)
        self.dataset_preparer.add_custom_classes(class_names)
        self.dataset_preparer.prepare_annotation_data(
            images_dir, annotations_file, format_type
        )
        
        # 데이터셋 분할
        splits = self.dataset_preparer.split_dataset()
        
        # YAML 설정 생성
        yaml_path = self.dataset_preparer.create_yaml_config()
        
        self.results['dataset'] = {
            'yaml_path': str(yaml_path),
            'splits': splits,
            'classes': class_names
        }
        
        return yaml_path
    
    def run_training(self, base_model="yolo11n.pt", epochs=100, 
                    batch_size=16, learning_rate=0.01):
        """
        학습 실행
        
        Args:
            base_model: 기본 모델
            epochs: 학습 에폭
            batch_size: 배치 크기
            learning_rate: 학습률
        """
        print("\n🎓 학습 시작...")
        
        self.fine_tuner = YOLOFineTuner(base_model)
        self.fine_tuner.configure_training(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # 학습 실행
        training_results = self.fine_tuner.train(
            data_yaml=self.results['dataset']['yaml_path'],
            project_name=self.project_name
        )
        
        # 검증
        val_metrics = self.fine_tuner.validate(
            self.results['dataset']['yaml_path']
        )
        
        self.results['training'] = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'metrics': {
                'mAP50': float(val_metrics.box.map50),
                'mAP50-95': float(val_metrics.box.map),
                'precision': float(val_metrics.box.mp),
                'recall': float(val_metrics.box.mr)
            }
        }
        
        return training_results
    
    def evaluate_model(self, test_images_dir):
        """
        모델 평가
        
        Args:
            test_images_dir: 테스트 이미지 디렉토리
        """
        print("\n📈 모델 평가 중...")
        
        if self.fine_tuner.model is None:
            print("❌ 학습된 모델이 없습니다.")
            return
        
        # 커스텀 검출기 생성
        self.detector = CustomObjectDetector(
            model_path=self.fine_tuner.model,
            class_names=self.results['dataset']['classes']
        )
        
        # 테스트 이미지 평가
        test_results = []
        test_images = list(Path(test_images_dir).glob("*.jpg")) + \
                     list(Path(test_images_dir).glob("*.png"))
        
        for img_path in test_images[:5]:  # 샘플로 5개만
            detections = self.detector.detect(str(img_path))
            test_results.append({
                'image': str(img_path),
                'detections': len(detections),
                'classes_found': list(set(d['class'] for d in detections))
            })
        
        self.results['evaluation'] = test_results
        
        return test_results
    
    def generate_report(self):
        """
        파인튜닝 결과 리포트 생성
        """
        print("\n📝 리포트 생성 중...")
        
        report = f"""
        ====================================
        파인튜닝 결과 리포트
        ====================================
        
        프로젝트: {self.project_name}
        날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        1. 데이터셋 정보
        ----------------
        클래스 수: {len(self.results['dataset']['classes'])}
        클래스: {', '.join(self.results['dataset']['classes'])}
        
        2. 학습 결과
        ------------
        Epochs: {self.results['training']['epochs']}
        Batch Size: {self.results['training']['batch_size']}
        Learning Rate: {self.results['training']['learning_rate']}
        
        성능 지표:
        - mAP@0.5: {self.results['training']['metrics']['mAP50']:.4f}
        - mAP@0.5-0.95: {self.results['training']['metrics']['mAP50-95']:.4f}
        - Precision: {self.results['training']['metrics']['precision']:.4f}
        - Recall: {self.results['training']['metrics']['recall']:.4f}
        
        3. 평가 결과
        ------------
        테스트 이미지 수: {len(self.results.get('evaluation', []))}
        """
        
        # 리포트 저장
        report_path = f"reports/{self.project_name}_report.txt"
        os.makedirs("reports", exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 리포트 저장: {report_path}")
        print(report)
        
        return report


def main():
    """
    파인튜닝 예제 실행
    """
    print("🚀 YOLO11 파인튜닝 시스템")
    print("="*50)
    
    # 자동 파인튜닝 파이프라인 생성
    pipeline = AutoFineTuningPipeline("my_custom_detector")
    
    # 예제: 커스텀 클래스 정의
    custom_classes = [
        "custom_object_1",
        "custom_object_2",
        "custom_object_3"
    ]
    
    print("\n커스텀 클래스:")
    for i, cls in enumerate(custom_classes):
        print(f"  {i}: {cls}")
    
    print("\n💡 사용 방법:")
    print("1. images/ 폴더에 학습 이미지 준비")
    print("2. annotations.json 파일에 COCO 형식 어노테이션 준비")
    print("3. pipeline.prepare_dataset() 실행")
    print("4. pipeline.run_training() 실행")
    print("5. pipeline.evaluate_model() 실행")
    print("6. pipeline.generate_report() 실행")
    
    print("\n📌 파인튜닝으로 다음이 가능합니다:")
    print("- 특정 객체에 대한 검출 정확도 향상")
    print("- 새로운 클래스 추가")
    print("- 도메인 특화 모델 생성")
    print("- 검출 속도 최적화")


if __name__ == "__main__":
    main()