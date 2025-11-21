"""
실시간 학습 및 모니터링 시스템
Active Learning과 Online Fine-tuning을 통한 지속적 성능 개선
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import json
import time
from datetime import datetime
import threading
import queue
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from collections import deque
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore')


class ActiveLearningManager:
    """
    Active Learning 관리자
    불확실한 샘플을 선별하여 학습 효율 극대화
    """
    
    def __init__(self, uncertainty_threshold=0.3, sample_buffer_size=100):
        """
        초기화
        
        Args:
            uncertainty_threshold: 불확실성 임계값
            sample_buffer_size: 샘플 버퍼 크기
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.sample_buffer = deque(maxlen=sample_buffer_size)
        self.selected_samples = []
        self.learning_stats = {
            'total_samples': 0,
            'uncertain_samples': 0,
            'selected_for_training': 0
        }
        
    def calculate_uncertainty(self, detections):
        """
        검출 결과의 불확실성 계산
        
        Args:
            detections: 검출 결과 리스트
        
        Returns:
            불확실성 점수 (0~1)
        """
        if not detections:
            return 1.0  # 아무것도 검출 못한 경우 최대 불확실성
        
        confidences = [d.get('confidence', 0) for d in detections]
        
        # 여러 메트릭을 조합한 불확실성 계산
        uncertainty_metrics = {
            'low_confidence': 1.0 - np.mean(confidences) if confidences else 1.0,
            'high_variance': np.std(confidences) if len(confidences) > 1 else 0,
            'detection_count': 1.0 / (1 + len(detections))  # 검출 수가 적을수록 불확실
        }
        
        # 가중 평균
        weights = {'low_confidence': 0.5, 'high_variance': 0.3, 'detection_count': 0.2}
        uncertainty = sum(weights[k] * v for k, v in uncertainty_metrics.items())
        
        return min(1.0, uncertainty)
    
    def should_add_sample(self, image_path, detections):
        """
        샘플을 학습 데이터에 추가할지 결정
        
        Args:
            image_path: 이미지 경로
            detections: 검출 결과
        
        Returns:
            추가 여부 (bool)
        """
        uncertainty = self.calculate_uncertainty(detections)
        
        self.learning_stats['total_samples'] += 1
        
        if uncertainty > self.uncertainty_threshold:
            self.learning_stats['uncertain_samples'] += 1
            
            # 중복 체크 (이미지 해시 기반)
            img_hash = self._get_image_hash(image_path)
            
            if img_hash not in [s['hash'] for s in self.sample_buffer]:
                self.sample_buffer.append({
                    'path': image_path,
                    'detections': detections,
                    'uncertainty': uncertainty,
                    'timestamp': datetime.now(),
                    'hash': img_hash
                })
                
                self.learning_stats['selected_for_training'] += 1
                return True
        
        return False
    
    def _get_image_hash(self, image_path):
        """이미지 해시 계산"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_training_batch(self, batch_size=16):
        """
        학습용 배치 선택
        
        Args:
            batch_size: 배치 크기
        
        Returns:
            선택된 샘플 리스트
        """
        # 불확실성이 높은 순으로 정렬
        sorted_samples = sorted(self.sample_buffer, 
                              key=lambda x: x['uncertainty'], 
                              reverse=True)
        
        # 상위 batch_size개 선택
        batch = sorted_samples[:batch_size]
        
        # 선택된 샘플 제거
        for sample in batch:
            if sample in self.sample_buffer:
                self.sample_buffer.remove(sample)
        
        return batch
    
    def get_statistics(self):
        """학습 통계 반환"""
        return {
            **self.learning_stats,
            'buffer_size': len(self.sample_buffer),
            'selection_rate': self.learning_stats['selected_for_training'] / 
                            max(1, self.learning_stats['total_samples'])
        }


class OnlineFineTuner:
    """
    온라인 파인튜닝 시스템
    실시간으로 모델을 업데이트
    """
    
    def __init__(self, base_model="yolo11n.pt", update_frequency=50):
        """
        초기화
        
        Args:
            base_model: 기본 모델
            update_frequency: 업데이트 주기 (샘플 수)
        """
        self.base_model = base_model
        self.current_model = YOLO(base_model)
        self.update_frequency = update_frequency
        self.update_count = 0
        self.training_queue = queue.Queue()
        self.is_training = False
        
        # 성능 추적
        self.performance_history = {
            'timestamps': [],
            'mAP': [],
            'precision': [],
            'recall': [],
            'loss': []
        }
        
        # 모델 버전 관리
        self.model_versions = []
        self.current_version = 0
        
        print(f"🔄 온라인 파인튜닝 시스템 초기화")
        print(f"   기본 모델: {base_model}")
        print(f"   업데이트 주기: {update_frequency} samples")
    
    def add_training_sample(self, image_path, annotations):
        """
        학습 샘플 추가
        
        Args:
            image_path: 이미지 경로
            annotations: 어노테이션 데이터
        """
        self.training_queue.put({
            'image': image_path,
            'annotations': annotations,
            'timestamp': datetime.now()
        })
    
    def start_training_thread(self):
        """백그라운드 학습 스레드 시작"""
        if not self.is_training:
            self.is_training = True
            thread = threading.Thread(target=self._training_loop, daemon=True)
            thread.start()
            print("🎯 백그라운드 학습 스레드 시작")
    
    def _training_loop(self):
        """백그라운드 학습 루프"""
        batch = []
        
        while self.is_training:
            try:
                # 큐에서 샘플 가져오기
                sample = self.training_queue.get(timeout=1)
                batch.append(sample)
                
                # 배치가 차면 학습
                if len(batch) >= self.update_frequency:
                    self._perform_update(batch)
                    batch = []
                    
            except queue.Empty:
                # 큐가 비어있으면 대기
                time.sleep(1)
            except Exception as e:
                print(f"❌ 학습 오류: {e}")
    
    def _perform_update(self, batch):
        """
        모델 업데이트 수행
        
        Args:
            batch: 학습 샘플 배치
        """
        print(f"\n🔄 모델 업데이트 시작 (배치 크기: {len(batch)})")
        
        try:
            # 임시 데이터셋 생성
            temp_dataset = self._create_temp_dataset(batch)
            
            # 파인튜닝 수행 (짧은 에폭)
            results = self.current_model.train(
                data=temp_dataset,
                epochs=5,  # 빠른 업데이트를 위해 짧게
                batch=len(batch),
                verbose=False,
                device=0 if torch.cuda.is_available() else 'cpu'
            )
            
            # 성능 기록
            self.performance_history['timestamps'].append(datetime.now())
            self.performance_history['mAP'].append(results.box.map if hasattr(results, 'box') else 0)
            self.performance_history['loss'].append(results.loss if hasattr(results, 'loss') else 0)
            
            # 모델 버전 저장
            self.save_model_version()
            
            self.update_count += 1
            print(f"✅ 모델 업데이트 완료 (버전: {self.current_version})")
            
        except Exception as e:
            print(f"❌ 업데이트 실패: {e}")
    
    def _create_temp_dataset(self, batch):
        """임시 데이터셋 생성"""
        # 실제 구현에서는 YOLO 형식으로 변환
        # 여기서는 경로만 반환
        temp_dir = Path("temp_dataset") / f"batch_{self.update_count}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장 로직...
        
        return str(temp_dir / "dataset.yaml")
    
    def save_model_version(self):
        """모델 버전 저장"""
        version_path = f"model_versions/v{self.current_version}.pt"
        os.makedirs("model_versions", exist_ok=True)
        
        self.current_model.save(version_path)
        
        self.model_versions.append({
            'version': self.current_version,
            'path': version_path,
            'timestamp': datetime.now(),
            'performance': {
                'mAP': self.performance_history['mAP'][-1] if self.performance_history['mAP'] else 0
            }
        })
        
        self.current_version += 1
    
    def rollback_to_version(self, version):
        """특정 버전으로 롤백"""
        for v in self.model_versions:
            if v['version'] == version:
                self.current_model = YOLO(v['path'])
                self.current_version = version
                print(f"✅ 버전 {version}으로 롤백 완료")
                return True
        
        print(f"❌ 버전 {version}을 찾을 수 없습니다")
        return False
    
    def get_best_model(self):
        """최고 성능 모델 반환"""
        if not self.model_versions:
            return self.current_model
        
        best_version = max(self.model_versions, 
                          key=lambda x: x['performance']['mAP'])
        
        return YOLO(best_version['path'])


class RealTimeMonitor:
    """
    실시간 성능 모니터링 시스템
    """
    
    def __init__(self, window_size=100):
        """
        초기화
        
        Args:
            window_size: 모니터링 윈도우 크기
        """
        self.window_size = window_size
        self.metrics_buffer = {
            'fps': deque(maxlen=window_size),
            'detections': deque(maxlen=window_size),
            'confidence': deque(maxlen=window_size),
            'processing_time': deque(maxlen=window_size)
        }
        
        self.alerts = []
        self.thresholds = {
            'min_fps': 20,
            'min_confidence': 0.5,
            'max_processing_time': 100  # ms
        }
        
        # 시각화 설정
        self.fig = None
        self.axes = None
        self.lines = {}
        
    def update_metrics(self, fps, detections, avg_confidence, processing_time):
        """
        메트릭 업데이트
        
        Args:
            fps: 초당 프레임 수
            detections: 검출 수
            avg_confidence: 평균 신뢰도
            processing_time: 처리 시간 (ms)
        """
        self.metrics_buffer['fps'].append(fps)
        self.metrics_buffer['detections'].append(detections)
        self.metrics_buffer['confidence'].append(avg_confidence)
        self.metrics_buffer['processing_time'].append(processing_time)
        
        # 알람 체크
        self._check_alerts()
    
    def _check_alerts(self):
        """성능 알람 체크"""
        current_metrics = {
            'fps': np.mean(self.metrics_buffer['fps']) if self.metrics_buffer['fps'] else 0,
            'confidence': np.mean(self.metrics_buffer['confidence']) if self.metrics_buffer['confidence'] else 0,
            'processing_time': np.mean(self.metrics_buffer['processing_time']) if self.metrics_buffer['processing_time'] else 0
        }
        
        # FPS 체크
        if current_metrics['fps'] < self.thresholds['min_fps']:
            self.alerts.append({
                'type': 'LOW_FPS',
                'value': current_metrics['fps'],
                'threshold': self.thresholds['min_fps'],
                'timestamp': datetime.now()
            })
        
        # 신뢰도 체크
        if current_metrics['confidence'] < self.thresholds['min_confidence']:
            self.alerts.append({
                'type': 'LOW_CONFIDENCE',
                'value': current_metrics['confidence'],
                'threshold': self.thresholds['min_confidence'],
                'timestamp': datetime.now()
            })
    
    def create_dashboard(self):
        """실시간 대시보드 생성"""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Real-Time Performance Monitor', fontsize=16)
        
        # FPS 그래프
        self.axes[0, 0].set_title('FPS')
        self.axes[0, 0].set_xlabel('Time')
        self.axes[0, 0].set_ylabel('Frames/sec')
        self.lines['fps'], = self.axes[0, 0].plot([], [], 'b-')
        
        # 검출 수 그래프
        self.axes[0, 1].set_title('Detections')
        self.axes[0, 1].set_xlabel('Time')
        self.axes[0, 1].set_ylabel('Count')
        self.lines['detections'], = self.axes[0, 1].plot([], [], 'g-')
        
        # 신뢰도 그래프
        self.axes[1, 0].set_title('Average Confidence')
        self.axes[1, 0].set_xlabel('Time')
        self.axes[1, 0].set_ylabel('Confidence')
        self.lines['confidence'], = self.axes[1, 0].plot([], [], 'r-')
        
        # 처리 시간 그래프
        self.axes[1, 1].set_title('Processing Time')
        self.axes[1, 1].set_xlabel('Time')
        self.axes[1, 1].set_ylabel('Time (ms)')
        self.lines['processing_time'], = self.axes[1, 1].plot([], [], 'm-')
        
        plt.tight_layout()
    
    def update_dashboard(self):
        """대시보드 업데이트"""
        if self.fig is None:
            self.create_dashboard()
        
        for key in self.metrics_buffer:
            if self.metrics_buffer[key]:
                x = list(range(len(self.metrics_buffer[key])))
                y = list(self.metrics_buffer[key])
                
                self.lines[key].set_data(x, y)
                
                # 축 범위 조정
                ax_idx = list(self.metrics_buffer.keys()).index(key)
                ax = self.axes[ax_idx // 2, ax_idx % 2]
                ax.relim()
                ax.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def generate_report(self):
        """성능 리포트 생성"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'alerts': self.alerts[-10:]  # 최근 10개 알람
        }
        
        for key in self.metrics_buffer:
            if self.metrics_buffer[key]:
                report['metrics'][key] = {
                    'mean': float(np.mean(self.metrics_buffer[key])),
                    'std': float(np.std(self.metrics_buffer[key])),
                    'min': float(np.min(self.metrics_buffer[key])),
                    'max': float(np.max(self.metrics_buffer[key]))
                }
        
        return report


class IntegratedLearningSystem:
    """
    통합 학습 시스템
    Active Learning + Online Fine-tuning + Real-time Monitoring
    """
    
    def __init__(self, base_model="yolo11n.pt"):
        """
        초기화
        
        Args:
            base_model: 기본 모델
        """
        self.base_model = base_model
        self.detector = YOLO(base_model)
        
        # 컴포넌트 초기화
        self.active_learner = ActiveLearningManager()
        self.online_tuner = OnlineFineTuner(base_model)
        self.monitor = RealTimeMonitor()
        
        # 상태
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        
        print("🚀 통합 학습 시스템 초기화 완료")
    
    def process_frame(self, frame):
        """
        프레임 처리
        
        Args:
            frame: 입력 프레임 (numpy array)
        
        Returns:
            처리된 프레임, 검출 결과
        """
        start_time = time.time()
        
        # 검출 수행
        results = self.detector(frame, verbose=False)
        
        # 검출 결과 파싱
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0])
                })
        
        # Active Learning 체크
        if self.active_learner.should_add_sample('current_frame', detections):
            # 학습 큐에 추가
            self.online_tuner.add_training_sample('current_frame', detections)
        
        # 메트릭 계산
        processing_time = (time.time() - start_time) * 1000  # ms
        fps = 1000 / processing_time if processing_time > 0 else 0
        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
        
        # 모니터링 업데이트
        self.monitor.update_metrics(fps, len(detections), avg_confidence, processing_time)
        
        # 프레임에 결과 그리기
        annotated_frame = self._annotate_frame(frame, detections)
        
        self.frame_count += 1
        
        return annotated_frame, detections
    
    def _annotate_frame(self, frame, detections):
        """프레임에 검출 결과 표시"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox'].astype(int)
            conf = det['confidence']
            
            # 바운딩 박스
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨
            label = f"Class {det['class']}: {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 통계 정보 표시
        stats = self.active_learner.get_statistics()
        info_text = f"Frames: {self.frame_count} | Uncertain: {stats['uncertain_samples']} | Buffer: {stats['buffer_size']}"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def start(self, video_source=0):
        """
        시스템 시작
        
        Args:
            video_source: 비디오 소스 (0: 웹캠, 파일 경로)
        """
        print("🎬 시스템 시작...")
        
        self.is_running = True
        self.start_time = time.time()
        
        # 백그라운드 학습 시작
        self.online_tuner.start_training_thread()
        
        # 모니터링 대시보드 생성
        self.monitor.create_dashboard()
        
        # 비디오 캡처
        cap = cv2.VideoCapture(video_source)
        
        print("Press 'q' to quit, 's' to save snapshot, 'r' to generate report")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 처리
            annotated_frame, detections = self.process_frame(frame)
            
            # 결과 표시
            cv2.imshow('Integrated Learning System', annotated_frame)
            
            # 대시보드 업데이트 (10프레임마다)
            if self.frame_count % 10 == 0:
                self.monitor.update_dashboard()
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_snapshot(annotated_frame)
            elif key == ord('r'):
                self._generate_system_report()
        
        # 정리
        cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        
        print("✅ 시스템 종료")
    
    def _save_snapshot(self, frame):
        """스냅샷 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"snapshots/snapshot_{timestamp}.jpg"
        os.makedirs("snapshots", exist_ok=True)
        cv2.imwrite(filename, frame)
        print(f"📸 스냅샷 저장: {filename}")
    
    def _generate_system_report(self):
        """시스템 리포트 생성"""
        report = {
            'system_info': {
                'runtime': time.time() - self.start_time,
                'frames_processed': self.frame_count,
                'model_version': self.online_tuner.current_version
            },
            'active_learning': self.active_learner.get_statistics(),
            'performance': self.monitor.generate_report(),
            'model_history': self.online_tuner.performance_history
        }
        
        # 리포트 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"reports/system_report_{timestamp}.json"
        os.makedirs("reports", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 리포트 생성: {filename}")
        
        # 요약 출력
        print("\n=== 시스템 요약 ===")
        print(f"실행 시간: {report['system_info']['runtime']:.1f}초")
        print(f"처리 프레임: {report['system_info']['frames_processed']}")
        print(f"불확실 샘플: {report['active_learning']['uncertain_samples']}")
        print(f"모델 버전: {report['system_info']['model_version']}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🤖 YOLO11 실시간 학습 시스템")
    print("=" * 60)
    
    # 시스템 생성
    system = IntegratedLearningSystem(base_model="yolo11n.pt")
    
    print("\n기능:")
    print("- Active Learning: 불확실한 샘플 자동 선별")
    print("- Online Fine-tuning: 실시간 모델 업데이트")
    print("- Real-time Monitoring: 성능 모니터링")
    print("- Auto-annotation: 자동 라벨링")
    
    print("\n사용법:")
    print("1. system.start(0) - 웹캠으로 시작")
    print("2. system.start('video.mp4') - 비디오 파일로 시작")
    
    # 예제 실행
    # system.start(0)  # 웹캠 사용


if __name__ == "__main__":
    main()