"""
도메인 특화 객체 검출 프로그램
- 특정 분야에 최적화된 검출 수행
- 커스텀 후처리 및 필터링
- 실시간 비디오 스트림 처리
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import json
from collections import defaultdict
import time
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN


class DomainSpecificDetector:
    """
    도메인별 특화 검출기
    특정 용도에 최적화된 검출 수행
    """
    
    # 도메인별 중요 클래스 정의
    DOMAIN_CLASSES = {
        'traffic': ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 
                   'traffic light', 'stop sign'],
        'retail': ['person', 'handbag', 'backpack', 'suitcase', 'bottle', 
                  'cup', 'cell phone'],
        'security': ['person', 'backpack', 'handbag', 'suitcase', 'knife', 
                    'cell phone', 'laptop'],
        'wildlife': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                    'bear', 'zebra', 'giraffe'],
        'kitchen': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                   'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                   'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
        'office': ['person', 'chair', 'laptop', 'mouse', 'keyboard', 'cell phone',
                  'book', 'clock', 'scissors'],
        'sports': ['person', 'bicycle', 'sports ball', 'baseball bat', 
                  'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                  'frisbee', 'skis', 'snowboard']
    }
    
    def __init__(self, domain='general', model_path='yolo11x.pt'):
        """
        초기화
        
        Args:
            domain: 도메인 타입 ('traffic', 'retail', 'security', 등)
            model_path: YOLO 모델 경로
        """
        self.domain = domain
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 도메인별 타겟 클래스 설정
        self.target_classes = self.DOMAIN_CLASSES.get(domain, None)
        
        # 객체 추적 정보
        self.tracking_data = defaultdict(list)
        self.object_counter = 0
        self.frame_count = 0
        
        # 알람 설정
        self.alerts = []
        self.alert_conditions = self._get_alert_conditions(domain)
        
        print(f"Domain-specific detector initialized for: {domain}")
        if self.target_classes:
            print(f"Monitoring classes: {', '.join(self.target_classes)}")
    
    def _get_alert_conditions(self, domain):
        """도메인별 알람 조건 설정"""
        conditions = {
            'traffic': {
                'crowding_threshold': 10,  # 10명 이상 밀집
                'vehicle_speed_threshold': 50,  # km/h
                'pedestrian_in_road': True
            },
            'retail': {
                'crowd_threshold': 15,
                'unattended_bag_time': 30,  # seconds
                'suspicious_behavior': True
            },
            'security': {
                'restricted_area_intrusion': True,
                'weapon_detection': True,
                'crowd_formation': True,
                'loitering_time': 60  # seconds
            },
            'wildlife': {
                'species_count_threshold': 5,
                'rare_species_detection': True
            }
        }
        return conditions.get(domain, {})
    
    def detect_domain_specific(self, image_path, 
                              conf_threshold=0.45,
                              apply_filters=True,
                              show_analytics=True):
        """
        도메인 특화 검출
        
        Args:
            image_path: 이미지 경로
            conf_threshold: 신뢰도 임계값
            apply_filters: 도메인 필터 적용 여부
            show_analytics: 분석 정보 표시 여부
        
        Returns:
            검출 결과 및 분석 정보
        """
        # 기본 검출
        results = self.model(image_path, conf=conf_threshold, verbose=False)[0]
        
        # 도메인 필터링
        if apply_filters and self.target_classes:
            filtered_detections = self._filter_by_domain(results)
        else:
            filtered_detections = self._parse_all_detections(results)
        
        # 도메인별 후처리
        processed_detections = self._domain_postprocess(filtered_detections)
        
        # 분석 수행
        analytics = {}
        if show_analytics:
            analytics = self._perform_analytics(processed_detections)
        
        # 알람 체크
        alerts = self._check_alerts(processed_detections, analytics)
        
        # 시각화
        self._visualize_domain_specific(image_path, processed_detections, 
                                       analytics, alerts)
        
        return {
            'detections': processed_detections,
            'analytics': analytics,
            'alerts': alerts,
            'domain': self.domain
        }
    
    def _filter_by_domain(self, results):
        """도메인별 필터링"""
        filtered = []
        
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                class_name = results.names[cls]
                
                if class_name in self.target_classes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    filtered.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                        'area': (x2 - x1) * (y2 - y1)
                    })
        
        return filtered
    
    def _parse_all_detections(self, results):
        """모든 검출 결과 파싱"""
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                class_name = results.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        return detections
    
    def _domain_postprocess(self, detections):
        """도메인별 후처리"""
        if self.domain == 'traffic':
            return self._traffic_postprocess(detections)
        elif self.domain == 'retail':
            return self._retail_postprocess(detections)
        elif self.domain == 'security':
            return self._security_postprocess(detections)
        else:
            return detections
    
    def _traffic_postprocess(self, detections):
        """교통 도메인 후처리"""
        # 차량과 보행자 분리
        vehicles = [d for d in detections 
                   if d['class'] in ['car', 'bus', 'truck', 'motorcycle']]
        pedestrians = [d for d in detections if d['class'] == 'person']
        
        # 차선 침범 체크, 거리 계산 등
        for det in detections:
            det['domain_info'] = {
                'type': 'vehicle' if det['class'] in ['car', 'bus', 'truck'] else 'other',
                'risk_level': self._calculate_traffic_risk(det, pedestrians)
            }
        
        return detections
    
    def _retail_postprocess(self, detections):
        """리테일 도메인 후처리"""
        # 고객 행동 분석
        persons = [d for d in detections if d['class'] == 'person']
        items = [d for d in detections if d['class'] in ['handbag', 'backpack']]
        
        for det in detections:
            det['domain_info'] = {
                'customer_density': len(persons),
                'interaction_zone': self._check_interaction_zone(det, items)
            }
        
        return detections
    
    def _security_postprocess(self, detections):
        """보안 도메인 후처리"""
        # 의심 객체 표시
        suspicious_items = ['knife', 'backpack', 'suitcase']
        
        for det in detections:
            is_suspicious = det['class'] in suspicious_items
            det['domain_info'] = {
                'suspicious': is_suspicious,
                'threat_level': 'high' if is_suspicious else 'low',
                'requires_attention': is_suspicious
            }
        
        return detections
    
    def _calculate_traffic_risk(self, vehicle, pedestrians):
        """교통 위험도 계산"""
        if not pedestrians:
            return 'low'
        
        # 차량과 보행자 간 거리 계산
        vehicle_center = vehicle['center']
        min_distance = float('inf')
        
        for ped in pedestrians:
            dist = distance.euclidean(vehicle_center, ped['center'])
            min_distance = min(min_distance, dist)
        
        # 거리 기반 위험도
        if min_distance < 50:
            return 'high'
        elif min_distance < 100:
            return 'medium'
        else:
            return 'low'
    
    def _check_interaction_zone(self, person, items):
        """상호작용 영역 체크"""
        if person['class'] != 'person':
            return False
        
        person_center = person['center']
        for item in items:
            dist = distance.euclidean(person_center, item['center'])
            if dist < 100:  # 픽셀 단위
                return True
        return False
    
    def _perform_analytics(self, detections):
        """분석 수행"""
        analytics = {
            'total_objects': len(detections),
            'class_distribution': {},
            'density_map': None,
            'clusters': []
        }
        
        # 클래스별 분포
        for det in detections:
            cls = det['class']
            analytics['class_distribution'][cls] = \
                analytics['class_distribution'].get(cls, 0) + 1
        
        # 밀도 분석
        if len(detections) > 3:
            centers = np.array([d['center'] for d in detections])
            
            # DBSCAN 클러스터링
            clustering = DBSCAN(eps=100, min_samples=2).fit(centers)
            labels = clustering.labels_
            
            # 클러스터 정보
            unique_labels = set(labels)
            for label in unique_labels:
                if label != -1:  # -1은 노이즈
                    cluster_points = centers[labels == label]
                    analytics['clusters'].append({
                        'id': label,
                        'size': len(cluster_points),
                        'center': cluster_points.mean(axis=0).tolist()
                    })
        
        # 도메인별 추가 분석
        if self.domain == 'traffic':
            analytics['vehicle_count'] = sum(1 for d in detections 
                                            if d['class'] in ['car', 'bus', 'truck'])
            analytics['pedestrian_count'] = sum(1 for d in detections 
                                               if d['class'] == 'person')
        
        elif self.domain == 'retail':
            analytics['customer_count'] = sum(1 for d in detections 
                                             if d['class'] == 'person')
            analytics['avg_confidence'] = np.mean([d['confidence'] 
                                                  for d in detections])
        
        return analytics
    
    def _check_alerts(self, detections, analytics):
        """알람 조건 체크"""
        alerts = []
        
        if self.domain == 'traffic':
            # 과밀 체크
            if analytics.get('pedestrian_count', 0) > \
               self.alert_conditions.get('crowding_threshold', 10):
                alerts.append({
                    'type': 'crowding',
                    'severity': 'high',
                    'message': f"High pedestrian density: {analytics['pedestrian_count']} people"
                })
        
        elif self.domain == 'security':
            # 위험 물품 체크
            for det in detections:
                if det.get('domain_info', {}).get('suspicious', False):
                    alerts.append({
                        'type': 'suspicious_object',
                        'severity': 'high',
                        'message': f"Suspicious object detected: {det['class']}"
                    })
        
        elif self.domain == 'retail':
            # 고객 밀집도 체크
            if analytics.get('customer_count', 0) > \
               self.alert_conditions.get('crowd_threshold', 15):
                alerts.append({
                    'type': 'crowd',
                    'severity': 'medium',
                    'message': f"High customer density: {analytics['customer_count']} people"
                })
        
        return alerts
    
    def _visualize_domain_specific(self, image_path, detections, 
                                  analytics, alerts):
        """도메인 특화 시각화"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 메인 검출 이미지
        ax1.imshow(image_rgb)
        ax1.set_title(f"{self.domain.upper()} Domain Detection", 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 색상 매핑
        color_map = {
            'person': 'blue',
            'car': 'green',
            'bus': 'orange',
            'truck': 'brown',
            'bicycle': 'cyan',
            'motorcycle': 'purple'
        }
        
        # 바운딩 박스 그리기
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = color_map.get(det['class'], 'red')
            
            # 도메인 정보에 따른 스타일 조정
            linewidth = 2
            linestyle = '-'
            alpha = 0.8
            
            if det.get('domain_info', {}).get('suspicious', False):
                color = 'red'
                linewidth = 3
                linestyle = '--'
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=linewidth, edgecolor=color,
                                facecolor='none', alpha=alpha,
                                linestyle=linestyle)
            ax1.add_patch(rect)
            
            # 라벨
            label = f"{det['class']} {det['confidence']:.2f}"
            if 'domain_info' in det:
                if 'risk_level' in det['domain_info']:
                    label += f" [{det['domain_info']['risk_level']}]"
            
            ax1.text(x1, y1-5, label, fontsize=9, color='white',
                    bbox=dict(boxstyle="round,pad=0.3",
                            facecolor=color, alpha=0.7))
        
        # 클러스터 표시
        for cluster in analytics.get('clusters', []):
            circle = plt.Circle(cluster['center'], 50,
                               color='yellow', fill=False,
                               linewidth=2, linestyle='--', alpha=0.5)
            ax1.add_patch(circle)
            ax1.text(cluster['center'][0], cluster['center'][1],
                    f"C{cluster['id']}", fontsize=12,
                    color='yellow', fontweight='bold')
        
        # 분석 정보 표시
        ax2.axis('off')
        info_text = f"Domain: {self.domain.upper()}\n"
        info_text += f"Total Objects: {analytics['total_objects']}\n\n"
        
        info_text += "Class Distribution:\n"
        for cls, count in analytics['class_distribution'].items():
            info_text += f"  • {cls}: {count}\n"
        
        if self.domain == 'traffic':
            info_text += f"\nVehicles: {analytics.get('vehicle_count', 0)}\n"
            info_text += f"Pedestrians: {analytics.get('pedestrian_count', 0)}\n"
        
        if analytics.get('clusters'):
            info_text += f"\nClusters Found: {len(analytics['clusters'])}\n"
            for cluster in analytics['clusters']:
                info_text += f"  • Cluster {cluster['id']}: {cluster['size']} objects\n"
        
        # 알람 표시
        if alerts:
            info_text += "\n⚠️ ALERTS:\n"
            for alert in alerts:
                info_text += f"  • [{alert['severity'].upper()}] {alert['message']}\n"
        
        ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5",
                        facecolor='lightgray', alpha=0.8))
        
        # 신뢰도 분포 그래프
        if detections:
            confidences = [d['confidence'] for d in detections]
            ax2_sub = fig.add_axes([0.55, 0.3, 0.35, 0.2])
            ax2_sub.hist(confidences, bins=10, color='skyblue', edgecolor='black')
            ax2_sub.set_xlabel('Confidence')
            ax2_sub.set_ylabel('Count')
            ax2_sub.set_title('Confidence Distribution')
            ax2_sub.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def process_video_stream(self, video_source, output_path=None,
                           show_live=True, save_alerts=True):
        """
        비디오 스트림 처리
        
        Args:
            video_source: 비디오 파일 경로 또는 웹캠 인덱스
            output_path: 출력 비디오 경로
            show_live: 실시간 표시 여부
            save_alerts: 알람 저장 여부
        """
        cap = cv2.VideoCapture(video_source)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        alert_log = []
        frame_count = 0
        
        print(f"Processing video stream for {self.domain} domain...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 매 N 프레임마다 검출 수행
            if frame_count % 3 == 0:  # 3프레임마다
                # 임시 파일로 저장
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                # 검출 수행
                result = self.detect_domain_specific(
                    temp_path,
                    show_analytics=False
                )
                
                # 프레임에 결과 그리기
                for det in result['detections']:
                    x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                    
                    # 색상 설정
                    color = (0, 255, 0)  # 기본 녹색
                    if det.get('domain_info', {}).get('suspicious', False):
                        color = (0, 0, 255)  # 빨간색
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{det['class']} {det['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 알람 표시
                if result['alerts']:
                    for i, alert in enumerate(result['alerts']):
                        alert_text = f"[{alert['severity']}] {alert['message']}"
                        cv2.putText(frame, alert_text, (10, 30 + i*30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                  (0, 0, 255), 2)
                        
                        if save_alerts:
                            alert_log.append({
                                'frame': frame_count,
                                'alert': alert
                            })
                
                # 도메인 정보 표시
                info_text = f"Domain: {self.domain} | Objects: {len(result['detections'])}"
                cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 출력 저장
            if output_path:
                out.write(frame)
            
            # 실시간 표시
            if show_live:
                cv2.imshow(f'{self.domain.upper()} Domain Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # 정리
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # 알람 로그 저장
        if save_alerts and alert_log:
            with open(f'{self.domain}_alerts.json', 'w') as f:
                json.dump(alert_log, f, indent=2)
            print(f"Alert log saved: {self.domain}_alerts.json")
        
        print(f"Video processing complete. Total frames: {frame_count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Domain-Specific Object Detection')
    parser.add_argument('--image', '-i', type=str,
                       help='Input image path')
    parser.add_argument('--video', '-v', type=str,
                       help='Input video path or webcam index')
    parser.add_argument('--domain', '-d', type=str, default='general',
                       choices=['general', 'traffic', 'retail', 'security',
                               'wildlife', 'kitchen', 'office', 'sports'],
                       help='Detection domain')
    parser.add_argument('--confidence', '-c', type=float, default=0.45,
                       help='Confidence threshold')
    parser.add_argument('--output', '-o', type=str,
                       help='Output path')
    
    args = parser.parse_args()
    
    # 검출기 생성
    detector = DomainSpecificDetector(domain=args.domain)
    
    if args.video:
        # 비디오 처리
        video_source = int(args.video) if args.video.isdigit() else args.video
        detector.process_video_stream(video_source, output_path=args.output)
    elif args.image:
        # 이미지 처리
        result = detector.detect_domain_specific(
            args.image,
            conf_threshold=args.confidence
        )
        print(f"\nDetection complete for {args.domain} domain")
        print(f"Found {len(result['detections'])} relevant objects")
        if result['alerts']:
            print(f"Generated {len(result['alerts'])} alerts")
    else:
        print("Please provide either --image or --video argument")


if __name__ == "__main__":
    main()