"""
YOLO 모델 테스트 및 비교 도구
- 다양한 모델 성능 비교
- 정확도 및 속도 측정
- 시각화 리포트 생성
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import os
from typing import List, Dict
import seaborn as sns
from tqdm import tqdm
import urllib.request


class ModelComparator:
    """
    YOLO 모델 비교 도구
    다양한 모델의 성능을 체계적으로 비교
    """
    
    def __init__(self):
        self.models = {}
        self.results = []
        self.test_images = []
        
    def add_model(self, name, model_path):
        """모델 추가"""
        print(f"Loading {name}...")
        self.models[name] = YOLO(model_path)
        
    def prepare_test_suite(self):
        """테스트 이미지 준비"""
        # 테스트 이미지 URL 목록
        test_urls = [
            ("https://raw.githubusercontent.com/ultralytics/assets/main/bus.jpg", "street_scene.jpg"),
            ("https://raw.githubusercontent.com/ultralytics/assets/main/zidane.jpg", "sports.jpg"),
        ]
        
        os.makedirs("test_images", exist_ok=True)
        
        for url, filename in test_urls:
            filepath = f"test_images/{filename}"
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    self.test_images.append(filepath)
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
            else:
                self.test_images.append(filepath)
        
        print(f"Test suite ready with {len(self.test_images)} images")
    
    def run_comprehensive_test(self, conf_threshold=0.5):
        """
        종합 테스트 실행
        
        Args:
            conf_threshold: 신뢰도 임계값
        
        Returns:
            테스트 결과 DataFrame
        """
        print("\n" + "="*60)
        print("Running Comprehensive Model Comparison")
        print("="*60)
        
        all_results = []
        
        for img_path in tqdm(self.test_images, desc="Processing images"):
            img_name = Path(img_path).stem
            
            for model_name, model in self.models.items():
                # 워밍업 (첫 실행은 느림)
                _ = model(img_path, verbose=False)
                
                # 실제 측정
                metrics = self._test_single_model(model, model_name, 
                                                 img_path, conf_threshold)
                metrics['image'] = img_name
                all_results.append(metrics)
        
        # DataFrame 생성
        df = pd.DataFrame(all_results)
        self.results = df
        
        return df
    
    def _test_single_model(self, model, model_name, image_path, conf_threshold):
        """단일 모델 테스트"""
        # 시간 측정
        times = []
        for _ in range(3):  # 3회 측정 평균
            start = time.time()
            results = model(image_path, conf=conf_threshold, verbose=False)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        
        # 검출 결과 분석
        result = results[0]
        num_detections = 0
        class_counts = {}
        confidences = []
        
        if result.boxes is not None:
            num_detections = len(result.boxes)
            
            for box in result.boxes:
                cls = int(box.cls[0])
                class_name = result.names[cls]
                conf = float(box.conf[0])
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidences.append(conf)
        
        # 메트릭 계산
        metrics = {
            'model': model_name,
            'inference_time': avg_time,
            'fps': 1 / avg_time,
            'detections': num_detections,
            'unique_classes': len(class_counts),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0,
            'std_confidence': np.std(confidences) if confidences else 0
        }
        
        # 모델 크기 정보
        model_info = self._get_model_info(model_name)
        metrics.update(model_info)
        
        return metrics
    
    def _get_model_info(self, model_name):
        """모델 정보 추출"""
        info = {
            'model_size': 'Unknown',
            'parameters': 0,
            'complexity': 'Unknown'
        }
        
        # 모델 이름에서 크기 추론
        if 'n' in model_name.lower():
            info['model_size'] = 'Nano'
            info['parameters'] = '3.2M'
            info['complexity'] = 'Low'
        elif 's' in model_name.lower():
            info['model_size'] = 'Small'
            info['parameters'] = '11.2M'
            info['complexity'] = 'Low-Medium'
        elif 'm' in model_name.lower():
            info['model_size'] = 'Medium'
            info['parameters'] = '25.9M'
            info['complexity'] = 'Medium'
        elif 'l' in model_name.lower():
            info['model_size'] = 'Large'
            info['parameters'] = '43.7M'
            info['complexity'] = 'Medium-High'
        elif 'x' in model_name.lower():
            info['model_size'] = 'Extra Large'
            info['parameters'] = '68.2M'
            info['complexity'] = 'High'
        
        return info
    
    def generate_report(self, output_dir='comparison_report'):
        """비교 리포트 생성"""
        if self.results is None or len(self.results) == 0:
            print("No results to generate report")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. 속도 비교 차트
        self._plot_speed_comparison(output_dir)
        
        # 2. 정확도 비교 차트
        self._plot_accuracy_comparison(output_dir)
        
        # 3. 효율성 매트릭스
        self._plot_efficiency_matrix(output_dir)
        
        # 4. 종합 리포트
        self._generate_summary_report(output_dir)
        
        print(f"Report generated in {output_dir}/")
    
    def _plot_speed_comparison(self, output_dir):
        """속도 비교 차트"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # FPS 비교
        model_avg = self.results.groupby('model').agg({
            'fps': 'mean',
            'inference_time': 'mean'
        }).reset_index()
        
        ax1.bar(model_avg['model'], model_avg['fps'], color='skyblue')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('FPS')
        ax1.set_title('Average FPS Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # 추론 시간 비교
        ax2.bar(model_avg['model'], model_avg['inference_time'] * 1000, 
               color='lightcoral')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Average Inference Time')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/speed_comparison.png', dpi=150)
        plt.close()
    
    def _plot_accuracy_comparison(self, output_dir):
        """정확도 비교 차트"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 검출 수 비교
        model_avg = self.results.groupby('model').agg({
            'detections': 'mean',
            'avg_confidence': 'mean'
        }).reset_index()
        
        ax1.bar(model_avg['model'], model_avg['detections'], color='lightgreen')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average Detections')
        ax1.set_title('Average Number of Detections')
        ax1.tick_params(axis='x', rotation=45)
        
        # 신뢰도 비교
        ax2.bar(model_avg['model'], model_avg['avg_confidence'], color='gold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Average Detection Confidence')
        ax2.set_ylim([0, 1])
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=150)
        plt.close()
    
    def _plot_efficiency_matrix(self, output_dir):
        """효율성 매트릭스"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 효율성 점수 계산 (속도와 정확도의 조합)
        model_scores = self.results.groupby('model').agg({
            'fps': 'mean',
            'avg_confidence': 'mean',
            'detections': 'mean'
        }).reset_index()
        
        # 정규화
        model_scores['fps_norm'] = (model_scores['fps'] - model_scores['fps'].min()) / \
                                   (model_scores['fps'].max() - model_scores['fps'].min())
        model_scores['conf_norm'] = model_scores['avg_confidence']
        model_scores['efficiency_score'] = (model_scores['fps_norm'] + 
                                           model_scores['conf_norm']) / 2
        
        # 산점도
        scatter = ax.scatter(model_scores['fps'], 
                           model_scores['avg_confidence'],
                           s=model_scores['detections'] * 20,
                           c=model_scores['efficiency_score'],
                           cmap='viridis', alpha=0.6)
        
        # 라벨 추가
        for idx, row in model_scores.iterrows():
            ax.annotate(row['model'], 
                       (row['fps'], row['avg_confidence']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10)
        
        ax.set_xlabel('FPS (Frames Per Second)')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Model Efficiency Matrix\n(Bubble size = Detection Count)')
        
        # 색상 바
        cbar = plt.colorbar(scatter)
        cbar.set_label('Efficiency Score')
        
        # 격자 추가
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/efficiency_matrix.png', dpi=150)
        plt.close()
    
    def _generate_summary_report(self, output_dir):
        """종합 리포트 생성"""
        # 통계 요약
        summary = self.results.groupby('model').agg({
            'fps': ['mean', 'std'],
            'detections': ['mean', 'std'],
            'avg_confidence': ['mean', 'std'],
            'inference_time': ['mean', 'std']
        }).round(3)
        
        # HTML 리포트 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .best {{ background-color: #90EE90; }}
                .worst {{ background-color: #FFB6C1; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>YOLO Model Comparison Report</h1>
            <p>Generated: {pd.Timestamp.now()}</p>
            
            <h2>Summary Statistics</h2>
            {summary.to_html()}
            
            <h2>Speed Comparison</h2>
            <img src="speed_comparison.png" alt="Speed Comparison">
            
            <h2>Accuracy Comparison</h2>
            <img src="accuracy_comparison.png" alt="Accuracy Comparison">
            
            <h2>Efficiency Matrix</h2>
            <img src="efficiency_matrix.png" alt="Efficiency Matrix">
            
            <h2>Recommendations</h2>
            <ul>
                <li><strong>For Real-time Applications:</strong> Use YOLOv11n or YOLOv11s</li>
                <li><strong>For Accuracy:</strong> Use YOLOv11x or YOLOv11l</li>
                <li><strong>For Balance:</strong> Use YOLOv11m</li>
            </ul>
            
            <h2>Detailed Results</h2>
            {self.results.to_html()}
        </body>
        </html>
        """
        
        with open(f'{output_dir}/report.html', 'w') as f:
            f.write(html_content)
        
        # CSV 저장
        self.results.to_csv(f'{output_dir}/detailed_results.csv', index=False)
        
        # JSON 저장
        self.results.to_json(f'{output_dir}/results.json', orient='records', indent=2)
        
        print(f"Summary report saved: {output_dir}/report.html")


def run_full_comparison():
    """전체 비교 실행"""
    print("Starting YOLO Model Comparison Tool")
    print("="*60)
    
    # 비교기 생성
    comparator = ModelComparator()
    
    # 모델 추가
    models_to_test = [
        ('YOLOv11n', 'yolo11n.pt'),
        ('YOLOv11s', 'yolo11s.pt'),
        ('YOLOv11m', 'yolo11m.pt'),
        ('YOLOv11l', 'yolo11l.pt'),
        ('YOLOv11x', 'yolo11x.pt'),
    ]
    
    for name, path in models_to_test:
        try:
            comparator.add_model(name, path)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    if not comparator.models:
        print("No models loaded. Exiting.")
        return
    
    # 테스트 이미지 준비
    comparator.prepare_test_suite()
    
    # 종합 테스트 실행
    results_df = comparator.run_comprehensive_test(conf_threshold=0.5)
    
    # 결과 출력
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    print(results_df.groupby('model')[['fps', 'detections', 'avg_confidence']].mean())
    
    # 리포트 생성
    comparator.generate_report()
    
    # 최적 모델 추천
    print("\n" + "="*60)
    print("Model Recommendations")
    print("="*60)
    
    # 가장 빠른 모델
    fastest = results_df.groupby('model')['fps'].mean().idxmax()
    print(f"✓ Fastest Model: {fastest}")
    
    # 가장 정확한 모델
    most_accurate = results_df.groupby('model')['avg_confidence'].mean().idxmax()
    print(f"✓ Most Accurate: {most_accurate}")
    
    # 가장 많이 검출하는 모델
    most_detections = results_df.groupby('model')['detections'].mean().idxmax()
    print(f"✓ Most Detections: {most_detections}")
    
    print("\nComparison complete! Check 'comparison_report' folder for detailed results.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Model Comparison Tool')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick comparison with fewer models')
    parser.add_argument('--custom', nargs='+',
                       help='Custom model paths to compare')
    
    args = parser.parse_args()
    
    if args.custom:
        # 커스텀 모델 비교
        comparator = ModelComparator()
        for i, path in enumerate(args.custom):
            comparator.add_model(f"Model_{i+1}", path)
        comparator.prepare_test_suite()
        comparator.run_comprehensive_test()
        comparator.generate_report()
    else:
        # 전체 비교 실행
        run_full_comparison()