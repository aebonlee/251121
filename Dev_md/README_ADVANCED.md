# 고급 YOLO11 객체 검출 시스템

기존 YOLO11 검출기보다 더 정확하고 강력한 객체 검출 시스템입니다.

## 🚀 주요 특징

### 1. **Advanced Detector** (`advanced_detector.py`)
- **다중 모델 앙상블**: YOLO11x, YOLO11l, YOLO11m 모델을 결합하여 정확도 향상
- **세그멘테이션 지원**: 객체의 정확한 윤곽 검출
- **고급 NMS**: 더 정교한 Non-Maximum Suppression
- **모델 비교 기능**: 5가지 모델 (nano ~ xlarge) 성능 비교

### 2. **Domain-Specific Detector** (`domain_specific_detector.py`)
- **7가지 도메인 특화 검출**:
  - 🚗 **Traffic**: 교통 모니터링 (차량, 보행자, 신호등)
  - 🛒 **Retail**: 리테일 분석 (고객, 제품, 행동 패턴)
  - 🔒 **Security**: 보안 감시 (의심 객체, 침입 감지)
  - 🦁 **Wildlife**: 야생동물 모니터링
  - 🍳 **Kitchen**: 주방 객체 검출
  - 💼 **Office**: 사무실 환경 분석
  - ⚽ **Sports**: 스포츠 분석

- **실시간 알람 시스템**: 도메인별 위험 상황 감지
- **클러스터링 분석**: 객체 밀집도 및 패턴 분석
- **비디오 스트림 처리**: 실시간 비디오 분석

### 3. **Test & Compare Tool** (`test_and_compare.py`)
- **성능 벤치마킹**: FPS, 정확도, 검출 수 비교
- **시각화 리포트**: HTML 리포트 자동 생성
- **효율성 매트릭스**: 속도 vs 정확도 분석
- **모델 추천**: 용도별 최적 모델 제안

## 📋 요구사항

```bash
pip install -r requirements.txt
```

필수 패키지:
- ultralytics>=8.3.0
- torch>=2.0.0
- opencv-python>=4.8.0
- scikit-learn>=1.3.0
- scipy>=1.10.0
- pandas>=2.0.0
- seaborn>=0.12.0

## 🎯 사용 방법

### 1. 고급 검출 (Advanced Detection)

#### 단일 모델 검출
```bash
python advanced_detector.py -i image.jpg -m single -c 0.5
```

#### 앙상블 검출 (더 정확함)
```bash
python advanced_detector.py -i image.jpg -m ensemble -c 0.5
```

#### 세그멘테이션 포함
```bash
python advanced_detector.py -i image.jpg -m segmentation --segmentation
```

#### 모델 비교
```bash
python advanced_detector.py -i image.jpg --compare
```

### 2. 도메인 특화 검출

#### 교통 모니터링
```bash
python domain_specific_detector.py -i traffic.jpg -d traffic
```

#### 보안 감시
```bash
python domain_specific_detector.py -i security_cam.jpg -d security
```

#### 실시간 비디오 처리
```bash
# 웹캠
python domain_specific_detector.py -v 0 -d security

# 비디오 파일
python domain_specific_detector.py -v video.mp4 -d traffic -o output.mp4
```

### 3. 모델 성능 비교

#### 전체 모델 비교
```bash
python test_and_compare.py
```

#### 커스텀 모델 비교
```bash
python test_and_compare.py --custom model1.pt model2.pt model3.pt
```

## 📊 성능 비교

| 모델 | FPS | 정확도 | 파라미터 | 용도 |
|------|-----|--------|----------|------|
| YOLOv11n | ~100 | 중간 | 3.2M | 실시간 처리 |
| YOLOv11s | ~80 | 중상 | 11.2M | 속도-정확도 균형 |
| YOLOv11m | ~50 | 높음 | 25.9M | 일반 용도 |
| YOLOv11l | ~30 | 매우 높음 | 43.7M | 높은 정확도 필요 |
| YOLOv11x | ~20 | 최고 | 68.2M | 최고 정확도 |

## 🔧 고급 기능

### 앙상블 검출
- 여러 모델의 예측을 결합하여 정확도 향상
- Weighted voting 방식 사용
- 오탐지 감소

### 도메인 특화 후처리
- 도메인별 중요 객체 우선 검출
- 컨텍스트 기반 필터링
- 행동 패턴 분석

### 실시간 알람
- 과밀 감지
- 의심 행동 탐지
- 위험 상황 알림

## 📈 출력 예시

### Advanced Detection
```
Processing: sample.jpg
Mode: ensemble, Confidence: 0.5

==================================================
Detection Statistics
==================================================
Total objects detected: 15

Class-wise detection:
  person         :   5 objects (conf: min=0.523, avg=0.742, max=0.912)
  car            :   3 objects (conf: min=0.612, avg=0.823, max=0.945)
  bus            :   1 objects (conf: min=0.887, avg=0.887, max=0.887)

Overall confidence: 0.784 (±0.142)
```

### Domain-Specific Detection (Traffic)
```
Domain: TRAFFIC
Total Objects: 12
Vehicles: 7
Pedestrians: 5

⚠️ ALERTS:
  • [HIGH] High pedestrian density: 5 people
  • [MEDIUM] Vehicle near pedestrian crossing
```

## 🎨 시각화 기능

- **바운딩 박스**: 색상으로 클래스 구분
- **세그멘테이션 마스크**: 객체 윤곽 표시
- **클러스터 표시**: 밀집 영역 하이라이트
- **신뢰도 히트맵**: 검출 신뢰도 시각화
- **분석 대시보드**: 실시간 통계 표시

## 📁 프로젝트 구조

```
yolo11_detector/
├── first/                      # 기본 검출기
│   ├── yolo_detector.py
│   └── requirements.txt
├── advanced_detector.py        # 고급 검출기
├── domain_specific_detector.py # 도메인 특화 검출
├── test_and_compare.py        # 성능 비교 도구
├── requirements.txt            # 고급 기능 패키지
└── README_ADVANCED.md         # 고급 기능 문서
```

## 💡 사용 팁

1. **정확도 우선**: `ensemble` 모드 + YOLOv11x 사용
2. **속도 우선**: `single` 모드 + YOLOv11n 사용
3. **균형**: `single` 모드 + YOLOv11m 사용
4. **특정 용도**: 해당 도메인 검출기 사용

## 🔍 문제 해결

### CUDA 메모리 부족
- 더 작은 모델 사용 (nano, small)
- 배치 크기 감소
- CPU 모드 사용: `--device cpu`

### 느린 추론 속도
- GPU 사용 확인
- 이미지 크기 축소
- 더 작은 모델 사용

## 📝 라이선스

MIT License

## 🤝 기여

Issues와 Pull Request는 언제나 환영합니다!