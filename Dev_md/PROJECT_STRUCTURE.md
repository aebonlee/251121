# 📁 YOLO11 프로젝트 구조 및 파일 설명

## 🗂️ 전체 프로젝트 구조

```
yolo11_detector/
│
├── 📂 first/                          # Phase 1: 기본 검출기
│   ├── yolo_detector.py               # 기본 YOLO 검출 프로그램
│   ├── demo.py                        # 간단한 데모 스크립트
│   ├── test_detector.py               # 테스트 스크립트
│   ├── requirements.txt               # 기본 패키지 목록
│   └── yolo_detector_tutorial.ipynb   # 기초 튜토리얼 노트북
│
├── 📂 Dev_md/                          # 개발 문서
│   ├── DEVELOPMENT_LOG.md             # 상세 개발일지
│   ├── PROJECT_STRUCTURE.md          # 프로젝트 구조 설명 (현재 파일)
│   ├── README_backup.md               # 원본 README 백업
│   └── README_ADVANCED.md            # 고급 기능 문서 백업
│
├── 📂 test_images/                     # 테스트 이미지 (자동 생성)
│   ├── bus.jpg                        # 샘플 이미지 1
│   └── zidane.jpg                     # 샘플 이미지 2
│
├── 📂 detection_results/               # 검출 결과 저장 (자동 생성)
│   └── *.json                         # 검출 결과 JSON 파일들
│
├── 📂 comparison_report/               # 성능 비교 리포트 (자동 생성)
│   ├── speed_comparison.png           # 속도 비교 차트
│   ├── accuracy_comparison.png        # 정확도 비교 차트
│   ├── efficiency_matrix.png          # 효율성 매트릭스
│   ├── report.html                    # HTML 리포트
│   └── results.json                   # 상세 결과 데이터
│
├── 🐍 advanced_detector.py            # 고급 검출기 (앙상블, 세그멘테이션)
├── 🐍 domain_specific_detector.py     # 도메인 특화 검출기
├── 🐍 test_and_compare.py            # 모델 성능 비교 도구
├── 📓 advanced_yolo_tutorial.ipynb    # 고급 기능 상세 튜토리얼
├── 📄 requirements.txt                 # 전체 프로젝트 패키지 목록
├── 📄 README.md                        # 메인 프로젝트 문서
├── 📄 README_ADVANCED.md              # 고급 기능 설명서
└── 📄 .gitignore                       # Git 제외 파일 설정
```

## 📝 주요 파일 상세 설명

### 1️⃣ Phase 1: 기본 검출기 (`first/` 폴더)

#### `yolo_detector.py`
- **목적**: YOLO11 기본 객체 검출
- **주요 클래스**: `YOLODetector`
- **기능**:
  - 3가지 도형으로 라벨링 (사각형, 원, 다각형)
  - 자동 도형 선택 모드
  - 클래스별 색상 자동 할당

#### `test_detector.py`
- **목적**: 검출기 테스트 및 검증
- **기능**:
  - 샘플 이미지 자동 다운로드
  - 4가지 모드 테스트 (auto, rectangle, circle, polygon)

### 2️⃣ Phase 2: 고급 검출기

#### `advanced_detector.py`
- **목적**: 정확도 향상을 위한 고급 기법
- **주요 클래스**: `AdvancedYOLODetector`
- **핵심 기능**:
  ```python
  # 앙상블 검출
  ensemble_models = [YOLO('yolo11l.pt'), YOLO('yolo11m.pt')]
  
  # 세그멘테이션
  seg_model = YOLO('yolo11x-seg.pt')
  ```
- **메서드**:
  - `ensemble_detect()`: 다중 모델 앙상블
  - `detect_with_segmentation()`: 픽셀 단위 검출
  - `compare_models()`: 모델 성능 비교

#### `domain_specific_detector.py`
- **목적**: 특정 분야 최적화
- **주요 클래스**: `DomainSpecificDetector`
- **지원 도메인** (7가지):
  ```python
  DOMAINS = {
      'traffic': [...],     # 교통 모니터링
      'retail': [...],      # 리테일 분석
      'security': [...],    # 보안 감시
      'wildlife': [...],    # 야생동물 관찰
      'kitchen': [...],     # 주방 환경
      'office': [...],      # 사무실 분석
      'sports': [...]       # 스포츠 분석
  }
  ```
- **특수 기능**:
  - DBSCAN 클러스터링
  - 실시간 알람 시스템
  - 비디오 스트림 처리

#### `test_and_compare.py`
- **목적**: 체계적인 성능 벤치마킹
- **주요 클래스**: `ModelComparator`
- **측정 메트릭**:
  - FPS (Frames Per Second)
  - 추론 시간
  - 검출 정확도
  - 효율성 점수
- **출력**:
  - HTML 리포트
  - 비교 차트
  - CSV/JSON 데이터

### 3️⃣ 학습 자료

#### `advanced_yolo_tutorial.ipynb`
- **목적**: 상세한 학습 가이드
- **구성**: 8개 파트
  1. 환경 설정
  2. YOLO11 기본 이해
  3. 기본 객체 검출
  4. 앙상블 기법
  5. 세그멘테이션
  6. 도메인 특화 검출
  7. 성능 비교
  8. 통합 시스템
- **특징**:
  - 각 코드 블록마다 상세 주석
  - 실행 가능한 예제
  - 시각화 포함

## 🔧 설정 파일

### `requirements.txt`
```txt
ultralytics>=8.3.0    # YOLO11
torch>=2.0.0          # PyTorch
opencv-python>=4.8.0  # OpenCV
numpy>=1.24.0         # NumPy
matplotlib>=3.6.0     # 시각화
scikit-learn>=1.3.0   # ML 도구
pandas>=2.0.0         # 데이터 분석
scipy>=1.10.0         # 과학 계산
seaborn>=0.12.0       # 고급 시각화
tqdm>=4.65.0          # 진행 표시
```

### `.gitignore`
- Python 캐시 파일
- YOLO 모델 파일 (*.pt)
- 출력 이미지
- IDE 설정 파일
- Jupyter 체크포인트

## 🚀 실행 순서

### 초급자 경로
1. `first/yolo_detector_tutorial.ipynb` - 기초 학습
2. `first/demo.py` - 간단한 실습
3. `first/test_detector.py` - 테스트

### 중급자 경로
1. `advanced_yolo_tutorial.ipynb` - 고급 기법 학습
2. `advanced_detector.py` - 앙상블/세그멘테이션
3. `test_and_compare.py` - 성능 비교

### 고급자 경로
1. `domain_specific_detector.py` - 도메인 특화
2. 커스텀 도메인 추가
3. 실시간 비디오 처리

## 💡 사용 팁

### 모델 선택 가이드
| 용도 | 추천 모델 | 설정 |
|------|-----------|------|
| 실시간 처리 | yolo11n.pt | conf=0.5 |
| 균형 | yolo11m.pt | conf=0.45 |
| 높은 정확도 | yolo11x.pt | conf=0.4 |
| 최고 정확도 | 앙상블 | conf=0.35 |

### 도메인 선택
```bash
# 교통 모니터링
python domain_specific_detector.py -i traffic.jpg -d traffic

# 보안 감시
python domain_specific_detector.py -i security.jpg -d security

# 리테일 분석
python domain_specific_detector.py -i store.jpg -d retail
```

## 📊 성능 지표

### 모델별 성능 (RTX 3060 기준)
| 모델 | FPS | mAP | 파라미터 |
|------|-----|-----|----------|
| YOLOv11n | 100+ | 37.3 | 3.2M |
| YOLOv11s | 80+ | 44.9 | 11.2M |
| YOLOv11m | 50+ | 50.2 | 25.9M |
| YOLOv11l | 30+ | 52.9 | 43.7M |
| YOLOv11x | 20+ | 54.7 | 68.2M |

## 🔍 문제 해결

### 자주 발생하는 문제

1. **CUDA 메모리 부족**
   ```python
   # 해결책: 작은 모델 사용
   model = YOLO('yolo11n.pt')
   ```

2. **느린 추론 속도**
   ```python
   # 해결책: GPU 확인
   torch.cuda.is_available()
   ```

3. **모델 다운로드 실패**
   ```bash
   # 수동 다운로드
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
   ```

## 📚 추가 자료

- [Ultralytics Docs](https://docs.ultralytics.com/)
- [YOLO11 Paper](https://arxiv.org/abs/yolo11)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

**Last Updated**: 2024.11.21  
**Author**: aebonlee  
**License**: MIT