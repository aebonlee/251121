# YOLO11 객체 검출 및 라벨링 프로그램

YOLO11(Ultralytics)을 사용하여 이미지에서 객체를 검출하고, 각 객체에 대해 다양한 도형(사각형, 원, 다각형)으로 라벨링하는 파이썬 프로그램입니다.

## 주요 기능

- **YOLO11 기반 객체 검출**: 최신 YOLO 모델을 사용한 정확한 객체 검출
- **다양한 라벨링 도형 지원**:
  - 사각형 (Rectangle)
  - 원 (Circle)  
  - 다각형 (8각형 Polygon)
  - 자동 선택 모드 (각 객체마다 다른 도형 자동 할당)
- **신뢰도 표시**: 각 검출된 객체의 신뢰도 점수 표시
- **색상 자동 할당**: 각 클래스별로 다른 색상 자동 적용

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/aebonlee/251121.git
cd 251121
```

### 2. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 사용법
```bash
python yolo_detector.py -i [이미지파일경로]
```

### 옵션 설명
- `-i, --image`: 입력 이미지 파일 경로 (필수)
- `-o, --output`: 출력 이미지 파일 경로 (선택, 기본값: input_labeled.jpg)
- `-m, --model`: YOLO 모델 파일 경로 (선택, 기본값: yolo11n.pt)
- `-s, --shape`: 라벨링 도형 타입 (선택)
  - `rectangle`: 모든 객체를 사각형으로 표시
  - `circle`: 모든 객체를 원으로 표시
  - `polygon`: 모든 객체를 8각형으로 표시
  - `auto`: 객체마다 다른 도형 자동 선택 (기본값)

### 사용 예제

#### 1. 자동 도형 선택 (각 객체마다 다른 도형)
```bash
python yolo_detector.py -i sample.jpg
```

#### 2. 모든 객체를 사각형으로 표시
```bash
python yolo_detector.py -i sample.jpg -s rectangle
```

#### 3. 출력 파일명 지정
```bash
python yolo_detector.py -i sample.jpg -o result.jpg -s auto
```

### 테스트 실행
```bash
python test_detector.py
```
테스트 스크립트는 샘플 이미지를 다운로드하고 모든 도형 타입으로 테스트를 수행합니다.

## 프로젝트 구조
```
yolo11_detector/
├── yolo_detector.py      # 메인 검출 프로그램
├── demo.py               # 간단한 데모 스크립트
├── test_detector.py      # 테스트 스크립트
├── requirements.txt      # 필요 패키지 목록
└── README.md            # 프로젝트 문서
```

## 지원 모델

기본적으로 YOLO11n(nano) 모델을 사용하며, 다음 모델들도 지원합니다:
- yolo11n.pt (Nano - 가장 빠름)
- yolo11s.pt (Small)
- yolo11m.pt (Medium)
- yolo11l.pt (Large)
- yolo11x.pt (Extra Large - 가장 정확함)

## 검출 가능한 객체 클래스

COCO 데이터셋 기반 80가지 객체 클래스:
- 사람, 자전거, 자동차, 오토바이, 비행기, 버스, 기차, 트럭
- 보트, 신호등, 소화전, 정지 표지판, 주차 미터기, 벤치
- 새, 고양이, 개, 말, 양, 소, 코끼리, 곰, 얼룩말, 기린
- 배낭, 우산, 핸드백, 넥타이, 가방, 프리스비, 스키, 스노보드
- 스포츠 공, 연, 야구 방망이, 야구 글러브, 스케이트보드
- 서핑보드, 테니스 라켓, 병, 와인잔, 컵, 포크, 나이프
- 숟가락, 그릇, 바나나, 사과, 샌드위치, 오렌지, 브로콜리
- 당근, 핫도그, 피자, 도넛, 케이크, 의자, 소파, 화분
- 침대, 식탁, 화장실, TV, 노트북, 마우스, 리모컨, 키보드
- 휴대폰, 전자레인지, 오븐, 토스터, 싱크대, 냉장고
- 책, 시계, 꽃병, 가위, 테디베어, 헤어드라이어, 칫솔

## 출력 예시

프로그램 실행 시 다음과 같은 정보가 출력됩니다:
```
라벨링된 이미지가 저장되었습니다: output.jpg
검출된 객체 수: 5
  - person: 95.3% 신뢰도
  - car: 89.7% 신뢰도
  - bus: 92.1% 신뢰도
```

## 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (선택사항, CPU에서도 동작 가능)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 문의사항

Issues 탭에서 문제를 보고하거나 기능을 제안해주세요.
