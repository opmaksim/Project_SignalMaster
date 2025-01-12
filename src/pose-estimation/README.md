## **구성 파일 설명**

### 1. **`train-yolo11n-pose.py`**: YOLO-Pose 모델 학습
이 스크립트는 **YOLO11n-Pose 모델**을 데이터셋을 사용해 학습합니다.

#### 주요 기능:
- **모델 로드**: `yolo11n-pose.pt` 사전 학습된 YOLO-Pose 모델 사용.
- **학습 파라미터**:
  - `epochs=50`: 50번의 에포크 동안 학습.
  - `batch=4`: 배치 크기 4.
  - `imgsz=(640, 480)`: 입력 이미지 크기를 640x480으로 고정.
- **데이터셋**: `data.yaml` 파일에서 데이터셋 경로와 클래스 정보를 로드.

#### 실행 예제:
```bash
python train-yolo11n-pose.py
```

---

### 2. **`quantization-fp16.py`**: 모델 양자화
**FP16 양자화**를 통해 모델 추론 속도를 최적화합니다.

#### 주요 기능:
- **모델 로드**: 학습된 모델(`best.pt`)을 로드.
- **TensorRT 엔진 변환**:
  - `format="engine"`: TensorRT 엔진 형식으로 변환.
  - `half=True`: FP16 양자화를 적용.
  - `workspace=1024`: 작업 공간 크기를 1024MB로 설정.
- **이미지 크기**: 640x480 해상도로 고정.

#### 실행 예제:
```bash
python quantization-fp16.py
```

#### 출력 파일:
- `best.engine`: Jetson Nano에서 최적화된 TensorRT 엔진 파일.

---

### 3. **`run.py`**: 실시간 추론 및 RC카 제어
웹캠을 통해 YOLO-Pose 모델로 실시간 추론을 수행하며, 추론 결과를 Arduino에 전달하여 RC카를 제어합니다.

#### 주요 기능:
- **클래스 라벨**:
  - `GO`, `LEFT`, `RIGHT`, `STOP`, `SLOW` 5가지 신호를 정의.
- **모델 로드**:
  - 양자화된 TensorRT 엔진(`best.engine`) 로드.
- **Arduino와 통신**:
  - UART를 사용하여 `/dev/ttyACM0` 포트로 신호 전송.
  - `baudrate=9600` 설정으로 안정적인 데이터 전송.
- **OpenCV를 통한 웹캠 사용**:
  - 실시간으로 프레임을 읽고, YOLO-Pose 모델로 추론 수행.
  - 클래스 ID에 따라 명령을 설정하고 Arduino로 전송.

#### 스레드 사용:
- **비동기 통신**:
  - Arduino와의 통신은 스레드로 실행되어 메인 루프의 효율성을 높임.

#### 실행 예제:
```bash
python run.py
```

#### 출력 예제:
웹캠에서 포착된 데이터가 추론되어 Arduino에 다음과 같은 명령이 전송됩니다:
```text
GO
LEFT
RIGHT
STOP
SLOW
```


## **시스템 구조**

### 데이터 흐름:
1. **YOLO-Pose 학습**:
   - `train-yolo11n-pose.py`로 모델을 학습.
   - 학습된 모델(`best.pt`)을 TensorRT 엔진으로 변환.
2. **실시간 추론 및 제어**:
   - 웹캠을 통해 입력 프레임을 실시간 추론.
   - 추론 결과를 UART를 통해 Arduino로 전송.
3. **Arduino**:
   - UART로 수신된 명령에 따라 RC카의 모터를 제어.

### 파일 관계:
```plaintext
train-yolo11n-pose.py  -->  best.pt
quantization-fp16.py   -->  best.engine
run.py                 -->  실시간 제어