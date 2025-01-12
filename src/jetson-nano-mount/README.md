# Signal-Master 프로젝트 🔪 젯슨 나노

Updated at 📅 2025-01-16 01:01:57

- [Signal-Master 프로젝트 🔪 젯슨 나노](#signal-master-프로젝트--젯슨-나노)
  - [🔰 사용 지침](#-사용-지침)
  - [📁 디렉터리 구조](#-디렉터리-구조)
  - [➡️ How-to](#️-how-to)
  - [➡️ Issues](#️-issues)
  - [📁 Environment Files](#-environment-files)
    - [🗄️ Dockerfile](#️-dockerfile)
    - [🗄️ devcontainer.json](#️-devcontainerjson)
    - [🗄️ settings.json](#️-settingsjson)
    - [🗄️ service.sh](#️-servicesh)
  - [📁 Python Files](#-python-files)
    - [🪶 Script 🔪 **pyskl\_skeleton\_dataset\_downloader\_and\_keypoint\_checker.py**](#-script--pyskl_skeleton_dataset_downloader_and_keypoint_checkerpy)
    - [🗄️ **config/project\_config.py**](#️-configproject_configpy)
    - [🗄️ **validation/cuda.py**](#️-validationcudapy)
  - [♾️ **Model Lifecycle Pipeline**](#️-model-lifecycle-pipeline)
    - [🪶 Script 🔪 **labeling.py**](#-script--labelingpy)
    - [🪶 Script 🔪 **train-yolo11n-pose.py**](#-script--train-yolo11n-posepy)
    - [🪶 Script 🔪 **quantization-fp16.py**](#-script--quantization-fp16py)
    - [🪶 Script 🔪 **run.py**](#-script--runpy)
  - [💱 **교통 수신호 행동 인식 모델 방향성 설정**](#-교통-수신호-행동-인식-모델-방향성-설정)
    - [➡️ 결론](#️-결론)
    - [분석 배경](#분석-배경)
    - [RGB 기반 모델](#rgb-기반-모델)
    - [Optical Flow 기반 모델](#optical-flow-기반-모델)
    - [Skeleton 기반 모델](#skeleton-기반-모델)
      - [❗ GCN 기반 모델 사용 시 도전 과제](#-gcn-기반-모델-사용-시-도전-과제)
    - [선택한 모델](#선택한-모델)
      - [Pose3D.pdf](#pose3dpdf)
      - [X3D.pdf](#x3dpdf)
      - [PYSKL.pdf](#pysklpdf)
    - [구현 및 테스트](#구현-및-테스트)
    - [📁 (행동 인식 모델) 디렉터리 구조](#-행동-인식-모델-디렉터리-구조)

## 🔰 사용 지침

1. 레포지토리를 클론하고 프로젝트 디렉토리로 이동:

   ```bash
   #!/bin/bash
   git clone https://github.com/opmaksim/Project_SignalMaster.git
   cd jetson-nano-mount
   ```

2. 개발 컨테이너를 빌드 및 시작:

   - 1️⃣

     ```bash
     #!/bin/bash
     code .
     ```

   - 2️⃣

     %VSCode (F1 Key)> Dev Containers: Reopen in Container

3. 젯슨 나노 호스트에서, 서비스 초기화 및 관리:

   ```bash
   #!/bin/bash
   ./service.sh init
   ```

## 📁 디렉터리 구조

tree **jetson-nano-mount/**  
├── 📂 .devcontainer  
├── 📂 .vscode  
├── 📂 howto  
├── 📂 issues  
├── 📂 requirements  
├── 📂 resource  
├── 📂 scripts  
├── [service.sh](service.sh)  
├── [run.sh](run.py)  
└── 📂 src  
&nbsp;&nbsp;&nbsp;&nbsp;└── 📂 signal_masters  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 📂 config  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 📂 ml  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── 📂 models  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 📂 tests  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── 📂 validation

## ➡️ How-to

- [Intellisense for Ultralytics package](howto/intellisense_for_ultralytics_package.md)

## ➡️ Issues

- [Python - Issues](issues/python-Issues.md)

## 📁 Environment Files

### 🗄️ [Dockerfile](.devcontainer/Dockerfile)

Jetson Nano의 CUDA 및 TensorRT 환경에 최적화된 Docker 이미지를 빌드.

- **주요 내용**:

  - 베이스 이미지: `ultralytics/ultralytics:latest-jetson-jetpack4`.

    **YOLOv8+ 모델을 사용할 수 있는 호환 이미지 사용.** (Python version)

  - GPU 및 디바이스 접근을 위한 사용자 권한 구성.
  - Jetson Nano의 CUDA 버전(10.2) 및 cuDNN(8.2.1)과 호환 보장.

    - CUDA 버전: `10.2.300`.
    - cuDNN 버전: `8.2.1`.
    - JetPack: `4.6.5` (L4T 버전: 32.7.5).

&nbsp;

### 🗄️ [devcontainer.json](.devcontainer/devcontainer.json)

개발 컨테이너 환경을 정의하며, Docker 런타임 설정 및 확장을 포함.

- **주요 내용**:

  - NVIDIA 런타임(`--runtime nvidia`)으로 GPU 접근 지원 **(컨테이너에 GPU 장치 권한 허용)**.
  - 비디오 입력 및 시리얼 통신을 위한 디바이스 마운트 **(컨테이너에 장치 권한 허용)**

    - `/dev/video0`
    - `/dev/ttyACM0`, `/dev/ttyACM1`, `/dev/ttyACM2`.

  - `postCreateCommand`를 통해 개발에 필요한 추가 Python 의존성 설치.

  - 컨테이너에서 개발 중에 GUI 화면을 호스트에 표시하기 위한 X11 포워딩 설정

    - `--volume /tmp/.X11-unix:/tmp/.X11-unix`.
    - `--volume /home/jsnano/.Xauthority:/tmp/.Xauthority:rw`.

  - IPC 및 네트워크를 호스트와 공유하여 성능 및 프로세스 간 통신 최적화.

---

&nbsp;

### 🗄️ [settings.json](.vscode/settings.json)

Python 개발을 위한 VSCode 워크스페이스 설정을 구성.

- **주요 내용**:
  - Python 포매터: `black`.
  - 저장 시 자동 포맷팅 활성화.
  - YOLO 관련 경로(`/ultralytics`) 추가로 IntelliSense 지원 강화.

&nbsp;

### 🗄️ [service.sh](service.sh)

시스템 부팅 시 YOLO 애플리케이션이 Docker 컨테이너에서 자동으로 시작되도록 서비스 생성.

- **주요 내용**:

  - `signal-masters.service`를 systemd에 정의.
  - Docker 컨테이너를 자동으로 재시작하고 YOLO 애플리케이션 스크립트(`run.py`) 실행.
  - 서비스 관리 명령어 제공 (`init`, `start`, `stop`, `status`, `enable`, `disable`).

- ❔ 명령어

  | 명령어    | 설명                               |
  | --------- | ---------------------------------- |
  | `init`    | 서비스를 생성, 활성화, 시작.       |
  | `start`   | 서비스를 즉시 시작.                |
  | `stop`    | 서비스를 중지.                     |
  | `enable`  | 부팅 시 서비스 자동 시작 활성화.   |
  | `disable` | 부팅 시 서비스 자동 시작 비활성화. |
  | `status`  | 서비스 상태 확인.                  |

  **사용 예시**

  ```bash
  #!/bin/bash
  ./service.sh init       # 서비스를 생성하고 시작
  ./service.sh start      # 서비스를 시작
  ./service.sh stop       # 서비스를 중지
  ./service.sh status     # 서비스 상태 확인
  ```

&nbsp;

---

## 📁 Python Files

### 🪶 Script 🔪 [**pyskl_skeleton_dataset_downloader_and_keypoint_checker.py**](scripts/pyskl_skeleton_dataset_downloader_and_keypoint_checker.py)

---

❔ **주요 기능**

- 동작인식 모델을 위한 데이터셋 다운로드, 키포인트 형상 검증, 데이터셋 관리.

  ➡️📍 동작인식 모델을 사용하기 위해 COCO 데이터 셋 (키포인트 17개) 을 사용하는 Yolo Pose 를 사용하자!

&nbsp;

❔ **데이터셋 정보**

- 지원되는 데이터셋 목록:

  | 키(key)         | 데이터셋 이름            | 예상 키포인트 수 | 다운로드 URL                                                                                  |
  | --------------- | ------------------------ | ---------------- | --------------------------------------------------------------------------------------------- |
  | `ntu60_hrnet`   | NTU60 HRNet 2D Skeleton  | 17               | [다운로드 링크](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl)   |
  | `ntu60_3danno`  | NTU60 3D Annotation      | 25               | [다운로드 링크](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl)  |
  | `ntu120_hrnet`  | NTU120 HRNet 2D Skeleton | 17               | [다운로드 링크](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl)  |
  | `ntu120_3danno` | NTU120 3D Annotation     | 25               | [다운로드 링크](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl) |
  | `gym_hrnet`     | GYM HRNet 2D Skeleton    | 17               | [다운로드 링크](https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl)         |
  | `ucf101_hrnet`  | UCF101 HRNet 2D Skeleton | 17               | [다운로드 링크](https://download.openmmlab.com/mmaction/pyskl/data/ucf101/ucf101_hrnet.pkl)   |
  | `hmdb51_hrnet`  | HMDB51 HRNet 2D Skeleton | 17               | [다운로드 링크](https://download.openmmlab.com/mmaction/pyskl/data/hmdb51/hmdb51_hrnet.pkl)   |

&nbsp;

❔ **함수 설명**

- `download_if_not_exists(dataset_info: DatasetInfo)`

  데이터셋이 존재하지 않을 경우 다운로드.

  - **매개변수**: `dataset_info` (DatasetInfo) - 데이터셋 정보 (이름, 상대 경로, 다운로드 URL 포함).
  - **출력**: 다운로드 상태 메시지.

- `check_keypoints_shape(dataset_info: DatasetInfo)`

  데이터셋의 키포인트 형상이 예상값과 일치하는지 확인.

  - **매개변수**: `dataset_info` (DatasetInfo) - 데이터셋 정보 (이름, 예상 키포인트 수 포함).
  - **출력**: 형상이 맞으면 성공 메시지, 불일치 시 오류 발생.

- `process_datasets(datasets: dict[str, DatasetInfo], dataset_keys: str | list[str])`

  선택된 데이터셋 처리.

  - 데이터셋 다운로드 및 키포인트 형상 검증 수행.
  - **매개변수**:
    - `datasets` (dict) - 데이터셋 정보를 포함하는 딕셔너리.
    - `dataset_keys` (str 또는 li st) - 처리할 데이터셋 키 또는 `*` (모든 데이터셋 처리).

&nbsp;

❔ **사용 예시**

- **모든 데이터셋 처리**

  ```python
  process_datasets(datasets=datasets, dataset_keys="*")
  ```

- **특정 데이터셋 처리**

  ```python
  process_datasets(datasets=datasets, dataset_keys=["ntu60_hrnet", "gym_hrnet"])
  ```

&nbsp;

❔ **참고사항**

- **데이터셋 경로**:

  - 기본 경로: `external/pyskl/data`.

---

&nbsp;

### 🗄️ [**config/project_config.py**](src/signal_masters/config/project_config.py)

❔ **주요 기능**

- 프로젝트 디렉토리를 동적으로 초기화하고, 필수 디렉토리를 관리하는 클래스 제공.

&nbsp;

❔ **주요 함수 설명**

- **`ProjectConfig` 클래스**

  프로젝트 디렉토리 구조를 관리하는 정적 클래스.

  - **프로젝트 루트 경로 초기화**

    - 실행 환경에 따라 프로젝트 루트 디렉토리를 동적으로 설정.

  - **디렉토리 생성**

    - `ml` (machine learning), `models`, `resource`와 같은 주요 프로젝트 디렉토리를 자동 생성.

  - **클래스 사용 방법**
    - `ProjectConfig.init()` 메서드 호출 후 경로 변수에 접근 가능.
    - 주요 경로 정보:
      - ProjectConfig.project_root
      - ProjectConfig.ml_dir
      - ProjectConfig.ai_models_dir
      - ProjectConfig.resource_dir

  **사용 예시**:

  ```python
  from signal_masters.config.project_config import ProjectConfig

  # 프로젝트 설정 초기화
  ProjectConfig.init()

  # 프로젝트 디렉토리 경로 확인
  print(ProjectConfig.ml_dir)
  ```

&nbsp;

❔ **주요 메서드 설명**

- **`init_project_root()`**

  - 프로젝트 루트 경로 초기화.

- **`init_directories()`**

  - 프로젝트에 필요한 주요 디렉토리 자동 생성.

- **`init()`**
  - 모든 초기화 작업 통합 수행.

&nbsp;

❔ **참고사항**

- 프로젝트 디렉토리 내 `workspaces` 디렉토리 존재 필요.
  해당 조건은 VSCode의 도커 환경(DevContainer)에서 사용하면 기본적으로 충족됨.
- `init()` 메서드 호출 없이 경로 변수 접근 불가.

&nbsp;

---

### 🗄️ [**validation/cuda.py**](src/signal_masters/validation/cuda.py)

❔ **주요 기능**

- 딥러닝 환경에서 CUDA와 YOLO 모델의 상태 점검, OpenCV CUDA 지원 여부 확인.

&nbsp;

❔ **주요 함수 설명**

- **`check_cuda_and_yolo_inference()`**

  PyTorch 및 CUDA 상태 점검과 YOLO 모델을 활용한 샘플 이미지 추론 수행.

  - **PyTorch 및 CUDA 상태 점검**

    - PyTorch 버전 출력.
    - CUDA 활성화 여부 확인.
    - cuDNN 버전 출력 (CUDA 활성화 시).
    - CUDA 디바이스에서 실행 중인 프로세스 정보 출력.

  - **YOLO 모델 추론**
    - 사전 학습된 YOLO 모델(`yolo11n-pose.pt`) 로드.
    - 샘플 이미지(`bus.jpg`)에서 YOLO 추론 수행 및 결과 출력.

  **사용 방법**:

  ```python
  from signal_masters.validation.cuda import check_cuda_and_yolo_inference

  check_cuda_and_yolo_inference()
  ```

&nbsp;

- **`check_opencv_cuda()`**

  OpenCV의 CUDA 지원 여부 및 디바이스 정보 확인.

  - **OpenCV 버전 및 CUDA 상태 점검**

    - OpenCV 버전 출력.
    - CUDA 활성화 여부 확인.

  - **CUDA 디바이스 정보 출력**

    - CUDA 지원 디바이스 수 출력.
    - 각 디바이스의 이름, 메모리 용량 등 정보 표시.

  - **CUDA 비활성화 처리**
    - CUDA가 지원되지 않을 경우 관련 메시지 출력.

  **사용 방법**:

  ```python
  from signal_masters.validation.cuda import check_opencv_cuda

  check_opencv_cuda()
  ```

&nbsp;

---

## ♾️ **Model Lifecycle Pipeline**

### 🪶 Script 🔪 [**labeling.py**](src/signal_masters/ml/labeling.py)

❔ **주요 기능**

- YOLO11n-Pose 모델을 사용해 이미지에서 키포인트를 추출하고 YOLO 형식의 라벨 파일 생성.

  - 이미지 디렉토리에서 `.png` 파일을 읽고, 각 파일에 대해 라벨 파일 생성.
  - 클래스 ID는 파일 이름에 포함된 숫자 범위를 기준으로 매핑.
  - 키포인트를 YOLO 형식으로 변환 후 바운딩 박스와 함께 라벨에 저장.

&nbsp;

---

### 🪶 Script 🔪 [**train-yolo11n-pose.py**](src/signal_masters/ml/train-yolo11n-pose.py)

❔ **주요 기능**

- **YOLO11n-Pose 모델**을 사용해 사용자 정의 데이터셋으로 학습 수행.

&nbsp;

❔ **설정**

- **모델 파일**: `yolo11n-pose.pt`.
- **학습 파라미터**:
  - 데이터셋 설정 파일: `data.yaml`.
  - 학습 에포크: `50`.
  - 배치 크기: `4`.
  - 이미지 크기: `(640, 480)`.
- **출력 파일**: `best.pt`

&nbsp;

---

### 🪶 Script 🔪 [**quantization-fp16.py**](src/signal_masters/ml/quantization-fp16.py)

❔ **주요 기능**

- 학습된 모델 양자화 (모델 추론 속도 최적화)

  Fine-tuning 된 YOLO11n-Pose 모델을 TensorRT 엔진(`.engine`)으로 변환.

&nbsp;

❔ **설정**

- **모델 파일**: `best.pt`.

  **train-yolo11n-pose.py** 로부터 학습된 모델 파일.

- **양자화 형식**: **FP16**.

  - **INT8** 형식은 현재 TensorRT 버전에서는 지원하지 않음.

    🔗 [TensorRT Export Failure: AssertionError with Range Operator (Error Occurred During Model Quantization)](issues/python-Issues.md)

- **이미지 크기**: `(640, 480)`.
- **TensorRT 작업 공간 크기**: `1024MB`.
- **출력 파일**: `best.engine`

  Jetson Nano 디바이스에 최적화된 TensorRT 엔진 파일.

&nbsp;

⚠️ **주의사항**

- Jetson Nano에서 remote-ssh를 통해 VS Code 를 실행한 상태에서, 양자화를 진행할 경우 메모리 부족 문제로 작업이 멈춤.

  이를 방지하기 위해 Jetson Nano 호스트에서 다음 명령어를 사용하여 Docker 컨테이너 실행 후 작업 수행 필요:

  ```bash
  sudo docker run --runtime nvidia -it --rm --network=host --ipc=host --gpus all \
    -v ~/repo/signal-masters:/workspace \
    -w /workspace \
    --device=/dev/video0 \
    signal-masters-image
  ```

&nbsp;

---

### 🪶 Script 🔪 [**run.py**](run.py)

❔ **주요 기능**

- **TensorRT 엔진 파일 로드**:

  - `best.engine` 파일(최적화된 모델 파일)을 사용해 YOLO-Pose 모델을 초기화.

- **웹캠을 통한 실시간 추론**:

  - OpenCV로 웹캠에서 프레임을 실시간으로 읽음.
  - YOLO-Pose 모델로 추론 수행.
  - 추론 결과에서 클래스 ID를 분석하여 명령 생성.

- **Arduino와의 시리얼 통신**:

  - 스레드를 활용해 비동기적으로 명령을 전송.
  - 생성된 명령(예: `GO`, `LEFT`)을 Arduino로 전송하여 RC카 제어.

    Arduino 는 UART 로 수신된 명령에 따라 RC 카의 모터를 제어.

&nbsp;

❔ **설정**

- **모델 파일**: `best.engine`.

  **quantization-fp16.py** 로부터 양자화된 모델 파일.

- **클래스 라벨** (5가지 신호):
  - `0`: GO
  - `1`: LEFT
  - `2`: RIGHT
  - `3`: STOP
  - `4`: SLOW.
- **시리얼 포트**: `/dev/ttyACM0`.
- **시리얼 통신 속도**: `9600 bps`. (baudrate per seconds)

&nbsp;

❔ **참고사항**

- 🚣 젯슨 나노 호스트의 signal-masters 서비스에서 실행하는 파일.

&nbsp;

---

## 💱 **교통 수신호 행동 인식 모델 방향성 설정**

### ➡️ 결론

- 젯슨 나노 환경에서는 RGB 또는 Optical Flow 기반 모델보다 **스켈레톤 기반** 접근이 적합.
- **PoseConv3D**는 경량 백본(**X3D S**)을 사용할 경우 젯슨 나노 환경에서 효율적으로 작동할 수 있음.
- **PYSKL**의 활용과 함께 동일한 키포인트 데이터를 사용하는 방식을 통해 개발 및 테스트를 진행하는 방향으로 연구를 설정함.

&nbsp;

---

### 분석 배경

- 젯슨 나노(Jetson Nano)와 같은 소형 하드웨어 환경에서 행동 인식 모델을 효과적으로 작동시키기 위해 다양한 접근법을 분석해야 함.
- RGB 기반, Optical Flow 기반, 그리고 스켈레톤(skeleton) 기반 모델을 비교하며, 각각의 장단점을 논문을 통해 검토.

&nbsp;

---

### RGB 기반 모델

RGB 데이터는 풍부한 정보를 포함하지만 계산량이 많고 메모리 소모가 커 소형 하드웨어 환경에서 부적합.

- RGB 기반 모델은 시각적 컨텍스트를 활용할 수 있으나, 배경과 조명의 변화에 민감하게 반응함.
- **젯슨 나노의 하드웨어 스펙(0.5 TFLOPS)** 상 효율적으로 작동하기 어려움.

&nbsp;

---

### Optical Flow 기반 모델

Optical Flow는 움직임을 나타내는 프레임 간의 픽셀 이동 정보를 활용.

- Optical Flow 계산은 모든 프레임의 픽셀 이동 정보를 계산하므로 계산량이 커 비효율적임.
- 젯슨 나노 환경에서는 비효율적.

&nbsp;

---

### Skeleton 기반 모델

Skeleton 기반 접근은 계산량이 적고 행동 인식에서 중요한 정보를 포함.

> "GCN-based methods are subject to limitations in robustness, interoperability, and scalability." ([2022-04-02] PoseC3D.pdf)

> "While GCN directly handles coordinates of human joints, its recognition ability is significantly affected by the distribution shift of coordinates, which can often occur when applying a different pose estimator to acquire the coordinates." ([2022-04-02] PoseC3D.pdf)

> "A small perturbation in coordinates often leads to completely different predictions." ([2022-04-02] PoseC3D.pdf)

- PoseConv3D([2022-04-02] PoseC3D.pdf)에서 제안된 방식처럼 3D 히트맵을 활용하면, 스켈레톤 데이터의 공간적, 시간적 특징을 효과적으로 학습 가능.
- 다중 인물 시나리오에서 계산량이 선형적으로 증가하며, 다른 데이터 모달리티와의 융합이 복잡함.

#### ❗ GCN 기반 모델 사용 시 도전 과제

GCN 기반 모델을 사용하려면 다음과 같은 추가 작업이 필요함:

- 각 이미지에서 관절 데이터를 포즈 인식 모델을 통해 추출해야 하며, 이를 키포인트로 매핑하는 작업이 필요. 이 과정에서 데이터 포맷 변환 및 추가적인 처리 단계가 복잡하게 얽힘.
- 젯슨 나노에서 실행 가능한 모델로 EfficientGCN을 고려했으나, 해당 모델의 FLOPS가 적합한지 확인 후에도 커스텀 데이터셋에 대한 라벨링 작업이 어렵고, 키포인트 매핑을 별도로 처리해야 하는 복잡성으로 인해 사용하지 않음.

&nbsp;

---

### 선택한 모델

#### [Pose3D.pdf](papers/%5B2022-04-02%5D%20PoseC3D.pdf)

GCN 기반 모델의 한계를 극복하며, 스켈레톤 데이터를 효과적으로 활용하는 CNN 기반 모델.

> "Compared to GCN-based methods, PoseConv3D is more effective in learning spatiotemporal features, more robust against pose estimation noises, and generalizes better in cross-dataset settings." ([2022-04-02] PoseC3D.pdf)

> "PoseConv3D takes as input 2D poses obtained by modern pose estimators shown in Figure 1. The 2D poses are represented by stacks of heatmaps of skeleton joints rather than coordinates operated on a human skeleton graph. The heatmaps at different timesteps will be stacked along the temporal dimension to form a 3D heatmap volume." ([2022-04-02] PoseC3D.pdf)

- PoseConv3D는 GCN의 단점을 극복하기 위해 3D 히트맵 볼륨을 입력으로 사용. 히트맵은 각 스켈레톤 조인트의 위치를 3차원 볼륨으로 표현하여 CNN 기반 아키텍처가 시간 및 공간적 특징을 학습할 수 있도록 설계.
- X3D 백본을 사용하여 경량화된 구조로 구현 가능하며, **X3D S(0.6G FLOPS)**를 사용할 경우 젯슨 나노 환경에서도 효율적으로 작동 가능.

#### [X3D.pdf](papers/%5B2022-04-09%5D%20X3D.pdf)

경량화된 3D CNN 아키텍처. 공간 및 시간 차원에서 효율적으로 확장 가능.

> 🌟 "X3D achieves state-of-the-art performance while requiring 4.8× and 5.5× fewer multiply-adds and parameters for similar accuracy as previous work." ([2022-04-09] X3D.pdf)

**X3D S** 버전은 0.6G FLOPS로 젯슨 나노의 연산 능력과 적합.

#### [PYSKL.pdf](papers/%5B2022-05-19%5D%20PYSKL.pdf)

Skeleton 기반 행동 인식을 위한 다양한 알고리즘을 지원하는 PyTorch 기반 오픈소스 툴박스.

> "In contrast to existing open-source skeleton action recognition projects that include only one or two algorithms, PYSKL implements six different algorithms a unified framework with both the latest and original good practices to ease the comparison of efficacy and efficiency." ([2022-05-19] PYSKL.pdf)

- PYSKL은 최신 및 기존의 우수한 학습 방식들을 통합하여, 다양한 Skeleton 기반 행동 인식 알고리즘(PoseC3D 포함)을 비교하고 연구하기 용이한 환경을 제공하며, 특히 X3D 백본을 지원하여 경량화된 학습 및 테스트 환경을 구현할 수 있음.
- 데이터 전처리 및 커스텀 데이터셋 적용을 지원하여 효율적 연구 가능.

&nbsp;

---

### 구현 및 테스트

1. **스켈레톤 데이터 기반 모델 설정:**

   - 동일한 키포인트를 사용하는 데이터셋으로 복잡성 최소화.
   - PYSKL 툴을 사용하여 데이터 전처리 및 학습.

2. **라벨링된 데이터셋 활용:**

   - PYSKL 내 제공된 학습된 모델을 기반으로 초기 테스트 수행.

3. **추후 연구 방향:**

   - 카메라 실시간 프레임 데이터를 기반으로 모델 테스트.
   - 커스텀 데이터셋 라벨링 및 학습.

&nbsp;

### 📁 (행동 인식 모델) 디렉터리 구조

tree **testing/**  
└── 📂 mmaction_or_pyskl-files  
&nbsp;&nbsp;&nbsp;&nbsp;├── [ipython_test.py](testing/mmaction_or_pyskl-files/ipython_test.py)  
&nbsp;&nbsp;&nbsp;&nbsp;├── [ipython_test_yolo.py](testing/mmaction_or_pyskl-files/ipython_test_yolo.py)  
&nbsp;&nbsp;&nbsp;&nbsp;├── [latest.pth](testing/mmaction_or_pyskl-files/latest.pth) -> [epoch_21.pth](testing/mmaction_or_pyskl-files/epoch_21.pth)  
&nbsp;&nbsp;&nbsp;&nbsp;├── [pyproject-pyskly.toml](testing/mmaction_or_pyskl-files/pyproject-pyskly.toml)  
&nbsp;&nbsp;&nbsp;&nbsp;├── [pyproject.toml](testing/mmaction_or_pyskl-files/pyproject.toml)  
&nbsp;&nbsp;&nbsp;&nbsp;├── [temp_question.txt](testing/mmaction_or_pyskl-files/temp_question.txt)  
&nbsp;&nbsp;&nbsp;&nbsp;├── 📂 tools  
&nbsp;&nbsp;&nbsp;&nbsp;└── 📂 x3d_shallow_ntu60_xsub  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── [do_posec3d.py](testing/mmaction_or_pyskl-files/x3d_shallow_ntu60_xsub/do_posec3d.py)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── [epcoh_24_eval_output.txt](testing/mmaction_or_pyskl-files/x3d_shallow_ntu60_xsub/epcoh_24_eval_output.txt)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── [epoch_24.pth](testing/mmaction_or_pyskl-files/x3d_shallow_ntu60_xsub/epoch_24.pth)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── [joint.py](testing/mmaction_or_pyskl-files/x3d_shallow_ntu60_xsub/joint.py)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── [last_pred.pkl](testing/mmaction_or_pyskl-files/x3d_shallow_ntu60_xsub/last_pred.pkl)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── [README.md](testing/mmaction_or_pyskl-files/x3d_shallow_ntu60_xsub/README.md)
