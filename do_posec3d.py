# %% Setup
### requirees package: imageio[ffmpeg] (??)
# pytubefix not works for almost Youtube shorts. Download fails ; "<video_id> is unavailable"
# ➡️ yt-dlp ; https://pypi.org/project/yt-dlp/
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import cv2
import torch
import torch.nn
import yt_dlp
from mmcv.runner import load_checkpoint
from pyskl.models import Recognizer3D, build_model
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Directory setup
remote_video_save_directory: Path = Path.home() / "remote_videos"
remote_video_save_directory.mkdir(parents=True, exist_ok=True)
captured_video_save_directory: Path = Path.home() / "remote_videos" / "captured"
captured_video_save_directory.mkdir(parents=True, exist_ok=True)
pyskl_path: Path = Path.home() / "repo/intel-edge-academy-6/external/pyskl"
pyskl_data_path: Path = pyskl_path / "data"

action_classes = [
    "drink water",  # 1
    "eat meal/snack",  # 2
    "brushing teeth",  # 3
    "brushing hair",  # 4
    "drop",  # 5
    "pickup",  # 6
    "throw",  # 7
    "sitting down",  # 8
    "standing up (from sitting position)",  # 9
    "clapping",  # 10
    "reading",  # 11
    "writing",  # 12
    "tear up paper",  # 13
    "wear jacket",  # 14
    "take off jacket",  # 15
    "wear a shoe",  # 16
    "take off a shoe",  # 17
    "wear on glasses",  # 18
    "take off glasses",  # 19
    "put on a hat/cap",  # 20
    "take off a hat/cap",  # 21
    "cheer up",  # 22
    "hand waving",  # 23
    "kicking something",  # 24
    "reach into pocket",  # 25
    "hopping (one foot jumping)",  # 26
    "jump up",  # 27
    "make a phone call/answer phone",  # 28
    "playing with phone/tablet",  # 29
    "typing on a keyboard",  # 30
    "pointing to something with finger",  # 31
    "taking a selfie",  # 32
    "check time (from watch)",  # 33
    "rub two hands together",  # 34
    "nod head/bow",  # 35
    "shake head",  # 36
    "wipe face",  # 37
    "salute",  # 38
    "put the palms together",  # 39
    "cross hands in front",  # 40
    "sneeze/cough",  # 41
    "staggering",  # 42
    "falling",  # 43
    "touch head (headache)",  # 44
    "touch chest (stomachache/heart pain)",  # 45
    "touch back (backache)",  # 46
    "touch neck (neckache)",  # 47
    "nausea or vomiting condition",  # 48
    "use a fan (with hand or paper)/feeling warm",  # 49
    "punching/slapping other person",  # 50
    "kicking other person",  # 51
    "pushing other person",  # 52
    "pat on back of other person",  # 53
    "point finger at the other person",  # 54
    "hugging other person",  # 55
    "giving something to other person",  # 56
    "touch other person's pocket",  # 57
    "handshaking",  # 58
    "walking towards each other",  # 59
    "walking apart from each other",  # 60
]

# YOLO Pose 모델 로드
model = YOLO("yolo11m-pose.pt")

# PoseC3D 모델 로드
checkpoint_path = (
    Path.home()
    / "repo/intel-edge-academy-6/external/pyskl/work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint/best_top1_acc_epoch_21.pth"
)
posec3d_model: Recognizer3D = build_model(
    {
        "type": "Recognizer3D",
        "backbone": {
            "type": "X3D",
            "gamma_d": 1,
            "in_channels": 17,  # 관절 포인트 수
            "base_channels": 24,
            "num_stages": 3,
            "se_ratio": None,
            "use_swish": False,
            "stage_blocks": (2, 5, 3),
            "spatial_strides": (2, 2, 2),
        },
        "cls_head": {
            "type": "I3DHead",
            "in_channels": 216,
            "num_classes": 60,  # NTU-60 데이터셋 기반
            "dropout": 0.5,
        },
        "test_cfg": {"average_clips": "prob"},
    }
)
load_checkpoint(posec3d_model, str(checkpoint_path), map_location="cuda")
posec3d_model = posec3d_model.cuda().eval()

# 카메라 열기
cap = cv2.VideoCapture(0)

# 48 프레임을 저장할 큐
frame_queue = deque(maxlen=48)


def preprocess_keypoints(result):
    """
    YOLO Pose 결과로부터 keypoints 데이터를 추출하고, PoseC3D 입력으로 적합한 형식으로 변환합니다.
    """
    if result.keypoints is not None and result.keypoints.data.size(0) > 0:
        keypoints_tensor = result.keypoints.data[:, :, :2]  # (N, 17, 2), x, y 좌표 추출
        keypoints_tensor = keypoints_tensor[0]  # 첫 번째 사람만 선택 (N=1)
        keypoints_tensor = keypoints_tensor.permute(1, 0)  # (17, 2) -> (2, 17)
        return keypoints_tensor.unsqueeze(0)  # 차원 확장 (1, 2, 17)
    return None


try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLO Pose 추론
        results = model.track(frame, persist=True, verbose=False)

        # Keypoints 처리
        keypoints = preprocess_keypoints(results[0])
        if keypoints is not None:
            frame_queue.append(keypoints)

        # 48 프레임이 쌓이면 행동 인식 수행
        if len(frame_queue) == 48:
            # 큐의 프레임을 쌓아 입력 텐서로 만듦 (48 프레임)
            input_tensor = torch.stack(
                list(frame_queue), dim=2
            ).cuda()  # (1, 2, 17, 48)

            # 차원 재배열 및 배치 추가
            input_tensor = (
                input_tensor.permute(0, 2, 3, 1).unsqueeze(-1).unsqueeze(-1)
            )  # (1, 17, 48, 1, 1)

            # PoseC3D 모델로 추론 수행
            with torch.no_grad():
                output = posec3d_model(return_loss=False, imgs=input_tensor)

            action = output.argmax()  # dim 사용 없이 argmax()로 인덱스 찾기
            action_label = action_classes[action]  # 해당 인덱스를 동작 클래스에 대응

            # 프레임에 동작 라벨 표시
            cv2.putText(
                frame,
                f"Action: {action_label}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        # YOLO 결과 시각화
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Pose Tracking", annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
