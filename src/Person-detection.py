import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # YOLOv8 모델 경로 설정

# Mediapipe 포즈 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 비디오 파일 열기
cap = cv2.VideoCapture(0)  # 동영상 파일 경로 설정

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8을 이용해 사람 탐지
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()  # 탐지 결과 가져오기

    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) == 0:  # 사람(class_id == 0)만 처리
            # 탐지된 사람 영역 잘라내기
            person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            if person_roi.size == 0:
                continue

            # Mediapipe 포즈 랜드마크 추출
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            result = pose.process(person_rgb)

            if result.pose_landmarks:
                # 원본 프레임에 랜드마크 그리기 (ROI 좌표 보정)
                for landmark in result.pose_landmarks.landmark:
                    lx = int(x1 + landmark.x * (x2 - x1))
                    ly = int(y1 + landmark.y * (y2 - y1))
                    cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)

    # 결과 출력
    cv2.imshow('YOLOv8 & Mediapipe Pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()