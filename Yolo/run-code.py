import cv2
from ultralytics import YOLO
import serial
import time

# 학습된 모델 로드
model = YOLO("runs/pose/yolo11n-pose-custom8/weights/best.pt")

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)

last_command_time = 0
command_interval = 0.5  # 명령을 보내는 최소 간격(0.5초)

def send_command_to_arduino(command):
    global last_command_time
    current_time = time.time()
    
    # 마지막 명령 후 일정 시간이 지나야 새로운 명령을 전송
    if current_time - last_command_time >= command_interval:
        arduino.write(bytes(command + '\n', 'utf-8'))
        last_command_time = current_time

# 클래스 ID에 따른 라벨 정의
class_labels = {
    0: "GO",
    1: "RIGHT",
    2: "LEFT",
    3: "STOP",
    4: "SLOW"
}

# 웹캠 시작
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠, 다른 장치를 사용 중이면 해당 장치 번호로 변경

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 영상을 읽을 수 없습니다.")
        break

    # 프레임에서 포즈 예측 수행
    results = model.predict(source=frame, save=False)

    detected = False  # 감지 여부 확인
    # 결과에서 클래스 ID에 따라 출력
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)  # 클래스 ID
                if class_id in class_labels:
                    detected = True
                    send_command_to_arduino(class_labels[class_id] + "\n")
    # 감지되지 않은 경우 기본값으로 'go' 출력
    if not detected:
        send_command_to_arduino(class_labels[0] + "\n")

    # 결과 시각화 (키포인트 및 바운딩 박스를 표시한 프레임)
    annotated_frame = results[0].plot()

    # 프레임에 예측 결과 출력
    cv2.imshow("YOLO11n-Pose Real-time Inference", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 창 닫기
cap.release()
cv2.destroyAllWindows()
