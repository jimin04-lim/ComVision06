import cv2
import numpy as np
import os
from sort import Sort

# 1. 필수 파일 존재 여부 확인
if not os.path.exists("yolov3.weights") or not os.path.exists("yolov3.cfg"):
    print("❌ 에러: yolov3.weights 또는 yolov3.cfg 파일이 폴더에 없습니다!")
    exit()

# YOLO 모델 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# SORT 추적기 초기화
mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

# 비디오 로드
video_path = "slow_traffic_small.mp4" 
if not os.path.exists(video_path):
    print(f"❌ 에러: {video_path} 비디오 파일이 없습니다!")
    exit()

cap = cv2.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        print("비디오 재생이 끝났거나 프레임을 읽을 수 없습니다.")
        break
    
    frame_count += 1
    height, width, channels = frame.shape

    # YOLO 입력용 블롭(Blob) 생성
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    
    # 2. 객체 검출 (모든 클래스, 신뢰도 0.2 이상으로 대폭 낮춤)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # 테스트를 위해 신뢰도를 0.2로 낮추고, 모든 사물을 검출하도록 변경
            if confidence > 0.2: 
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                
                boxes.append([x, y, x+w, y+h])
                confidences.append(float(confidence))

    # NMS (겹치는 박스 제거)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    dets = []
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x1, y1, x2, y2 = boxes[i]
            dets.append([x1, y1, x2, y2, confidences[i]])

    # 디버깅 출력: 현재 프레임에서 몇 개가 검출되었는지 확인
    print(f"[Frame {frame_count}] YOLO 검출 개수: {len(dets)}개")

    # SORT 알고리즘에 맞게 numpy 배열로 변환
    dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))

    # SORT 객체 추적 업데이트
    trackers = mot_tracker.update(dets)

    # 시각화 (바운딩 박스 그리기)
    for d in trackers:
        x1, y1, x2, y2, track_id = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # 빨간색 박스
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Multi-Object Tracking", frame)
    
    if cv2.waitKey(1) == 27: # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()