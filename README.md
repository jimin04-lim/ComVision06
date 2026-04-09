# ComVision 06주차 실습
# OpenCV 실습 과제

## 0601. SORT알고리즘을 활용한 다중 객체 추적기 구현
- **설명**:
  - SORT 알고리즘을 사용하여 비디오에서 다중 객체를 실시간으로 추적하는 프로그램
- **요구사항**:
  - 객체 검출기 구현: YOLOv3와 같은 사전 훈련된 객체 검출 모델을 사용하여 각 프레임에서 객체 검출
    - OpenCV의 DNN 모듈을 사용하여 YOLOv3 모델을 로드, 각 프레임에서 객체 검출 가능
  - mathworks.comSORT 추적기 초기화: 검출된 객체의 경계 상자를 입력으로 받아 SORT 추적기 초기화
  - 객체추적: 각 프레임마다 검출된 객체와 기존 추적 객체를 연관시켜서 추적 유지
  - 결과 시각화: 추적된 각 객체에 고유 ID를 부여하고, 해당 ID와 경계 상자를 비디오 프레임에 표시하여 실시간 출력
  - SORT 알고리즘: 칼만 필터와 헝가리안 알고리즘을 사용한 객체의 상태 예측, 데이터 연관 수행
  - 추적 성능 향상: 객체의 appearance 정보를 활용하는 Deep SORT와 같은 확장 알고리즘을 사용하여 추적 성능 향상
- **코드**
  ```python
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
  ```
- **주요코드**
  ```python
  cv2.dnn.readNet(...): 미리 학습된 YOLOv3 가중치 파일과 설정 파일을 불러와 객체 검출 모델을 메모리에 올립니다.

  mot_tracker = Sort(...): SORT 객체를 초기화합니다. 내부적으로 칼만 필터(Kalman Filter)를 사용하여 이전 프레임의 정보로 현재 객체의 위치를 예측하고, 헝가리안 알고리즘(Hungarian Algorithm)을 통해 예측된 위치와 실제 검출된 객체의 위치를 매칭(Data Association)합니다.
  
  mot_tracker.update(dets): 프레임마다 검출된 바운딩 박스 정보(dets)를 입력하여, 기존에 추적하던 객체인지 새로운 객체인지 판별한 후 고유 ID가 포함된 추적 결과를 반환합니다.
  
  추가 개념 (Deep SORT): 기본 SORT 알고리즘은 객체가 가려졌을 때(Occlusion) ID가 바뀌는 한계가 있습니다. 이를 해결하기 위해 딥러닝으로 추출한 객체의 외형적 특징(Appearance 정보)까지 매칭에 활용하여 성능을 대폭 향상시킨 알고리즘이 바로 'Deep SORT'입니다.

  ```
- **결과물**:
<img width="894" height="554" alt="image" src="https://github.com/user-attachments/assets/73bb7e47-a7de-4319-ad49-1edaf753f95f" />
<img width="1164" height="766" alt="image" src="https://github.com/user-attachments/assets/3b6da559-df5e-487d-ab4f-76ae2619cb66" />




## 0602. Mediapip를 활용한 얼굴 랜드마크 추출 및 시각화
- **설명**: Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크 추출하고, 이를 실시간 영상에 시각화하는 프로그램 구현
- **요구사항**:
  - Mediapipe의 FaceMesh 모듈을 사용하여 얼굴 랜드마크 검출기를 초기화
    - solutions.face_mesh 사용
  - OpenCV를 사용하여 웹캠으로부터 실시간 영상 캡처
  - 검출된 얼굴 랜드마크를 실시간 영상에 점으로 표시
    - OpenCV의 circle 함수를 사용해 랜드마크 시각화 가능(단, 좌표 정규화가 되어 있어 이미지 크기에 맞는 변환 필요)
  - ESC 키를 누르면 프로그램 종료
- **코드**
  ```python
  ```
- **주요코드**
  ```python
  mp.solutions.face_mesh.FaceMesh(...): Mediapipe의 468개 랜드마크 기반 고정밀 얼굴 인식 모델을 불러옵니다. 초기화 시 인식할 최대 얼굴 수와 추적 신뢰도를 설정할 수 있습니다.

  face_mesh.process(rgb_frame): 입력된 이미지에서 얼굴의 랜드마크 위치를 추론합니다. 이 함수는 반드시 RGB 포맷의 이미지를 인자로 받아야 합니다.
  
  x, y = int(lm.x * w), int(lm.y * h): 가장 중요한 정규화 좌표 변환 부분입니다. Mediapipe가 반환하는 좌표는 이미지 해상도에 무관하게 0~1 사이로 정규화되어 있기 때문에, 해당 값에 원본 이미지의 너비(w)와 높이(h)를 곱해야만 cv2.circle()로 화면에 정확히 점을 찍을 수 있습니다.

  ```
- **결과물**:



