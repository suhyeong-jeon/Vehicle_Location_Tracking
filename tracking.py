from collections import defaultdict

from ultralytics import YOLO
import cv2
import numpy as np


def video_tracking(model, cap, track_history):
    cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)  # cv2 윈도우 창 생성
    cv2.resizeWindow('YOLOv8 Tracking', 640, 640)  # cv2 윈도우 창 크기 변경

    while cap.isOpened():
        success, frame = cap.read()

        f_h, f_w, f_c = frame.shape # h = 1080 w = 1920 c = 3

        center_x = int(f_w/2)
        center_y = int(f_h/2)

        if success:
            results = model.track(frame, persist=True)

            boxes = results[0].boxes.xywh.cpu()
            predicted_boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() # tensor -> 리스트로 변환

            left_track_id = []
            front_track_id = []
            right_track_id = []

            annotated_frame = results[0].plot() # frame에 대해 car를 추적한 결과를 시각적으로 표현한 정보. 즉 bounding box와 confidence가 표시 된 frame임

            cv2.putText(annotated_frame, "L-D : ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, "F-D : ", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, "R-D : ", (1150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)

            for box, track_id, predicted_box in zip(boxes, track_ids, predicted_boxes):
                x, y, w, h = box
                x1, y1, x2, y2 = predicted_box
                predicted_box_size = (y2-y1) * (x2-x1)

                track = track_history[track_id] # track_id에 대한 새로운 list가 defaultdict 내부에 생성됨
                # print('track : ', track)
                track.append((float(x), float(y))) # track에 bounding box의 x, y좌표 저장 -> YOLO에서 xywh포멧의 x, y는 bounding box의 중심 좌표임.
                # print('append track : ', track)
                # print('track len : ', len(track))
                if len(track) > 30:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2)) # 3차원 배열로 변환
                # print(points)
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                # print(x1, y1, x2, y2)

                if predicted_box_size >= 25000:
                    # print("감지됨 : ", track_id)
                    # x에 대한 위치 확인 -> 사면체, 전방
                    if x < (f_w//5) * 2 and y > (f_h//3) : # 좌측 상단
                        left_track_id.append(track_id)
                        cv2.putText(annotated_frame, f"L-D : {str(left_track_id)[1:-1]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        # print(f'좌측 {track_id} : {(y2-y1) * (x2-x1)}')

                    elif x > (f_w//5)*3 and y > (f_h//3):
                        right_track_id.append(track_id)
                        cv2.putText(annotated_frame, f"R-D : {str(right_track_id)[1:-1]}", (1150, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        # print(f'우측 {track_id} : {(y2 - y1) * (x2 - x1)}')
                    elif (x >= (f_w//5) * 2 and x <= (f_w//5)*3) and y > (f_h//3):
                        front_track_id.append(track_id)
                        cv2.putText(annotated_frame, f"F-D : {str(front_track_id)[1:-1]}", (600, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        # print(f'정면 {track_id} : {(y2 - y1) * (x2 - x1)}')

            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    video_path = "D:\Download/sample.mp4"
    track_history = defaultdict(lambda: []) # default값을 list로 줌
    model = YOLO('./best.pt')
    cap = cv2.VideoCapture(video_path)

    video_tracking(model, cap, track_history)