from ultralytics import YOLO
import cv2
import glob
import os


def photo_prediction(model, input_path, input_type):
    input_path_list = glob.glob(os.path.join(input_path, f'*.{input_type}'))

    for path in input_path_list:
        results = model.predict(
            path,
            save=False,
            imgsz=640,
            conf=0.65,
            device='0',
            # stream=True,
        )
        # print(results)

        for result in results:
            image_path = result.path
            boxes = result.boxes.xyxy
            cls = result.boxes.cls # test 사진 안에 detected된 물체의 class number가 적힌 tensor
            conf = result.boxes.conf
            cls_dict = result.names


            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 축소된 이미지의 비율에 맞춘 box를 만들기 위해 height, weight, channel이 필요함
            h, w, c = image.shape

            image = cv2.resize(image, (640, 640))

            for box, cls_number, conf in zip(boxes, cls, conf):
                conf_number = float(conf.item())
                cls_number_int = int(cls_number.item())
                cls_name = cls_dict[cls_number_int]

                x1, y1, x2, y2 = box

                x1_int = int(x1.int())
                y1_int = int(y1.int())
                x2_int = int(x2.item())
                y2_int = int(y2.item())
                print(f"Coordinate : {x1_int, y1_int, x2_int, y2_int}\nclass : {cls_name}")

                # 위에서 이미지를 640 * 640 크기로 resize했으니 bounding box들의 크기 비율 또한 축소시켜 사진에 맞춰줘야함
                scale_factor_x = 640 / w
                scale_factor_y = 640 / h
                x1_scale = int(x1_int * scale_factor_x)
                y1_scale = int(y1_int * scale_factor_y)
                x2_scale = int(x2_int * scale_factor_x)
                y2_scale = int(y2_int * scale_factor_y)

                print(f"Box size : {(x2_scale - x1_scale) * (y2_scale - y1_scale)}\n")

                image = cv2.rectangle(image, (x1_scale, y1_scale), (x2_scale, y2_scale), (0, 255, 255), 3)

            cv2.imshow("Test", image)
            cv2.waitKey(0)

def video_prediction(model, input_path, input_type):
    input_path_list = glob.glob(os.path.join(input_path, f'*.{input_type}'))

    for path in input_path_list:
        cap = cv2.VideoCapture(path)

        if not cap.isOpened(): # 비디오 파일이 정상적으로 열렸는지 확인
            print("Error: Could not open video file.")
            exit()

        fps = cap.get(cv2.CAP_PROP_FPS) # 비디오의 프레임 수 가져옴

        cv2.namedWindow("YOLO Prediction", cv2.WINDOW_NORMAL) # cv2 윈도우 창 생성
        cv2.resizeWindow('YOLO Prediction', 640, 640) # cv2 윈도우 창 크기 변경

        while True:
            ret, frame = cap.read()

            if not ret: # 프레임을 제대로 읽지 못한 경우 종류
                break

            # 프레임에 모델 적용
            results = model.predict(
                frame,
                save=False,
                imgsz=640,
                conf=0.65,
                device='0',
                stream=True,
            )

            for result in results:
                image_path = result.path
                boxes = result.boxes.xyxy
                cls = result.boxes.cls  # test 사진 안에 detected된 물체의 class number가 적힌 tensor
                conf = result.boxes.conf
                cls_dict = result.names

                h, w, c = frame.shape

                frame = cv2.resize(frame, (640, 640))

                for box, cls_number, conf in zip(boxes, cls, conf):
                    conf_number = float(conf.item())
                    cls_number_int = int(cls_number.item())
                    cls_name = cls_dict[cls_number_int]

                    x1, y1, x2, y2 = box

                    x1_int = int(x1.int())
                    y1_int = int(y1.int())
                    x2_int = int(x2.item())
                    y2_int = int(y2.item())
                    print(f"Coordinate : {x1_int, y1_int, x2_int, y2_int}\nclass : {cls_name}")

                    # 위에서 이미지를 640 * 640 크기로 resize했으니 bounding box들의 크기 비율 또한 축소시켜 사진에 맞춰줘야함
                    scale_factor_x = 640 / w
                    scale_factor_y = 640 / h
                    x1_scale = int(x1_int * scale_factor_x)
                    y1_scale = int(y1_int * scale_factor_y)
                    x2_scale = int(x2_int * scale_factor_x)
                    y2_scale = int(y2_int * scale_factor_y)

                    print(f"Box size : {(x2_scale - x1_scale) * (y2_scale - y1_scale)}\n")

                    frame = cv2.rectangle(frame, (x1_scale, y1_scale), (x2_scale, y2_scale), (0, 255, 255), 3)
            
            cv2.imshow("YOLO Prediction", frame) # Bounding Box표시까지 한 Frame을 cv2에 출력

            if cv2.waitKey(1) & 0xFF == ord('q'): # q 키 누르면 cv2 종료
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    model = YOLO('./runs/detect/train6/weights/best.pt')
    # image_path = 'D:/Computer_Visoin_Projects/CarDetection/transformed_data/val/images'
    input_path = 'D:\Download'
    input_type = 'mp4'

    # photo_prediction(model, input_path, input_type)
    video_prediction(model, input_path, input_type)