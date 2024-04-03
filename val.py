from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./runs/detect/train6/weights/best.pt')

    source = 'D:/Download/sample.mp4'

    model.predict(source, save=True, imgsz=640, conf=0.7)
