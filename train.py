from ultralytics import YOLO
import torch

if __name__=='__main__':
    print("GPU : ", torch.cuda.is_available())
    model = YOLO('yolov8x.pt')

    result = model.train(data='D:/Computer_Visoin_Projects/CarDetection/ultralytics-main/ultralytics-main/ultralytics/cfg/datasets/car_detection.yaml', epochs=100, imgsz=640, batch=8, optimizer="SGD", device=0,
                         degrees=7, patience=100, mixup=0.1)
