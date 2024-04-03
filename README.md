# Vehicle Location Tracking

#### * Created an AI model that detects and tracks the location of the car.   
#### * Created an algorithm that detects cars location(Left, Front, Right)   

### Training, Validation Dataset : [NUSCENES by Motional](https://www.nuscenes.org/nuscenes)
### Utilized Label Studio to detect and track a target class.
### Test Dataset : [Driving Downtown - Las Vegas 4K - USA](https://www.youtube.com/watch?v=DL703lh_my8&t=48s)   


#### model = YOLO('yolov8x.pt')
#### epochs=100, imgsz=640, batch=8, optimizer='SGD'   


### Custom Data Augmentation
#### degrees = 7 mixup=0.1, default Yolov8 augmentations.   


### Review
#### The model needs more training.
