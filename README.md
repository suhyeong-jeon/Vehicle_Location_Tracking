# Vehicle Location Tracking   

<p align="center"><img src="https://github.com/suhyeong-jeon/Vehicle_Location_Tracking/assets/70623959/50e8e7c9-5e5b-43cb-be21-58fea42f302a"></p>
<p align="center"><img src="https://github.com/suhyeong-jeon/Vehicle_Location_Tracking/assets/70623959/5d18dbd2-0272-4011-a21c-10b036e4265f"></p>

#### * Created an AI model that detects and tracks the location of the car.
#### * Created an algorithm that detects cars location(Left-Detection, Front-Detection, Right-Detection)   

- - -

### Training, Validation Dataset : [NUSCENES by Motional](https://www.nuscenes.org/nuscenes)
### Utilized Label Studio to detect and track a target class.
### Test Dataset : [Driving Downtown - Las Vegas 4K - USA](https://www.youtube.com/watch?v=DL703lh_my8&t=48s)   

- - -

#### model = YOLO('yolov8x.pt')
#### epochs=100, imgsz=640, batch=8, optimizer='SGD'   

- - -

### Custom Data Augmentation
#### degrees = 7 mixup=0.1, default Yolov8 augmentations.   

- - -

### Review
#### The model needs more training. The accuracy is not that good and it would be better add extra data to train the model.   
