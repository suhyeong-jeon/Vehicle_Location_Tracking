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

### Custom Data Augmentations
#### degrees = 7 mixup=0.1, default Yolov8 augmentations.   

- - -

### Review
#### The model needs more training. The accuracy is not that good. Therefore, it would be better add extra data to train the model.   


#### 학습 데이터의 표본이 많이 부족해 추가적인 NUSCENES를 다운받아 라벨링해 학습하면 정확도가 훨씬 증가할 것으로 보인다. val_loss도 확연히 줄어들고 있었음.   

#### 차들의 방향은 주행중인 차량의 전방 카메라의 대략적인 중앙 좌표 (x_f, y_f)를 기준으로 x는 5등분 y는 3등분을 했다.   
#### 동영상의 각 프레임을 모델에 넣어 감지한 object의 bounding box의 xyxy좌표를 얻어 bounding box의 넓이를 계산해 일정 값 이상이면 차와 가깝다고 판단해, 해당 차의 위치를 판별하는 알고리즘이 실행된다.   
#### 먼저 object의 bounding box의 xywh좌표를 얻은 후 x값이 전방 카메라의 중앙 좌표 x_f//5 보다 작다면 진행 차량의 왼쪽, x_f//5보다 크다면 진행 차량의 오른쪽, 그 사이의 값이라면 정면에 위치한다는 정보를 변수에 저장해 OpenCV2로 카메라에 표시하게 된다.
