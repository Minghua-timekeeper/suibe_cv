from ultralytics import YOLO

model = YOLO("yolov8s.pt")

data = "ultralytics/datasets/Helmet.yaml"

model.train(data=data, workers=0, epochs=300, batch=8)
