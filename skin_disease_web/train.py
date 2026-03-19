from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data=r"C:\Users\acer\Downloads\archive (1)\ham_full_split_split",
    epochs=50,
    imgsz=224,
    batch=16,
    device="cpu"
)