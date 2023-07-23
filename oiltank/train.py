from ultralytics import YOLO

def main():
    model = YOLO("ultralytics/ultralytics/cfg/models/v5/yolov5.yaml")
    model.info()
    model.train(data="C://Users/laboratory/repository/oiltank/oiltank_dataset/data.yaml", epochs=500)

if __name__ == "__main__":
    main()
