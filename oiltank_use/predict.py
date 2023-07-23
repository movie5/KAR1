from ultralytics import YOLO
import pandas as pd
import os
def yolo_to_x1y1x2y2x3y3x4y4(yolo_coords):
    """
    YOLO 형식의 bounding box 좌표를 x1, y1, x2, y2, x3, y3, x4, y4 형식으로 변환하는 함수
    """
    x, y, w, h = yolo_coords
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y - h/2
    x3, y3 = x - w/2, y + h/2
    x4, y4 = x + w/2, y + h/2
    return x1, y1, x2, y2, x3, y3, x4, y4

def main():
    # Load a model
    model = YOLO('runs/detect/train22/weights/best.pt')  # pretrained YOLOv8n model
    results = model.predict(source='oiltank_dataset/test_images', save=True, save_txt=True)

    data = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            file_name = os.path.splitext(os.path.basename(result.path))[0][:8]
            conf = box.conf[0]
            x1, y1, x2, y2, x3, y3, x4, y4 = yolo_to_x1y1x2y2x3y3x4y4(box.xywh[0])
            data.append([file_name, conf, x1, y1, x2, y2, x3, y3, x4, y4])

    df = pd.DataFrame(data, columns=['File', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'])
    df.to_csv('predicted_labels.csv', index=False)

if __name__ == "__main__":
    main()
