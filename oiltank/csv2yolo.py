import os
import pandas as pd

def convert_coordinates(coords_str, image_width, image_height):
    coords = coords_str.split(',')
    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, coords)
    x = (x1 + x2 + x3 + x4) / 4
    y = (y1 + y2 + y3 + y4) / 4
    width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
    height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

    # 정규화된 좌표 계산
    x_normalized = x / image_width
    y_normalized = y / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height

    return x_normalized, y_normalized, width_normalized, height_normalized

def create_yolo_annotations(image_folder, csv_file, target_image_size):
    df = pd.read_csv(csv_file)
    image_ids = df['image_id'].unique()

    for image_id in image_ids:
        txt_filename = os.path.splitext(image_id)[0] + '.txt'
        txt_path = os.path.join(image_folder, txt_filename)

        image_path = os.path.join(image_folder, image_id)
        image_width, image_height = target_image_size

        with open(txt_path, 'w') as txt_file:
            image_df = df[df['image_id'] == image_id]
            for index, row in image_df.iterrows():
                type_name = row['type_name']
                if type_name == 'oil tank':
                    x, y, width, height = convert_coordinates(row['object_imcoords'], image_width, image_height)
                    txt_file.write(f'0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n')

if __name__ == "__main__":
    dataset_folder = "oiltank_dataset"
    train_image_folder = os.path.join(dataset_folder, "train_images")
    val_image_folder = os.path.join(dataset_folder, "val_images")
    train_csv_file = os.path.join(dataset_folder, "train_dataset.csv")
    val_csv_file = os.path.join(dataset_folder, "val_dataset.csv")

    target_image_size = (1024, 1024)

    create_yolo_annotations(train_image_folder, train_csv_file, target_image_size)
    create_yolo_annotations(val_image_folder, val_csv_file, target_image_size)
