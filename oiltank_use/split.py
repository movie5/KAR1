import os
import shutil

# 검증 데이터셋의 비율 설정
validation_ratio = 0.2

# 폴더 경로 설정
train_images_path = 'oiltank_dataset/train_images'
train_labels_path = 'oiltank_dataset/train_labels'
val_images_path = 'oiltank_dataset/val_images'
val_labels_path = 'oiltank_dataset/val_labels'

# train_images 폴더에서 파일들을 읽어옴
train_image_files = os.listdir(train_images_path)

# 검증 데이터셋의 크기 계산
num_validation_samples = int(len(train_image_files) * validation_ratio)

# 검증 데이터셋의 파일들을 val_images로 이동시킴
for file_name in train_image_files[:num_validation_samples]:
    if file_name.endswith('.tif') or file_name.endswith('.png') or file_name.endswith('.kml'):
        shutil.move(os.path.join(train_images_path, file_name), os.path.join(val_images_path, file_name))

# 검증 데이터셋에 해당하는 파일들의 이름을 추출
validation_file_names = [file_name[:8] for file_name in train_image_files[:num_validation_samples]]

# train_labels 폴더에서 검증 데이터셋에 해당하는 json 파일을 val_labels로 이동시킴
for file_name in os.listdir(train_labels_path):
    if file_name[:8] in validation_file_names:
        shutil.move(os.path.join(train_labels_path, file_name), os.path.join(val_labels_path, file_name))
