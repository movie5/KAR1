import json
import pandas as pd
import glob

train_images = sorted(glob.glob('oiltank_dataset/train_images/*'))
train_jsons = sorted(glob.glob('oiltank_dataset/train_labels/*'))
val_images = sorted(glob.glob('oiltank_dataset/val_images/*'))
val_jsons = sorted(glob.glob('oiltank_dataset/val_labels/*'))

len(train_images), len(train_jsons), len(val_images), len(val_jsons)

train_meta = []
for j in train_jsons:
    json_obj = json.load(open(j))
    for f in json_obj['features']:
        # properties
        object_imcoords = f['properties']['object_imcoords']
        object_angle = f['properties']['object_angle']
        image_id = f['properties']['image_id']
        ingest_time = f['properties']['ingest_time']
        type_name = f['properties']['type_name']
        # Add to list
        train_meta.append([image_id, type_name, object_imcoords, object_angle, ingest_time])

# Make dataframe for train dataset
df_train = pd.DataFrame(train_meta,
                        columns=['image_id', 'type_name', 'object_imcoords', 'object_angle', 'ingest_time'])

val_meta = []
for j in val_jsons:
    json_obj = json.load(open(j))
    for f in json_obj['features']:
        # properties
        object_imcoords = f['properties']['object_imcoords']
        object_angle = f['properties']['object_angle']
        image_id = f['properties']['image_id']
        ingest_time = f['properties']['ingest_time']
        type_name = f['properties']['type_name']
        # Add to list
        val_meta.append([image_id, type_name, object_imcoords, object_angle, ingest_time])

# Make dataframe for validation dataset
df_val = pd.DataFrame(val_meta,
                      columns=['image_id', 'type_name', 'object_imcoords', 'object_angle', 'ingest_time'])

# Optionally, you can save the dataframes as CSV files if needed
df_train.to_csv('train_dataset.csv', index=False)
df_val.to_csv('val_dataset.csv', index=False)