import os
import sys
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.bin"
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.csv2xml import csv2xml
from keras_retinanet.paths import load_config
import numpy as np
import pandas as pd
import tensorflow as tf
import zipfile
import xml.etree.ElementTree as ET
import csv


# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set tf backend to allow memory to grow, instead of claiming everything


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def dv_xml_to_csv(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    header = ('datetime', 'depth', 'active')
    table = []

    for element in root.iter("frames"):
        for frame in element:
            active = frame.get("active")
            depth = frame.get("depth")
            time = frame.get('time')
            row = time, float(depth), active
            table.append(row)
    out_df = pd.DataFrame(table, columns=header)
    return out_df


def read_zip_files(path_to_data):
    list_of_files = []
    zip_dict = {}
    zip_files = [x for x in os.listdir(path_to_data) if x.endswith("active.zip")]
    for z in zip_files:
        archive = zipfile.ZipFile(os.path.join(path_to_data, z), "r")
        zip_dict[z] = archive.namelist()
        list_of_files += archive.namelist()
    return zip_dict, list_of_files


def select_data_from_xml_file(list_of_files, xml_file):
    df = dv_xml_to_csv(xml_file)
    image_list = [i[:17] for i in list_of_files]
    df = df[df["datetime"].isin(image_list)]
    return df


def predict_image(model, filename):
    image = read_image_bgr(filename)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    return boxes, scores, labels


def DV_predict(model, path_to_data, orientation, xml_file):
    path_to_data = os.path.join(path_to_data, orientation)
    dict, list_of_files = read_zip_files(path_to_data)
    df = select_data_from_xml_file(list_of_files, xml_file)
    anno = []
    for ind in df.index:
        date = df["datetime"][ind]
        z = [key for key, value in dict.items() if str(date)+".jpg" in value]
        archive = zipfile.ZipFile(os.path.join(path_to_data, z[0]), "r")
        img = archive.open(str(date)+".jpg")
        boxes, scores, labels = predict_image(model, img)
        anno_row = [date, df["depth"][ind], 0, 0, 0, 0, 0, 0]
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score > 0.05:
                anno_row = [date, df["depth"][ind], 0, 0, 0, 0, 0, 0]
                anno_row[2:6] = box
                anno_row[6] = labels_to_names[label]
                anno_row[7] = score
                anno.append(anno_row)
        if anno_row[7] < 0.05:
            anno.append([date, df["depth"][ind], 0, 0, 0, 0, 0, 0])
    anno_df = pd.DataFrame(anno, columns=['datetime', 'depth', 'x0', 'y0', 'x1', 'y1', 'label', 'score'])
    output_csv = xml_file.split(".")[0] + "_" + orientation + ".csv"
    anno_df.to_csv(output_csv, index=False)
    csv2xml(xml_file, output_csv, orientation)
    return anno_df


if __name__ == '__main__':
    PARAMS = load_config(config_path=os.path.join(os.path.dirname(__file__), 'detect_config_local.yaml'))
    labels_to_names = {}
    with open(PARAMS["classes"], mode='r') as inp:
        reader = csv.reader(inp)
        labels_to_names = {int(rows[1]): rows[0] for rows in reader}

    model = models.load_model(PARAMS["snapshot_path"], backbone_name='resnet50')

    for orientation in PARAMS['orientation']:
        DV_predict(model, PARAMS['path_to_data'], orientation, PARAMS["xml_file"])


