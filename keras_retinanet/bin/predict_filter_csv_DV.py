import os
import sys
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.bin"
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.read_zip_files import read_zip_files
from keras_retinanet.utils.filter_empty_img import filter_empty_img, filter_empty_img_zip
from keras_retinanet.paths import load_config
import numpy as np
import pandas as pd
import tensorflow as tf
import zipfile
import time
import xml.etree.ElementTree as ET
import csv
from keras_retinanet.utils.csv2xml import csv2xml

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set tf backend to allow memory to grow, instead of claiming everything


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


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


def select_data_from_xml_file(list_of_files, xml_file, pred_filter):
    df = dv_xml_to_csv(xml_file)
    image_list = [i[:17] for i in list_of_files]
    df = df[df["datetime"].isin(image_list)]

    if pred_filter is not None:
        filtered = pd.read_csv(pred_filter)
        filtered = filtered[["datetime", "Pred"]]
        filtered["datetime"] = filtered["datetime"].astype(int)
        df["datetime"] = df["datetime"].astype(int)
        df = pd.merge(df, filtered, on="datetime")
        df = df[df["Pred"] == "fish"]
        print("Number of fish images:", len(df))
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


def DV_predict(model, path_to_data, score_threshold, orientation, path_to_xml_file, pred_filter):
    path_to_data = os.path.join(path_to_data, orientation)
    dict, list_of_files = read_zip_files(path_to_data)
    df = select_data_from_xml_file(list_of_files, path_to_xml_file, pred_filter)
    anno = []
    t1 = time.time()

    output_csv = path_to_xml_file.split(".")[0] + "_" + orientation + ".csv"
    with open(output_csv, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['datetime', 'depth', 'x0', 'y0', 'x1', 'y1', 'label', 'score'])

        for ind in df.index:
            date = df["datetime"][ind]
            z = [key for key, value in dict.items() if str(date)+".jpg" in value]
            archive = zipfile.ZipFile(os.path.join(path_to_data, z[0]), "r")
            img = archive.open(str(date)+".jpg")
            boxes, scores, labels = predict_image(model, img)
            #anno_row = [date, df["depth"][ind], 0, 0, 0, 0, 0, 0]
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score > score_threshold:
                    anno_row = [date, df["depth"][ind], 0, 0, 0, 0, 0, 0]
                    anno_row[2:6] = box
                    anno_row[6] = labels_to_names[label]
                    anno_row[7] = score
                    writer.writerow(anno_row)
                    anno.append(anno_row)
                else:
                    anno.append([date, df["depth"][ind], 0, 0, 0, 0, 0, 0])
        anno_df = pd.DataFrame(anno, columns=['datetime', 'depth', 'x0', 'y0',
                                              'x1', 'y1', 'label', 'score'])
    print("Prediction time per image", (time.time() - t1) / len(df))
    anno_df.to_csv(output_csv, index=False)
    print("DV_predict processing time", time.time() - start)
    return output_csv


if __name__ == '__main__':
    start = time.time()
    PARAMS = load_config(config_path=os.path.join(os.path.dirname(__file__),
                                                  'detect_config_predict_filter_csv_DV.yaml'))
    snapshot_path = PARAMS["snapshot_path"]
    labels_to_names = PARAMS["classes"]

    stations = PARAMS["stations"] if PARAMS["stations"] else os.listdir(PARAMS['path_to_data'])
    for station in stations:
        print("Station:", station)
        path_to_data = os.path.join(PARAMS['path_to_data'], station)
        path_to_xml_file = os.path.join(path_to_data, PARAMS["xml_file"])
        path_to_output = os.path.join(PARAMS["output"], station)
        pred_filter = os.path.join(PARAMS["output"], station, station + "_filtered.csv")
        makedirs(os.path.dirname(pred_filter))
        csv_file_paths = []
        for orientation in PARAMS['orientation']:
            if os.path.isfile(pred_filter):
                print("Images have already been filtered")
            else:
                filter_empty_img_zip(PARAMS["path_to_filter_snapshot"],
                                     os.path.join(path_to_data, orientation), pred_filter)
            csvpath = DV_predict(model, path_to_data, PARAMS['score_threshold'],
                                 orientation, path_to_xml_file, pred_filter)
            csv_file_paths.append(csvpath)
        csv2xml(path_to_xml_file, csv_file_paths, PARAMS['orientation'])
