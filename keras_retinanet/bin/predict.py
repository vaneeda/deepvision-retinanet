import os
import sys
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.bin"
import pandas as pd
import numpy as np
import zipfile
import time
import glob
import tensorflow as tf
from keras_retinanet.paths import load_config
from pathlib import Path
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.bubbleplot import convert_annotations_to_num_instances_per_class, plot_save_bubbleplot_histogram



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


def predict_image(model, filename, min_img_size):
    image = read_image_bgr(filename)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=min_img_size)
    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    return boxes, scores, labels


def DV_predict_2018(model, path_to_data, orientation):
    anno = []
    t1 = time.time()
    path_to_data = os.path.join(path_to_data, orientation)
    for z in glob.glob(path_to_data+"/*active"):
        for i in os.listdir(z):
            date = int(i.split(".")[0])
            img = os.path.join(z, i)
            boxes, scores, labels = predict_image(model, img)
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score > 0.05:
                    anno_row = [date, 0, 0, 0, 0, 0, 0]
                    anno_row[1:5] = box
                    anno_row[5] = labels_to_names[label]
                    anno_row[6] = score
                    anno.append(anno_row)
    anno_df = pd.DataFrame(anno, columns=['datetime', 'x0', 'y0', 'x1', 'y1', 'label', 'score'])
    print("Prediction time per image", (time.time() - t1) / len(anno_df["datetime"].unique()))
    print("DV_predict processing time", time.time() - t1)
    return anno_df


def DV_predict(model, path_to_data, orientation):
    anno = []
    t1 = time.time()
    path_to_data = os.path.join(path_to_data, orientation)
    for z in glob.glob(path_to_data+"/*active.zip"):
        for i in zipfile.ZipFile(z, "r").namelist():
            date = int(i.split(".")[0])
            img = zipfile.ZipFile(z, "r").open(i)
            boxes, scores, labels = predict_image(model, img, PARAMS["min_img_size"])
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score > 0.05:
                    anno_row = [date, 0, 0, 0, 0, 0, 0]
                    anno_row[1:5] = box
                    anno_row[5] = labels_to_names[label]
                    anno_row[6] = score
                    anno.append(anno_row)
    anno_df = pd.DataFrame(anno, columns=['datetime', 'x0', 'y0', 'x1', 'y1', 'label', 'score'])
    print("Prediction time per image", (time.time() - t1) / len(anno_df["datetime"].unique()))
    print("DV_predict processing time", time.time() - t1)
    return anno_df


# Allow relative imports when being executed as script.
if __name__ == "__main__":
    PARAMS = load_config(config_path=os.path.join(os.path.dirname(__file__), 'detect_config_local_meso.yaml'))
    labels_to_names = PARAMS["classes"]
    model = models.load_model(PARAMS["snapshot_path"], backbone_name='resnet50', compile=False)
    model = models.convert_model(model)
#     model = models.convert_model(
#         model,
# #        nms=args.nms,
# #        class_specific_filter=args.class_specific_filter,
#         nms_threshold=args.nms_threshold,
# #        score_threshold=args.score_threshold,
#         # max_detections=args.max_detections,
#         # parallel_iterations=args.parallel_iterations
#     )

    model_name = os.path.basename(os.path.dirname(PARAMS["snapshot_path"]))
    num = os.path.basename(PARAMS["snapshot_path"]).split(".")[0][-2:]
    labels_to_names = PARAMS["classes"]
    #stations = os.listdir(args.path_to_data)
    stations = PARAMS["stations"]
    for station in stations:
        print("Station:", station)
        path_to_data = os.path.join(PARAMS["path_to_data"], station)
        path_to_xml_file = os.path.join(path_to_data, PARAMS["xml_file"])
        path_to_output = os.path.join(PARAMS["output"], station, model_name)
        makedirs(path_to_output)
        # filter = os.path.join(PARAMS["path_to_filter"],
        #                                   station + "_filtered.csv")
        for orientation in PARAMS["orientation"]:
            output_csv = os.path.join(path_to_output,
                                      station + "_" + orientation + "_predictions_inf_" + str(num) + ".csv")
            if not os.path.exists(output_csv):
                anno_df = DV_predict(model, path_to_data, orientation)
                anno_df = anno_df.sort_values(by="datetime")
                anno_df.to_csv(output_csv, index=False)
            else:
                anno_df = pd.read_csv(output_csv)

            if PARAMS["opt_thresholds"]:
                anno_df["score"] = [j if j > PARAMS["opt_thresholds"][i] else 0 for i, j in
                                          zip(anno_df["label"], anno_df["score"])]
                anno_df = anno_df[anno_df["score"] > 0]
                name = "_opt_thresholds"
            else:
                anno_df = anno_df[anno_df["score"] >= PARAMS["score_threshold"]]
                name = "_score_threshold_" + str(PARAMS["score_threshold"])

            pred, labels = convert_annotations_to_num_instances_per_class(anno_df)
            pred.to_csv(output_csv.split(".")[0] + name + "_one-hot.csv", index=False)
            path_to_save_images = os.path.join(path_to_output, station + "_" + orientation + name + "_inf_" + str(num))

            print(pred[labels].sum())
            plot_save_bubbleplot_histogram(pred, labels, path_to_xml_file, path_to_save_images)
