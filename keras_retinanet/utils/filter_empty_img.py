from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense
import numpy as np
import pandas as pd
import time
import os
import zipfile
from keras_retinanet.utils.read_zip_files import read_zip_files


def filter_empty_img(path_to_checkpoint, path_to_data, pred_filter):
    """ Filters images into "empty", "fish" and "krill" classes
    Parameters
    ----------
    :param path_to_checkpoint: path to weights (trained on MobileNet)
    :param path_to_data: path to images to be filtered
    :param pred_filter: path to csv results file (to be created)

    :return: csv file with predictions

    """
    start= time.time()
    label_map = {0: 'empty', 1: 'fish', 2: 'krill'}
    # create the base pre-trained model
    base_model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3,
                           include_top=False, weights='imagenet', input_tensor=None, pooling='avg')

    # add a global spatial average pooling layer
    x = base_model.output
    test_pred = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=test_pred)
    model.load_weights(str(path_to_checkpoint))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    img_width, img_height = 256, 256
    #img_width, img_height = 417, 350  #Use with geometrically corrected images

    if any(fname.endswith(ext) for ext in ['jpg', 'png'] for fname in os.listdir(path_to_data)):
        images = []
        for img in os.listdir(path_to_data):
            img = os.path.join(path_to_data, img)
            img = image.load_img(img, target_size=(img_width, img_height))
            img = image.img_to_array(img)
            img /= 255
            img = np.expand_dims(img, axis=0)
            images.append(img)

        # stack up images list to pass for prediction
        images = np.vstack(images)
        predictions_prob = model.predict(images, batch_size=10)
        prob = np.array(predictions_prob)
        datetime = [i[:17] for i in os.listdir(path_to_data)]

    else:
        datagen = image.ImageDataGenerator(rescale=1. / 255)
        data_generator = datagen.flow_from_directory(path_to_data, shuffle=False)
        predictions_prob = model.predict(data_generator)
        prob = np.array(predictions_prob)
        datetime = [(i.split("/")[1])[:17] for i in data_generator.filenames]

    predictions_classes = np.argmax(predictions_prob, axis=-1)  # multiple categories
    predictions = [label_map[k] for k in predictions_classes]
    df = pd.DataFrame(prob, columns=['p(' + label_map[k] + ')' for k in label_map])
    df.insert(0, 'datetime', datetime)
    df['Pred'] = predictions
    print("Total filtering time:", time.time()-start)
    df.to_csv(pred_filter, index=None)
    return df


def filter_empty_img_zip(path_to_checkpoint, path_to_data, pred_filter):
    """ Filters images into "empty", "fish" and "krill" classes
    Parameters
    ----------
    :param path_to_checkpoint: path to weights (trained on MobileNet)
    :param path_to_data: path to images to be filtered
    :param pred_filter: path to csv results file (to be created)

    :return: csv file with predictions

    """
    from PIL import Image
    start = time.time()
    label_map = {0: 'empty', 1: 'fish', 2: 'krill'}
    img_width, img_height = 256, 256
    # create the base pre-trained model
    base_model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3,
                           include_top=False, weights='imagenet', input_tensor=None, pooling='avg')

    # add a global spatial average pooling layer
    x = base_model.output
    test_pred = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=test_pred)
    model.load_weights(str(path_to_checkpoint))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    dict, list_of_files = read_zip_files(path_to_data)
    predictions_prob_list = []
    for i in list_of_files:
        z = [key for key, value in dict.items() if str(i) in value]
        archive = zipfile.ZipFile(os.path.join(path_to_data, z[0]), "r")
        img = archive.open(str(i))
        img = Image.open(img)
        img = img.resize((img_width, img_height))
        img = image.img_to_array(img)
        img /= 255
        img = np.expand_dims(img, axis=0)
        predictions_prob = model.predict(img)
        predictions_prob_list.append(predictions_prob[0])
    prob = np.array(predictions_prob_list)

    datetime = [i[:17] for i in list_of_files]
    predictions_classes = np.argmax(prob, axis=-1)  # multiple categories
    predictions = [label_map[k] for k in predictions_classes]
    df = pd.DataFrame(prob, columns=['p(' + label_map[k] + ')' for k in label_map])
    df.insert(0, 'datetime', datetime)
    df['Pred'] = predictions
    print("Total filtering time:", time.time() - start)
    df.to_csv(pred_filter, index=None)
    return df
