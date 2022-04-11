import numpy as np
from matplotlib import pyplot as plt
import os


def calculate_F1_precision_recall(thr, num_classes, num_annotations, all_scores, all_true_positives):
    """
    Calculates F1_scores, precision, recall for a given score threshold, for each class and for all classes

    :param thr : float (between 0 and 1)
                 Score threshold
    :param num_classes:  int
                         Number of classes
    :param num_annotations: list
                         List of number of annotations per class,
    :param all_scores: dict
                       Dictionary of scores for all instances detected, one array per class
    :param all_true_positives: dict
                       Dictionary of true positives, one array per class

    :return: np.arrays
             F1_score, precision, recall
    """

    TP = np.zeros(num_classes)
    num_detections = np.zeros(num_classes)
    precision = np.zeros(num_classes+1)
    recall = np.zeros(num_classes+1)
    F1_score = np.zeros(num_classes + 1)

    for label in range(num_classes):
        TP[label] = sum(all_true_positives[label][all_scores[label] > thr])
        num_detections[label] = len(all_scores[label][all_scores[label] > thr])
        precision[label] = TP[label] / num_detections[label]
        recall[label] = TP[label] / num_annotations[label]
        F1_score[label] = 2 * TP[label] / (num_annotations[label] + num_detections[label])

    F1_score[num_classes] = 2 * np.sum(TP) / (sum(num_annotations) + np.sum(num_detections))
    precision[num_classes] = np.sum(TP) / np.sum(num_detections)
    recall[num_classes] = np.sum(TP) / sum(num_annotations)
    return F1_score, precision, recall


def plot_each_species(score_range, x, label_dict, var_name, dataset_path, opt_score_thres=None):
    colours = ['blue', 'red', 'green', 'orange']
    dataset = (os.path.basename(dataset_path)).split(".")[0]
    plt.figure()
    for key, value in label_dict.items():
        if opt_score_thres is not None:
            label = value + " (opt. score thr. = {0:.3f})".format(opt_score_thres[key])
        else:
            label = value
        plt.plot(score_range, x[key], c=colours[key], label= label)
    plt.xlabel('Score threshold', fontsize=12)
    plt.ylabel(var_name, fontsize=12)
    plt.title(var_name + " for each species")
    plt.legend()
    path_to_save_plot = os.path.dirname(dataset_path)+"/plots/"
    if not os.path.isdir(path_to_save_plot):
        os.makedirs(path_to_save_plot)
    plt.savefig(path_to_save_plot+var_name+"_"+str(dataset))


def plot_F1_precision_recall(score_range, F1_score, precision, recall, num_classes, dataset_path):
    dataset = (os.path.basename(dataset_path)).split(".")[0]
    plt.figure()
    optimal_score_threshold = score_range[np.argmax(F1_score[num_classes])]
    plt.plot(score_range, F1_score[num_classes], c='black',label="F1_score (opt. score thr. = {0:.3f})".format(optimal_score_threshold))
    plt.plot(score_range, precision[num_classes], c='cyan',label="precision")
    plt.plot(score_range, recall[num_classes], c='magenta',label="recall")
    plt.xlabel('Score threshold', fontsize=12)

    plt.title("All species: F1, Precision, Recall")
    plt.legend()
    path_to_save_plot = os.path.dirname(dataset_path) + "/plots/"
    if not os.path.isdir(path_to_save_plot):
        os.makedirs(path_to_save_plot)
    plt.savefig(path_to_save_plot+"/All_species_F1_precision_recall_" + str(dataset))
    plt.show()