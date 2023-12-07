from pylab import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset':'dejavuserif'})
from matplotlib import gridspec
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def convert_annotations_to_num_instances_per_class(pred):
    """Takes annotations file and score threshold and returns csv file with number of instances per class """

    integer_encoded = LabelEncoder().fit_transform(pred["label"])
    encoded = to_categorical(integer_encoded)
    fish_species = sorted(pred["label"].unique())
    for i, fish in enumerate(fish_species):
        pred[fish] = encoded[:, i]
    pred = pred[['datetime']+fish_species]
    pred = pred.groupby(by=["datetime"])[fish_species].sum()
    pred = pred.reset_index()
    return pred, fish_species


def dv_xml_to_csv(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    header = ('datetime', 'time_elapsed', 'folder', 'depth', 'active')
    table = []
    count = 0
    for element in root.iter("frames"):
        for frame in element:
            active = frame.get("active")
            depth = frame.get("depth").replace(",", ".")
            time = frame.get('time')
            folder = frame.get('folder')
            t = datetime((int(time[:4])), int(time[4:6]), int(time[6:8]), int(time[8:10]),
                         int(time[10:12]), int(time[12:14]), int(time[14:]) * 1000)
            if count == 0:
                t0 = t
            time_elapsed = (t - t0).total_seconds()
            row = time, time_elapsed, folder, -float(depth), active
            table.append(row)
            count += 1

    out_df = pd.DataFrame(table, columns=header)
    #csv_file = xml_file.split(".")[0]+".csv"
    #out_df.to_csv(csv_file,index=False)
    return out_df


def prepare_csv(pred, path_to_xml_file):
    deep_vision = dv_xml_to_csv(path_to_xml_file)
    deep_vision = deep_vision.drop(columns=["active"])
    deep_vision["datetime"] = pd.to_numeric(deep_vision["datetime"])
    pred["datetime"] = pd.to_numeric(pred["datetime"])
    pred = pd.merge(deep_vision, pred, on="datetime", how="outer")
    pred.fillna(0, inplace=True)
    pred = pred.reset_index()
    return pred


def plot_save_bubbleplot_histogram(pred, classes, cm, path_to_xml_file, path_to_save_images=None):
    pred = prepare_csv(pred, path_to_xml_file)
    pred["time (min)"] = (pred['time_elapsed'] / 60).astype(int)
    ST = pred.groupby('time (min)')[classes].sum()
    ST = ST.reset_index()
    figure = plt.figure(figsize=(16, 10))
    if "depth" not in pred.columns:
        ST.plot('time (min)', classes, kind='bar', stacked=True, color=cm, fontsize=6)
    else:
        ST["avg_depth"] = pred.groupby('time (min)')[['depth']].mean()
        ST = ST.reset_index()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax0 = plt.subplot(gs[0])
        #ax0.set_title("Station "+str(station))
        ST.plot('time (min)', classes, kind='bar', stacked=True, color=cm, ax=ax0, fontsize=6)
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ST.plot(kind='line', x="time (min)", y='avg_depth', color='gray', ax=ax1,alpha=0.3, fontsize=10)
        for i in classes:
            sct = scatter(ST["time (min)"], ST["avg_depth"], c=cm[i],
                  s=ST[i], linewidths=2, edgecolor=cm[i], label=i)
        sct.set_alpha(0.5)

        ax0.set_xlabel('Time elapsed (min)',fontsize=20)
        ax0.set_ylabel('Fish count per min',fontsize=20)
        ax0.set_xlim(xmin=0)
        ax0.xaxis.set_major_locator(MultipleLocator(5))
        # For the minor ticks, use no labels; default NullFormatter.
        ax0.xaxis.set_minor_locator(MultipleLocator(1))
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        # ax0.yaxis.set_major_locator(MultipleLocator(100))
        # ax0.yaxis.set_minor_locator(MultipleLocator(50))
        # ax0.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax1.set_ylabel('Depth (m)')
        # ax1.yaxis.set_major_locator(MultipleLocator(100))
        # ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        #plt.setp(ax0.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)

    if path_to_save_images is not None:
       # plt.savefig(path_to_save_images + score_folder +
       #          '/ST'+str(station) + '_histogram+bubbleplot_hq_transparent.svg',
       #      dpi=300, bbox_inches='tight',transparent=True)
       plt.savefig(path_to_save_images + '_histogram+bubbleplot_hq_transparent.svg',
            dpi=300, bbox_inches='tight',transparent=True)
    else:
       plt.show()
