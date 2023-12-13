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
            t = datetime.strptime(str(time), "%Y%m%d%H%M%S%f")
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
    pred["t (min)"] = (pred['time_elapsed'] / 60).astype(int)
    pred = pred.drop(columns=["folder", "time_elapsed"])
    cl = []
    dic = {"datetime": [], "t (min)": [], "avg_depth": []}
    for idx, group in pred.groupby("t (min)"):
        dic["datetime"].append(group["datetime"].iloc[0])
        dic["t (min)"].append(idx)
        dic["avg_depth"].append(group["depth"].mean())
        v = [i for i in group[classes].sum().to_dict().values()]
        cl.append(v)
    df = pd.DataFrame(dic)
    df[classes] = cl

    df["time"] = [datetime.strptime(str(t), "%Y%m%d%H%M%S%f").time().isoformat(timespec='minutes') for t in df["datetime"]]

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax0 = plt.subplot(gs[0])
    #ax0.set_title("Station "+str(station))
    df.plot("time", classes, kind='bar', stacked=True, color=cm, ax=ax0, fontsize=6)
    ax1 = plt.subplot(gs[1], sharex=ax0)

    df.plot(kind='line', x="time", y='avg_depth', rot=90, color='gray', ax=ax1, alpha=0.3, fontsize=10)
    for i in classes:
        sct = scatter(df["time"], df["avg_depth"], c=cm[i],
                      s=df[i], linewidths=2, edgecolor=cm[i], label=i)
    sct.set_alpha(0.5)
    ax0.set_xlabel('time')
    ax0.set_ylabel('Fish count per min')
    ax0.set_xlim(xmin=0)
    ax1.set_ylabel('Depth (m)')
    ax1.set_ylim(ymax=0)
    ax0.xaxis.set_minor_locator(MultipleLocator(1))
    interval_in_sec = 5
    start_min = datetime.strptime(str(df["datetime"].iloc[0]), "%Y%m%d%H%M%S%f").minute % 10
    if start_min == 0:
        ii = 0
    else:
        if 1 <= start_min <= interval_in_sec:
            ii = interval_in_sec - start_min
        else:
            ii = 10 - start_min

    x_ticks = ax0.get_xticks()
    ax0.set_xticks(x_ticks[ii::interval_in_sec])
    #  Display every nth x-tick label
    ax0.set_xticklabels(df['time'][ii::interval_in_sec])

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.subplots_adjust(hspace=.0)

    if path_to_save_images is not None:
       plt.savefig(path_to_save_images + '_histogram+bubbleplot_hq_transparent.svg',
           transparent=True)
    else:
       plt.show()


