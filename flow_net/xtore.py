import os
import glob
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import cv2
import joblib
import re
from tqdm import tqdm
h,w = 256, 256
def events2Tore3C(events, k, frameSize):
    x = events[:,0]
    y = events[:,1]
    ts = events[:,2]
    ts = ts - np.min(ts)
    max_time = np.max(ts)
    sampleTimes = [max_time]
    toreFeature = np.inf * np.ones((frameSize[0], frameSize[1], k))
    Xtore = np.zeros((frameSize[0], frameSize[1], k, len(sampleTimes)), dtype=np.single)
    priorSampleTime = -np.inf

    for sampleLoop, currentSampleTime in enumerate(sampleTimes):
        addEventIdx = (ts >= priorSampleTime) & (ts < currentSampleTime)

        newTore = np.full((frameSize[0], frameSize[1], k), np.inf)
        for i, j, t in zip(x[addEventIdx], y[addEventIdx], ts[addEventIdx]):
            i = frameSize[0]-1-i
            j = frameSize[1]-1-j
            newTore[i, j] = np.sort(np.partition(np.append(newTore[i, j], currentSampleTime - t), k)[:k])

        toreFeature += (currentSampleTime - priorSampleTime)
        toreFeature = np.sort(np.concatenate((toreFeature, newTore), axis=2), axis=2)[:, :, :k]

        Xtore[:, :, :, sampleLoop] = toreFeature.astype(np.single)

        priorSampleTime = currentSampleTime

    # Scale the Tore surface
    minTime = 150
    maxTime = 5e6

    Xtore[np.isnan(Xtore)] = maxTime
    Xtore[Xtore > maxTime] = maxTime

    Xtore = np.log(Xtore + 1)
    Xtore = Xtore - np.log(minTime + 1)
    Xtore[Xtore < 0] = 0

    return Xtore

def convert_events_to_tore(csv_file, output_dir):
    ac = re.search(r'subject\d+_group\d+_time\d+', csv_file).group(0)
    filename = os.path.basename(csv_file).replace('.csv', '.npy')
    output_path = os.path.join(output_dir, ac, filename)
    # try:
    #     _ = np.load(output_path)
    #     return
    # except Exception as e:
    #     print(f"Corrupted file ：{output_path}")
    #     os.remove(output_path)
    # if os.path.exists(output_path):
    #     print(f"File {output_path} already exists. Skipping conversion.")
    #     return
    subject_output_dir = os.path.dirname(output_path)
    os.makedirs(subject_output_dir, exist_ok=True)
    events = pd.read_csv(csv_file, header=None, dtype=np.float32,
                         names=['v', 'u', 'in_pixel_time', 'off_pixel_time', 'polarity'])
    events.dropna(inplace=True)
    events['in_pixel_time'] = events['in_pixel_time'].astype(np.int32)
    events['u'] = events['u'].astype(np.int32)
    events['v'] = events['v'].astype(np.int32)
    events = events[(events['v'] <= 1279) &
                    (events['u'] <= 799)]
    events = events[['v', 'u', 'in_pixel_time']].values
    sorted_indices = np.argsort(events[:, 2])
    events = events[sorted_indices]


    img = events2Tore3C(events, 4, [1280, 800]).squeeze(-1)
    img = np.pad(img, ((0, 0), (240, 240), (0, 0)), 'constant')  # [1280, 1280, 3]
    img_resize = cv2.resize(img, (h, w), interpolation=cv2.INTER_NEAREST)
    arr_normalized = (img_resize / 10.407669)  # Normalize to the [0, 255] range
    # output_path = 'output_image.png'
    # cv2.imwrite(output_path, (arr_normalized*255).astype(np.uint8))
    np.save(output_path, arr_normalized)



def process_subject_files(subject_dir, output_base_dir):
    """
    Process all CSV files for each subject folder independently.
    """
    csv_files = glob.glob(os.path.join(subject_dir, '**/event_camera/events/event*.csv'), recursive=True)
    # Use tqdm to show progress while processing CSV files per subject
    for csv_file in tqdm(csv_files, desc=f"Processing {subject_dir}", ncols=100):
        convert_events_to_tore(csv_file, output_base_dir)
    # print(f"complete {subject_dir}")


def process_all_subjects(base_dir, output_base_dir):
    """
    Use Parallel to speed up CSV processing for each subject folder.
    """
    subject_dirs = glob.glob(os.path.join(base_dir, 'subject*/'))  # Get all subject directories
    # Use Parallel to process CSV files in each subject folder
    Parallel(n_jobs=-1)(delayed(process_subject_files)(subject_dir, output_base_dir) for subject_dir in
                       tqdm(subject_dirs, desc="Processing Subjects", ncols=100))


if __name__ == "__main__":
    # # Define base input and output paths
    base_dir = "../data_event/data_event_raw"
    output_base_dir = "../data_event/data_event_out/xtore_256"
    #
    # # Start processing all subject folders
    process_all_subjects(base_dir, output_base_dir)
    # convert_events_to_tore('../data_event/data_event_raw/subject01_group1_time3/event_camera/events/event0015.csv',output_base_dir)
