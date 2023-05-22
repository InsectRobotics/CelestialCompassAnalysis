"""
create_csv.py

Brittle script to produce all plots from the tod and canopy recordings. The
directory structure is assumed so this will break if anything changes.
"""

import image_processing as imp
import models as md
import _rosbag as rb
import analysis

import numpy as np
import pandas as pd
import os

default_nb_samples = 360

data_base = os.path.abspath(os.path.join(os.getcwd(), '..'))
csv_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'csv'))

default_raw_file = 'raw_dataset.csv'
default_pooled_file = 'pooled_dataset.csv'
default_error_file = 'error_dataset.csv'
default_collections = ['sardinia_data', 'south_africa_data']
skip_dirs = ["elevation", "path_integration", "sun_tables", "track_test"]
solar_visibility = [
    "visible",
    "covered",  # inferable
    "hidden"
]
anti_solar_visibility = [
    "visible",
    "covered",  # information not completely corrupted
    "hidden"
]
clouds_level = [
    "clear",
    "thin broken",
    "thick broken",
    "thin solid",
    "thin uniform",
    "mixed broken",
    "thick solid",
    "thick uniform"
]
canopy_level = [
    "clear",
    "trees on horizon",
    "trees at zenith",
    "trees with openings",
    "trees on one side",
    "building on one side"
]

sky_conditions = {  # sun status, anti-sun status, clouds level, canopy level
    'Friday_13-05-22_18-00-12_CEST': [0, 0, 0, 0],
    'Friday_13-05-22_18-30-11_CEST': [0, 0, 0, 0],
    'Friday_13-05-22_19-00-23_CEST': [0, 0, 0, 0],
    'Friday_13-05-22_19-30-09_CEST': [0, 0, 0, 0],
    'Friday_13-05-22_20-00-18_CEST': [0, 0, 0, 0],
    'Friday_13-05-22_20-30-23_CEST': [0, 0, 0, 0],
    'Friday_13-05-22_21-00-22_CEST': [0, 0, 0, 0],
    'Monday_16-05-22_11-19-06_CEST': [1, 0, 3, 2],
    'Saturday_14-05-22_06-32-27_CEST': [0, 0, 0, 0],
    'Saturday_14-05-22_07-00-12_CEST': [0, 0, 0, 0],
    'Saturday_14-05-22_07-30-13_CEST': [0, 0, 0, 0],
    'Saturday_14-05-22_08-00-13_CEST': [0, 0, 0, 0],
    'Saturday_14-05-22_08-30-18_CEST': [0, 0, 0, 0],
    'Saturday_14-05-22_09-00-11_CEST': [0, 0, 0, 0],
    'Sunday_15-05-22_17-05-48_CEST': [2, 0, 0, 3],
    'Sunday_15-05-22_17-41-05_CEST': [2, 0, 0, 3],
    'Thursday_12-05-22_09-08-12_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_10-00-48_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_11-01-03_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_12-03-06_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_13-00-12_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_14-09-27_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_15-09-27_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_16-01-54_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_17-01-44_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_18-00-11_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_19-00-20_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_20-02-32_CEST': [0, 0, 0, 0],
    'Thursday_12-05-22_21-03-00_CEST': [0, 0, 0, 0],
    'Thursday_19-05-22_11-30-21_CEST': [0, 0, 0, 0],
    'Thursday_19-05-22_12-00-16_CEST': [0, 0, 0, 0],
    'Thursday_19-05-22_12-30-11_CEST': [0, 0, 0, 0],
    'Thursday_19-05-22_13-00-10_CEST': [0, 0, 0, 0],
    'Thursday_19-05-22_13-30-25_CEST': [0, 0, 0, 0],
    'Thursday_19-05-22_14-00-15_CEST': [0, 0, 0, 0],
    'Thursday_19-05-22_14-30-14_CEST': [0, 0, 0, 0],
    'Thursday_19-05-22_15-00-14_CEST': [0, 0, 0, 0],
    'Thursday_19-05-22_16-33-48_CEST': [0, 2, 0, 4],
    'Thursday_19-05-22_16-54-50_CEST': [2, 0, 0, 4],
    'Thursday_19-05-22_17-20-24_CEST': [0, 2, 0, 5],

    'Friday_11-11-22_14-25-46_SAST': [1, 2, 6, 1],
    'Friday_11-11-22_16-01-02_SAST': [2, 2, 6, 1],
    'Friday_18-11-22_09-22-15_SAST': [1, 0, 3, 0],
    'Friday_18-11-22_09-34-45_SAST': [1, 0, 3, 0],
    'Monday_14-11-22_07-35-27_SAST': [0, 0, 0, 1],
    'Monday_14-11-22_08-31-21_SAST': [0, 0, 0, 1],
    'Monday_14-11-22_09-31-28_SAST': [0, 0, 0, 1],
    'Monday_14-11-22_10-33-11_SAST': [0, 0, 1, 1],
    'Monday_14-11-22_11-30-27_SAST': [0, 0, 1, 1],
    'Monday_14-11-22_12-32-42_SAST': [0, 0, 1, 1],
    'Monday_14-11-22_13-31-22_SAST': [0, 0, 2, 1],
    'Monday_14-11-22_14-30-10_SAST': [0, 0, 1, 1],
    'Monday_14-11-22_15-32-43_SAST': [0, 0, 1, 1],
    'Monday_14-11-22_16-31-03_SAST': [0, 0, 1, 1],
    'Monday_14-11-22_17-30-59_SAST': [0, 0, 0, 1],
    'Monday_14-11-22_18-30-49_SAST': [0, 0, 0, 1],
    'Monday_14-11-22_19-30-02_SAST': [0, 0, 0, 1],
    'Monday_21-11-22_05-31-37_SAST': [1, 1, 4, 0],
    'Monday_21-11-22_06-00-11_SAST': [1, 1, 4, 0],
    'Monday_21-11-22_06-31-27_SAST': [2, 2, 7, 0],
    'Monday_21-11-22_07-01-53_SAST': [2, 2, 7, 0],
    'Monday_21-11-22_08-01-37_SAST': [2, 2, 7, 0],
    'Monday_21-11-22_14-26-58_SAST': [0, 0, 3, 0],
    'Monday_28-11-22_08-29-06_SAST': [2, 1, 6, 2],
    'Monday_28-11-22_08-48-02_SAST': [1, 2, 6, 1],
    'Monday_28-11-22_13-36-03_SAST': [1, 2, 6, 1],
    'Saturday_12-11-22_10-20-48_SAST': [1, 1, 4, 0],
    'Saturday_12-11-22_11-18-20_SAST': [1, 1, 4, 0],
    'Saturday_12-11-22_13-35-07_SAST': [1, 1, 5, 0],
    'Saturday_12-11-22_14-31-01_SAST': [1, 1, 4, 0],
    'Saturday_12-11-22_15-23-06_SAST': [2, 2, 7, 0],
    'Saturday_12-11-22_16-16-00_SAST': [1, 0, 1, 0],
    'Saturday_26-11-22_10-17-03_SAST': [1, 1, 4, 0],
    'Saturday_26-11-22_14-03-34_SAST': [1, 0, 3, 0],
    'Saturday_26-11-22_16-02-49_SAST': [0, 0, 6, 0],
    'Sunday_13-11-22_08-01-27_SAST': [0, 0, 0, 0],
    'Sunday_13-11-22_09-00-04_SAST': [0, 1, 1, 0],
    'Sunday_13-11-22_10-00-06_SAST': [0, 0, 0, 0],
    'Sunday_13-11-22_11-00-09_SAST': [0, 0, 0, 0],
    'Sunday_13-11-22_12-00-05_SAST': [1, 1, 1, 0],
    'Sunday_13-11-22_13-00-06_SAST': [1, 0, 2, 0],
    'Sunday_13-11-22_14-00-04_SAST': [0, 0, 2, 0],
    'Sunday_13-11-22_15-00-05_SAST': [0, 0, 2, 0],
    'Sunday_13-11-22_16-00-03_SAST': [0, 0, 2, 0],
    'Sunday_13-11-22_17-00-05_SAST': [0, 0, 1, 0],
    'Sunday_20-11-22_14-28-16_SAST': [0, 0, 2, 0],
    'Sunday_20-11-22_14-36-46_SAST': [0, 0, 2, 0],
    'Sunday_20-11-22_14-45-36_SAST': [0, 0, 2, 0],
    'Sunday_27-11-22_15-14-02_SAST': [1, 1, 2, 3],
    'Sunday_27-11-22_15-30-38_SAST': [0, 2, 2, 2],
    'Sunday_27-11-22_15-46-11_SAST': [0, 2, 2, 2],
    'Sunday_27-11-22_16-02-54_SAST': [1, 1, 5, 4],
    'Tuesday_22-11-22_10-23-15_SAST': [2, 2, 7, 0],
    'Tuesday_22-11-22_13-07-52_SAST': [1, 1, 5, 0],
    'Tuesday_22-11-22_14-16-59_SAST': [1, 1, 4, 0],
    'Wednesday_23-11-22_11-00-00_SAST': [2, 1, 6, 0],
    'Wednesday_23-11-22_12-01-04_SAST': [0, 2, 6, 0],
    'Wednesday_23-11-22_13-01-47_SAST': [0, 2, 6, 0]
}
tilt_condition = {
    'Friday_13-05-22_18-00-12_CEST': 0.00,
    'Friday_13-05-22_18-30-11_CEST': 0.00,
    'Friday_13-05-22_19-00-23_CEST': 0.00,
    'Friday_13-05-22_19-30-09_CEST': 0.00,
    'Friday_13-05-22_20-00-18_CEST': 0.00,
    'Friday_13-05-22_20-30-23_CEST': 0.00,
    'Friday_13-05-22_21-00-22_CEST': 0.00,
    'Monday_16-05-22_11-19-06_CEST': 0.00,
    'Saturday_14-05-22_06-32-27_CEST': 0.00,
    'Saturday_14-05-22_07-00-12_CEST': 0.00,
    'Saturday_14-05-22_07-30-13_CEST': 0.00,
    'Saturday_14-05-22_08-00-13_CEST': 0.00,
    'Saturday_14-05-22_08-30-18_CEST': 0.00,
    'Saturday_14-05-22_09-00-11_CEST': 0.00,
    'Sunday_15-05-22_17-05-48_CEST': 0.00,
    'Sunday_15-05-22_17-41-05_CEST': 0.00,
    'Thursday_12-05-22_09-08-12_CEST': 0.00,
    'Thursday_12-05-22_10-00-48_CEST': 0.00,
    'Thursday_12-05-22_11-01-03_CEST': 0.00,
    'Thursday_12-05-22_12-03-06_CEST': 0.00,
    'Thursday_12-05-22_13-00-12_CEST': 0.00,
    'Thursday_12-05-22_14-09-27_CEST': 0.00,
    'Thursday_12-05-22_15-09-27_CEST': 0.00,
    'Thursday_12-05-22_16-01-54_CEST': 0.00,
    'Thursday_12-05-22_17-01-44_CEST': 0.00,
    'Thursday_12-05-22_18-00-11_CEST': 0.00,
    'Thursday_12-05-22_19-00-20_CEST': 0.00,
    'Thursday_12-05-22_20-02-32_CEST': 0.00,
    'Thursday_12-05-22_21-03-00_CEST': 0.00,
    'Thursday_19-05-22_11-30-21_CEST': 0.00,
    'Thursday_19-05-22_12-00-16_CEST': 0.00,
    'Thursday_19-05-22_12-30-11_CEST': 0.00,
    'Thursday_19-05-22_13-00-10_CEST': 0.00,
    'Thursday_19-05-22_13-30-25_CEST': 0.00,
    'Thursday_19-05-22_14-00-15_CEST': 0.00,
    'Thursday_19-05-22_14-30-14_CEST': 0.00,
    'Thursday_19-05-22_15-00-14_CEST': 0.00,
    'Thursday_19-05-22_16-33-48_CEST': 0.00,
    'Thursday_19-05-22_16-54-50_CEST': 0.00,
    'Thursday_19-05-22_17-20-24_CEST': 0.00,

    'Friday_11-11-22_14-25-46_SAST': 0.00,
    'Friday_11-11-22_16-01-02_SAST': 0.00,
    'Friday_18-11-22_09-22-15_SAST': 0.00,
    'Friday_18-11-22_09-34-45_SAST': 0.00,
    'Monday_14-11-22_07-35-27_SAST': 0.00,
    'Monday_14-11-22_08-31-21_SAST': 0.00,
    'Monday_14-11-22_09-31-28_SAST': 0.00,
    'Monday_14-11-22_10-33-11_SAST': 0.00,
    'Monday_14-11-22_11-30-27_SAST': 0.00,
    'Monday_14-11-22_12-32-42_SAST': 0.00,
    'Monday_14-11-22_13-31-22_SAST': 0.00,
    'Monday_14-11-22_14-30-10_SAST': 0.00,
    'Monday_14-11-22_15-32-43_SAST': 0.00,
    'Monday_14-11-22_16-31-03_SAST': 0.00,
    'Monday_14-11-22_17-30-59_SAST': 0.00,
    'Monday_14-11-22_18-30-49_SAST': 0.00,
    'Monday_14-11-22_19-30-02_SAST': 0.00,
    'Monday_21-11-22_05-31-37_SAST': 0.00,
    'Monday_21-11-22_06-00-11_SAST': 0.00,
    'Monday_21-11-22_06-31-27_SAST': 0.00,
    'Monday_21-11-22_07-01-53_SAST': 0.00,
    'Monday_21-11-22_08-01-37_SAST': 0.00,
    'Monday_21-11-22_14-26-58_SAST': 0.00,
    'Monday_28-11-22_08-29-06_SAST': 0.00,
    'Monday_28-11-22_08-48-02_SAST': 0.00,
    'Monday_28-11-22_13-36-03_SAST': 0.00,
    'Saturday_12-11-22_10-20-48_SAST': 0.00,
    'Saturday_12-11-22_11-18-20_SAST': 0.00,
    'Saturday_12-11-22_13-35-07_SAST': 0.00,
    'Saturday_12-11-22_14-31-01_SAST': 0.00,
    'Saturday_12-11-22_15-23-06_SAST': 0.00,
    'Saturday_12-11-22_16-16-00_SAST': 0.00,
    'Saturday_26-11-22_10-17-03_SAST': 0.00,
    'Saturday_26-11-22_14-03-34_SAST': 0.00,
    'Saturday_26-11-22_16-02-49_SAST': 0.00,
    'Sunday_13-11-22_08-01-27_SAST': 0.00,
    'Sunday_13-11-22_09-00-04_SAST': 0.00,
    'Sunday_13-11-22_10-00-06_SAST': 0.00,
    'Sunday_13-11-22_11-00-09_SAST': 0.00,
    'Sunday_13-11-22_12-00-05_SAST': 0.00,
    'Sunday_13-11-22_13-00-06_SAST': 0.00,
    'Sunday_13-11-22_14-00-04_SAST': 0.00,
    'Sunday_13-11-22_15-00-05_SAST': 0.00,
    'Sunday_13-11-22_16-00-03_SAST': 0.00,
    'Sunday_13-11-22_17-00-05_SAST': 0.00,
    'Sunday_20-11-22_14-28-16_SAST': 0.20,
    'Sunday_20-11-22_14-36-46_SAST': 4.40,
    'Sunday_20-11-22_14-45-36_SAST': 8.00,
    'Sunday_27-11-22_15-14-02_SAST': 0.00,
    'Sunday_27-11-22_15-30-38_SAST': 0.00,
    'Sunday_27-11-22_15-46-11_SAST': 0.00,
    'Sunday_27-11-22_16-02-54_SAST': 0.00,
    'Tuesday_22-11-22_10-23-15_SAST': 0.00,
    'Tuesday_22-11-22_13-07-52_SAST': 0.00,
    'Tuesday_22-11-22_14-16-59_SAST': 0.00,
    'Wednesday_23-11-22_11-00-00_SAST': 0.00,
    'Wednesday_23-11-22_12-01-04_SAST': 0.00,
    'Wednesday_23-11-22_13-01-47_SAST': 0.00
}


def read_csv_dataset(*data_set, in_file=None, remove_outliers=False):
    if in_file is None:
        in_file = default_raw_file

    if os.path.exists(in_file):
        data_dir = os.path.dirname(in_file)
        in_file = in_file.split(os.path.sep)[-1]
    else:
        data_dir = csv_dir

    data = pd.read_csv(os.path.join(data_dir, in_file), low_memory=False)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    s = np.clip(data[["I135", "I045", "I090", "I000"]].astype('float32') / analysis.MAX_INT, np.finfo(float).eps, 1.)
    data["POL"] = (s["I000"] - s["I090"]) / (s["I000"] + s["I090"] + np.finfo(float).eps)
    data["INT"] = (s["I000"] + s["I090"]) / 2
    data["INT-POL"] = data["INT"] - data["POL"]
    data = data[abs(data["POL"]) < 0.8]

    # remove outliers
    if remove_outliers:

        sessions = np.unique(data["session"])
        condition = []
        for session in sessions:
            condition.append(data["session"] == session)
            recordings = np.unique(data["rotation"])
            for recording in recordings:
                condition.append(data["rotation"] == recording)
                units = np.unique(data["device"])
                for unit in units:
                    condition.append(data["device"] == unit)

                    dsr = data[np.all(condition, axis=0)]

                    if dsr.shape[0] < 7:
                        condition.pop()
                        continue

                    i_imu = np.argsort(dsr["IMU"].to_numpy())

                    # fig, ax = plt.subplot_mosaic([["POL", "INT", "INT-POL"]], figsize=(7, 2))

                    for si in ["POL", "INT", "INT-POL"]:
                        s = dsr[si].to_numpy()

                        # ax[si].plot(dsr["IMU"].to_numpy()[i_imu], s[i_imu])

                        if np.all(np.isclose(s, np.mean(s))):
                            data.loc[np.all(condition, axis=0), si] = np.nan
                            continue

                        s[i_imu] = analysis.butter_low_pass_filter(np.r_[s[i_imu[-50:]], s[i_imu], s[i_imu[:50]]])[50:-50]
                        data.loc[np.all(condition, axis=0), si] = s

                        # ax[si].plot(dsr["IMU"].to_numpy()[i_imu], s[i_imu])

                    # fig.show()

                    # print(f"{session}, {recording}, {unit}, {dsr.shape}")

                    condition.pop()
                condition.pop()

            condition.pop()

        data = data.dropna(axis=0)

    data = data[data["rotation"] != 'pol_op_recording']
    data["rotation"] = data["rotation"].astype(int)

    if data_set is not None or len(data_set) > 0:
        data_list = []
        for ds in data_set:
            data_list.append(data[data["collection"] == ds.replace('_data', '')])
        data = pd.concat(data_list)

    return data


def create_full_csv(*data_dirs, out_file=None):
    if len(data_dirs) < 1:
        data_dirs = default_collections

    print(f"CREATE FULL CSV")

    if out_file is None:
        out_file = os.path.join(csv_dir, default_raw_file)
    elif not out_file.endswith('.csv'):
        out_file += '.csv'

    dir_path = os.path.dirname(os.path.realpath(out_file))

    # Make plot directory if it doesn't exist. Existing files will
    # be overwritten.
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    dataframes = []
    for dataset in data_dirs:
        print(f"DATASET: {dataset}")
        data_dir = os.path.join(data_base, dataset)

        image_dir = os.path.join(dir_path, 'sessions')
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        # Change into data directory - tod_data, canopy_data
        datasets = os.listdir(data_dir)
        for s in datasets:
            if s in skip_dirs:
                print(f"Skipping session: {s}")
                continue

            session_dir = os.path.join(data_dir, s)
            if not os.path.isdir(session_dir):
                continue

            os.chdir(session_dir)

            days = os.listdir(os.getcwd())
            for d in days:
                session_d = os.path.join(session_dir, d)
                if not os.path.isdir(session_d):
                    continue

                os.chdir(session_d)

                rec_sessions = os.listdir(os.getcwd())
                for r in rec_sessions:
                    session_path = os.path.join(data_dir, s, d, r)
                    if not os.path.isdir(session_path):
                        continue

                    print(session_path)
                    try:
                        dataframes.append(pd.DataFrame(rb.read_bagfile(session_path, image_dir)))
                    except Exception as e:
                        print(f"Warning: this session was skipped because it probably contains irrelevant data.\n{e}")

    df = pd.concat(dataframes)
    df.to_csv(out_file, index=False)
    print(f"File saved in: {out_file}.")


def create_pooled_csv(*data_dirs, in_file=None, out_file=None):

    if len(data_dirs) < 1:
        data_dirs = default_collections

    if in_file is None:
        in_file = os.path.join(csv_dir, default_raw_file)
        print(f"Input file was not provided. Default input file is: '{in_file}'.")
    elif not os.path.exists(in_file):
        in_file = os.path.join(csv_dir, default_raw_file)
        print(f"Input file does not exist. Input file was replaced by its default: '{in_file}'.")
    else:
        print(f"Input file found: '{in_file}'.")

    dataset_clean = {
        "session": [], "time": [], "location": [], "unit_type": [],
        "sun_azimuth": [], "sun_elevation": [], "direction": [], "response": [], "collection": [], "rotation": [],
        "solar_visibility": [], "anti_solar_visibility": [], "clouds": [], "occlusions": [], "tilt": []
    }
    centre = {}

    df = read_csv_dataset(*data_dirs, in_file=in_file)
    # print(df.columns.to_numpy(dtype=str))

    # Each experiment is represented by a unique image
    for session in np.unique(df["session"]):
        dff = df[df["session"] == session]
        print(f"{session}", end=': ')

        sun_azi = np.deg2rad(np.mean(dff["sun_azimuth"].to_numpy()))
        sun_ele = np.deg2rad(np.mean(dff["sun_elevation"].to_numpy()))

        session = session.replace(".jpg", "")
        if session not in centre:
            if session in imp.sun_centres:
                centre[session] = np.deg2rad(imp.sun_centres[session][4])
            else:
                dfu = dff[np.all([
                    dff["unit"] != 7,
                    dff["unit"] != 5,
                    dff["unit"] != 4,
                ], axis=0)]

                pol_res_, int_res_, _, _, _, _ = analysis.get_sensor_responses(dfu, default_nb_samples)
                x_imu = np.linspace(0, 2 * np.pi, 360, endpoint=False)
                y_inp = int_res_ - pol_res_

                est_sun = np.angle(np.nanmean(np.exp(1j * x_imu) * y_inp))

                centre[session] = (est_sun - sun_azi + np.pi) % (2 * np.pi) - np.pi

        print(f"sun = {np.rad2deg(sun_azi):.2f}, centre = {np.rad2deg(centre[session]):.2f}; "
              f"sun = '{solar_visibility[sky_conditions[session][0]]}', "
              f"anti-sun = '{anti_solar_visibility[sky_conditions[session][1]]}', "
              f"clouds = '{clouds_level[sky_conditions[session][2]]}', "
              f"canopy = '{canopy_level[sky_conditions[session][3]]}'")

        # nb_recordings x nb_samples
        pol_res, int_res, p00, p45, p90, m45 = analysis.get_sensor_responses_per_rotation(
            dff, default_nb_samples, imu_drift=centre[session])

        units = {"POL": pol_res, "INT": int_res,
                 "I000": p00, "I045": p45, "I090": p90, "I135": m45}
        nu = len(units)
        for i in range(pol_res.shape[0]):
            dataset_clean["session"].extend([session] * pol_res.shape[1] * nu)
            dataset_clean["time"].extend([dff["timestamp"].to_numpy()[0]] * pol_res.shape[1] * nu)
            dataset_clean["location"].extend([dff["location"].to_numpy()[0]] * pol_res.shape[1] * nu)
            dataset_clean["sun_azimuth"].extend([np.rad2deg(sun_azi)] * pol_res.shape[1] * nu)
            dataset_clean["sun_elevation"].extend([np.rad2deg(sun_ele)] * pol_res.shape[1] * nu)
            dataset_clean["collection"].extend([dff["collection"].to_numpy()[0]] * pol_res.shape[1] * nu)
            dataset_clean["rotation"].extend([i + 1] * pol_res.shape[1] * nu)
            dataset_clean["solar_visibility"].extend([sky_conditions[session][0]] * pol_res.shape[1] * nu)
            dataset_clean["anti_solar_visibility"].extend([sky_conditions[session][1]] * pol_res.shape[1] * nu)
            dataset_clean["clouds"].extend([sky_conditions[session][2]] * pol_res.shape[1] * nu)
            dataset_clean["occlusions"].extend([sky_conditions[session][3]] * pol_res.shape[1] * nu)
            dataset_clean["tilt"].extend([tilt_condition[session]] * pol_res.shape[1] * nu)
            for ut in units:
                dataset_clean["unit_type"].extend([ut] * pol_res.shape[1])
                dataset_clean["direction"].extend(
                    (np.linspace(0, 360, pol_res.shape[1], endpoint=False) + 180) % 360 - 180)
                dataset_clean["response"].extend(units[ut][i])

    pooled_df = pd.DataFrame(dataset_clean)

    if out_file is None:
        out_dir = os.path.dirname(in_file)
        out_file = os.path.join(out_dir, default_pooled_file)
    elif not out_file.endswith('.csv'):
        out_file += '.csv'
    pooled_df.to_csv(out_file, index=False, float_format='%.4f')
    print(f"File saved in: {out_file}.")

    return pooled_df


def create_errors_csv(*data_dirs, pooled_df=None, out_file=None):

    if len(data_dirs) < 1:
        data_dirs = default_collections

    if isinstance(pooled_df, str) and os.path.exists(os.path.realpath(pooled_df)):
        print(f"Loading input file: '{pooled_df}'.")
        pooled_df = pd.read_csv(pooled_df)
    elif pooled_df is None:
        pooled_path = os.path.join(csv_dir, default_pooled_file)
        if os.path.exists(pooled_path):
            print(f"Loading default input file: '{pooled_path}'.")
            pooled_df = pd.read_csv(pooled_path)
        else:
            print(f"Creating new (pooled dataset) file from default raw file: '{pooled_path}'.")
            pooled_df = create_pooled_csv(*data_dirs, out_file=pooled_path)

    data_frame = {
        "session": [], "collection": [], "time": [], "location": [], "spatial_resolution": [], "model": [],
        "sun_azimuth": [], "sun_elevation": [],
        "mean_error": [], "mean_error_sd": [], "max_error": [], "max_error_sd": [],
        "mean_absolute_error": [], "mean_absolute_error_sd": [],
        "root_mean_square_error": [], "root_mean_square_error_sd": [],
        "solar_visibility": [], "anti_solar_visibility": [], "clouds": [], "occlusions": [], "tilt": [], "rotation": []
    }

    error_df = None
    df = pooled_df.assign(direction=pooled_df["direction"] % 360)

    for spatial_resolution in range(3, 61):
        for session in np.unique(df["session"]):
            dfs = df[df["session"] == session]
            print(f"{session} (resolution = {spatial_resolution})", end=': ')

            sun_azi = np.deg2rad(np.mean(dfs["sun_azimuth"].to_numpy()))
            sun_ele = np.deg2rad(np.mean(dfs["sun_elevation"].to_numpy()))

            pol_res = dfs[dfs["unit_type"] == "POL"]
            int_res = dfs[dfs["unit_type"] == "INT"]
            s000_res = dfs[dfs["unit_type"] == "I000"]
            s045_res = dfs[dfs["unit_type"] == "I045"]
            s090_res = dfs[dfs["unit_type"] == "I090"]
            s135_res = dfs[dfs["unit_type"] == "I135"]

            pol_res = pol_res.pivot(index="rotation", columns="direction", values="response").to_numpy()
            int_res = int_res.pivot(index="rotation", columns="direction", values="response").to_numpy()
            s_res = [
                s000_res.pivot(index="rotation", columns="direction", values="response").to_numpy(),
                s045_res.pivot(index="rotation", columns="direction", values="response").to_numpy(),
                s090_res.pivot(index="rotation", columns="direction", values="response").to_numpy(),
                s135_res.pivot(index="rotation", columns="direction", values="response").to_numpy()
                ]

            # nb_recordings x nb_samples
            ang_pol, x_pol = analysis.compute_sensor_output_from_responses(
                pol_res, int_res, spatial_resolution, polarisation=True, intensity=False)
            ang_int, x_int = analysis.compute_sensor_output_from_responses(
                pol_res, int_res, spatial_resolution, polarisation=False, intensity=True)
            ang_inp, x_inp = analysis.compute_sensor_output_from_responses(
                pol_res, int_res, spatial_resolution, polarisation=True, intensity=True)
            ang_frz, x_frz = analysis.compute_sensor_output_from_responses(
                pol_res, int_res, spatial_resolution, algorithm=md.four_zeros)
            ang_eig, x_eig = analysis.compute_sensor_output_from_responses(
                s_res, int_res, spatial_resolution, algorithm=md.eigenvectors)

            # # break ambiguity in eigenvectors result
            # mu_eig, = compare(ang_eig, x_eig + sun_azi)
            # ang_eig[abs(mu_eig) > np.pi/2] += np.pi
            # ang_eig = circ_norm(ang_eig)

            error_types = {"mean error": "mne", "mean absolute error": "mae",
                           "root mean square error": "rmse", "max error": "mxe"}
            unit_ang = {"POL": {"ang": ang_pol, "x": x_pol + sun_azi},
                        "INT": {"ang": ang_int, "x": x_int + sun_azi},
                        "INT-POL": {"ang": ang_inp, "x": x_inp + sun_azi},
                        "FZ": {"ang": ang_frz, "x": x_frz + sun_azi},
                        "EIG": {"ang": ang_eig, "x": x_eig + sun_azi}}
            error_stats = {"POL": {}, "INT": {}, "INT-POL": {}, "FZ": {}, "EIG": {}}

            for ut in unit_ang:
                for et in error_types:
                    mu, sd = analysis.compare(unit_ang[ut]["ang"], unit_ang[ut]["x"], error_type=et, std=True, axis=1)
                    error_stats[ut][et] = np.rad2deg([mu, sd])

            nb_recordings = error_stats["POL"]["mean error"][0].shape[0]
            nb_algs = len(unit_ang)

            data_frame["session"].extend([session] * nb_algs * nb_recordings)
            data_frame["time"].extend([dfs["time"].to_numpy()[0]] * nb_algs * nb_recordings)
            data_frame["location"].extend([dfs["location"].to_numpy()[0]] * nb_algs * nb_recordings)
            data_frame["spatial_resolution"].extend([spatial_resolution] * nb_algs * nb_recordings)
            data_frame["sun_azimuth"].extend([np.rad2deg(sun_azi)] * nb_algs * nb_recordings)
            data_frame["sun_elevation"].extend([np.rad2deg(sun_ele)] * nb_algs * nb_recordings)
            data_frame["collection"].extend([dfs["collection"].to_numpy()[0]] * nb_algs * nb_recordings)
            data_frame["solar_visibility"].extend([sky_conditions[session][0]] * nb_algs * nb_recordings)
            data_frame["anti_solar_visibility"].extend([sky_conditions[session][1]] * nb_algs * nb_recordings)
            data_frame["clouds"].extend([sky_conditions[session][2]] * nb_algs * nb_recordings)
            data_frame["occlusions"].extend([sky_conditions[session][3]] * nb_algs * nb_recordings)
            data_frame["tilt"].extend([tilt_condition[session]] * nb_algs * nb_recordings)
            data_frame["rotation"].extend(list(np.arange(nb_recordings)) * nb_algs)
            for ut in unit_ang:
                data_frame["model"].extend([ut] * nb_recordings)

            for et in error_types:
                error_name = et.replace(" ", "_")

                # errors to print
                error_print = error_types[et] in ["mae", "rmse"]

                if error_print:
                    print(f"{error_types[et].upper()} = ", end="")

                for i_ut, ut in enumerate(unit_ang):

                    # units to print
                    unit_print = ut in ["FZ"]
                    if error_print and unit_print:
                        print(f"({ut}) {np.nanmean(error_stats[ut][et][0]):.2f} "
                              f"({np.nanmean(error_stats[ut][et][1]):.2f})", end="")
                        if i_ut < 2:
                            print(", ", end="")
                        else:
                            print("; ", end="")

                    data_frame[error_name].extend(list(error_stats[ut][et][0]))
                    data_frame[f"{error_name}_sd"].extend(list(error_stats[ut][et][1]))

            print(f"sun = {np.rad2deg(sun_azi):.2f}")

        error_df = pd.DataFrame(data_frame)

        if out_file is None:
            out_file = os.path.join(csv_dir, default_error_file)
        elif not out_file.endswith('.csv'):
            out_file += '.csv'
        error_df.to_csv(out_file, index=False, float_format='%.4f')
        print(f"File saved in: {out_file}.")

    return error_df


if __name__ == "__main__":
    import warnings
    import argparse

    warnings.simplefilter('ignore')

    parser = argparse.ArgumentParser(
        description="Summarise data in CSV files."
    )

    parser.add_argument("-t", dest="type", type=str, required=True,
                        choices=['raw', 'pooled', 'error'],
                        help="The type of CSV dataset to create.")
    parser.add_argument("-c", dest="collection", type=str, required=False, default="all",
                        choices=['sardinia', 'south_africa', 'all'],
                        help="The data collection to use as input.")
    parser.add_argument("-i", dest="input", type=str, required=False, default=None,
                        help="Input file path.")
    parser.add_argument("-o", dest="output", type=str, required=False, default=None,
                        help="Desired output file path.")

    args = parser.parse_args()
    csv_type = args.type
    collection = args.collection
    infile = args.input
    outfile = args.output

    if collection == "all":
        collection = default_collections
    else:
        collection = [collection.repace(" ", "_") + '_data']

    if csv_type == 'raw':
        create_full_csv(*collection, out_file=outfile)
    elif csv_type == 'pooled':
        create_pooled_csv(*collection, in_file=infile, out_file=outfile)
    elif csv_type == 'error':
        create_errors_csv(*collection, pooled_df=infile, out_file=outfile)
