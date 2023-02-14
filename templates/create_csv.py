"""
create_csv.py

Brittle script to produce all plots from the tod and canopy recordings. The
directory structure is assumed so this will break if anything changes.
"""

from image_processing import get_observer_from_file_name, extract_sun_vector_from_name
from rosbag_to_json import rosbag_to_dict
from datetime import datetime
from copy import copy

import os
import shutil
import rosbag
import numpy as np
import pandas as pd
import skylight as skl

skip_dirs = ["elevation", "path_integration", "sun_tables", "track_test"]


def read_bagfile(session, images_dir):
    timestamp = datetime.strptime(session.split(os.path.sep)[-1], '%H-%M__%A_%d-%m-%y')
    category = session.split(os.path.sep)[-3].replace("_data", "")

    os.chdir(session)

    image_file = [x for x in os.listdir() if ".jpg" in x]

    if len(image_file) == 0:
        print("Warning: missing sky image file from the session directory.")
        image_file = None
    elif len(image_file) > 1:
        print("Warning: multiple image files in directory.")
    else:
        image_file = image_file[0]

    if image_file is not None:
        shutil.copy2(image_file, images_dir)

    recordings = [x for x in os.listdir() if ".bag" in x]

    if len(recordings) == 0:
        print("There are no rosbag files in the specified directory.")

    full_data = dict()
    for rec in recordings:
        bag = rosbag.Bag(rec)
        rec_data = rosbag_to_dict(bag)
        full_data[rec] = rec_data

    data_frame = {"IMU": [], "sun_azimuth": [], "sun_elevation": [],
                  "S0": [], "S1": [], "S2": [], "S3": [], "unit": [], "recording": [],
                  "location": [], "timestamp": [], "longitude": [], "latitude": [],
                  "category": [], "image_file": []}

    image_name = image_file.replace('.jpg', '')
    obs = get_observer_from_file_name(image_name)
    sun_predict = extract_sun_vector_from_name(image_name)
    sun_model = skl.Sun(obs)

    for c, key in enumerate(full_data.keys()):
        data = full_data[key]

        pol_op_keys = ["pol_op_{}".format(x) for x in range(8)]
        po = []
        for k in pol_op_keys:
            po.append(data[k])
        po = np.array(po)

        imu = np.array(data["yaw"])
        imu_drift = np.angle(np.squeeze(sun_predict)) - sun_model.az
        imu = (imu - imu_drift + np.pi) % (2 * np.pi) - np.pi

        imus = np.linspace(imu[:-1], imu[1:], 8, endpoint=False)
        yaws = (imus.T - data['azimuths']).T

        # # unit 5 photoreceptor 1 is faulty
        # # unit 7 is faulty, so it is ignored
        # i_valid = [0, 1, 2, 3, 4, 5, 6]
        # imu = yaws[i_valid]  # .reshape((-1,))
        # pol = po[i_valid, 1:]  # .reshape((-1, 4))
        imu = yaws
        pol = po[:, 1:]

        sun_azi = np.rad2deg((sun_model.az + np.pi) % (2 * np.pi) - np.pi)
        sun_ele = np.rad2deg((sun_model.alt + np.pi) % (2 * np.pi) - np.pi)
        for unit in range(imu.shape[0]):
            data_frame["IMU"].extend(np.rad2deg(imu[unit]))
            data_frame["sun_azimuth"].extend([sun_azi] * imu.shape[1])
            data_frame["sun_elevation"].extend([sun_ele] * imu.shape[1])
            for i in range(4):
                data_frame[f"S{i}"].extend(pol[unit, :, i])
            data_frame["unit"].extend([unit] * imu.shape[1])
            data_frame["recording"].extend([key.split('.')[0]] * imu.shape[1])
            data_frame["category"].extend([category] * imu.shape[1])
            data_frame["location"].extend([obs.city] * imu.shape[1])
            data_frame["timestamp"].extend([obs.date] * imu.shape[1])
            data_frame["longitude"].extend([obs.lon] * imu.shape[1])
            data_frame["latitude"].extend([obs.lat] * imu.shape[1])
            data_frame["image_file"].extend([image_file] * imu.shape[1])

    return data_frame


if __name__ == "__main__":
    data_dirs = ['sardinia_data', 'south_africa_data']
    # data_dirs = ['south_africa_data']
    data_base = os.path.abspath(os.path.join(os.getcwd(), '..'))
    out_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'csv'))
    print(f"CREATE_CSV")

    # Make plot directory if it doesn't exist. Existing files will
    # be overwritten.
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for dataset in data_dirs:
        print(f"DATASET: {dataset}")
        data_dir = os.path.join(data_base, dataset)

        image_dir = os.path.join(out_dir, dataset)
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        # Change into data directory - tod_data, canopy_data
        datasets = os.listdir(data_dir)
        dataframes = []
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
                        dataframes.append(pd.DataFrame(read_bagfile(session_path, image_dir)))
                    except Exception as e:
                        print(f"Warning: this session was skipped because it probably contains irrelevant data.\n{e}")

        df = pd.concat(dataframes)
        print(df)
        df.to_csv(os.path.join(out_dir, dataset, f"{dataset}.csv"), index=False)

