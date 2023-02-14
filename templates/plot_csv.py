from scipy.signal import butter, filtfilt
from scipy.stats import circmean
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import seaborn as sb

import os

# 'GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
# 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'
# mpl.use('Qt5Agg')  # QtAgg, Qt5Agg, TkAgg, WebAgg
# mpl.use('QtAgg')  # non-gui: agg, pdf, pgf, ps, svg, template?

MAX_INT = 11000.
min_rs, max_rs = 3, 20
norm_lines = mpl.colors.Normalize(0.5, max_rs + 0.5)
cmap_lines = mpl.colormaps['tab20']

default_datasets = ['sardinia_data', 'south_africa_data']
# default_datasets = ['sardinia_data']
# default_datasets = ['south_africa_data']
data_base = os.path.abspath(os.path.join(os.getcwd(), '..', 'csv'))
out_base = os.path.abspath(os.path.join(os.getcwd(), '..', 'plots', 'csv'))
if not os.path.exists(out_base):
    os.makedirs(out_base)

cutoff = 0.03
fs = 2.0
order = 1
default_nb_samples = 360


def butter_low_pass_filter(x):
    beta, alpha = butter(order, cutoff, fs=fs, btype='low', analog=False)
    return filtfilt(beta, alpha, x)


def read_dataset(data_set, remove_outliers=False):
    data_dir = os.path.join(data_base, data_set)
    data = pd.read_csv(os.path.join(data_dir, f"{data_set}.csv"), low_memory=False)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # s = np.sqrt(np.clip(data[["S0", "S1", "S2", "S3"]].astype('float32') / MAX_INT, np.finfo(float).eps, 1.))
    s = np.clip(data[["S0", "S1", "S2", "S3"]].astype('float32') / MAX_INT, np.finfo(float).eps, 1.)
    data["POL"] = (s["S3"] - s["S2"]) / (s["S2"] + s["S3"] + np.finfo(float).eps)
    data["INT"] = (s["S2"] + s["S3"]) / 2
    data["INT-POL"] = data["INT"] - data["POL"]
    data = data[abs(data["POL"]) < 0.8]

    # remove outliers
    if remove_outliers:

        sessions = np.unique(data["image_file"])
        condition = []
        for session in sessions:
            condition.append(data["image_file"] == session)
            recordings = np.unique(data["recording"])
            for recording in recordings:
                condition.append(data["recording"] == recording)
                units = np.unique(data["unit"])
                for unit in units:
                    condition.append(data["unit"] == unit)

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

                        s[i_imu] = butter_low_pass_filter(np.r_[s[i_imu[-50:]], s[i_imu], s[i_imu[:50]]])[50:-50]
                        data.loc[np.all(condition, axis=0), si] = s

                        # ax[si].plot(dsr["IMU"].to_numpy()[i_imu], s[i_imu])

                    # fig.show()

                    # print(f"{session}, {recording}, {unit}, {dsr.shape}")

                    condition.pop()
                condition.pop()

            condition.pop()

        data = data.dropna(axis=0)

    data = data[data["recording"] != 'pol_op_recording']
    data["recording"] = data["recording"].astype(int)

    return data


def compute_sensor_output(data, ring_size=8, nb_samples=360, polarisation=True, intensity=False):
    # i_valid = [[0, 1, 2, 3, 4, 6]]
    i_valid = [[1, 2]]

    recordings = np.unique(data["recording"])
    nb_recordings = len(recordings)
    sol_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pol_angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False)
    imu_angles = (np.linspace(0, 2 * np.pi, nb_samples, endpoint=False) + np.pi) % (2 * np.pi) - np.pi
    all_pol_angles = np.zeros((nb_recordings, nb_samples), dtype='float32')

    for i, rec in enumerate(recordings):
        data_rec = data[data["recording"] == rec]
        data_rec = data_rec[np.any(data_rec["unit"].to_numpy()[:, None] == i_valid, axis=1)]

        imu = np.deg2rad(data_rec["IMU"].to_numpy())
        azi = np.deg2rad(data_rec["sun_azimuth"].to_numpy())
        pol = data_rec["POL"].to_numpy()
        iny = data_rec["INT"].to_numpy()
        inp = data_rec["INT-POL"].to_numpy()
        # imu += azi

        for s in range(nb_samples):
            # get the desired absolute rotation of the compass (with respect to north)
            rotate = imu_angles[s]

            # calculate the distance between the orientation of each unit in the desired compass orientation
            # and the orientation of each unit in the dataset
            a_dist = abs((imu[:, None] + pol_angles[None, :] - rotate + np.pi) % (2 * np.pi) - np.pi)

            # keep the dataset entries with the 3 closest orientations for each sensor
            i_closest = np.argsort(a_dist, axis=0)

            # calculate the median response of the POL and INT units on the preferred angles
            pol_i = np.median(pol[i_closest[:3]], axis=0)
            int_i = np.median(iny[i_closest[:3]], axis=0)
            inp_i = np.median(inp[i_closest[:3]], axis=0)

            # calculate the SOL responses
            sol = np.zeros(8, dtype='float32')
            if polarisation and not intensity:
                sol = pol2sol(-pol_i, pol_prefs=pol_angles, sol_prefs=sol_angles)
            elif intensity and not polarisation:
                sol = pol2sol(int_i, pol_prefs=pol_angles, sol_prefs=sol_angles)
            elif intensity and polarisation:
                sol = pol2sol(inp_i, pol_prefs=pol_angles, sol_prefs=sol_angles)
            elif not intensity and not polarisation:
                sol_i = pol2sol(int_i, pol_prefs=pol_angles, sol_prefs=sol_angles)  # intensity
                sol_p = pol2sol(pol_i, pol_prefs=pol_angles, sol_prefs=sol_angles + np.pi)  # polarisation
                sol = (sol_i + sol_p) / 2
            all_pol_angles[i, s], sigma = sol2angle(sol)
            # all_pol_angles[i, s] -= np.mean(azi)  # subtract the sun azimuth to correct for the true north

            # print(np.rad2deg(rotate), np.rad2deg(all_pol_angles[i, s]))
    # drift = circmean((all_pol_angles - imu_angles + np.pi) % (2 * np.pi) - np.pi, axis=1)
    # all_pol_angles -= drift[:, None]

    return np.squeeze(all_pol_angles + np.pi) % (2 * np.pi) - np.pi, np.squeeze(imu_angles)


def compute_sensor_output_per_recording(data_frame, ring_size, polarisation=True, intensity=False):
    ang_x = data_frame.groupby("recording").apply(lambda x: compute_sensor_output(
        x, ring_size=ring_size, polarisation=polarisation, intensity=intensity))

    ang = np.array([x[0] for x in ang_x.to_numpy()], dtype='float64')
    imu = np.array([x[1] for x in ang_x.to_numpy()], dtype='float64')

    return ang, imu


def compute_sensor_output_from_responses(res_pol, res_int, ring_size=8, polarisation=True, intensity=True):

    nb_recordings, nb_samples = res_pol.shape
    sol_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pol_angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False)
    imu_angles = (np.linspace(0, 2 * np.pi, nb_samples, endpoint=False) + np.pi) % (2 * np.pi) - np.pi
    all_pol_angles = np.zeros((nb_recordings, nb_samples), dtype='float32')

    for i in range(nb_recordings):

        pol = res_pol[i]
        iny = res_int[i]

        for s in range(nb_samples):
            # get the desired absolute rotation of the compass (with respect to north)
            rotate = imu_angles[s]

            # calculate the distance between the orientation of each unit in the desired compass orientation
            # and the orientation of each unit in the dataset
            a_dist = abs((imu_angles[:, None] - pol_angles[None, :] + rotate + np.pi) % (2 * np.pi) - np.pi)

            # keep the dataset entries with the 3 closest orientations for each sensor
            i_closest = np.argsort(a_dist, axis=0)

            # calculate the median response of the POL and INT units on the preferred angles
            pol_i = np.median(pol[i_closest[:3]], axis=0)
            int_i = np.median(iny[i_closest[:3]], axis=0)

            # calculate the SOL responses
            sol = np.zeros(8, dtype='float32')
            if polarisation and not intensity:
                sol = pol2sol(-pol_i, pol_prefs=pol_angles, sol_prefs=sol_angles)
            elif intensity and not polarisation:
                sol = pol2sol(int_i, pol_prefs=pol_angles, sol_prefs=sol_angles)
            elif intensity and polarisation:
                sol = pol2sol(int_i - pol_i, pol_prefs=pol_angles, sol_prefs=sol_angles)
            elif not intensity and not polarisation:
                sol_i = pol2sol(int_i, pol_prefs=pol_angles, sol_prefs=sol_angles)  # intensity
                sol_p = pol2sol(-pol_i, pol_prefs=pol_angles, sol_prefs=sol_angles)  # polarisation
                sol = (sol_i + sol_p) / 2
            all_pol_angles[i, s], sigma = sol2angle(sol)

    imu_angles = np.array([imu_angles] * nb_recordings)
    return np.squeeze(all_pol_angles + np.pi) % (2 * np.pi) - np.pi, np.squeeze(imu_angles)


def get_sensor_responses(data, imu_samples=360, imu_drift=0., nb_nearest=11):

    i_valid = [[0, 1, 2, 3, 6]]
    # i_valid = [[1, 2]]

    imu_angles = (np.linspace(0, 2 * np.pi, imu_samples, endpoint=False) + np.pi) % (2 * np.pi) - np.pi
    all_pol_res = np.zeros(imu_samples, dtype='float32')
    all_int_res = np.zeros(imu_samples, dtype='float32')

    if 'unit' in data.columns.to_list():
        data_rec = data[np.any(data["unit"].to_numpy()[:, None] == i_valid, axis=1)]

        imu_pol = np.deg2rad(data_rec["IMU"].to_numpy())
        imu_iny = imu_pol
        pol = data_rec["POL"].to_numpy()
        iny = data_rec["INT"].to_numpy()
    else:
        imu_pol = np.deg2rad(data[data["unit_type"] == "POL"]["direction"].to_numpy())
        imu_iny = np.deg2rad(data[data["unit_type"] == "INT"]["direction"].to_numpy())

        pol = data[data["unit_type"] == "POL"]["response"].to_numpy()
        iny = data[data["unit_type"] == "INT"]["response"].to_numpy()

    for s in range(imu_samples):
        # get the desired absolute rotation of the compass (with respect to north)
        rotate = imu_angles[s] + imu_drift

        # calculate the distance between the orientation of each unit in the desired compass orientation
        # and the orientation of each unit in the dataset
        p_dist = abs((imu_pol + rotate + np.pi) % (2 * np.pi) - np.pi)
        i_dist = abs((imu_iny + rotate + np.pi) % (2 * np.pi) - np.pi)

        # keep the dataset entries with the nearest orientations for each sensor
        p_nearest = np.argsort(p_dist, axis=0)[:nb_nearest]
        i_nearest = np.argsort(i_dist, axis=0)[:nb_nearest]

        # calculate the median response of the POL and INT units on the preferred angles
        all_pol_res[s] = np.median(pol[p_nearest], axis=0)
        all_int_res[s] = np.median(iny[i_nearest], axis=0)

    return all_pol_res, all_int_res


def get_sensor_responses_per_recording(data, imu_samples=360, imu_drift=0., nb_nearest=11):
    ang_x = data.groupby("recording").apply(lambda x: get_sensor_responses(
        x, imu_samples=imu_samples, imu_drift=imu_drift, nb_nearest=nb_nearest))

    pol_res = np.array([x[0] for x in ang_x.to_numpy()], dtype='float64')
    int_res = np.array([x[1] for x in ang_x.to_numpy()], dtype='float64')

    return pol_res, int_res


def pol2sol(pol, sol_prefs, pol_prefs):
    n_sol = len(sol_prefs)
    n_pol = len(pol_prefs)

    sol = np.sum((n_sol / n_pol) * np.cos(pol_prefs[:, None] - sol_prefs[None, :]) * pol[:, None], axis=0)
    # sol = np.zeros(n_sol)
    # for z in range(n_sol):
    #     sol[z] = np.sum((n_sol / n_pol) * np.cos(pol_prefs - sol_prefs[z]) * pol)
    #     # for j in range(n_pol):
    #     #     sol[z] += (n_sol / n_pol) * np.cos(pol_prefs[j] - sol_prefs[z]) * pol[j]

    return sol


def sol2angle(sol):
    z = np.sum(sol * np.exp(1j * np.linspace(0, 2 * np.pi, len(sol), endpoint=False)))

    angle = np.angle(z)
    confidence = np.abs(z)

    return angle, confidence


def plot_all(*datasets, reset_clean=False, reset_mae=True):
    if len(datasets) < 1:
        datasets = default_datasets

    reset_clean = reset_clean or not os.path.exists(os.path.join(data_base, "clean_all.csv"))
    reset_mae = reset_mae or not os.path.exists(os.path.join(data_base, "mae_all.csv")) or reset_clean

    if reset_mae:
        mae_df = None
        data_frame = {
            "session": [], "time": [], "location": [], "category": [], "ring_size": [], "unit_type": [],
            "sun_azimuth": [], "sun_elevation": [], "MAE": [], "dataset": []
        }
    else:
        mae_df = pd.read_csv(os.path.join(data_base, "mae_all.csv"))
        data_frame = {}

    if reset_clean:
        clean_df = None
        dataset_clean = {
            "session": [], "time": [], "location": [], "category": [], "unit_type": [],
            "sun_azimuth": [], "sun_elevation": [], "direction": [], "response": [], "dataset": [], "recording": []
        }
    else:
        clean_df = pd.read_csv(os.path.join(data_base, "clean_all.csv"))
        dataset_clean = {}
    centre = {}

    if reset_clean:
        print("\nRESET CLEAN DATASET:\n--------------------")

        for dataset in datasets:
            df = read_dataset(dataset)
            print(f"DATASET: {dataset}, {df.shape}")
            # print(df.columns.to_numpy(dtype=str))

            # Each experiment is represented by a unique image
            for image_file in np.unique(df["image_file"]):
                dff = df[df["image_file"] == image_file]
                print(f"{image_file}", end=': ')

                sun_azi = np.deg2rad(np.mean(dff["sun_azimuth"].to_numpy()))
                sun_ele = np.deg2rad(np.mean(dff["sun_elevation"].to_numpy()))

                image_file = image_file.replace(".jpg", "")
                if image_file not in centre:
                    dfu = dff[np.all([
                        dff["unit"] != 7,
                        dff["unit"] != 5,
                        dff["unit"] != 4,
                    ], axis=0)]

                    pol_res_, int_res_ = get_sensor_responses(dfu, default_nb_samples)
                    x_imu = np.linspace(0, 2 * np.pi, 360, endpoint=False)
                    y_inp = int_res_ - pol_res_

                    est_sun = np.angle(np.nanmean(np.exp(1j * x_imu) * y_inp))

                    centre[image_file] = (est_sun - sun_azi + np.pi) % (2 * np.pi) - np.pi

                    if abs(centre[image_file]) > np.deg2rad(45):
                        centre[image_file] = 0.

                print(f"sun = {np.rad2deg(sun_azi):.2f}, centre = {np.rad2deg(centre[image_file]):.2f}; "
                      f"category = {dff['category'].to_numpy()[0]}")

                # nb_recordings x nb_samples
                pol_res, int_res = get_sensor_responses_per_recording(dff, default_nb_samples,
                                                                      imu_drift=centre[image_file])

                for i in range(pol_res.shape[0]):
                    dataset_clean["session"].extend([image_file] * pol_res.shape[1] * 2)
                    dataset_clean["time"].extend([dff["timestamp"].to_numpy()[0]] * pol_res.shape[1] * 2)
                    dataset_clean["location"].extend([dff["location"].to_numpy()[0]] * pol_res.shape[1] * 2)
                    dataset_clean["category"].extend([dff['category'].to_numpy()[0]] * pol_res.shape[1] * 2)
                    dataset_clean["sun_azimuth"].extend([np.rad2deg(sun_azi)] * pol_res.shape[1] * 2)
                    dataset_clean["sun_elevation"].extend([np.rad2deg(sun_ele)] * pol_res.shape[1] * 2)
                    dataset_clean["dataset"].extend([dataset] * pol_res.shape[1] * 2)
                    dataset_clean["recording"].extend([i + 1] * pol_res.shape[1] * 2)
                    dataset_clean["unit_type"].extend(["INT"] * pol_res.shape[1] + ["POL"] * pol_res.shape[1])
                    dataset_clean["direction"].extend(
                        (np.linspace(0, 360, pol_res.shape[1], endpoint=False) + 180) % 360 - 180)
                    dataset_clean["response"].extend(int_res[i])
                    dataset_clean["direction"].extend(
                        (np.linspace(0, 360, pol_res.shape[1], endpoint=False) + 180) % 360 - 180)
                    dataset_clean["response"].extend(pol_res[i])

        clean_df = pd.DataFrame(dataset_clean)
        clean_df.to_csv(os.path.join(data_base, "clean_all.csv"), index=False)

        print(f"Saved clean dataset at: {os.path.join(data_base, 'clean_all.csv')}\n")

    print("\nPLOTTING DATA:\n--------------")
    for ring_size in range(3, 61):

        for dataset in datasets:

            df = clean_df[clean_df["dataset"] == dataset]
            for image_file in np.unique(df["session"]):
                dfs = df[df["session"] == image_file]
                dfs["direction"] = dfs["direction"] % 360
                print(f"{image_file} (ring = {ring_size})", end=': ')

                sun_azi = np.deg2rad(np.mean(dfs["sun_azimuth"].to_numpy()))
                sun_ele = np.deg2rad(np.mean(dfs["sun_elevation"].to_numpy()))

                pol_res = dfs[dfs["unit_type"] == "POL"]
                int_res = dfs[dfs["unit_type"] == "INT"]

                pol_res = pol_res.pivot(index="recording", columns="direction", values="response").to_numpy()
                int_res = int_res.pivot(index="recording", columns="direction", values="response").to_numpy()

                # nb_recordings x nb_samples
                ang_pol, x_pol = compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=True, intensity=False)
                ang_int, x_int = compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=False, intensity=True)
                ang_inp, x_inp = compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=True, intensity=True)
                ang_ipp, x_ipp = compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=False, intensity=False)

                category = dfs['category'].to_numpy()[0]
                if reset_mae:

                    mae_pol = np.rad2deg(compare(x_pol + sun_azi, ang_pol, absolute=True))
                    mae_int = np.rad2deg(compare(x_int + sun_azi, ang_int, absolute=True))
                    mae_inp = np.rad2deg(compare(x_inp + sun_azi, ang_inp, absolute=True))
                    mae_ipp = np.rad2deg(compare(x_ipp + sun_azi, ang_ipp, absolute=True))

                    data_frame["session"].extend([image_file] * 4)
                    data_frame["time"].extend([dfs["time"].to_numpy()[0]] * 4)
                    data_frame["location"].extend([dfs["location"].to_numpy()[0]] * 4)
                    data_frame["category"].extend([category] * 4)
                    data_frame["ring_size"].extend([ring_size] * 4)
                    data_frame["sun_azimuth"].extend([np.rad2deg(sun_azi)] * 4)
                    data_frame["sun_elevation"].extend([np.rad2deg(sun_ele)] * 4)
                    data_frame["dataset"].extend([dataset] * 4)
                    data_frame["unit_type"].extend(["INT", "POL", "INT-POL", "INT+POL"])
                    data_frame["MAE"].extend([mae_int, mae_pol, mae_inp, mae_ipp])
                else:
                    dfi = mae_df[mae_df["session"] == image_file]

                    mae_pol = dfi[dfi["unit_type"] == "POL"]
                    mae_int = dfi[dfi["unit_type"] == "INT"]
                    mae_inp = dfi[dfi["unit_type"] == "INT-POL"]
                    mae_ipp = dfi[dfi["unit_type"] == "INT+POL"]

                print(f"MAE = {mae_int:.2f} (INT), {mae_pol:.2f} (POL), "
                      f"{mae_inp:.2f} (INT-POL), {mae_ipp:.2f} (INT+POL); "
                      f"sun = {np.rad2deg(sun_azi):.2f}; "
                      f"category = {category}")

                mosaic = [["image", "image", "int", "pol", "int-pol", "."],
                          ["image", "image", "int_pred", "pol_pred", "int-pol_pred", "int+pol_pred"]]

                fig, ax = plt.subplot_mosaic(mosaic, figsize=(12, 4),
                                             subplot_kw={'projection': 'polar'})
                gridspec = ax["image"].get_subplotspec().get_gridspec()
                ax["image"].remove()

                subfig = fig.add_subfigure(gridspec[:, :2])

                ax["image"] = subfig.add_subplot(111)

                plot_image(image_file + ".jpg", dataset, axis=ax["image"])

                for r in range(pol_res.shape[0]):

                    x_imu = np.linspace(0, 2 * np.pi, default_nb_samples, endpoint=False)
                    y_pol = pol_res[r]
                    y_iny = int_res[r]
                    y_inp = int_res[r] - pol_res[r]

                    plot_responses_over_imu(
                        x_imu, y_pol, axis=ax["pol"],
                        c=r/pol_res.shape[0], sun=sun_azi, x_ticks=False, filtered=False, negative=True)
                    plot_responses_over_imu(
                        x_imu, y_iny, axis=ax["int"],
                        c=r/pol_res.shape[0], sun=sun_azi, x_ticks=False, filtered=False)
                    plot_responses_over_imu(
                        x_imu, y_inp, axis=ax["int-pol"],
                        c=r/pol_res.shape[0], sun=sun_azi, x_ticks=False, filtered=False)

                plot_prediction_error_over_imu(x_pol + sun_azi, ang_pol, axis=ax["pol_pred"], x_ticks=False)
                plot_prediction_error_over_imu(x_int + sun_azi, ang_int, axis=ax["int_pred"], x_ticks=False)
                plot_prediction_error_over_imu(x_inp + sun_azi, ang_inp, axis=ax["int-pol_pred"], x_ticks=False)
                plot_prediction_error_over_imu(x_ipp + sun_azi, ang_ipp, axis=ax["int+pol_pred"], x_ticks=False)

                # Add text
                ax["pol_pred"].set_title("POL")
                ax["int_pred"].set_title("INT")
                ax["int-pol_pred"].set_title("INT-POL")
                ax["int+pol_pred"].set_title("INT+POL")
                # ax["pol_pred"].set_ylabel("predictions")
                fig.tight_layout()

                outpath = os.path.join(out_base, "all")
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                outpath = os.path.join(outpath, image_file.replace(".jpg", f"_rs{ring_size:02d}.svg"))
                fig.savefig(outpath, bbox_inches="tight")
                plt.show()
        if reset_mae:
            mae_df = pd.DataFrame(data_frame)
            mae_df.to_csv(os.path.join(data_base, "mae_all.csv"), index=False)


def plot_summary_mae(*datasets, category="tod", fig=None, ax=None, show=False, save=False):
    if len(datasets) < 1:
        datasets = default_datasets

    figs, axs = [], []
    for dataset in datasets:
        df = read_dataset(dataset)
        df = df[df["unit"] != 7]
        print(f"DATASET: {dataset}, {df.shape}")
        # print(df.columns.to_numpy(dtype=str))

        categories = np.unique(df["category"])
        if category not in categories:
            print(f"Warning: Category '{category}' was not found in dataset '{dataset}'.")
            continue

        dfc = df[df["category"] == category]
        out_path = os.path.join(out_base, f"{category}_{dataset}.svg")

        if category == "tod":
            if fig is None or ax is None:
                mosaic = [["mse_pol", "mae_int", "mae_diff"]]
                fig, ax = plt.subplot_mosaic(mosaic, figsize=(7, 2))

            for ring_size in range(min_rs, max_rs + 1):
                print("RING_SIZE:", ring_size)

                nb_intervals = 2
                nb_hours = 15
                hours = np.linspace(7 - 1 / nb_intervals, 21, nb_hours * nb_intervals)
                mae_pol, mae_int = [[]] * nb_hours * nb_intervals, [[]] * nb_hours * nb_intervals
                mae_diff = [[]] * nb_hours * nb_intervals

                for i, hour in enumerate(hours):
                    dft = dfc["timestamp"].dt.hour + dfc["timestamp"].dt.minute / 60
                    dfh = dfc[np.all([hour - .5 / nb_intervals < dft, dft <= hour + .5 / nb_intervals], axis=0)]

                    print(f"Hour: {hour}")

                    for image_file in np.unique(dfh["image_file"]):
                        dff = dfh[dfh["image_file"] == image_file]
                        print(image_file, dff.shape, end=': ')

                        ang_x_pol = dff.groupby("recording").apply(lambda x: compute_sensor_output(
                            x, ring_size=ring_size, polarisation=True, intensity=False))
                        ang_x_int = dff.groupby("recording").apply(lambda x: compute_sensor_output(
                            x, ring_size=ring_size, polarisation=False, intensity=True))

                        ang_pol = np.array([x[0] for x in ang_x_pol.to_numpy()], dtype='float64')
                        x_pol = np.array([x[1] for x in ang_x_pol.to_numpy()], dtype='float64')

                        ang_int = np.array([x[0] for x in ang_x_int.to_numpy()], dtype='float64')
                        x_int = np.array([x[1] for x in ang_x_int.to_numpy()], dtype='float64')

                        mae_pol[i].append(np.rad2deg(compare(ang_pol, x_pol)))
                        mae_int[i].append(np.rad2deg(compare(ang_int, x_int)))
                        mae_diff[i].append(mae_pol[i][-1] - mae_int[i][-1])
                        print(f"MAE = {mae_pol[i][-1]:.2f} (POL), {mae_int[i][-1]:.2f} (INT)")

                    if len(mae_pol[i]) > 0 and len(mae_int[i]) > 0:
                        mae_pol[i] = np.mean(mae_pol[i])
                        mae_int[i] = np.mean(mae_int[i])
                        mae_diff[i] = np.mean(mae_diff[i])
                    else:
                        mae_pol[i] = np.nan
                        mae_int[i] = np.nan
                        mae_diff[i] = np.nan
                mae_pol = np.array(mae_pol)
                mae_int = np.array(mae_int)
                mae_diff = np.array(mae_diff)

                plot_mse_over_time(hours, mae_pol, c=ring_size, axis=ax["mse_pol"])
                plot_mse_over_time(hours, mae_int, c=ring_size, axis=ax["mae_int"], y_ticks=False)
                plot_mse_diff_over_time(hours, mae_diff, c=ring_size, axis=ax["mae_diff"])

            fig.tight_layout()
            if show:
                axc = fig.add_axes([0.73, 0.89, 0.2, 0.02])
                mpl.colorbar.ColorbarBase(axc, orientation='horizontal', cmap=cmap_lines, norm=norm_lines,
                                          ticks=[3, 8, 14, 20])
                fig.show()

            if save:
                fig.savefig(out_path, bbox_inches="tight")
        elif category in ["canopy", "cloud_cover", "tilt"]:
            dfc2 = df[df["category"] == "tod"]
            dfc2 = dfc2[np.all([dfc2["timestamp"].dt.hour > 7, dfc2["timestamp"].dt.hour < 19], axis=0)]

            mae = {"MAE": [], "category": [], "sensor": [], "ring size": []}

            def update_data(ang, x, cat, sens_type, r_size, axis=None):
                mae["MAE"].extend(np.rad2deg(compare(ang, x, axis=axis)))
                mae["category"].extend([cat] * ang.shape[0])
                mae["sensor"].extend([sens_type] * ang.shape[0])
                mae["ring size"].extend([r_size] * ang.shape[0])

                return np.mean(mae['MAE'][-ang.shape[0]:-1])

            step = 2
            rs_start = ((min_rs - 1) // step + 1) * step  # the next multiple of {step}
            for ring_size in range(rs_start, max_rs + 1, step):
                print("RING_SIZE:", ring_size)
                print(f"{category.capitalize().replace('_', ' ')} >>")
                for image_file in np.unique(dfc["image_file"]):
                    dff = dfc[dfc["image_file"] == image_file]
                    print(image_file, dff.shape, end=': ')

                    ang_pol, x_pol = compute_sensor_output_per_recording(dff, ring_size, True, False)
                    ang_int, x_int = compute_sensor_output_per_recording(dff, ring_size, False, True)

                    mae_pol_v = update_data(ang_pol, x_pol, image_file, "POL", ring_size, axis=-1)
                    mae_int_v = update_data(ang_int, x_int, image_file, "INT", ring_size, axis=-1)

                    print(f"MAE = {mae_pol_v:.2f} (POL), {mae_int_v:.2f} (INT)")

                print("TOD >>")
                if category == "tilt":
                    dfc2 = dfc2[dfc2["timestamp"].dt.hour == 14]

                for image_file in np.unique(dfc2["image_file"]):
                    dff = dfc2[dfc2["image_file"] == image_file]
                    print(image_file, dff.shape, end=': ')

                    ang_pol, x_pol = compute_sensor_output_per_recording(dff, ring_size, True, False)
                    ang_int, x_int = compute_sensor_output_per_recording(dff, ring_size, False, True)

                    mae_pol_v = update_data(ang_pol, x_pol, "TOD", "POL", ring_size, axis=-1)
                    mae_int_v = update_data(ang_int, x_int, "TOD", "INT", ring_size, axis=-1)

                    print(f"MAE = {mae_pol_v:.2f} (POL), {mae_int_v:.2f} (INT)")

            mae = pd.DataFrame(mae)
            # print(mae)

            # palette = np.array(sb.color_palette('tab20', 20))[2:]
            palette = {rs: cmap_lines(norm_lines(rs)) for rs in range(min_rs, max_rs + 1)}
            cats = np.unique(mae["category"])

            print(np.array(cats))

            mosaic = []
            for cati in cats:
                mosaic.append([f"{cati}_pol", f"{cati}_int"])

            fig, ax = plt.subplot_mosaic(mosaic, figsize=(4, 2 * len(cats)))
            for cati in cats:
                mae_c = mae[mae["category"] == cati]
                sb.boxplot(mae_c[mae_c["sensor"] == "POL"], x="ring size", y="MAE",
                           palette=palette, ax=ax[f"{cati}_pol"])
                sb.boxplot(mae_c[mae_c["sensor"] == "INT"], x="ring size", y="MAE",
                           palette=palette, ax=ax[f"{cati}_int"])

                ax[f"{cati}_pol"].set_yscale('log', base=2)
                ax[f"{cati}_pol"].set_yticks([1, 4, 15, 60])
                ax[f"{cati}_pol"].set_yticklabels(["1", "4", "15", "60"])
                ax[f"{cati}_pol"].set_ylim([1, 90])
                ax[f"{cati}_pol"].legend([], [], frameon=False)

                ax[f"{cati}_int"].set_yscale('log', base=2)
                ax[f"{cati}_int"].set_ylabel('')
                ax[f"{cati}_int"].set_yticks([1, 4, 15, 60])
                ax[f"{cati}_int"].set_yticklabels(["", "", "", ""])
                ax[f"{cati}_int"].set_ylim([1, 90])
                ax[f"{cati}_int"].legend([], [], frameon=False)

                if cati != cats[-1]:
                    ax[f"{cati}_pol"].set_xlabel(None)
                    ax[f"{cati}_int"].set_xlabel(None)
                    ax[f"{cati}_pol"].set_xticklabels([""] * len(np.unique(mae_c["ring size"])))
                    ax[f"{cati}_int"].set_xticklabels([""] * len(np.unique(mae_c["ring size"])))

            fig.tight_layout()

            if show:
                fig.show()

            if save:
                fig.savefig(out_path, bbox_inches="tight")
        else:
            print(f"Warning: Category '{category}' is not supported yet.")
            continue

        figs.append(fig)
        axs.append(ax)

    return zip(figs, axs)
        # # Each experiment is represented by a unique image
        # for image_file in np.unique(df["image_file"]):
        #     dff = df[df["image_file"] == image_file]
        #     print(image_file, dff.shape, end=': ')
        #
        #     ang_pol, x_pol = compute_sensor_output(dff, polarisation=True, intensity=False)
        #     ang_int, x_int = compute_sensor_output(dff, polarisation=False, intensity=True)
        #     mse_pol = np.rad2deg(compare(ang_pol, x_pol))
        #     mae_int = np.rad2deg(compare(ang_int, x_int))
        #     print(f"MAE = {mse_pol:.2f} (POL), {mae_int:.2f} (INT)")
        #
        #     fig, ax = create_plot()
        #
        #     plot_image(image_file, dataset, axis=ax["image"])
        #     for unit in range(8):
        #         if unit in [0, 3, 4, 5, 6, 7]:
        #             continue
        #         x_imu = dff[dff["unit"] == unit]["IMU"].to_numpy()
        #         y_pol = dff[dff["unit"] == unit]["POL"].to_numpy()
        #         y_iny = dff[dff["unit"] == unit]["INT"].to_numpy()
        #
        #         plot_responses_over_imu(x_imu, y_pol, axis=ax["pol"], c=unit, x_ticks=False)
        #         plot_responses_over_imu(x_imu, y_iny, axis=ax["int"], c=unit, x_ticks=False)
        #
        #     for r in range(ang_pol.shape[0]):
        #         plot_prediction_over_imu(x_pol, ang_pol[r], axis=ax["pol_pred"], c=r)
        #         plot_prediction_over_imu(x_int, ang_int[r], axis=ax["int_pred"], c=r)
        #
        #     # Add text
        #     ax["pol"].set_ylabel("POL")
        #     ax["int"].set_ylabel("INT")
        #     ax["pol_pred"].set_ylabel("predictions")
        #     fig.tight_layout()
        #
        #     plt.show()


def plot_image(image_name, data_set=None, axis=None, draw_sun=True):

    if axis is None:
        axis = plt.subplot(111)

    if image_name is not None:
        if draw_sun:
            directory = "sun"
        else:
            directory = data_set
        img = mpimg.imread(os.path.join(data_base, directory, image_name))
        axis.imshow(img, origin="lower")
        axis.set_title(image_name.split(".")[0].replace("_", " "))
    else:
        axis.set_title("No image found")

    axis.set_xticks([])
    axis.set_yticks([])

    return axis


def plot_responses_over_imu(x, y, sun=None, negative=False, axis=None, c=0, y_padding=.1, x_ticks=True,
                            y_min=-1, y_max=1, filtered=False):

    if axis is None:
        axis = plt.subplot(111)

    axis.set_theta_direction(-1)
    axis.set_theta_zero_location("N")

    x = (x + np.pi) % (2 * np.pi) - np.pi
    i_sort = np.argsort(x)

    x = x[i_sort]
    r = y[i_sort]

    arrow = np.mean(np.exp(1j * x) * r * np.power(-1, float(negative)))

    if filtered and len(r) > 10:
        r = butter_low_pass_filter(np.r_[r[-50:], r, r[:50]])[50:-50]

    axis.plot(x, r, color='k', lw=0.5, alpha=1 - c)

    if sun is not None:
        axis.quiver(sun, -1, np.sin(sun), np.cos(sun), scale=.02, width=.02, color='lightgreen', alpha=0.5)

    axis.quiver(np.angle(arrow), 0, np.imag(arrow) / abs(arrow), np.real(arrow) / abs(arrow),
                scale=5, color=f"{1 - c:.2f}")

    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 0], s=20, facecolor=f'C{c+2}', marker='.')
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 1], s=20, facecolor=f'C{c+4}', marker='.')
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 2], s=20, facecolor=f'C{c+6}', marker='.')
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 3], s=20, facecolor=f'C{c+8}', marker='.')

    axis.set_yticks([-1, 0, 1])
    axis.set_yticklabels([""] * 3)
    axis.set_xticks([-3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axis.set_xticklabels(["", "W", "", "N", "", "E", "", "S"])

    if not x_ticks:
        axis.set_xticklabels([""] * 8)

    # ax.set_ylim([-y_padding, 2 * np.pi + y_padding])
    axis.set_ylim([y_min - y_padding, y_max + y_padding])
    axis.set_xlim([-np.pi, np.pi])

    return axis, np.angle(arrow)


def plot_prediction_over_imu(x, y, axis=None, x_ticks=True):

    if axis is None:
        axis = plt.subplot(111)

    x = (x + np.pi) % (2 * np.pi) - np.pi
    y = (y + np.pi) % (2 * np.pi) - np.pi

    i_sort = np.argsort(x[0])
    x = x[:, i_sort]
    r = y[:, i_sort]

    x_mean = x[0]
    r_mean = circmean(r, axis=0, nan_policy='omit')

    for r_i in [rr for rr in r] + [r_mean]:
        for i in range(r.shape[1] - 1):

            while r_i[i + 1] - r_i[i] > 0.75 * np.pi:
                r_i[i + 1] -= 2 * np.pi

            while r_i[i + 1] - r_i[i] < -0.75 * np.pi:
                r_i[i + 1] += 2 * np.pi

    axis.plot(x, r, color='grey', lw=0.5)
    axis.plot(x, r + 2 * np.pi, color=f'grey', lw=0.5)
    axis.plot(x, r - 2 * np.pi, color=f'grey', lw=0.5)

    axis.plot(x_mean, r_mean, color='black', lw=2)
    axis.plot(x_mean, r_mean + 2 * np.pi, color='black', lw=2)
    axis.plot(x_mean, r_mean - 2 * np.pi, color='black', lw=2)

    axis.set_yticks([-np.pi, 0, np.pi])
    axis.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    axis.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axis.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    if not x_ticks:
        axis.set_xticks([])

    axis.set_ylim([-np.pi, np.pi])
    axis.set_xlim([-np.pi, np.pi])

    return axis


def plot_prediction_error_over_imu(x, y, axis=None, x_ticks=True):

    if axis is None:
        axis = plt.subplot(111)

    # axis.set_theta_direction(-1)
    axis.set_theta_zero_location("N")

    x = (x + np.pi) % (2 * np.pi) - np.pi
    y = (y + np.pi) % (2 * np.pi) - np.pi

    i_sort = np.argsort(x[0])
    x = x[:, i_sort]
    r = y[:, i_sort]

    e = (r - x + np.pi) % (2 * np.pi) - np.pi
    x_mean = x[0]
    e_mean = (circmean(e, axis=0, nan_policy='omit') + np.pi) % (2 * np.pi) - np.pi

    for i in range(x.shape[0]):
        axis.plot(x[i], abs(e[i]), color='k', lw=1.5, alpha=1 - i / x.shape[0])
    axis.plot(x_mean, abs(e_mean), color='red', lw=2)

    axis.set_yticks([0, np.pi/4, np.pi/2])
    axis.set_yticklabels([""] * 3)
    axis.set_xticks([-3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axis.set_xticklabels(["", "E", "", "N", "", "W", "", "S"])

    if not x_ticks:
        axis.set_xticklabels([""] * 8)

    axis.set_ylim([-0.1, np.pi/2 + 0.1])
    axis.set_xlim([-np.pi, np.pi])

    return axis


def plot_mse_over_time(hours, mse, c, y_ticks=True, axis=None):
    if axis is None:
        axis = plt.subplot(111)

    params = {"label": c} if not y_ticks else {}
    params["lw"] = 1
    axis.plot(hours[~np.isnan(mse)], mse[~np.isnan(mse)], color=cmap_lines(norm_lines(c)), **params)
    axis.set_xlim([hours[~np.isnan(mse)].min(), hours[~np.isnan(mse)].max()])
    axis.set_xticks([7, 12, 17, 21])
    axis.set_xticklabels(["7 am", "12 pm", "5 pm", "9 pm"])
    axis.set_yscale('log', base=2)
    axis.set_yticks([1, 4, 15, 60])
    if y_ticks:
        axis.set_yticklabels(["1", "4", "15", "60"])
    else:
        axis.set_yticklabels(["", "", "", ""])
    axis.set_ylim([1, 90])


def plot_mse_diff_over_time(hours, d_mse, c, y_ticks=True, axis=None):
    if axis is None:
        axis = plt.subplot(111)

    params = {"label": c} if not y_ticks else {}
    params["lw"] = 1
    axis.plot(hours[~np.isnan(d_mse)], d_mse[~np.isnan(d_mse)], color=cmap_lines(norm_lines(c)), **params)
    axis.set_xlim([hours[~np.isnan(d_mse)].min(), hours[~np.isnan(d_mse)].max()])
    axis.set_xticks([7, 12, 17, 21])
    axis.set_xticklabels(["7 am", "12 pm", "5 pm", "9 pm"])
    axis.set_yscale('symlog', base=2)
    axis.set_yticks([-15, 0, 15])
    if y_ticks:
        axis.set_yticklabels(["-15", "0", "15"])
    else:
        axis.set_yticklabels(["", "", ""])
    axis.set_ylim([-60, 60])


def compare(x, ground_truth, absolute=True, axis=None):
    ang_distance = (x - ground_truth + np.pi) % (2 * np.pi) - np.pi
    if absolute:
        ang_distance = abs(ang_distance)
    return (circmean(ang_distance, axis=axis, nan_policy='omit') + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    import warnings

    warnings.simplefilter('ignore')

    plot_all()
    # figure, axes = None, None
    # plot_summary_mae(category="tod", fig=figure, ax=axes, show=True, save=True)
    # plot_summary_mae(category="canopy", fig=figure, ax=axes, show=True, save=True)
    # plot_summary_mae('south_africa_data', category="cloud_cover", fig=figure, ax=axes, show=True, save=True)
    # plot_summary_mae('south_africa_data', category="canopy", fig=figure, ax=axes, show=True, save=True)
    # plot_summary_mae('south_africa_data', category="tilt", fig=figure, ax=axes, show=True, save=True)
