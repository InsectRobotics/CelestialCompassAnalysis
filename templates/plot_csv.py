import _rosbag as rb
import circstats as cs
import models as md
import analysis

from dtw import dtw

import scipy.optimize as so
import scipy.signal as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import seaborn as sb

import os

min_rs, max_rs = 3, 20
norm_lines = mpl.colors.Normalize(0.5, max_rs + 0.5)
cmap_lines = mpl.colormaps['tab20']

default_collections = ['sardinia', 'south_africa']
default_raw = "raw_dataset.csv"
default_pooled = "pooled_dataset.csv"
default_error = "error_dataset.csv"
data_base = os.path.abspath(os.path.join(os.getcwd(), '..', 'csv'))
out_base = os.path.abspath(os.path.join(os.getcwd(), '..', 'csv', 'plots'))
if not os.path.exists(out_base):
    os.makedirs(out_base)

cmap_dict = {
    "ring_size": "viridis",
    "ele_bins": "viridis",
    "sun_elevation": "viridis",
    "cloud_status": "tab10",
    "canopy_status": "Dark2"
}
i_colour = '#ff5555ff'
p_colour = '#cd55ffff'
c_colour = '#557dffff'
z_colour = '#ffffffff'
e_colour = '#555555ff'
box_colour = {
    "INT": i_colour,
    "POL": p_colour,
    "INT-POL": c_colour,
    "FZ": z_colour,
    "EIG": e_colour
}

figure_2_selected = [
    'Sunday_20-11-22_14-28-16_SAST',
    'Sunday_27-11-22_16-02-54_SAST'
]
solar_elevation_selected = [
    'Monday_14-11-22_07-35-27_SAST',
    'Monday_14-11-22_09-31-28_SAST',
    'Monday_14-11-22_12-32-42_SAST',
    'Monday_14-11-22_13-31-22_SAST',
    'Monday_14-11-22_15-32-43_SAST',
    'Monday_14-11-22_17-30-59_SAST',
    'Monday_14-11-22_19-30-02_SAST'
]
clouds_selected = [
    'Monday_14-11-22_09-31-28_SAST',  # no clouds
    'Sunday_13-11-22_11-00-09_SAST',
    'Monday_14-11-22_12-32-42_SAST',  # thin broken clouds
    'Sunday_13-11-22_12-00-05_SAST',
    'Sunday_13-11-22_14-00-04_SAST',  # thick broken clouds
    'Sunday_13-11-22_15-00-05_SAST',
    'Monday_21-11-22_14-26-58_SAST',  # thin solid clouds
    'Friday_18-11-22_09-34-45_SAST',
    'Saturday_26-11-22_10-17-03_SAST',  # thin uniform cloud
    'Saturday_12-11-22_10-20-48_SAST',
    'Tuesday_22-11-22_13-07-52_SAST',  # mixed broken clouds
    'Saturday_12-11-22_13-35-07_SAST',
    'Friday_11-11-22_16-01-02_SAST',  # thick solid clouds
    'Wednesday_23-11-22_11-00-00_SAST',
    'Saturday_12-11-22_15-23-06_SAST',  # thick uniform clouds
    'Monday_21-11-22_07-01-53_SAST'
]
occlusions_selected = [
    'Saturday_26-11-22_16-02-49_SAST',  # no occlusion
    'Monday_21-11-22_05-31-37_SAST',
    'Monday_14-11-22_16-31-03_SAST',  # trees occupying the horizon
    'Monday_28-11-22_08-48-02_SAST',
    'Monday_28-11-22_08-29-06_SAST',  # trees occupying the centre
    'Monday_16-05-22_11-19-06_CEST',
    'Sunday_15-05-22_17-05-48_CEST',  # trees with openings
    'Sunday_15-05-22_17-41-05_CEST',
    'Thursday_19-05-22_16-33-48_CEST',  # trees occupying one side
    'Thursday_19-05-22_16-54-50_CEST',
    'Thursday_19-05-22_17-20-24_CEST'  # building occupying one side
]
lunar_4_selected = [
    'Thursday_12-05-22_18-00-11_CEST',
    'Thursday_12-05-22_19-00-20_CEST',
    'Thursday_12-05-22_20-02-32_CEST',
    'Thursday_12-05-22_21-03-00_CEST'
]
lunar_3_selected = [
    'Friday_13-05-22_19-30-09_CEST',
    'Friday_13-05-22_20-00-18_CEST',
    'Friday_13-05-22_20-30-23_CEST',
    'Friday_13-05-22_21-00-22_CEST'
]
four_zeros_selected = eigenvectors_selected = [
    "Monday_14-11-22_08-31-21_SAST",
    "Monday_21-11-22_06-00-11_SAST",
    "Sunday_27-11-22_15-46-11_SAST"
]

figure_metrics = {
    '3a': "root mean square error",
    '3b': "mean error sd",
    '4b': "root mean square error",
    '4c': "mean error sd",
    '5b': "root mean square error",
    '5c': "mean error sd",
    '6b': "root mean square error",
    '6c': "mean error sd",
    '7': "root mean square error",
    'S2a': "root mean square error",
    'S2b': "mean error sd",
    'S2c': "root mean square error",
    'S2d': "mean error sd",
}


def plot_image(image_name, axis=None, draw_sun=True, dataset_base=None):

    if axis is None:
        axis = plt.subplot(111)

    if dataset_base is None:
        dataset_base = data_base

    if image_name is not None:
        if draw_sun and not os.path.exists(os.path.join(dataset_base, "sun", image_name)):
            draw_sun = False
        if draw_sun:
            directory = "sun"
        else:
            directory = "sessions"
        img = mpimg.imread(os.path.join(dataset_base, directory, image_name))
        axis.imshow(img, origin="lower")
        axis.set_title(image_name.split(".")[0].replace("_", " "), fontsize=8)
    else:
        axis.set_title("No image found", fontsize=8)

    axis.set_xticks([])
    axis.set_yticks([])

    return axis


def plot_circ_error_bars(theta, rho, errors, color='black', edge_color='black', alpha=1., axis=None):
    q25, q75 = cs.circ_norm(np.asanyarray(errors))
    q25 += 2 * np.pi
    q75 -= 2 * np.pi
    while q75 < theta and not np.isclose(q75, theta):
        q75 += 2 * np.pi
    while q25 > theta and not np.isclose(q25, theta):
        q25 -= 2 * np.pi

    nb_samples = int(np.rad2deg(q75 - q25))

    if nb_samples % 2 == 0:  # ensure its odd
        nb_samples += 1

    ang = np.linspace(q25, q75, nb_samples)

    axis.plot(ang, rho * np.ones_like(ang), color=edge_color, lw=1, alpha=alpha, zorder=1)
    axis.plot(theta, rho, color=edge_color, markerfacecolor=color, alpha=alpha, zorder=1,
              marker=(3, 0, -np.rad2deg(theta)), markersize=3)


def plot_responses_over_imu(x, y, sun=None, negative=False, prediction=None, error=None, axis=None,
                            c=0, color="red", y_padding=.1, x_ticks=True, y_min=-1, y_max=1, filtered=False):

    if axis is None:
        axis = plt.subplot(111)

    max_recording = 15

    axis.set_theta_direction(-1)
    axis.set_theta_zero_location("N")

    x = (x + np.pi) % (2 * np.pi) - np.pi
    i_sort = np.argsort(x)

    x = x[i_sort]
    r = y[i_sort]

    if filtered and len(r) > 10:
        r = analysis.butter_low_pass_filter(np.r_[r[-50:], r, r[:50]])[50:-50]

    axis.plot(x, r, color=color[:-2] + '88', lw=0.5, alpha=1 - c / max_recording, zorder=2)

    if sun is not None:
        axis.quiver(sun, -1, np.sin(sun), np.cos(sun), scale=.02, width=.02, color='lightgreen', zorder=0)

    if prediction is None:
        arrow = np.mean(np.exp(1j * x) * r * np.power(-1, float(negative)))
    else:
        arrow = np.exp(1j * prediction)

    if error is None:
        axis.quiver(np.angle(arrow), 1, np.imag(arrow) / abs(arrow), np.real(arrow) / abs(arrow),
                    scale=6 - 6 * c / max_recording, color=f"{1 - c / max_recording:.2f}", zorder=1)
    else:
        plot_circ_error_bars(prediction, c / max_recording + 1, error,
                             color=color, edge_color=color, alpha=1 - c / max_recording, axis=axis)

    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 0], s=20, facecolor=f'C{c+2}', marker='.')
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 1], s=20, facecolor=f'C{c+4}', marker='.')
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 2], s=20, facecolor=f'C{c+6}', marker='.')
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 3], s=20, facecolor=f'C{c+8}', marker='.')

    axis.set_yticks([-1, 0])
    axis.set_yticklabels([""] * 2)
    axis.set_xticks([-3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axis.set_xticklabels(["", "W", "", "N", "", "E", "", "S"])
    axis.spines["polar"].set_visible(False)

    if not x_ticks:
        axis.set_xticks([])

    # ax.set_ylim([-y_padding, 2 * np.pi + y_padding])
    axis.set_ylim([y_min - y_padding, y_max + 1 + y_padding])
    axis.set_xlim([-np.pi, np.pi])

    return axis, np.angle(arrow)


def plot_responses(*collections, calculate_predictions=True, figure=None, ring_size=20, dataset_path=None):

    if len(collections) < 1:
        collections = default_collections

    if dataset_path is None:
        dataset_path = os.path.join(data_base, default_pooled)
    dataset_base = os.path.dirname(dataset_path)
    clean_df = pd.read_csv(dataset_path)

    df = clean_df[np.any([clean_df["collection"] == c for c in collections], axis=0)]

    if figure is None:
        sessions = np.unique(df["session"])
    elif figure in [2, '2', "2d"]:
        sessions = figure_2_selected
    elif figure in [4, '4', "4a"]:
        sessions = solar_elevation_selected
    elif figure in [5, '5', "5a"]:
        sessions = clouds_selected
    elif figure in [6, '6', "6a"]:
        sessions = occlusions_selected
    elif figure in ['S4b']:
        sessions = lunar_4_selected
    elif figure in ['S4c']:
        sessions = lunar_3_selected
    else:
        sessions = None

    if sessions is not None:
        mosaic = [[image_file, f"{image_file} int", f"{image_file} pol", f"{image_file} int-pol"]
                  for image_file in sessions]

        fig, ax = plt.subplot_mosaic(mosaic, figsize=(9, 2 * len(sessions)),
                                     subplot_kw={'projection': 'polar'})

        for i_s, session in enumerate(sessions):
            dfs = df[df["session"] == session]
            dfs["direction"] = dfs["direction"] % 360
            print(f"Session {i_s + 1: 2d}: {session}")

            sun_azi = np.deg2rad(np.mean(dfs["sun_azimuth"].to_numpy()))

            pol_res = dfs[dfs["unit_type"] == "POL"]
            int_res = dfs[dfs["unit_type"] == "INT"]

            pol_res = pol_res.pivot(index="rotation", columns="direction", values="response").to_numpy()
            int_res = int_res.pivot(index="rotation", columns="direction", values="response").to_numpy()

            ax[session].remove()
            ax[session] = plt.subplot(len(sessions), 4, i_s * 4 + 1)

            plot_image(session + ".jpg", axis=ax[session])

            if calculate_predictions:
                # nb_recordings x nb_samples
                ang_pol, x_pol = analysis.compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=True, intensity=False)
                ang_int, x_int = analysis.compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=False, intensity=True)
                ang_inp, x_inp = analysis.compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=True, intensity=True)

                pol_q25, pol_q50, pol_q75 = cs.circ_quantiles(ang_pol - x_pol, axis=-1)
                int_q25, int_q50, int_q75 = cs.circ_quantiles(ang_int - x_int, axis=-1)
                inp_q25, inp_q50, inp_q75 = cs.circ_quantiles(ang_inp - x_inp, axis=-1)
            else:
                pol_q25, pol_q50, pol_q75 = [[None] * pol_res.shape[0]] * 3
                int_q25, int_q50, int_q75 = [[None] * pol_res.shape[0]] * 3
                inp_q25, inp_q50, inp_q75 = [[None] * pol_res.shape[0]] * 3

            for r in range(pol_res.shape[0]):
                x_imu = np.linspace(0, 2 * np.pi, pol_res.shape[1], endpoint=False)
                y_pol = pol_res[r]
                y_iny = int_res[r]
                y_inp = int_res[r] - pol_res[r]

                plot_responses_over_imu(
                    x_imu, y_pol, axis=ax[f"{session} pol"], color=p_colour,
                    prediction=pol_q50[r], error=(pol_q25[r], pol_q75[r]),
                    c=r, sun=sun_azi, x_ticks=False, filtered=False, negative=True)
                plot_responses_over_imu(
                    x_imu, y_iny, axis=ax[f"{session} int"], color=i_colour,
                    prediction=int_q50[r], error=(int_q25[r], int_q75[r]),
                    c=r, sun=sun_azi, x_ticks=False, filtered=False)
                plot_responses_over_imu(
                    x_imu, y_inp, axis=ax[f"{session} int-pol"], color=c_colour,
                    prediction=inp_q50[r], error=(inp_q25[r], inp_q75[r]),
                    c=r, sun=sun_azi, x_ticks=False, filtered=False)

        # Add text
        ax[f"{sessions[0]} pol"].set_title("POL", fontsize=8)
        ax[f"{sessions[0]} int"].set_title("INT", fontsize=8)
        ax[f"{sessions[0]} int-pol"].set_title("INT-POL", fontsize=8)
        fig.tight_layout()
    else:
        print(f"Figure {figure} is not supported for this type of plot.")
        fig = None

    return fig


def plot_error_boxes(*collections, figure=None, model="INT-POL", metric=None, default_resolution=8):

    if len(collections) < 1:
        collections = default_collections

    if metric is None:
        metric = figure_metrics[f'{figure}']

    metric = metric.lower().replace(" ", "_")

    error_df = pd.read_csv(os.path.join(data_base, default_error))

    # select dataset from the requested dataset
    error_df = error_df[np.any([error_df["collection"] == c for c in collections], axis=0)]

    fig = None

    if figure not in ["model-comparison"]:
        fig = plt.figure(figure, figsize=(3, 2))

    if figure in ["3a", "3b", "spatial_resolution"]:
        df_select = error_df[np.all([
            error_df["clouds"] < 3,
            error_df["occlusions"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["sun_elevation"] >= 10,
            error_df["sun_elevation"] <= 80,
            error_df["model"] == model,
            np.any([
                error_df["spatial_resolution"] == 3,
                error_df["spatial_resolution"] == 4,
                error_df["spatial_resolution"] == 5,
                error_df["spatial_resolution"] == 8,
                error_df["spatial_resolution"] == 16,
                error_df["spatial_resolution"] == 24,
                error_df["spatial_resolution"] == 60,
            ], axis=0)
        ], axis=0)]

        sb.boxplot(df_select, x="spatial_resolution", y=metric, palette='viridis')

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 15, 60], ["1", "4", "15", "60"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-60, -15, -4, 0, 4, 15, 60], ["-60", "-15", "-4", "0", "4", "15", "60"])
            plt.ylim([-90, 90])
    elif figure in ["4b", "4c", "solar-elevation"]:  # solar elevation

        # sun elevation
        df_select = error_df[np.all([
            error_df["clouds"] < 3,
            error_df["occlusions"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["spatial_resolution"] == default_resolution,
            error_df["model"] == model
        ], axis=0)]

        ele_bins = np.linspace(-10, 90, 21)
        ele_labels = np.linspace(-5, 90, 20, dtype=int)
        df_select["ele_bins"] = pd.cut(x=df_select["sun_elevation"], bins=ele_bins, labels=ele_labels, include_lowest=True)

        sb.boxplot(df_select, x="ele_bins", y=metric, palette='viridis')

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 15, 60], ["1", "4", "15", "60"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-60, -15, -4, 0, 4, 15, 60], ["-60", "-15", "-4", "0", "4", "15", "60"])
            plt.ylim([-90, 90])
    elif figure in ["5b", "5c", "clouds"]:

        # cloud cover
        df_select = error_df[np.all([
            error_df["occlusions"] < 2,
            error_df["sun_elevation"] >= 15,
            np.isclose(error_df["tilt"], 0),
            error_df["spatial_resolution"] == default_resolution,
            error_df["model"] == model
        ], axis=0)]

        order = [0, 1, 2, 5, 3, 4, 6, 7]
        sb.boxplot(df_select, x="clouds", y=metric, color=box_colour[model], order=order)

        print(df_select.groupby(by="clouds")[metric].median())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-64, -16, -4, 0, 4, 16, 64], ["-64", "-16", "-4", "0", "4", "16", "64"])
            plt.ylim([-90, 90])
    elif figure in ["6b", "6c", "occlusions"]:

        # sun elevation
        df_select = error_df[np.all([
            error_df["clouds"] < 5,
            error_df["sun_elevation"] >= 15,
            np.isclose(error_df["tilt"], 0),
            error_df["spatial_resolution"] == default_resolution,
            error_df["model"] == model
        ], axis=0)]

        sb.boxplot(df_select, x="occlusions", y=metric, color=box_colour[model])

        print(df_select.groupby(by="occlusions")[metric].median())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-64, -16, -4, 0, 4, 16, 64], ["-64", "-16", "-4", "0", "4", "16", "64"])
            plt.ylim([-90, 90])
    elif figure in [7, "7", "S2a", "S2b", "S2c", "S2d", "model-comparison"]:

        fig = plt.figure(figure, figsize=(3, 2))

        clouds = [[0, 1, 2], [3, 4, 5, 6, 7]]
        occlusion = [[0, 1], [2, 3, 4, 5]]
        solar = [[0, 1], [2]]

        df_select = error_df[np.all([
            error_df["sun_elevation"] >= 15,
            np.isclose(error_df["tilt"], 0),
            error_df["spatial_resolution"] == default_resolution
        ], axis=0)]

        sky_condition = df_select["clouds"].to_numpy() * np.nan
        sky_condition[np.all([
            np.any([df_select["clouds"] == c for c in clouds[0]], axis=0),
            np.any([df_select["occlusions"] == c for c in occlusion[0]], axis=0),
            np.any([df_select["solar_visibility"] == c for c in solar[0]], axis=0)
        ], axis=0)] = 0  # clear
        sky_condition[np.all([
            np.any([df_select["clouds"] == c for c in clouds[1]], axis=0),
            np.any([df_select["occlusions"] == c for c in occlusion[0]], axis=0),
            np.any([df_select["solar_visibility"] == c for c in solar[0]], axis=0)
        ], axis=0)] = 1  # cloudy
        sky_condition[np.all([
            np.any([df_select["clouds"] == c for c in clouds[0]], axis=0),
            np.any([df_select["occlusions"] == c for c in occlusion[1]], axis=0),
            np.any([df_select["solar_visibility"] == c for c in solar[0]], axis=0)
        ], axis=0)] = 2  # occluded
        sky_condition[np.any([
            np.any([df_select["solar_visibility"] == c for c in solar[1]], axis=0)
        ], axis=0)] = 3  # solar or anti-solar meridian is hidden
        df_select["sky_condition"] = sky_condition
        df_select = df_select.dropna(axis=0)

        ax = plt.subplot(111)

        sb.boxplot(df_select, x="sky_condition", y=metric, hue="model", palette=box_colour,
                   hue_order=["INT-POL", "POL", "INT", "EIG", "FZ"], fliersize=.5, ax=ax)

        print(df_select.groupby(by=["model", "sky_condition"])[metric].median())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-64, -16, -4, 0, 4, 16, 64], ["-64", "-16", "-4", "0", "4", "16", "64"])
            plt.ylim([-90, 90])

        ax.legend_.remove()
    elif figure in ["location"]:

        df_select = error_df[np.all([
            error_df["cloud_status"] < 6,
            error_df["canopy_status"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["sun_elevation"] >= 15,
            error_df["unit_type"] == model,
            # error_df["category"] == "tod",
            error_df["ring_size"] == default_resolution
        ], axis=0)]

        sb.boxplot(df_select, x="location", y=metric, palette='Dark2')

        print(df_select.groupby(by="location")[metric].median())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 15, 60], ["1", "4", "15", "60"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-60, -15, -4, 0, 4, 15, 60], ["-60", "-15", "-4", "0", "4", "15", "60"])
            plt.ylim([-90, 90])
    else:
        print(f"Figure {figure} is not supported for this type of plot.")
        fig = None

    if fig is not None:
        fig.tight_layout()

    return fig


def plot_error_lines(*collections, figure=None, model="INT-POL", metric=None, default_resolution=8):

    if len(collections) < 1:
        collections = default_collections

    if metric is None:
        metric = figure_metrics[figure]

    metric = metric.lower().replace(" ", "_")

    error_df = pd.read_csv(os.path.join(data_base, default_error))

    # select dataset from the requested dataset
    error_df = error_df[np.any([error_df["collection"] == c for c in collections], axis=0)]

    fig = plt.figure(figure, figsize=(3, 2))

    if figure in ["3a", "3b", "spatial_resolution"]:
        df_select = error_df[np.all([
            error_df["clouds"] < 3,
            error_df["occlusions"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["sun_elevation"] >= 15,
            error_df["model"] == model
        ], axis=0)]

        # y_q25 = df_select.groupby("spatial_resolution")[metric].quantile(.25)
        y_q50 = df_select.groupby("spatial_resolution")[metric].median()
        # y_q75 = df_select.groupby("spatial_resolution")[metric].quantile(.75)
        size = df_select.shape[0]
        print(size)
        df_sample = df_select.iloc[np.random.permutation(size)[:500]]

        sb.regplot(df_sample, x="spatial_resolution", y=metric, fit_reg=False, x_jitter=.5,
                   scatter_kws={'color': 'black', 'alpha': 0.1, 's': 5})
        # plt.fill_between(y_q50.index.to_numpy(), y_q25.to_numpy(), y_q75.to_numpy(),
        #                  facecolor='black', alpha=0.2)
        plt.plot(y_q50.index.to_numpy(), y_q50.to_numpy(), color='black', lw=2)

        # sb.boxplot(df_select, x="spatial_resolution", y=metric, palette='viridis')
        print(df_select.groupby(by="spatial_resolution")[metric].median())

        plt.yscale('symlog', base=2)
        plt.xscale('log', base=6)
        plt.xlim(np.min(y_q50.index.to_numpy()), np.max(y_q50.index.to_numpy()))
        plt.xticks([3, 5, 8, 12, 20, 36, 60], ['3', '5', '8', '12', '20', '36', '60'])
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-64, -16, -4, 0, 4, 16, 64], ["-64", "-16", "-4", "0", "4", "16", "64"])
            plt.ylim([-90, 90])
    elif figure in ["4b", "4c", "solar-elevation"]:  # solar elevation

        # sun elevation
        df_select = error_df[np.all([
            error_df["clouds"] < 5,
            error_df["occlusions"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["spatial_resolution"] == default_resolution,
            error_df["model"] == model
        ], axis=0)]

        ele_bins = np.linspace(-10, 90, 21)
        ele_labels = np.linspace(-5, 90, 20, dtype=int)
        df_select["ele_bins"] = pd.cut(x=df_select["sun_elevation"], bins=ele_bins, labels=ele_labels, include_lowest=True)

        # y_q25 = df_select.groupby("ring_size")[metric].quantile(.25)
        y_q50 = df_select.groupby("ele_bins")[metric].median()
        # y_q75 = df_select.groupby("ring_size")[metric].quantile(.75)
        size = df_select.shape[0]
        df_sample = df_select.iloc[np.random.permutation(size)[:500]]

        print(y_q50.index.to_numpy())
        sb.regplot(df_sample, x="sun_elevation", y=metric, fit_reg=False, x_jitter=1,
                   scatter_kws={'color': 'black', 'alpha': 0.1, 's': 5})
        # plt.fill_between(y_q50.index.to_numpy(), y_q25.to_numpy(), y_q75.to_numpy(),
        #                  facecolor='black', alpha=0.2)
        plt.plot(y_q50.index.to_numpy(), y_q50.to_numpy(), color='black', lw=2)

        # sb.boxplot(df_select, x="ring_size", y=metric, palette='viridis')
        print(df_select.groupby(by="ele_bins")[metric].median())

        plt.yscale('symlog', base=2)
        # plt.xscale('symlog', base=2)
        plt.xlim(np.min(y_q50.index.to_numpy()), np.max(y_q50.index.to_numpy()))
        plt.xticks([0, 15, 30, 60, 85], ['0', '', '30', '60', '85'])
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-64, -16, -4, 0, 4, 16, 64], ["-64", "-16", "-4", "0", "4", "16", "64"])
            plt.ylim([-90, 90])
    else:
        print(f"Figure {figure} is not supported for this type of plot.")
        fig = None

    if fig is not None:
        fig.tight_layout()

    return fig


def plot_pd_response_distribution(session=None):
    scale = 1.0 / 12000.0

    data = rb.read_pd_bagfile(session)

    def objective(x, a, b, c):
        return a * np.minimum(np.exp(-0.5 * np.square((x - b) / abs(c))), .8)

    angles_3 = np.array(data["map"][2]["angle"]) - 90
    pd_3_min = abs(np.array(data["map"][2]["min_response"]))
    pd_3_max = abs(np.array(data["map"][2]["max_response"]))
    pd_3 = np.mean([pd_3_min, pd_3_max], axis=0) * scale
    pd_3_sigma = np.std([pd_3_min, pd_3_max], axis=0)
    angles_4 = np.array(data["map"][3]["angle"]) - 90
    pd_4_min = abs(np.array(data["map"][3]["min_response"]))
    pd_4_max = abs(np.array(data["map"][3]["max_response"]))
    pd_4 = np.mean([pd_4_min, pd_4_max], axis=0) * scale
    pd_4_sigma = np.std([pd_4_min, pd_4_max], axis=0)

    run_3_1 = abs(np.array(data["1"]["response"])) * scale
    run_3_2 = abs(np.array(data["2"]["response"])) * scale
    time_3_1 = abs(np.array(data["1"]["time"]))
    time_3_2 = abs(np.array(data["2"]["time"]))

    alignment_3_1 = dtw(run_3_1, pd_3, keep_internals=True)
    angles_3_1 = angles_3[alignment_3_1.index2]

    alignment_3_2 = dtw(run_3_2, pd_3, keep_internals=True)
    angles_3_2 = angles_3[alignment_3_2.index2]

    popt_3, _ = so.curve_fit(objective, angles_3, pd_3, sigma=pd_3_sigma)
    print(f"PD 3: mean = {popt_3[1]:.2f}, SD = {popt_3[2]:.2f}")

    popt_4, _ = so.curve_fit(objective, angles_4, pd_4, sigma=pd_4_sigma)
    print(f"PD 4: mean = {popt_4[1]:.2f}, SD = {popt_4[2]:.2f}")

    fig = plt.figure("photodiode-response-distribution", figsize=(4, 7))

    ax1 = plt.subplot(311, polar=True)
    ax2 = plt.subplot(312, polar=True)
    ax3 = plt.subplot(313, polar=False)

    angles_3_1 = ss.savgol_filter(angles_3_1, 51, 3)
    ax1.plot(np.deg2rad(angles_3_1), run_3_1, c='C0', marker='x', ls='', lw=1)

    angles_3_2 = ss.savgol_filter(angles_3_2, 51, 3)
    ax1.plot(np.deg2rad(angles_3_2), run_3_2, c='C0', marker='+', ls='', lw=1)

    ax1.fill_between(np.deg2rad(angles_3), np.zeros_like(angles_3), pd_3, color='C0', alpha=0.2, label="unit 3")
    ax2.fill_between(np.deg2rad(angles_4), np.zeros_like(angles_4), pd_4, color='C1', alpha=0.2, label="unit 4")

    ax1.plot(np.deg2rad(angles_3), np.mean([pd_3, pd_4], axis=0), c='k', lw=3, alpha=0.7)
    ax2.plot(np.deg2rad(angles_4), np.mean([pd_3, pd_4], axis=0), c='k', lw=3, alpha=0.7)
    ax3.plot(np.deg2rad(angles_4), np.mean([pd_3, pd_4], axis=0), c='k', lw=3, alpha=0.7)

    for ax in [ax1, ax2]:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        ax.set_xlim(-np.pi/2, np.pi/2)
        ax.set_ylim(-0.01, 1.01)
        ax.set_yticks([])

    ax1.set_title("PD 3")
    ax2.set_title("PD 4")
    ax3.set_title("mean")

    fig.tight_layout()

    return fig


def plot_eigenvectors(figure=None, rotation=1, default_resolution=8):

    if figure is None:
        figure = 'EIG'

    clean_path = os.path.join(data_base, default_pooled)
    out_base_path = os.path.abspath(out_base)
    if not os.path.exists(out_base_path):
        os.makedirs(out_base_path)

    clean_df = pd.read_csv(clean_path)

    fig = plt.figure(figure, figsize=(2.5, 6))

    for s, session in enumerate(eigenvectors_selected):

        dfs = clean_df[clean_df["session"] == session]
        dfr = dfs[dfs["rotation"] == rotation]

        dfr_000 = dfr[dfr["unit_type"] == "I000"]
        dfr_045 = dfr[dfr["unit_type"] == "I045"]
        dfr_090 = dfr[dfr["unit_type"] == "I090"]
        dfr_135 = dfr[dfr["unit_type"] == "I135"]

        s1 = dfr_000["response"].to_numpy()
        s2 = dfr_045["response"].to_numpy()
        s3 = dfr_090["response"].to_numpy()
        s4 = dfr_135["response"].to_numpy()

        sun = dfr_000["sun_azimuth"].to_numpy().mean()

        for r in range(0, 360, 360):
            # implement and test the eigenvectors algorithm
            x = cs.circ_norm(np.linspace(0, 2 * np.pi, 360, endpoint=False) + np.deg2rad(r))
            ix = np.argsort(x)
            xi = x[ix]
            s1i = s1[ix]
            s2i = s2[ix]
            s3i = s3[ix]
            s4i = s4[ix]
            pol_prefs = cs.circ_norm(np.linspace(0, 2 * np.pi, default_resolution, endpoint=False))

            s1i_samples = np.interp(pol_prefs, xi, s1i)
            s2i_samples = np.interp(pol_prefs, xi, s2i)
            s3i_samples = np.interp(pol_prefs, xi, s3i)
            s4i_samples = np.interp(pol_prefs, xi, s4i)

            est_sun, phi, d, p = md.eigenvectors(s1i_samples, s2i_samples, s3i_samples, s4i_samples,
                                                 pol_prefs=pol_prefs, verbose=True)

            e, = np.rad2deg(analysis.compare(est_sun, np.deg2rad(sun + r)))

            print(f"sun: {sun + r:.2f}, error: {e:.2f}")
            # ----------------------------------------------

            ax1 = plt.subplot(311 + s, polar=True)
            ax1.set_theta_direction(-1)
            ax1.set_theta_zero_location("N")

            azi = pol_prefs
            ele = np.full_like(pol_prefs, np.pi / 4)

            plt.plot([np.deg2rad(sun + r)] * 2, [-np.pi / 2, np.pi / 2], 'g', lw=3)
            plt.plot([est_sun] * 2, [-np.pi / 2, np.pi / 2], 'k:')
            plt.quiver(azi, ele, p[0], p[1], scale=5)
            plt.scatter(azi, ele, c=phi, s=5, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)

            plt.yticks([])
            plt.xticks([])
            plt.ylim(-np.pi / 2, np.pi / 2)
            plt.colorbar()

    fig.tight_layout()

    return fig


def plot_four_zeros(figure=None, rotation=1, default_resolution=8):

    if figure is None:
        figure = 'FZ'

    clean_path = os.path.join(data_base, default_pooled)
    out_base_path = os.path.abspath(out_base)
    if not os.path.exists(out_base_path):
        os.makedirs(out_base_path)

    clean_df = pd.read_csv(clean_path)

    fig = plt.figure(figure, figsize=(2, 6))

    for ses, session in enumerate(four_zeros_selected):
        dfs = clean_df[clean_df["session"] == session]
        dfr = dfs[dfs["rotation"] == rotation]
        dfp = dfr[dfr["unit_type"] == "POL"]

        p = dfp["response"].to_numpy()
        s = dfp["sun_azimuth"].to_numpy().mean()

        # implement and test four zeros
        x = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        ix = np.argsort(x)
        xi = x[ix]
        pi = p[ix]
        pol_prefs = cs.circ_norm(np.linspace(0, 2 * np.pi, default_resolution, endpoint=False), 2 * np.pi, 0)

        p_samples = np.interp(pol_prefs, xi, pi)
        alpha, y, z0, z1, z2 = md.four_zeros(p_samples, pol_prefs, verbose=True)

        r0, r1, r2 = np.abs([z0, z1, z2])
        a0, a1, a2 = np.angle([z0, z1, z2])

        p_z0 = r0 * np.cos(a0 - np.linspace(0, 0 * np.pi, 360, endpoint=False))
        p_z1 = r1 * np.cos(a1 - np.linspace(0, 2 * np.pi, 360, endpoint=False))
        p_z2 = r2 * np.cos(a2 - np.linspace(0, 4 * np.pi, 360, endpoint=False))

        e, = np.rad2deg(analysis.compare(alpha, np.deg2rad(s)))
        print(f"sun: {s:.2f}, error: {e:.2f}")

        ax1 = plt.subplot(311 + ses, polar=True)
        ax1.set_theta_direction(-1)
        ax1.set_theta_zero_location("N")

        plt.plot(xi, pi - p_z0, color='gray')
        plt.plot([np.deg2rad(s)] * 2, [-1, 1], 'g', lw=3)
        plt.plot([alpha % (2 * np.pi)] * 2, [-1, 1], 'k:')
        plt.plot(xi, p_z1 + p_z2, color='k')
        plt.plot(y, np.zeros(4), 'ro')
        plt.ylim([-1, 1])
        plt.yticks([0], [""])
        plt.xticks([])


    plt.tight_layout()

    return fig


if __name__ == "__main__":
    import warnings
    import argparse

    warnings.simplefilter('ignore')

    parser = argparse.ArgumentParser(
        description="Plot figures for manuscript."
    )

    out_extensions = ['png', 'jpeg', 'jpg', 'svg', 'pdf']
    parser.add_argument("-f", dest="figure", type=str, required=True,
                        choices=['2d', '3a', '3b', '4a', '4b', '4c', '5a', '5b', '5c', '6a', '6b', '6c', '7', '9h',
                                 'S2a', 'S2b', 'S2c', 'S2d', 'S3a', 'S3b', 'S4b', 'S4c'],
                        help="The figure number followed by the subfigure letter.")
    parser.add_argument("-i", dest="input", type=str, required=False, default=None,
                        help="The input file of the dataset to use.")
    parser.add_argument("-o", dest="output", type=str, required=False, default=None, choices=out_extensions,
                        help="Desired output file or file-extension.")

    args = parser.parse_args()
    figure_no = args.figure
    in_file = args.input
    outfile = args.output

    if outfile is not None and outfile.lower() in out_extensions:
        outfile = os.path.join(out_base, f"fig_{figure_no}.{outfile.lower()}")

    plot = {
        '2d': lambda: plot_responses(figure=2),
        '3a': lambda: plot_error_lines(figure="3a"),
        '3b': lambda: plot_error_lines(figure="3b"),
        '4a': lambda: plot_responses(figure=4),
        '4b': lambda: plot_error_lines(figure="4b", default_resolution=8),
        '4c': lambda: plot_error_lines(figure="4c", default_resolution=8),
        '5a': lambda: plot_responses(figure=5),
        '5b': lambda: plot_error_boxes(figure="5b", default_resolution=8),
        '5c': lambda: plot_error_boxes(figure="5c", default_resolution=8),
        '6a': lambda: plot_responses(figure=6),
        '6b': lambda: plot_error_boxes(figure="6b", default_resolution=8),
        '6c': lambda: plot_error_boxes(figure="6c", default_resolution=8),
        '7': lambda: plot_error_boxes(figure=7, default_resolution=8),
        '9h': lambda: plot_pd_response_distribution(),
        'S2a': lambda: plot_error_boxes(figure="S2a", default_resolution=8),
        'S2b': lambda: plot_error_boxes(figure="S2b", default_resolution=8),
        'S2c': lambda: plot_error_boxes(figure="S2c", default_resolution=36),
        'S2d': lambda: plot_error_boxes(figure="S2d", default_resolution=36),
        'S3a': lambda: plot_eigenvectors(figure='S3a', default_resolution=8),
        'S3b': lambda: plot_four_zeros(figure='S3b', default_resolution=8),
        'S4b': lambda: plot_responses(figure="S4b"),
        'S4c': lambda: plot_responses(figure="S4c")
    }

    fig_out = plot[figure_no]()
    fig_out.show()

    if fig_out is not None and outfile is not None:
        fig_out.savefig(outfile, bbox_inches="tight")
