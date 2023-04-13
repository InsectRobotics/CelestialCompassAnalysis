from create_csv import read_csv_dataset, create_clean_csv, create_errors_csv, butter_low_pass_filter, compare
from create_csv import compute_sensor_output_per_recording, compute_sensor_output, compute_sensor_output_from_responses
from circstats import circ_mean, circ_median, circ_quantiles, circ_norm

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
cloud_level_selected = [
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
occlusion_level_selected = [
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


def plot_all(*datasets, reset_clean=True, reset_mae=True):
    if len(datasets) < 1:
        datasets = default_datasets

    reset_clean = reset_clean or not os.path.exists(os.path.join(data_base, "clean_all.csv"))
    reset_mae = reset_mae or not os.path.exists(os.path.join(data_base, "mae_all.csv")) or reset_clean

    if reset_clean:
        print("\nRESET CLEAN DATASET:\n--------------------")
        clean_df = create_clean_csv(*datasets)
    else:
        clean_df = pd.read_csv(os.path.join(data_base, "clean_all.csv"))

    if reset_mae:
        print("\nRESET MAE DATASET:\n------------------")
        create_errors_csv(*datasets, clean_df)

    print("\nPLOTTING DATA:\n--------------")
    for ring_size in range(60, 2, -1):

        for dataset in datasets:

            df = clean_df[clean_df["dataset"] == dataset]
            for image_file in np.unique(df["session"]):
                dfs = df[df["session"] == image_file]
                dfs["direction"] = dfs["direction"] % 360
                print(f"{image_file} (ring = {ring_size})", end=': ')

                sun_azi = np.deg2rad(np.mean(dfs["sun_azimuth"].to_numpy()))

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

                    x_imu = np.linspace(0, 2 * np.pi, pol_res.shape[1], endpoint=False)
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


def plot_responses(*datasets, calculate_predictions=True, figure=None, ring_size=20):

    if len(datasets) < 1:
        datasets = default_datasets
        dataset = "all"
    else:
        dataset = '+'.join(datasets)

    clean_df = pd.read_csv(os.path.join(data_base, "clean_all.csv"))

    df = clean_df[np.any([clean_df["dataset"] == ds for ds in datasets], axis=0)]

    if figure is None:
        image_files = np.unique(df["session"])
    elif figure in [2, '2']:
        image_files = figure_2_selected
    elif figure == "solar-elevation":
        image_files = solar_elevation_selected
    elif figure == "cloud-level":
        image_files = cloud_level_selected
    elif figure == "occlusion-level":
        image_files = occlusion_level_selected

    mosaic = [[image_file, f"{image_file} int", f"{image_file} pol", f"{image_file} int-pol"]
              for image_file in image_files]

    fig, ax = plt.subplot_mosaic(mosaic, figsize=(9, 2 * len(image_files)),
                                 subplot_kw={'projection': 'polar'})

    for imf, image_file in enumerate(image_files):
        dfs = df[df["session"] == image_file]
        dfs["direction"] = dfs["direction"] % 360
        print(f"{image_file}")

        sun_azi = np.deg2rad(np.mean(dfs["sun_azimuth"].to_numpy()))

        pol_res = dfs[dfs["unit_type"] == "POL"]
        int_res = dfs[dfs["unit_type"] == "INT"]

        pol_res = pol_res.pivot(index="recording", columns="direction", values="response").to_numpy()
        int_res = int_res.pivot(index="recording", columns="direction", values="response").to_numpy()

        ax[image_file].remove()
        ax[image_file] = plt.subplot(len(image_files), 4, imf * 4 + 1)

        plot_image(image_file + ".jpg", dataset, axis=ax[image_file])

        if calculate_predictions:
            # nb_recordings x nb_samples
            ang_pol, x_pol = compute_sensor_output_from_responses(
                pol_res, int_res, ring_size, polarisation=True, intensity=False)
            ang_int, x_int = compute_sensor_output_from_responses(
                pol_res, int_res, ring_size, polarisation=False, intensity=True)
            ang_inp, x_inp = compute_sensor_output_from_responses(
                pol_res, int_res, ring_size, polarisation=True, intensity=True)

            pol_q25, pol_q50, pol_q75 = circ_quantiles(ang_pol - x_pol, axis=-1)
            int_q25, int_q50, int_q75 = circ_quantiles(ang_int - x_int, axis=-1)
            inp_q25, inp_q50, inp_q75 = circ_quantiles(ang_inp - x_inp, axis=-1)
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
                x_imu, y_pol, axis=ax[f"{image_file} pol"], color=p_colour,
                prediction=pol_q50[r], error=(pol_q25[r], pol_q75[r]),
                c=r, sun=sun_azi, x_ticks=False, filtered=False, negative=True)
            plot_responses_over_imu(
                x_imu, y_iny, axis=ax[f"{image_file} int"], color=i_colour,
                prediction=int_q50[r], error=(int_q25[r], int_q75[r]),
                c=r, sun=sun_azi, x_ticks=False, filtered=False)
            plot_responses_over_imu(
                x_imu, y_inp, axis=ax[f"{image_file} int-pol"], color=c_colour,
                prediction=inp_q50[r], error=(inp_q25[r], inp_q75[r]),
                c=r, sun=sun_azi, x_ticks=False, filtered=False)

    # Add text
    ax[f"{image_files[0]} pol"].set_title("POL", fontsize=8)
    ax[f"{image_files[0]} int"].set_title("INT", fontsize=8)
    ax[f"{image_files[0]} int-pol"].set_title("INT-POL", fontsize=8)
    fig.tight_layout()

    outpath = os.path.join(out_base, "all")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if figure is None:
        figure = 'none'

    fig.savefig(os.path.join(outpath, f"{dataset}-{figure}-responses.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(outpath, f"{dataset}-{figure}-responses.png"), bbox_inches="tight")

    plt.show()


def plot_mae_circ(*datasets, unit_type="INT-POL", metric="mean absolute error", ring_size=20):

    global min_rs, max_rs

    if len(datasets) < 1:
        datasets = default_datasets

    metric = metric.replace(" ", "_")
    absolute = metric in ["mean_absolute_error", "root_mean_square_error", "max_error"] or "sd" in metric
    # scale = np.pi if "sd" in metric.lower() else 1.
    scale = 1.

    mae_df = pd.read_csv(os.path.join(data_base, "mae_all.csv"))

    # select dataset from the requested dataset
    mae_df = mae_df[np.any([mae_df["dataset"] == ds for ds in datasets], axis=0)]

    # exclude the INT+POL response
    mae_df = mae_df[mae_df["unit_type"] != "INT+POL"]

    # transform time to the correct format
    mae_df["time"] = pd.to_datetime(mae_df["time"])

    # correct the scale for SD data
    mae_df[metric] *= scale

    mosaic = [["ring_size", "sun_ele"],
              ["clouds", "canopies"]]
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(6, 5), subplot_kw={'projection': 'polar'})

    df_select = mae_df[np.all([
        mae_df["cloud_status"] < 3,
        mae_df["canopy_status"] < 2,
        np.isclose(mae_df["tilt"], 0),
        mae_df["sun_elevation"] >= 10,
        mae_df["sun_elevation"] <= 80,
        mae_df["category"] == "tod",
        mae_df["unit_type"] == unit_type
    ], axis=0)]

    plot_prediction_bars(df_select, x=metric, y="ring_size", cmap='viridis', axis=ax["ring_size"],
                         title="number of units", absolute=absolute)

    # sun elevation
    df_select = mae_df[np.all([
        mae_df["cloud_status"] < 3,
        mae_df["canopy_status"] < 2,
        np.isclose(mae_df["tilt"], 0),
        mae_df["ring_size"] == ring_size,
        mae_df["unit_type"] == unit_type
    ], axis=0)]

    ele_bins = np.linspace(-10, 90, 21)
    ele_labels = np.linspace(-5, 90, 20, dtype=int)
    df_select["ele_bins"] = pd.cut(x=df_select["sun_elevation"], bins=ele_bins, labels=ele_labels, include_lowest=True)

    plot_prediction_bars(df_select, x=metric, y="ele_bins", cmap='viridis', axis=ax["sun_ele"],
                         title="sun elevation", absolute=absolute)

    df_select = mae_df[np.all([
        mae_df["canopy_status"] < 2,
        np.isclose(mae_df["tilt"], 0),
        mae_df["ring_size"] == ring_size,
        mae_df["sun_elevation"] > 5,
        mae_df["unit_type"] == unit_type
    ], axis=0)]

    plot_prediction_bars(df_select, x=metric, y="cloud_status", cmap='tab10', axis=ax["clouds"],
                         title="cloud cover", absolute=absolute)

    df_select = mae_df[np.all([
        mae_df["cloud_status"] < 3,
        np.isclose(mae_df["tilt"], 0),
        mae_df["ring_size"] == ring_size,
        mae_df["sun_elevation"] > 5,
        mae_df["unit_type"] == unit_type
    ], axis=0)]

    plot_prediction_bars(df_select, x=metric, y="canopy_status", cmap='Dark2', axis=ax["canopies"],
                         title="canopy level", absolute=absolute)

    fig.tight_layout()

    outpath = os.path.join(out_base, "all")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if datasets == default_datasets:
        dataset = 'all'
    else:
        dataset = '+'.join(datasets)

    filename = f"{dataset}-{metric.lower()}-{unit_type.lower()}-u{ring_size:02d}-{'mae' if absolute else 'sun'}"

    fig.savefig(os.path.join(outpath, f"{filename}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(outpath, f"{filename}.png"), bbox_inches="tight")

    plt.show()


def plot_circ_summary(*datasets, plot_type, unit_type=None, default_ring_size=16):

    global min_rs, max_rs

    if len(datasets) < 1:
        datasets = default_datasets

    metric = "mean_error"

    error_df = pd.read_csv(os.path.join(data_base, "mae_all.csv"))

    # select dataset from the requested dataset
    error_df = error_df[np.any([error_df["dataset"] == ds for ds in datasets], axis=0)]

    # exclude the INT+POL response
    error_df = error_df[error_df["unit_type"] != "INT+POL"]

    # transform time to the correct format
    error_df["time"] = pd.to_datetime(error_df["time"])

    if unit_type is None:
        unit_type = ["INT", "POL", "INT-POL"]
    elif not isinstance(unit_type, list):
        unit_type = [unit_type]

    select = [np.any([
        error_df["unit_type"] == ut for ut in unit_type
    ], axis=0)]

    if plot_type == "units":

        select += [
            np.isclose(error_df["tilt"], 0),
            error_df["sun_elevation"] >= 10,
            error_df["sun_elevation"] <= 80,
            np.any([
                error_df["ring_size"] == 3,
                error_df["ring_size"] == 4,
                error_df["ring_size"] == 5,
                error_df["ring_size"] == 8,
                error_df["ring_size"] == 16,
                error_df["ring_size"] == 24,
                error_df["ring_size"] == 60,
            ], axis=0)
        ]

        df_select = error_df[np.all(select, axis=0)]

        category = "ring_size"
        cmap = "cloud_status"
        emap = "canopy_status"
        cat_list = np.unique(df_select["ring_size"])
    elif plot_type == "solar-elevation":
        select += [
            np.isclose(error_df["tilt"], 0),
            error_df["ring_size"] == default_ring_size
        ]

        df_select = error_df[np.all(select, axis=0)]

        ele_bins = np.linspace(-10, 90, 21)
        ele_labels = np.linspace(-5, 90, 20, dtype=int)
        df_select["ele_bins"] = pd.cut(x=df_select["sun_elevation"], bins=ele_bins, labels=ele_labels, include_lowest=True)

        category = "ele_bins"
        cmap = "cloud_status"
        emap = "canopy_status"
        cat_list = np.unique(df_select["ele_bins"])
    elif plot_type == "cloud-level":

        select += [
            error_df["sun_elevation"] >= 10,
            error_df["sun_elevation"] <= 80,
            np.isclose(error_df["tilt"], 0),
            error_df["ring_size"] == default_ring_size
        ]

        df_select = error_df[np.all(select, axis=0)]

        category = "cloud_status"
        cmap = "sun_elevation"
        emap = "canopy_status"
        cat_list = np.unique(df_select["cloud_status"])
    elif plot_type == "occlusion-level":

        select += [
            error_df["sun_elevation"] >= 10,
            error_df["sun_elevation"] <= 80,
            np.isclose(error_df["tilt"], 0),
            error_df["ring_size"] == default_ring_size
        ]

        df_select = error_df[np.all(select, axis=0)]

        category = "canopy_status"
        emap = "sun_elevation"
        cmap = "cloud_status"
        cat_list = np.unique(df_select["canopy_status"])
    else:
        return

    mosaic = []
    for cat in cat_list:
        mosaic.append([])
        for ut in unit_type:
            mosaic[-1].append(f"{cat}-{ut}")

    fig, ax = plt.subplot_mosaic(mosaic, figsize=(2 * len(unit_type), 2 * len(cat_list)),
                                 subplot_kw={'projection': 'polar'})

    show_title = True
    for cat in cat_list:
        print(cat)
        for ut in unit_type:
            axn = ax[f"{cat}-{ut}"]

            dfs = df_select[np.all([df_select[category] == cat, df_select["unit_type"] == ut], axis=0)]

            if show_title:
                kwargs = {"title": ut}
            else:
                kwargs = {}
            plot_prediction_bars2(dfs, x=metric, y="time", axis=axn, cmap=cmap, emap=emap, **kwargs)
        show_title = False

    fig.tight_layout()

    outpath = os.path.join(out_base, "all")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if datasets == default_datasets:
        dataset = 'all'
    else:
        dataset = '+'.join(datasets)

    filename = f"{dataset}-{plot_type}-summary"

    fig.savefig(os.path.join(outpath, f"{filename}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(outpath, f"{filename}.png"), bbox_inches="tight")

    plt.show()


def plot_error_boxes(*datasets, unit_type="INT-POL", metric="mean absolute error",
                     plot_type="solar-elevation", default_ring_size=8):

    if len(datasets) < 1:
        datasets = default_datasets

    metric = metric.lower().replace(" ", "_")

    error_df = pd.read_csv(os.path.join(data_base, "error_fz_eig.csv"))

    # select dataset from the requested dataset
    error_df = error_df[np.any([error_df["dataset"] == ds for ds in datasets], axis=0)]

    # exclude the INT+POL response
    error_df = error_df[error_df["unit_type"] != "INT+POL"]

    if datasets == default_datasets:
        dataset = 'all'
    else:
        dataset = '+'.join(datasets)

    filename = f"{dataset}-{metric}-{unit_type}-{plot_type}"

    if plot_type != "model-comparison":
        fig = plt.figure(filename, figsize=(3, 2))

    if plot_type == "units":
        df_select = error_df[np.all([
            error_df["cloud_status"] < 3,
            error_df["canopy_status"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["sun_elevation"] >= 10,
            error_df["sun_elevation"] <= 80,
            error_df["unit_type"] == unit_type,
            error_df["category"] == "tod",
            np.any([
                error_df["ring_size"] == 3,
                error_df["ring_size"] == 4,
                error_df["ring_size"] == 5,
                error_df["ring_size"] == 8,
                error_df["ring_size"] == 16,
                error_df["ring_size"] == 24,
                error_df["ring_size"] == 60,
            ], axis=0)
        ], axis=0)]

        sb.boxplot(df_select, x="ring_size", y=metric, palette='viridis')
        print(df_select.groupby(by="ring_size")[metric].median())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 15, 60], ["1", "4", "15", "60"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-60, -15, -4, 0, 4, 15, 60], ["-60", "-15", "-4", "0", "4", "15", "60"])
            plt.ylim([-90, 90])
    elif plot_type == "solar-elevation":  # solar elevation

        # sun elevation
        df_select = error_df[np.all([
            error_df["cloud_status"] < 3,
            error_df["canopy_status"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["ring_size"] == default_ring_size,
            error_df["unit_type"] == unit_type
        ], axis=0)]

        ele_bins = np.linspace(-10, 90, 21)
        ele_labels = np.linspace(-5, 90, 20, dtype=int)
        df_select["ele_bins"] = pd.cut(x=df_select["sun_elevation"], bins=ele_bins, labels=ele_labels, include_lowest=True)

        sb.boxplot(df_select, x="ele_bins", y=metric, palette='viridis')
        # print(df_select.groupby(by="ele_bins")[metric].median())
        print(df_select[df_select["ele_bins"] > 10].groupby(by="ele_bins")[metric].median().mean())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 15, 60], ["1", "4", "15", "60"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-60, -15, -4, 0, 4, 15, 60], ["-60", "-15", "-4", "0", "4", "15", "60"])
            plt.ylim([-90, 90])
    elif plot_type == "cloud-level":

        # cloud cover
        df_select = error_df[np.all([
            error_df["canopy_status"] < 2,
            error_df["sun_elevation"] >= 15,
            np.isclose(error_df["tilt"], 0),
            error_df["ring_size"] == default_ring_size,
            error_df["unit_type"] == unit_type
        ], axis=0)]

        order = [0, 1, 2, 5, 3, 4, 6, 7]
        sb.boxplot(df_select, x="cloud_status", y=metric, palette='tab10', order=order)

        print(df_select.groupby(by="cloud_status")[metric].median())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-64, -16, -4, 0, 4, 16, 64], ["-64", "-16", "-4", "0", "4", "16", "64"])
            plt.ylim([-90, 90])
    elif plot_type == "occlusion-level":

        # sun elevation
        df_select = error_df[np.all([
            error_df["cloud_status"] < 5,
            error_df["sun_elevation"] >= 15,
            np.isclose(error_df["tilt"], 0),
            error_df["ring_size"] == default_ring_size,
            error_df["unit_type"] == unit_type
        ], axis=0)]

        sb.boxplot(df_select, x="canopy_status", y=metric, palette='Dark2')

        print(df_select.groupby(by="canopy_status")[metric].median())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-64, -16, -4, 0, 4, 16, 64], ["-64", "-16", "-4", "0", "4", "16", "64"])
            plt.ylim([-90, 90])
    elif plot_type == "model-comparison":

        fig = plt.figure(filename, figsize=(6, 2))

        clouds = [[0, 1, 2], [3, 4, 5, 6, 7]]
        occlusion = [[0, 1], [2, 3, 4, 5]]
        solar = [[0, 1], [2]]
        ring_sizes = [default_ring_size, 36]

        df_select = error_df[np.all([
            error_df["sun_elevation"] >= 15,
            np.isclose(error_df["tilt"], 0),
            np.any([error_df["ring_size"] == default_ring_size,
                    error_df["ring_size"] == 36], axis=0)
        ], axis=0)]

        sky_condition = df_select["cloud_status"].to_numpy() * np.nan
        sky_condition[np.all([
            np.any([df_select["cloud_status"] == c for c in clouds[0]], axis=0),
            np.any([df_select["canopy_status"] == c for c in occlusion[0]], axis=0),
            np.any([df_select["sun_status"] == c for c in solar[0]], axis=0)
        ], axis=0)] = 0  # clear
        sky_condition[np.all([
            np.any([df_select["cloud_status"] == c for c in clouds[1]], axis=0),
            np.any([df_select["canopy_status"] == c for c in occlusion[0]], axis=0),
            np.any([df_select["sun_status"] == c for c in solar[0]], axis=0)
        ], axis=0)] = 1  # cloudy
        sky_condition[np.all([
            np.any([df_select["cloud_status"] == c for c in clouds[0]], axis=0),
            np.any([df_select["canopy_status"] == c for c in occlusion[1]], axis=0),
            np.any([df_select["sun_status"] == c for c in solar[0]], axis=0)
        ], axis=0)] = 2  # occluded
        sky_condition[np.any([
            np.any([df_select["sun_status"] == c for c in solar[1]], axis=0)
        ], axis=0)] = 3  # solar or anti-solar meridian is hidden
        df_select["sky_condition"] = sky_condition
        df_select = df_select.dropna(axis=0)

        for i, ring_size in enumerate(ring_sizes):
            ax = plt.subplot(121 + i)

            dfs = df_select[df_select["ring_size"] == ring_size]

            palette_body = {"POL": p_colour, "INT": i_colour, "INT-POL": c_colour,
                            "FZ": z_colour, "EIG": e_colour}
            sb.boxplot(dfs, x="sky_condition", y=metric, hue="unit_type", palette=palette_body,
                       hue_order=["INT-POL", "POL", "INT", "EIG", "FZ"], fliersize=.5, ax=ax)

            print(f"Number of devices: {ring_size}")
            print(dfs.groupby(by=["unit_type", "sky_condition"])[metric].median())

            plt.yscale('symlog', base=2)
            if "absolute" in metric or "square" in metric or "sd" in metric:
                plt.yticks([1, 4, 16, 64], ["1", "4", "16", "64"])
                plt.ylim([0, 90])
            else:
                plt.yticks([-64, -16, -4, 0, 4, 16, 64], ["-64", "-16", "-4", "0", "4", "16", "64"])
                plt.ylim([-90, 90])

            ax.legend_.remove()

    elif plot_type == "location":

        df_select = error_df[np.all([
            error_df["cloud_status"] < 6,
            error_df["canopy_status"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["sun_elevation"] >= 15,
            error_df["unit_type"] == unit_type,
            # error_df["category"] == "tod",
            error_df["ring_size"] == default_ring_size
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

    outpath = os.path.join(out_base, "all")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    fig.savefig(os.path.join(outpath, f"{filename}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(outpath, f"{filename}.png"), bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_error_lines(*datasets, unit_type="INT-POL", metric="mean absolute error",
                     plot_type="solar-elevation", default_ring_size=8):

    if len(datasets) < 1:
        datasets = default_datasets

    metric = metric.lower().replace(" ", "_")

    error_df = pd.read_csv(os.path.join(data_base, "mae_all.csv"))

    # select dataset from the requested dataset
    error_df = error_df[np.any([error_df["dataset"] == ds for ds in datasets], axis=0)]

    # exclude the INT+POL response
    error_df = error_df[error_df["unit_type"] != "INT+POL"]

    if datasets == default_datasets:
        dataset = 'all'
    else:
        dataset = '+'.join(datasets)

    filename = f"{dataset}-{metric}-{unit_type}-{plot_type}-line"

    fig = plt.figure(filename, figsize=(3, 2))

    if plot_type == "units":
        df_select = error_df[np.all([
            error_df["cloud_status"] < 3,
            error_df["canopy_status"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["sun_elevation"] >= 15,
            error_df["unit_type"] == unit_type,
            error_df["category"] == "tod",
        ], axis=0)]

        # y_q25 = df_select.groupby("ring_size")[metric].quantile(.25)
        y_q50 = df_select.groupby("ring_size")[metric].median()
        # y_q75 = df_select.groupby("ring_size")[metric].quantile(.75)
        size = df_select.shape[0]
        print(size)
        df_sample = df_select.iloc[np.random.permutation(size)[:500]]

        sb.regplot(df_sample, x="ring_size", y=metric, fit_reg=False, x_jitter=.5,
                   scatter_kws={'color': 'black', 'alpha': 0.1, 's': 5})
        # plt.fill_between(y_q50.index.to_numpy(), y_q25.to_numpy(), y_q75.to_numpy(),
        #                  facecolor='black', alpha=0.2)
        plt.plot(y_q50.index.to_numpy(), y_q50.to_numpy(), color='black', lw=2)

        # sb.boxplot(df_select, x="ring_size", y=metric, palette='viridis')
        print(df_select.groupby(by="ring_size")[metric].median())

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
    elif plot_type == "solar-elevation":  # solar elevation

        # sun elevation
        df_select = error_df[np.all([
            error_df["cloud_status"] < 5,
            error_df["canopy_status"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["ring_size"] == default_ring_size,
            error_df["unit_type"] == unit_type
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
    elif plot_type == "cloud-level":

        # cloud cover
        df_select = error_df[np.all([
            error_df["canopy_status"] < 2,
            error_df["sun_elevation"] >= 10,
            error_df["sun_elevation"] <= 80,
            np.isclose(error_df["tilt"], 0),
            error_df["ring_size"] == default_ring_size,
            error_df["unit_type"] == unit_type
        ], axis=0)]

        sb.boxplot(df_select, x="cloud_status", y=metric, palette='tab10')

        print(df_select.groupby(by="cloud_status")[metric].median())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 15, 60], ["1", "4", "15", "60"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-60, -15, -4, 0, 4, 15, 60], ["-60", "-15", "-4", "0", "4", "15", "60"])
            plt.ylim([-90, 90])
    elif plot_type == "occlusion-level":

        # sun elevation
        df_select = error_df[np.all([
            error_df["cloud_status"] < 3,
            error_df["sun_elevation"] >= 10,
            error_df["sun_elevation"] <= 80,
            np.isclose(error_df["tilt"], 0),
            error_df["ring_size"] == default_ring_size,
            error_df["unit_type"] == unit_type
        ], axis=0)]

        sb.boxplot(df_select, x="canopy_status", y=metric, palette='Dark2')

        print(df_select.groupby(by="canopy_status")[metric].median())

        plt.yscale('symlog', base=2)
        if "absolute" in metric or "square" in metric or "sd" in metric:
            plt.yticks([1, 4, 15, 60], ["1", "4", "15", "60"])
            plt.ylim([0, 90])
        else:
            plt.yticks([-60, -15, -4, 0, 4, 15, 60], ["-60", "-15", "-4", "0", "4", "15", "60"])
            plt.ylim([-90, 90])
    elif plot_type == "location":

        df_select = error_df[np.all([
            error_df["cloud_status"] < 6,
            error_df["canopy_status"] < 2,
            np.isclose(error_df["tilt"], 0),
            error_df["sun_elevation"] >= 10,
            error_df["sun_elevation"] <= 85,
            error_df["unit_type"] == unit_type,
            # error_df["category"] == "tod",
            error_df["ring_size"] == default_ring_size
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

    outpath = os.path.join(out_base, "all")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    fig.savefig(os.path.join(outpath, f"{filename}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(outpath, f"{filename}.png"), bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_mae(*datasets):

    global min_rs, max_rs

    if len(datasets) < 1:
        datasets = default_datasets

    mae_df = pd.read_csv(os.path.join(data_base, "mae_all_1.csv"))
    mae_df = mae_df[mae_df["unit_type"] != "INT+POL"]
    mae_df["time"] = pd.to_datetime(mae_df["time"])

    min_rs = np.min(mae_df["ring_size"])
    max_rs = np.max(mae_df["ring_size"])

    palette = {rs: cmap_lines(norm_lines(rs)) for rs in range(min_rs, max_rs + 1)}

    unit_type = "INT"
    metric = "MAE"
    # metric = "MAE_rel"
    # metric = "turtuosity"
    mosaic = [["ring_size", "ring_size"],
              ["tod", "tod"],
              ["clouds", "canopies"]]
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(6, 6))

    df_select = mae_df[np.all([
        mae_df["cloud status"] < 3,
        mae_df["canopy status"] < 2,
        np.isclose(mae_df["tilt"], 0),
        mae_df["sun_elevation"] >= 10,
        mae_df["sun_elevation"] <= 80,
        mae_df["category"] == "tod",
        mae_df["unit_type"] == unit_type
    ], axis=0)]

    sb.boxplot(df_select, x="ring_size", y=metric, ax=ax["ring_size"], palette=palette)

    ax["ring_size"].set_yscale('symlog', base=2)
    ax["ring_size"].set_yticks([1, 4, 15, 60])
    ax["ring_size"].set_yticklabels(["1", "4", "15", "60"])
    ax["ring_size"].set_ylim([0, 90])
    ax["ring_size"].legend([], [], frameon=False)

    palette = {rs: cmap_lines(norm_lines(rs)) for rs in np.linspace(-5, 90, 20)}

    # replace unit_type names to numbers
    # unit_type = np.where(mae_df["unit_type"][:, None] == [["INT", "POL", "INT-POL"]])[1]
    # mae_df = mae_df.assign(unit_type=unit_type)
    df_select = mae_df[np.all([
        mae_df["cloud status"] < 3,
        mae_df["canopy status"] < 2,
        np.isclose(mae_df["tilt"], 0),
        mae_df["ring_size"] == 20,
        mae_df["unit_type"] == unit_type
    ], axis=0)]

    ele_bins = np.linspace(-10, 90, 21)
    ele_labels = np.linspace(-5, 90, 20, dtype=int)
    df_select["ele_bins"] = pd.cut(x=df_select["sun_elevation"], bins=ele_bins, labels=ele_labels, include_lowest=True)

    sb.boxplot(df_select, x="ele_bins", y=metric, ax=ax["tod"], palette=palette)

    ax["tod"].set_yscale('symlog', base=2)
    ax["tod"].set_yticks([1, 4, 15, 60])
    ax["tod"].set_yticklabels(["1", "4", "15", "60"])
    ax["tod"].set_ylim([0, 90])
    ax["tod"].legend([], [], frameon=False)

    palette = {rs: cmap_lines(norm_lines(rs)) for rs in range(0, 10)}

    df_select = mae_df[np.all([
        mae_df["canopy status"] < 2,
        np.isclose(mae_df["tilt"], 0),
        mae_df["ring_size"] == 20,
        mae_df["sun_elevation"] > 5,
        mae_df["unit_type"] == unit_type
    ], axis=0)]

    sb.boxplot(df_select, x="cloud status", y=metric, ax=ax["clouds"], palette=palette)

    ax["clouds"].set_yscale('symlog', base=2)
    ax["clouds"].set_yticks([1, 4, 15, 60])
    ax["clouds"].set_yticklabels(["1", "4", "15", "60"])
    ax["clouds"].set_ylim([0, 90])
    ax["clouds"].legend([], [], frameon=False)

    df_select = mae_df[np.all([
        mae_df["cloud status"] < 3,
        np.isclose(mae_df["tilt"], 0),
        mae_df["ring_size"] == 20,
        mae_df["sun_elevation"] > 5,
        mae_df["unit_type"] == unit_type
    ], axis=0)]
    sb.boxplot(df_select, x="canopy status", y=metric, ax=ax["canopies"], palette=palette)

    ax["canopies"].set_yscale('symlog', base=2)
    ax["canopies"].set_yticks([1, 4, 15, 60])
    ax["canopies"].set_yticklabels(["", "", "", ""])
    ax["canopies"].set_ylabel(None)
    ax["canopies"].set_ylim([0, 90])
    ax["canopies"].legend([], [], frameon=False)

    plt.tight_layout()
    plt.show()


def plot_summary_mae(*datasets, category="tod", fig=None, ax=None, show=False, save=False):
    if len(datasets) < 1:
        datasets = default_datasets

    figs, axs = [], []
    for dataset in datasets:
        df = read_csv_dataset(dataset)
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
        axis.set_title(image_name.split(".")[0].replace("_", " "), fontsize=8)
    else:
        axis.set_title("No image found", fontsize=8)

    axis.set_xticks([])
    axis.set_yticks([])

    return axis


def plot_circ_error_bars(theta, rho, errors, color='black', edge_color='black', alpha=1., axis=None):
    q25, q75 = circ_norm(np.asanyarray(errors))
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
        r = butter_low_pass_filter(np.r_[r[-50:], r, r[:50]])[50:-50]

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


def plot_prediction_bars(data: pd.DataFrame, y, x, cmap='inferno', x_ticks=False, absolute=True, title=None, axis=None):

    if absolute:
        axis.set_theta_direction(1)
        axis.set_theta_zero_location("E")

        axis.quiver(0, -1, 2, 0, scale=.02, width=.02, color='lightgreen', zorder=0)

        y_max_tick = 5
    else:
        axis.set_theta_direction(-1)
        axis.set_theta_zero_location("N")

        axis.quiver(0, -1, 0, 2, scale=.02, width=.02, color='lightgreen', zorder=0)

        y_max_tick = 5

    data_temp = data.copy()
    data_temp = data_temp.assign(**{x: data_temp[x].apply(np.deg2rad).apply(circ_norm)})
    if absolute:
        data_temp = data_temp.assign(**{x: data_temp[x].apply(abs)})

    data_grouped = data_temp.groupby(by=y)[x].apply(circ_quantiles).dropna()

    ys = data_grouped.index.to_numpy()
    q25s, q50s, q75s = np.stack(data_grouped.to_numpy()).T

    y_min, y_max = np.maximum(np.min(ys), 0), np.max(ys)

    normal = mpl.colors.Normalize(y_min - 0.5, y_max + 0.5)
    cmap_rs = mpl.colormaps[cmap]

    for yi, q25, q50, q75 in zip(ys, q25s, q50s, q75s):
        color = cmap_rs(normal(yi))
        y_norm = y_max_tick * (yi - y_min + 0.1) / (y_max - y_min + 0.2)
        print(f"{title}: {yi:.0f}, MAE = {np.rad2deg(q50):.2f} ({np.rad2deg(q25):.2f}, {np.rad2deg(q75):.2f})")
        plot_circ_error_bars(q50, y_norm, [q25, q75], color=color, axis=axis)

    axis.set_yticks([-1, 0])
    axis.set_yticklabels([""] * 2)
    axis.set_xticks([-3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axis.set_xticklabels(["", "W", "", "N", "", "E", "", "S"])
    axis.spines["polar"].set_visible(False)

    if not x_ticks:
        axis.set_xticks([])

    # ax.set_ylim([-y_padding, 2 * np.pi + y_padding])
    if absolute:
        axis.set_ylim([-1 - (y_max_tick + 1) * 0.03, y_max_tick * 1.03 + 0.03])
        axis.set_xlim([0, np.pi])
    else:
        axis.set_ylim([-1 - (y_max_tick + 1) * 0.03, y_max_tick * 1.03 + 0.03])
        axis.set_xlim([-np.pi, np.pi])

    if title is not None:
        axis.set_title(title)


def plot_prediction_bars2(data: pd.DataFrame, y, x, cmap='viridis', emap=None, title=None, axis=None):

    axis.set_theta_direction(-1)
    axis.set_theta_zero_location("N")

    axis.quiver(0, -1, 0, 2, scale=.02, width=.02, color='lightgreen', zorder=0)

    y_max_tick = 5

    data_temp = data.copy()
    data_temp = data_temp.assign(**{x: data_temp[x].apply(np.deg2rad).apply(circ_norm)})

    data_grouped = data_temp.groupby(by=[y, cmap, emap])[x].apply(circ_quantiles).dropna()

    ys = data_grouped.index.get_level_values(y)
    ys = np.array([np.datetime64(yi, 'm').astype('long') for yi in ys])
    cs = data_grouped.index.get_level_values(cmap).to_numpy()
    es = data_grouped.index.get_level_values(emap).to_numpy()

    q25s, q50s, q75s = np.stack(data_grouped.to_numpy()).T

    c_min, c_max = np.min(cs), np.max(cs)
    e_min, e_max = np.min(es), np.max(es)

    c_normal = mpl.colors.Normalize(c_min - 0.5, 10 + 0.5)
    e_normal = mpl.colors.Normalize(e_min - 0.5, 10 + 0.5)
    cmap_rs = mpl.colormaps[cmap_dict[cmap]]
    emap_rs = mpl.colormaps[cmap_dict[emap]]

    for yi, ci, ei, q25, q50, q75 in zip(np.arange(len(ys)), cs, es, q25s, q50s, q75s):
        color = cmap_rs(c_normal(ci))
        edge_color = emap_rs(e_normal(ei))
        y_norm = y_max_tick * (yi + 0.1) / (len(ys) + 0.2)
        print(f"{title}: {yi:.0f}, MAE = {np.rad2deg(q50):.2f} ({np.rad2deg(q25):.2f}, {np.rad2deg(q75):.2f})")
        plot_circ_error_bars(q50, y_norm, [q25, q75], color=color, edge_color=edge_color, axis=axis)

    axis.set_yticks([-1, 0])
    axis.set_yticklabels([""] * 2)
    axis.set_xticks([-3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    axis.set_xticklabels(["", "W", "", "N", "", "E", "", "S"])
    axis.spines["polar"].set_visible(False)

    axis.set_xticks([])

    axis.set_ylim([-1 - (y_max_tick + 1) * 0.03, y_max_tick * 1.03 + 0.03])
    axis.set_xlim([-np.pi, np.pi])

    if title is not None:
        axis.set_title(title)


def plot_prediction_over_imu(x, y, axis=None, x_ticks=True):

    if axis is None:
        axis = plt.subplot(111)

    x = (x + np.pi) % (2 * np.pi) - np.pi
    y = (y + np.pi) % (2 * np.pi) - np.pi

    i_sort = np.argsort(x[0])
    x = x[:, i_sort]
    r = y[:, i_sort]

    x_mean = x[0]
    r_mean = circ_mean(r, axis=0, nan_policy='omit')

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
    e_mean = (circ_mean(e, axis=0, nan_policy='omit') + np.pi) % (2 * np.pi) - np.pi

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


if __name__ == "__main__":
    import warnings

    warnings.simplefilter('ignore')

    # metrics: mean error, mean absolute error, root mean square error, max error
    # plot_all()
    # plot_circ_summary(plot_type="units")
    # plot_circ_summary(plot_type="solar-elevation")
    # plot_circ_summary(plot_type="cloud-level")
    # plot_circ_summary(plot_type="occlusion-level")
    # plot_error_lines(plot_type="units", metric="mean error sd", default_ring_size=8)
    # plot_error_lines(plot_type="solar-elevation", metric="root mean square error", default_ring_size=8)
    # plot_error_boxes(plot_type="cloud-level", metric="mean error sd", default_ring_size=8)
    # plot_error_boxes(plot_type="occlusion-level", metric="mean error sd", default_ring_size=8)
    plot_error_boxes("south_africa_data", plot_type="model-comparison", metric="root mean square error", default_ring_size=8)
    # plot_error_boxes(plot_type="location", metric="mean error sd", default_ring_size=8)
    # plot_error_boxes(plot_type="dataset", metric="root mean square error", default_ring_size=8)
    # plot_mae_circ(unit_type="INT-POL", metric="root mean square error sd", ring_size=8)
    # plot_responses()
    # plot_responses('sardinia_data')
    # plot_responses(figure='solar-elevation')
    # plot_responses(figure='cloud-level')
    # plot_responses(figure='occlusion-level')

    # figure, axes = None, None
    # plot_summary_mae(category="tod", fig=figure, ax=axes, show=True, save=True)
    # plot_summary_mae(category="canopy", fig=figure, ax=axes, show=True, save=True)
    # plot_summary_mae('south_africa_data', category="cloud_cover", fig=figure, ax=axes, show=True, save=True)
    # plot_summary_mae('south_africa_data', category="canopy", fig=figure, ax=axes, show=True, save=True)
    # plot_summary_mae('south_africa_data', category="tilt", fig=figure, ax=axes, show=True, save=True)
