"""
plot_from_dict.py

Central plotting utilities (so bagfiles and json files can essentially
be plotted by the same code).

The main function provided by this module is produce_plot which will
take a data dictionary and produce a SkyCompass/IMU comparison plot
for a given recording session.
"""
from models import decode_sky_compass, unit2pol

from scipy.stats import circmean

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json
import argparse
import os

from skylight import Sun


def synchronise(d, zero=True, unwrap=True):
    """
    Zero each angle series and adjust values so wraps plot nicely.

    :param d: A series of angles (a single recording)
    :param zero: Shift series so angles start from zero
    :param unwrap: Unwrap angles which jump over 0/2pi boundary.
    :returns: The adjusted array.
    """
    d = np.array(d)
    if zero:
        d = d - d[0]
    d = d % (2 * np.pi)
    if unwrap:
        d = np.unwrap(d, np.pi, period=2 * np.pi)
        # for i in range(len(d) - 1):
        #     if (d[i] < 3 * np.pi / 4) and (d[i+1] > 7 * np.pi / 4):
        #         d[i+1] = d[i+1] - 2 * np.pi
        #     elif (d[i] > 7 * np.pi/4) and (d[i+1] < 3 * np.pi/4):
        #         d[i+1] = d[i+1] + 2 * np.pi

    return d


def produce_plot(data_dictionary, polarisation=True, intensity=False, ring_size=8, nb_samples=360, observer=None):

    mosaic = [["image", "responses"],
              ["image", "prediction"]]

    if observer is not None:
        sun = Sun(observer)
        print(np.rad2deg(sun.az))
        synchronise_sun = lambda x: x - sun.az
        synchronise_imu = lambda x: x
    else:
        synchronise_sun = synchronise
        synchronise_imu = synchronise

    fig, ax = plt.subplot_mosaic(mosaic, figsize=(10, 4))

    # Plot image if available
    plot_image(data_dictionary["image_filename"], ax=ax["image"])
    del data_dictionary["image_filename"]

    mse = []

    pref_angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False)
    imu_angles = (np.linspace(0, 2 * np.pi, nb_samples, endpoint=False) + np.pi) % (2 * np.pi) - np.pi
    all_pol_angles = np.zeros((len(data_dictionary.keys()), nb_samples), dtype='float32')
    print("MAE:", end=" ")
    for c, key in enumerate(data_dictionary.keys()):
        data = data_dictionary[key]

        # Extract data
        imu = synchronise_imu(np.array(data["yaw"]))
        pol_op_keys = ["pol_op_{}".format(x) for x in range(8)]
        po = []
        for k in pol_op_keys:
            po.append(data[k])

        imus = np.linspace(imu[:-1], imu[1:], 8, endpoint=False)
        yaws = (imus.T - data['azimuths']).T
        po = np.array(po)

        # unit 5 photoreceptor 1 is faulty, so it is ignored
        # unit 7 is faulty, so it is also ignored
        i_valid = [0, 1, 2, 3, 4, 6]
        imu = yaws[i_valid].reshape((-1,))
        pol = po[i_valid, 1:].reshape((-1, 4))
        plot_responses_over_imu(imu, pol, ax=ax["responses"], c=c, x_ticks=False)
        # print(imu.shape, pol.shape)

        pol_angles = np.zeros(nb_samples, dtype='float32')
        for sam in range(nb_samples):
            rotate = imu_angles[sam]
            a_dist = abs((imu[:, None] + pref_angles[None, :] - rotate + np.pi) % (2 * np.pi) - np.pi)
            i_closest = np.argmin(a_dist, axis=0)
            po = pol[i_closest]
            pol_angles[sam], pol_sensor_sigma = decode_sky_compass(po, pol_prefs=pref_angles,
                                                                   polarisation=polarisation, intensity=intensity)
        # print(pol_angles.shape, pol_angles)

        all_pol_angles[c] = synchronise_sun(pol_angles)
        plot_prediction_over_imu(imu_angles, all_pol_angles[c], ax=ax["prediction"], c=c)

        mse.append(compare(all_pol_angles[c], imu_angles))
        print(f"{np.rad2deg(mse[-1]):.2f}", end="\t")

    y = (circmean(all_pol_angles, axis=0) + np.pi) % (2 * np.pi) - np.pi
    plot_average_over_imu(imu_angles, y, ax=ax["prediction"])
    print(f"\nMean MAE: {np.rad2deg(np.nanmean(mse)):.2f} +/- {np.rad2deg(np.nanstd(mse)):.2f}")

    # Add text
    ax["responses"].set_ylabel("POL-OP\nresponses")
    ax["prediction"].set_ylabel("compass\npredictions")
    fig.tight_layout()

    return fig, mse


def plot_image(imagefile, ax=None):

    if ax is None:
        ax = plt.subplot(111)

    if imagefile is not None:
        img = mpimg.imread(imagefile)
        ax.imshow(img)
        ax.set_title(imagefile.split(".")[0].replace("_", " "))
    else:
        ax.set_title("No image found")

    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def plot_responses_over_imu(x, y, ax=None, c=0, y_padding=.1, x_ticks=True):

    if ax is None:
        ax = plt.subplot(111)

    r = unit2pol(y.T / 33000.).T
    # r = (y[:, 0] - y[:, 1]) / (y[:, 0] + y[:, 1] + np.finfo(float).eps)
    ax.scatter((x + np.pi) % (2 * np.pi) - np.pi, r, s=20, facecolor=f'C{c}', marker='.', label=f"PO-{c+1:02d}")
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 0], s=20, facecolor=f'C{c+2}', marker='.')
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 1], s=20, facecolor=f'C{c+4}', marker='.')
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 2], s=20, facecolor=f'C{c+6}', marker='.')
    # ax.scatter((x - np.pi) % (2 * np.pi) - np.pi, y[:, 3], s=20, facecolor=f'C{c+8}', marker='.')

    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    if not x_ticks:
        ax.set_xticks([])

    # ax.set_ylim([-y_padding, 2 * np.pi + y_padding])
    ax.set_ylim([-y_padding - 1, 1 + y_padding])
    ax.set_xlim([-np.pi, np.pi])

    return ax


def plot_prediction_over_imu(x, y, ax=None, c=0, x_ticks=True):

    if ax is None:
        ax = plt.subplot(111)

    y = (y + np.pi) % (2 * np.pi) - np.pi
    ax.scatter(x, y, s=20, facecolor=f'C{c}', marker='.', label=f"PO-{c+1:02d}")

    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    if not x_ticks:
        ax.set_xticks([])

    ax.set_ylim([-np.pi, np.pi])
    ax.set_xlim([-np.pi, np.pi])

    return ax


def plot_average_over_imu(x, y, ax=None, x_ticks=True):

    if ax is None:
        ax = plt.subplot(111)

    i_sort = np.argsort(x)
    y = (y + np.pi) % (2 * np.pi) - np.pi
    x_sort = x[i_sort]
    y_sort = y[i_sort]
    for i in range(1, len(y_sort)):
        while y_sort[i] - y_sort[i-1] > np.pi / 2:
            y_sort[i] -= 2 * np.pi
        while y_sort[i] - y_sort[i-1] < -np.pi / 2:
            y_sort[i] += 2 * np.pi

    for shift in np.linspace(-4 * np.pi, 4 * np.pi, 5):
        ax.plot(x_sort, y_sort + shift, 'k-', lw=2, alpha=0.7)

    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    if not x_ticks:
        ax.set_xticks([])

    ax.set_ylim([-np.pi, np.pi])
    ax.set_xlim([-np.pi, np.pi])

    return ax


def plot_angles(max_length, data=None, ax=None, x_padding=0, y_padding=np.pi/18, x_ticks=True):
    if ax is None:
        ax = plt.subplot(111)

    if data is not None:
        data = np.array(data)
        line, = ax.plot(data)
        c = line.get_color()
        for shift in np.linspace(-4 * np.pi, 4 * np.pi, 5)[[0, 1, 3, 4]]:
            ax.plot(data + shift, c=c)

    # General formatting
    # ax.set_yticks([0, np.pi, 2*np.pi])
    # ax.set_yticklabels([r"$0$", r"$\pi$", r"$2 \pi$"])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

    if not x_ticks:
        ax.set_xticks([])

    # ax.set_ylim([-y_padding, 2 * np.pi + y_padding])
    ax.set_ylim([-np.pi - y_padding, np.pi + y_padding])
    ax.set_xlim([-x_padding, max_length + x_padding - 1])

    return ax


def compare(x, ground_truth):
    ang_distance = (x - ground_truth + np.pi) % (2 * np.pi) - np.pi
    return np.mean(np.abs(ang_distance))
