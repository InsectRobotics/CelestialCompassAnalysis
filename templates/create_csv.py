"""
create_csv.py

Brittle script to produce all plots from the tod and canopy recordings. The
directory structure is assumed so this will break if anything changes.
"""

from image_processing import get_observer_from_file_name, extract_sun_vector_from_name, sun_centres
from rosbag_to_json import rosbag_to_dict
from circstats import circ_mean, circ_std, circ_norm

from scipy.signal import butter, filtfilt
from scipy.optimize import fsolve, newton
from datetime import datetime
from copy import copy

import os
import shutil
import rosbag
import numpy as np
import pandas as pd
import skylight as skl

MAX_INT = 11000.
cutoff = 0.03
fs = 2.0
order = 1
default_nb_samples = 360


data_base = os.path.abspath(os.path.join(os.getcwd(), '..'))
csv_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'csv'))
clean_path = os.path.join(csv_dir, "clean_all_units.csv")
mae_path = os.path.join(csv_dir, "error_fz_eig.csv")

default_datasets = ['sardinia_data', 'south_africa_data']
skip_dirs = ["elevation", "path_integration", "sun_tables", "track_test"]
sun_status = [
    "visible",
    "covered",  # inferable
    "hidden"
]
anti_sun_status = [
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


def butter_low_pass_filter(x):
    beta, alpha = butter(order, cutoff, fs=fs, btype='low', analog=False)
    return filtfilt(beta, alpha, x)


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


def four_zeros(po, pol_prefs, x0=None, verbose=False):
    z0 = 1 / len(pol_prefs) * np.sum(po * np.exp(0j * pol_prefs))
    z1 = 2 / len(pol_prefs) * np.sum(po * np.exp(1j * pol_prefs))
    z2 = 2 / len(pol_prefs) * np.sum(po * np.exp(2j * pol_prefs))

    r0, r1, r2 = np.abs([z0, z1, z2])
    a0, a1, a2 = np.angle([z0, z1, z2]) % (2 * np.pi)

    if verbose:
        print(f"z0: {r0:.2f} exp(i {np.rad2deg(a0):.2f}), "
              f"z1: {r1:.2f} exp(i {np.rad2deg(a1):.2f}), "
              f"z2: {r2:.2f} exp(i {np.rad2deg(a2):.2f})")

    if x0 is None:
        x0 = np.sort((a2 / 2 + np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4) % (2 * np.pi))

    f = lambda a: r1 * np.cos(a1 - a) + r2 * np.cos(a2 - 2 * a)
    df = lambda a: r1 * np.sin(a1 - a) + 2 * r2 * np.sin(a2 - 2 * a)

    # solve the equation to find the 4 zeros
    sol = newton(f, x0, fprime=df)

    # sort the solution
    y = np.sort(np.array(sol) % (2 * np.pi))

    # # 4 zeros algorithm
    # s = np.mean(y)
    # i, m = 0, 0
    # if not np.isclose(y[1] - y[0], y[3] - y[2]):
    #     i = 1
    #     m = 1
    # if y[i % 4] - y[(i + 3) % 4] < y[(i + 2) % 4] - y[(i + 1) % 4]:
    #     m += 2
    #
    # angle = (s + m * np.pi/2) % (2 * np.pi)

    # our alternative
    dy = abs(circ_norm(np.diff(y, append=y[0] + 2 * np.pi)))
    i = np.argmin(dy)
    angle = circ_norm(y[i] + 0.5 * dy[i])

    res = [angle]

    if verbose:
        print(f"predicted solar azimuth: {np.rad2deg(angle):.2f}, "
              f"zeros: ", *[f"{np.rad2deg(yi):.2f}" for yi in y])
        res += [y, z0, z1, z2]

    return tuple(res)


def eigenvectors(s000, s045, s090, s135, pol_prefs, verbose=False):
    phi_1, phi_2, phi_3, phi_4 = np.deg2rad([0, 45, 90, 135])  # in local frame

    # Equation (4), Zhao et al. (2016)
    t1 = s000 / s090
    t2 = s045 / s135

    # Equations (5)
    phi = circ_norm(0.5 * np.arctan2(
        (t2 - 1) * np.cos(2 * phi_1) + (1 - t1) * np.cos(2 * phi_2) +
        (t1 - t1 * t2) * np.cos(2 * phi_3) + (t1 * t2 - t2) * np.cos(2 * phi_4),
        (1 - t2) * np.sin(2 * phi_1) + (t1 - 1) * np.sin(2 * phi_2) +
        (t1 * t2 - t1) * np.sin(2 * phi_3) + (t2 - t1 * t2) * np.sin(2 * phi_4)
    ))
    d = (t1 - 1) / (np.cos(2 * phi - 2 * phi_1) - t1 * np.cos(2 * phi - 2 * phi_3))

    # Sturzl (2017)
    azi = pol_prefs  # the azimuth of the device
    ele = np.full_like(pol_prefs, np.pi/4)  # the elevation of the device

    # alpha = phi - azi  # AOP in global frame (original, from paper)
    # e_par = np.array([-np.cos(azi) * np.sin(ele), -np.sin(azi) * np.sin(ele), np.cos(ele)])  # parallel to e-vector
    # e_per = np.array([-np.sin(azi), np.cos(azi), np.zeros_like(azi)])  # perpendicular to e-vector
    # p = -np.cos(alpha) * e_par + np.sin(alpha) * e_per

    alpha = azi + phi  # AOP in global frame (my change)
    # p = np.array([np.sin(azi), np.cos(azi)])  # this points outwards (towards the facing direction)
    # p = np.array([np.cos(-azi), np.sin(-azi)])  # this points 90 deg from facing direction
    p = -np.array([np.sin(alpha), np.cos(alpha)])  # this points 90 deg from facing direction (inwards)

    # # compute eigenvectors
    # eig = p.dot(p.T)
    # eig[~np.isfinite(eig)] = 0.  # ensure that there are not infinites or NaNs
    # w, v = np.linalg.eig(eig)
    # iv = np.argmax(w)
    # vi = v[iv]

    # compute mean vector (equivalent)
    vi = np.mean(d * p, axis=1)

    z_sun = vi[1] + 1j * vi[0]
    res = [np.angle(z_sun)]

    if verbose:
        res += [phi, d, d * p]

    return tuple(res)


def compare(x, ground_truth, error_type=None, axis=None, std=False):
    ang_distance = circ_norm(x - ground_truth)
    # ang_distance_org = ang_distance.copy()

    if error_type is None:
        error_type = "mean error"

    if error_type in ["mean error", "mne"]:
        result = [np.nanmean(ang_distance, axis=axis)]
    elif error_type in ["mean absolute error", "mae"]:
        ang_distance = abs(ang_distance)
        result = [np.nanmean(ang_distance, axis=axis)]
    elif error_type in ["root mean square error", "rmse"]:
        ang_distance = np.square(ang_distance)
        result = [np.sqrt(np.nanmean(ang_distance, axis=axis))]
    elif error_type in ["max error", "mxe"]:
        ang_distance = abs(ang_distance)
        result = [np.nanmax(ang_distance, axis=axis)]
    else:
        print(f"Unknown error type: '{error_type}'")
        return None

    if std:
        result += [np.nanstd(ang_distance, axis=axis)]

    return result


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


def compute_sensor_output_from_responses(res_pol, res_int, ring_size=8, polarisation=True, intensity=True,
                                         algorithm=None):

    if algorithm is eigenvectors:
        res_000, res_045, res_090, res_135 = res_pol
        res_pol = res_090 - res_000
    else:
        res_000, res_045, res_090, res_135 = None, None, None, None
        s000, s045, s090, s135 = None, None, None, None
        s000_i, s045_i, s090_i, s135_i = None, None, None, None

    nb_recordings, nb_samples = res_pol.shape
    sol_angles = circ_norm(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    pol_angles = circ_norm(np.linspace(0, 2 * np.pi, ring_size, endpoint=False))
    imu_angles = circ_norm(np.linspace(0, 2 * np.pi, nb_samples, endpoint=False))
    all_pol_angles = np.zeros((nb_recordings, nb_samples), dtype='float32')

    for i in range(nb_recordings):

        pol = res_pol[i]
        iny = res_int[i]

        if algorithm is eigenvectors:
            s000 = res_000[i]
            s045 = res_045[i]
            s090 = res_090[i]
            s135 = res_135[i]

        for s in range(nb_samples):
            # get the desired absolute rotation of the compass (with respect to north)
            rotate = imu_angles[s]

            #
            # # calculate the distance between the orientation of each unit in the desired compass orientation
            # # and the orientation of each unit in the dataset
            # a_dist = abs(circ_norm(imu_angles[:, None] - pol_angles[None, :] + rotate))
            #
            # # keep the dataset entries with the 3 closest orientations for each sensor
            # i_closest = np.argsort(a_dist, axis=0)[:3]
            #
            # # calculate the median response of the POL and INT units on the preferred angles
            # pol_i = np.median(pol[i_closest], axis=0)
            # int_i = np.median(iny[i_closest], axis=0)

            imu_r = circ_norm(imu_angles + rotate)
            i_sort = np.argsort(imu_r)

            pol_i = np.interp(pol_angles, imu_r[i_sort], pol[i_sort])
            int_i = np.interp(pol_angles, imu_r[i_sort], iny[i_sort])
            if algorithm is eigenvectors:
                s000_i = np.interp(pol_angles, imu_r[i_sort], s000[i_sort])
                s045_i = np.interp(pol_angles, imu_r[i_sort], s045[i_sort])
                s090_i = np.interp(pol_angles, imu_r[i_sort], s090[i_sort])
                s135_i = np.interp(pol_angles, imu_r[i_sort], s135[i_sort])

            # calculate the SOL responses
            sol = np.zeros(8, dtype='float32')
            if algorithm is None:
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
            elif algorithm is four_zeros:
                all_pol_angles[i, s], = algorithm(pol_i, pol_prefs=pol_angles)
            elif algorithm is eigenvectors:
                all_pol_angles[i, s], = algorithm(s000_i, s045_i, s090_i, s135_i, pol_prefs=pol_angles)

    imu_angles = np.array([imu_angles] * nb_recordings)
    return np.squeeze(all_pol_angles + np.pi) % (2 * np.pi) - np.pi, np.squeeze(imu_angles)


def get_sensor_responses(data, imu_samples=360, imu_drift=0., nb_nearest=11):

    i_valid = [[0, 1, 2, 3, 6]]
    # i_valid = [[1, 2]]

    imu_angles = (np.linspace(0, 2 * np.pi, imu_samples, endpoint=False) + np.pi) % (2 * np.pi) - np.pi
    all_pol_res = np.zeros(imu_samples, dtype='float32')
    all_int_res = np.zeros(imu_samples, dtype='float32')
    all_000_res = np.zeros(imu_samples, dtype='float32')
    all_045_res = np.zeros(imu_samples, dtype='float32')
    all_090_res = np.zeros(imu_samples, dtype='float32')
    all_135_res = np.zeros(imu_samples, dtype='float32')

    if 'unit' in data.columns.to_list():
        data_rec = data[np.any(data["unit"].to_numpy()[:, None] == i_valid, axis=1)]

        imu_pol = np.deg2rad(data_rec["IMU"].to_numpy())
        imu_iny = imu_pol
        imu_000 = imu_pol
        imu_045 = imu_pol
        imu_090 = imu_pol
        imu_135 = imu_pol
        pol = data_rec["POL"].to_numpy()
        iny = data_rec["INT"].to_numpy()
        s000 = np.clip(data_rec["S3"].to_numpy() / MAX_INT, np.finfo(float).eps, 1)
        s045 = np.clip(data_rec["S1"].to_numpy() / MAX_INT, np.finfo(float).eps, 1)
        s090 = np.clip(data_rec["S2"].to_numpy() / MAX_INT, np.finfo(float).eps, 1)
        s135 = np.clip(data_rec["S0"].to_numpy() / MAX_INT, np.finfo(float).eps, 1)
    else:
        imu_pol = np.deg2rad(data[data["unit_type"] == "POL"]["direction"].to_numpy())
        imu_iny = np.deg2rad(data[data["unit_type"] == "INT"]["direction"].to_numpy())
        imu_000 = np.deg2rad(data[data["unit_type"] == "S000"]["direction"].to_numpy())
        imu_045 = np.deg2rad(data[data["unit_type"] == "S045"]["direction"].to_numpy())
        imu_090 = np.deg2rad(data[data["unit_type"] == "S090"]["direction"].to_numpy())
        imu_135 = np.deg2rad(data[data["unit_type"] == "S135"]["direction"].to_numpy())

        pol = data[data["unit_type"] == "POL"]["response"].to_numpy()
        iny = data[data["unit_type"] == "INT"]["response"].to_numpy()
        s000 = data[data["unit_type"] == "S000"]["response"].to_numpy()
        s045 = data[data["unit_type"] == "S045"]["response"].to_numpy()
        s090 = data[data["unit_type"] == "S090"]["response"].to_numpy()
        s135 = data[data["unit_type"] == "S135"]["response"].to_numpy()

    for s in range(imu_samples):
        # get the desired absolute rotation of the compass (with respect to north)
        rotate = imu_angles[s] + imu_drift

        # calculate the distance between the orientation of each unit in the desired compass orientation
        # and the orientation of each unit in the dataset
        p_dist = abs((imu_pol + rotate + np.pi) % (2 * np.pi) - np.pi)
        i_dist = abs((imu_iny + rotate + np.pi) % (2 * np.pi) - np.pi)
        s0_dist = abs((imu_000 + rotate + np.pi) % (2 * np.pi) - np.pi)
        s1_dist = abs((imu_045 + rotate + np.pi) % (2 * np.pi) - np.pi)
        s2_dist = abs((imu_090 + rotate + np.pi) % (2 * np.pi) - np.pi)
        s3_dist = abs((imu_135 + rotate + np.pi) % (2 * np.pi) - np.pi)

        # keep the dataset entries with the nearest orientations for each sensor
        p_nearest = np.argsort(p_dist, axis=0)[:nb_nearest]
        i_nearest = np.argsort(i_dist, axis=0)[:nb_nearest]
        s0_nearest = np.argsort(s0_dist, axis=0)[:nb_nearest]
        s1_nearest = np.argsort(s1_dist, axis=0)[:nb_nearest]
        s2_nearest = np.argsort(s2_dist, axis=0)[:nb_nearest]
        s3_nearest = np.argsort(s3_dist, axis=0)[:nb_nearest]

        # calculate the median response of the POL and INT units on the preferred angles
        all_pol_res[s] = np.median(pol[p_nearest], axis=0)
        all_int_res[s] = np.median(iny[i_nearest], axis=0)
        all_000_res[s] = np.median(s000[s0_nearest], axis=0)
        all_045_res[s] = np.median(s045[s1_nearest], axis=0)
        all_090_res[s] = np.median(s090[s2_nearest], axis=0)
        all_135_res[s] = np.median(s135[s3_nearest], axis=0)

    return all_pol_res, all_int_res, all_000_res, all_045_res, all_090_res, all_135_res


def get_sensor_responses_per_recording(data, imu_samples=360, imu_drift=0., nb_nearest=11):
    ang_x = data.groupby("recording").apply(lambda x: get_sensor_responses(
        x, imu_samples=imu_samples, imu_drift=imu_drift, nb_nearest=nb_nearest))

    pol_res = np.array([x[0] for x in ang_x.to_numpy()], dtype='float64')
    int_res = np.array([x[1] for x in ang_x.to_numpy()], dtype='float64')
    p00_res = np.array([x[2] for x in ang_x.to_numpy()], dtype='float64')
    p45_res = np.array([x[3] for x in ang_x.to_numpy()], dtype='float64')
    p90_res = np.array([x[4] for x in ang_x.to_numpy()], dtype='float64')
    m45_res = np.array([x[5] for x in ang_x.to_numpy()], dtype='float64')

    return pol_res, int_res, p00_res, p45_res, p90_res, m45_res


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


def read_csv_dataset(data_set, remove_outliers=False):
    data_dir = os.path.join(csv_dir, data_set)
    data = pd.read_csv(os.path.join(data_dir, f"{data_set}.csv"), low_memory=False)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # s = np.sqrt(np.clip(data[["S0", "S1", "S2", "S3"]].astype('float32') / MAX_INT, np.finfo(float).eps, 1.))
    s = np.clip(data[["S0", "S1", "S2", "S3"]].astype('float32') / MAX_INT, np.finfo(float).eps, 1.)
    data["POL"] = (s["S3"] - s["S2"]) / (s["S3"] + s["S2"] + np.finfo(float).eps)
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


def create_full_csv(*data_dirs):
    if len(data_dirs) < 1:
        data_dirs = default_datasets

    print(f"CREATE_CSV")

    # Make plot directory if it doesn't exist. Existing files will
    # be overwritten.
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    for dataset in data_dirs:
        print(f"DATASET: {dataset}")
        data_dir = os.path.join(data_base, dataset)

        image_dir = os.path.join(csv_dir, dataset)
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
        df.to_csv(os.path.join(csv_dir, dataset, f"{dataset}.csv"), index=False)


def create_clean_csv(*data_dirs):

    if len(data_dirs) < 1:
        data_dirs = default_datasets

    dataset_clean = {
        "session": [], "time": [], "location": [], "category": [], "unit_type": [],
        "sun_azimuth": [], "sun_elevation": [], "direction": [], "response": [], "dataset": [], "recording": [],
        "sun status": [], "anti-sun status": [], "cloud status": [], "canopy status": [], "tilt": []
    }
    centre = {}

    for dataset in data_dirs:
        df = read_csv_dataset(dataset)
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
                if image_file in sun_centres:
                    centre[image_file] = np.deg2rad(sun_centres[image_file][4])
                else:
                    dfu = dff[np.all([
                        dff["unit"] != 7,
                        dff["unit"] != 5,
                        dff["unit"] != 4,
                    ], axis=0)]

                    pol_res_, int_res_, _, _, _, _ = get_sensor_responses(dfu, default_nb_samples)
                    x_imu = np.linspace(0, 2 * np.pi, 360, endpoint=False)
                    y_inp = int_res_ - pol_res_

                    est_sun = np.angle(np.nanmean(np.exp(1j * x_imu) * y_inp))

                    centre[image_file] = (est_sun - sun_azi + np.pi) % (2 * np.pi) - np.pi

            print(f"sun = {np.rad2deg(sun_azi):.2f}, centre = {np.rad2deg(centre[image_file]):.2f}; "
                  f"category = {dff['category'].to_numpy()[0]}, "
                  f"sun = '{sun_status[sky_conditions[image_file][0]]}', "
                  f"anti-sun = '{anti_sun_status[sky_conditions[image_file][1]]}', "
                  f"clouds = '{clouds_level[sky_conditions[image_file][2]]}', "
                  f"canopy = '{canopy_level[sky_conditions[image_file][3]]}'")

            # nb_recordings x nb_samples
            pol_res, int_res, p00, p45, p90, m45 = get_sensor_responses_per_recording(
                dff, default_nb_samples, imu_drift=centre[image_file])

            units = {"POL": pol_res, "INT": int_res,
                     "S000": p00, "S045": p45, "S090": p90, "S135": m45}
            nu = len(units)
            for i in range(pol_res.shape[0]):
                dataset_clean["session"].extend([image_file] * pol_res.shape[1] * nu)
                dataset_clean["time"].extend([dff["timestamp"].to_numpy()[0]] * pol_res.shape[1] * nu)
                dataset_clean["location"].extend([dff["location"].to_numpy()[0]] * pol_res.shape[1] * nu)
                dataset_clean["category"].extend([dff['category'].to_numpy()[0]] * pol_res.shape[1] * nu)
                dataset_clean["sun_azimuth"].extend([np.rad2deg(sun_azi)] * pol_res.shape[1] * nu)
                dataset_clean["sun_elevation"].extend([np.rad2deg(sun_ele)] * pol_res.shape[1] * nu)
                dataset_clean["dataset"].extend([dataset] * pol_res.shape[1] * nu)
                dataset_clean["recording"].extend([i + 1] * pol_res.shape[1] * nu)
                dataset_clean["sun status"].extend([sky_conditions[image_file][0]] * pol_res.shape[1] * nu)
                dataset_clean["anti-sun status"].extend([sky_conditions[image_file][1]] * pol_res.shape[1] * nu)
                dataset_clean["cloud status"].extend([sky_conditions[image_file][2]] * pol_res.shape[1] * nu)
                dataset_clean["canopy status"].extend([sky_conditions[image_file][3]] * pol_res.shape[1] * nu)
                dataset_clean["tilt"].extend([tilt_condition[image_file]] * pol_res.shape[1] * nu)
                for ut in units:
                    dataset_clean["unit_type"].extend([ut] * pol_res.shape[1])
                    dataset_clean["direction"].extend(
                        (np.linspace(0, 360, pol_res.shape[1], endpoint=False) + 180) % 360 - 180)
                    dataset_clean["response"].extend(units[ut][i])

    clean_df = pd.DataFrame(dataset_clean)
    clean_df.to_csv(clean_path, index=False, float_format='%.4f')

    print(f"Saved clean dataset at: {clean_path}\n")

    return clean_df


def create_errors_csv(*data_dirs, clean_df=None):

    if len(data_dirs) < 1:
        data_dirs = default_datasets

    if clean_df is None:
        if os.path.exists(clean_path):
            clean_df = pd.read_csv(clean_path)
        else:
            clean_df = create_clean_csv(*data_dirs)

    data_frame = {
        "session": [], "dataset": [], "time": [], "location": [], "category": [], "ring_size": [], "unit_type": [],
        "sun_azimuth": [], "sun_elevation": [],
        "mean_error": [], "mean_error_sd": [], "max_error": [], "max_error_sd": [],
        "mean_absolute_error": [], "mean_absolute_error_sd": [],
        "root_mean_square_error": [], "root_mean_square_error_sd": [],
        "sun_status": [], "anti_sun_status": [], "cloud_status": [], "canopy_status": [], "tilt": [], "recording": []
    }

    mae_df = None
    for ring_size in range(3, 61):
        for dataset in data_dirs:

            df = clean_df[clean_df["dataset"] == dataset]
            df = df.assign(direction=df["direction"] % 360)
            for image_file in np.unique(df["session"]):
                dfs = df[df["session"] == image_file]
                print(f"{image_file} (ring = {ring_size})", end=': ')

                sun_azi = np.deg2rad(np.mean(dfs["sun_azimuth"].to_numpy()))
                sun_ele = np.deg2rad(np.mean(dfs["sun_elevation"].to_numpy()))

                pol_res = dfs[dfs["unit_type"] == "POL"]
                int_res = dfs[dfs["unit_type"] == "INT"]
                s000_res = dfs[dfs["unit_type"] == "S000"]
                s045_res = dfs[dfs["unit_type"] == "S045"]
                s090_res = dfs[dfs["unit_type"] == "S090"]
                s135_res = dfs[dfs["unit_type"] == "S135"]

                pol_res = pol_res.pivot(index="recording", columns="direction", values="response").to_numpy()
                int_res = int_res.pivot(index="recording", columns="direction", values="response").to_numpy()
                s_res = [
                    s000_res.pivot(index="recording", columns="direction", values="response").to_numpy(),
                    s045_res.pivot(index="recording", columns="direction", values="response").to_numpy(),
                    s090_res.pivot(index="recording", columns="direction", values="response").to_numpy(),
                    s135_res.pivot(index="recording", columns="direction", values="response").to_numpy()
                    ]

                # nb_recordings x nb_samples
                ang_pol, x_pol = compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=True, intensity=False)
                ang_int, x_int = compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=False, intensity=True)
                ang_inp, x_inp = compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=True, intensity=True)
                ang_ipp, x_ipp = compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, polarisation=False, intensity=False)
                ang_frz, x_frz = compute_sensor_output_from_responses(
                    pol_res, int_res, ring_size, algorithm=four_zeros)
                ang_eig, x_eig = compute_sensor_output_from_responses(
                    s_res, int_res, ring_size, algorithm=eigenvectors)

                # # break ambiguity in eigenvectors result
                # mu_eig, = compare(ang_eig, x_eig + sun_azi)
                # ang_eig[abs(mu_eig) > np.pi/2] += np.pi
                # ang_eig = circ_norm(ang_eig)

                category = dfs['category'].to_numpy()[0]

                error_types = {"mean error": "mne", "mean absolute error": "mae",
                               "root mean square error": "rmse", "max error": "mxe"}
                unit_ang = {"POL": {"ang": ang_pol, "x": x_pol + sun_azi},
                            "INT": {"ang": ang_int, "x": x_int + sun_azi},
                            "INT-POL": {"ang": ang_inp, "x": x_inp + sun_azi},
                            "INT+POL": {"ang": ang_ipp, "x": x_ipp + sun_azi},
                            "FZ": {"ang": ang_frz, "x": x_frz + sun_azi},
                            "EIG": {"ang": ang_eig, "x": x_eig + sun_azi}}
                error_stats = {"POL": {}, "INT": {}, "INT-POL": {}, "INT+POL": {}, "FZ": {}, "EIG": {}}

                for ut in unit_ang:
                    for et in error_types:
                        mu, sd = compare(unit_ang[ut]["ang"], unit_ang[ut]["x"], error_type=et, std=True, axis=1)
                        error_stats[ut][et] = np.rad2deg([mu, sd])

                nb_recordings = error_stats["POL"]["mean error"][0].shape[0]
                nb_algs = len(unit_ang)

                data_frame["session"].extend([image_file] * nb_algs * nb_recordings)
                data_frame["time"].extend([dfs["time"].to_numpy()[0]] * nb_algs * nb_recordings)
                data_frame["location"].extend([dfs["location"].to_numpy()[0]] * nb_algs * nb_recordings)
                data_frame["category"].extend([category] * nb_algs * nb_recordings)
                data_frame["ring_size"].extend([ring_size] * nb_algs * nb_recordings)
                data_frame["sun_azimuth"].extend([np.rad2deg(sun_azi)] * nb_algs * nb_recordings)
                data_frame["sun_elevation"].extend([np.rad2deg(sun_ele)] * nb_algs * nb_recordings)
                data_frame["dataset"].extend([dataset] * nb_algs * nb_recordings)
                data_frame["sun_status"].extend([sky_conditions[image_file][0]] * nb_algs * nb_recordings)
                data_frame["anti_sun_status"].extend([sky_conditions[image_file][1]] * nb_algs * nb_recordings)
                data_frame["cloud_status"].extend([sky_conditions[image_file][2]] * nb_algs * nb_recordings)
                data_frame["canopy_status"].extend([sky_conditions[image_file][3]] * nb_algs * nb_recordings)
                data_frame["tilt"].extend([tilt_condition[image_file]] * nb_algs * nb_recordings)
                data_frame["recording"].extend(list(np.arange(nb_recordings)) * nb_algs)
                for ut in unit_ang:
                    data_frame["unit_type"].extend([ut] * nb_recordings)

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

                print(f"sun = {np.rad2deg(sun_azi):.2f}; category = {category}")

        mae_df = pd.DataFrame(data_frame)
        mae_df.to_csv(mae_path, index=False, float_format='%.4f')

        print(f"Saved MAE dataset at: {mae_path}\n")

    return mae_df


if __name__ == "__main__":
    import warnings

    warnings.simplefilter('ignore')

    # create_full_csv('sardinia_data', 'south_africa_data')
    # create_clean_csv('sardinia_data', 'south_africa_data')
    create_errors_csv('sardinia_data', 'south_africa_data')
    # create_errors_csv('south_africa_data')
