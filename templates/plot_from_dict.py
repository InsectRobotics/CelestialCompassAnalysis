"""
plot_from_dict.py

Central plotting utilities (so bagfiles and json files can essentially
be plotted by the same code).

The main function provided by this module is produce_plot which will
take a data dictionary and produce a SkyCompass/IMU comparison plot
for a given recording session.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json
import argparse
import os
from scipy.stats import circmean

UNIT_ANGLES = np.deg2rad([45, 135, 0, 90], dtype='float64')  # -45, +45, 0, 90
POL_PREFS = np.deg2rad([0, 90, 180, 270, 45, 135, 225, 315], dtype='float64')

A = None
A_inv = A


def act(s, log=True):
    """
    Photoreceptor activation function. Gkanias et al. (2019) uses square
    root activation, however log activation appears to generate a slightly
    more stable angular representation. 

    :param s: The input to the photoreceptor.
    :param log: Set True to use Log activation rather than square root.
    :return: The firing rate of the photoreceptor given input s.
    """
    if not log:
        return np.sqrt(s)

    return np.log(s)


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


def decode_sky_compass(po, n_sol=8, pol_prefs=POL_PREFS, polarisation=True, intensity=False):
    """
    Sky compass decoding routine from Gkanias et al. (2019). This has been
    implemeneted from scratch using the equations presented as a demonstration
    that everything can be programmed from scratch without calling out to
    external libraries or FFT implementations.

    :param po: Pol-op unit responses
    :param n_sol: The number of SOL neurons in use
    :param pol_prefs: The preferred angles of each POL neuron. This is determined
                      by the hardware configuration at the time the recording was
                      taken.
    :returns: A list of azimuthal angles with a list of associated confidence
              in each.
    """
    # po structure
    # po[p] Unit p
    # po[p][t] readings for unit p at time t
    # po[p][t][d] reading for diode d of unit p at time t
    units, duration, diodes = po.shape

    sol_prefs = np.linspace(0, 2 * np.pi - (2 * np.pi / n_sol), n_sol)

    angular_outputs = []
    confidence_outputs = []

    for t in range(duration):  # For each timestep

        # ensure that the response is in [0, 1]
        # the highest observed value was 32768 (Thursday 17:20)
        response = np.clip(po[:, t] / 33000., 0., 1.)
        if not polarisation and not intensity:
            angle, sigma = pol2eig(response, pol_prefs)
        elif polarisation and not intensity:
            angle, sigma = pol2sol(response, sol_prefs, pol_prefs, unit_transform=unit2pol)
        elif not polarisation and intensity:
            angle, sigma = pol2sol(response, sol_prefs + np.pi, pol_prefs, unit_transform=unit2int)
        else:
            a_pol, c_pol = pol2sol(response, sol_prefs, pol_prefs, unit_transform=unit2pol)
            a_int, c_int = pol2sol(response, sol_prefs + np.pi, pol_prefs, unit_transform=unit2int)
            s_pol = 1. / c_pol ** 2
            s_int = 1. / c_int ** 2

            angle = (s_pol * a_int + s_int * a_pol) / (s_pol + s_int)
            sigma = (s_pol * s_int) / (s_pol + s_int)
        angular_outputs.append(angle)
        confidence_outputs.append(sigma)

    return angular_outputs, confidence_outputs


def pol2eig(po, pol_prefs):

    units, diodes = po.shape

    # Get photodiode responses for each unit
    phi, d = np.array([unit2eig(po[x]) for x in range(units)]).T

    pe = np.array([np.cos(phi + pol_prefs), np.sin(phi + pol_prefs), np.zeros_like(phi)])

    alpha = -pol_prefs
    gamma = np.full_like(alpha, np.deg2rad(45))

    c = np.zeros([3, 3, 8], dtype='float32')
    e = np.zeros_like(pe)
    for i, a, g in zip(range(len(alpha)), alpha, gamma):
        c[..., i] = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        c[..., i] = np.dot(c[..., i], np.array([[np.cos(g), 0, np.sin(g)], [0, 1, 0], [-np.sin(g), 0, np.cos(g)]]))

        e[:, i] = c[..., i].dot(pe[:, i])

    eig = e.dot(e.T)

    eigenvalues, eigenvectors = np.linalg.eigh(eig)

    i_star = np.argmin(eigenvalues)
    s_1, s_2, s_3 = eigenvectors[i_star]

    # print(eigenvalues)
    # print(eigenvectors)

    # import sys
    # sys.exit()

    # Compute argument and magnitude of complex conjugate of R (Eq. (4))
    angle = np.arctan2(s_2, s_1)
    confidence = 1.0

    return angle, confidence


def pol2sol(po, sol_prefs, pol_prefs, unit_transform):
    units, diodes = po.shape
    n_sol = len(sol_prefs)
    n_pol = len(pol_prefs)

    # Get photodiode responses for each unit
    r_pol = np.array([unit_transform(po[x]) for x in range(units)])

    # Init for sum
    r_sol = np.zeros(n_sol)
    R = 0
    for z in range(n_sol):
        for j in range(n_pol):  # Compute SOL responses (Eq. (2))
            aj = pol_prefs[j] - np.pi / 2
            r_sol[z] += (n_sol / n_pol) * np.sin(aj - sol_prefs[z]) * r_pol[j]

        # Accumulate R (Eq. (3))
        R += r_sol[z] * np.exp(-1j * 2 * np.pi * (z - 1) / n_sol)

    a = np.real(R)
    b = np.imag(R)

    # Compute argument and magnitude of complex conjugate of R (Eq. (4))
    angle = np.arctan2(-b, a)
    confidence = np.sqrt(a ** 2 + b ** 2)

    return angle, confidence


def unit2eig(unit_response):
    f = unit_response
    # q1, q2, q3 = np.linalg.inv(a.T.dot(a)).dot(a.T).dot(f)
    q1, q2, q3 = A_inv.dot(f)
    # q1 = i_0 - i_90
    # q2 = i_45 - i_135
    # q3 = (i_0 + i_45 + i_90 + i_135) / 2

    phi = 0.5 * np.arctan2(q2, q1)
    d = np.sqrt(np.square(q1) + np.square(q2)) / q3

    # print(f"phi = {np.rad2deg(phi):.2f}, d = {d:.2f}, responses = {unit_response}")

    return phi, d


def unit2pol(unit_response, log=True):
    r_h, r_v = unit2vh(unit_response, log)
    return (r_h - r_v) / (r_h + r_v)


def unit2int(unit_response, log=True):
    r_h, r_v = unit2vh(unit_response, log)
    # r = (r_h + r_v) / 2
    r = np.sqrt(r_h**2 + r_v**2)
    return r


def unit2vh(unit_response, log=True):
    # -45, +45, 0, 90
    _, _, s_v, s_h = unit_response
    return act(s_h, log), act(s_v, log)


def produce_plot(data_dictionary, polarisation=True, intensity=False):

    mosaic = [["image", "skycompass"],
              ["image", "imu_yaw"],
              ["image", "averages"]]

    fig, ax = plt.subplot_mosaic(mosaic, figsize=(10, 4))

    #
    # Plot image if available
    #
    plot_image(data_dictionary["image_filename"], ax=ax["image"])
    del data_dictionary["image_filename"]

    max_length = 0  # Maximum recording length in set
    min_length = 0  # Minimum recording length in set
    first = True  # First iteration flag

    # Corrected datastructures for computing averages
    corrected_skycompass_table = []
    corrected_imu_table = []

    mse = []

    print("MAE:", end=" ")
    for k in data_dictionary.keys():
        data = data_dictionary[k]

        # Extract data
        imu = synchronise(np.array(data["yaw"]))
        pol_op_keys = ["pol_op_{}".format(x) for x in range(8)]
        po = []
        for k in pol_op_keys:
            po.append(data[k])

        po = np.array(po)
        pol_sensor_angle = synchronise(
            np.array(decode_sky_compass(po, polarisation=polarisation, intensity=intensity)[0]))

        if max_length < len(imu):
            max_length = len(imu)

        if first:
            first = False
            min_length = len(imu)

        if min_length > len(imu):
            min_length = len(imu)

        # Plot cleaned angles
        plot_angles(max_length, data=pol_sensor_angle, ax=ax["skycompass"], x_ticks=False)
        plot_angles(max_length, data=imu, ax=ax["imu_yaw"], x_ticks=False)

        mse.append(compare(pol_sensor_angle, imu))
        print(f"{np.rad2deg(mse[-1]):.2f}", end="\t")

        corrected_skycompass_table.append(pol_sensor_angle)
        corrected_imu_table.append(imu)
    print(f"\nMean MAE: {np.rad2deg(np.nanmean(mse)):.2f} +/- {np.rad2deg(np.nanstd(mse)):.2f}")

    # Compute mean angle over all reacordings at each timestep.
    sc_means = []
    imu_means = []
    for t in range(min_length):
        sc_angles = [x[t] for x in corrected_skycompass_table]
        imu_angles = [x[t] for x in corrected_imu_table]
        scmean = circmean(sc_angles, nan_policy='omit')
        imumean = circmean(imu_angles, nan_policy='omit')
        sc_means.append(scmean)
        imu_means.append(imumean)

    # Plot (clean) averages
    plot_angles(max_length, data=synchronise(sc_means), ax=ax["averages"])
    plot_angles(max_length, data=synchronise(imu_means), ax=ax["averages"])

    # Add text
    # ax["skycompass"].text(0,5,"Sky Compass")
    # ax["imu_yaw"].set_ylim([-padding, 2*np.pi + padding])
    # ax["averages"].set_ylim([-padding, 2*np.pi + padding])

    ax["skycompass"].set_title("Sky Compass")
    ax["imu_yaw"].set_title("IMU (Yaw)")
    ax["averages"].set_title("Circular average")
    fig.tight_layout()

    return fig


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


def update_globals():
    global A, A_inv

    A = np.round(np.array([np.cos(2 * UNIT_ANGLES), np.sin(2 * UNIT_ANGLES), np.ones_like(UNIT_ANGLES)]).T, decimals=2)

    # A_inv = np.linalg.inv(A.T.dot(A)).dot(A.T)
    A_inv = np.linalg.pinv(A)


update_globals()
