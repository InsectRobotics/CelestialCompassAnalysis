from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from dtw import dtw

import rosbag
import numpy as np

import os.path
import sys
import re


def read_bagfile(session):
    if not os.path.isdir(session):
        print("-s must specify a directory")
        sys.exit()

    os.chdir(session)
    map_file = [x for x in os.listdir() if ".txt" in x][0]
    recordings = [x for x in os.listdir() if ".bag" in x]

    if len(recordings) == 0:
        print("There are no rosbag files in the specified directory.")
        sys.exit()

    if len(map_file) == 0:
        print("There are no mapping files in the specified directory.")
        sys.exit()

    full_data = {}

    for r in recordings:
        bag = rosbag.Bag(r)
        data_name = r.split('.')[0]
        full_data[data_name] = {
            'response': [],
            'time': []
        }
        for topic, msg, t in bag.read_messages():
            full_data[data_name]["response"].append(msg.data[2])
            full_data[data_name]["time"].append(t.to_time())

    full_data["map"] = {}
    with open(map_file, 'r') as f:
        unit_no = -1
        for _ in range(42):
            line = f.readline()
            if 'PD' in line:
                result = re.match(r"= PD ([0-9]+) =", line)
                if result is None:
                    continue
                unit_no = int(result.group(1)) - 1
                full_data["map"][unit_no] = {
                    "angle": [],
                    "min_response": [],
                    "max_response": []
                }
            elif unit_no >= 0:
                result = re.match(r"([0-9]+), \[(.*), (.*)\]", line)
                if result is None:
                    continue
                full_data["map"][unit_no]["angle"].append(float(result.group(1)))
                full_data["map"][unit_no]["min_response"].append(float(result.group(2)))
                full_data["map"][unit_no]["max_response"].append(float(result.group(3)))

    return full_data


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    scale = 1.0 / 12000.0

    calling_directory = os.getcwd()
    parser = argparse.ArgumentParser(
        description="Produce a recording-session plot for the pol units."
    )

    parser.add_argument("-s", dest="session", type=str, required=True,
                        help="The recording session directory."
                        )

    args = parser.parse_args()

    session = os.path.abspath(args.session)

    data = read_bagfile(session)

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

    popt_3, _ = curve_fit(objective, angles_3, pd_3, sigma=pd_3_sigma)
    print(f"PD 3: mean = {popt_3[1]:.2f}, SD = {popt_3[2]:.2f}")

    popt_4, _ = curve_fit(objective, angles_4, pd_4, sigma=pd_4_sigma)
    print(f"PD 4: mean = {popt_4[1]:.2f}, SD = {popt_4[2]:.2f}")

    # fig = plt.figure("photodiode-response-distribution.pdf", figsize=(5, 4))
    fig = plt.figure("photodiode-response-distribution", figsize=(7, 4))

    ax1 = plt.subplot(121, polar=True)
    ax2 = plt.subplot(122, polar=True)

    angles_3_1 = savgol_filter(angles_3_1, 51, 3)
    ax1.plot(np.deg2rad(angles_3_1), run_3_1, c='C0', marker='x', ls='', lw=1)

    angles_3_2 = savgol_filter(angles_3_2, 51, 3)
    ax1.plot(np.deg2rad(angles_3_2), run_3_2, c='C0', marker='+', ls='', lw=1)

    # plt.fill_between(angles_3, np.zeros_like(angles_3), objective(angles_3, *popt_3), color='C0', alpha=0.2)
    # plt.fill_between(angles_4, np.zeros_like(angles_4), objective(angles_4, *popt_4), color='C1', alpha=0.2)

    ax1.fill_between(np.deg2rad(angles_3), np.zeros_like(angles_3), pd_3, color='C0', alpha=0.2, label="unit 3")
    ax2.fill_between(np.deg2rad(angles_4), np.zeros_like(angles_4), pd_4, color='C1', alpha=0.2, label="unit 4")

    # plt.plot(angles_3, pd_3, c='C0', lw=2, label='unit 3')
    # plt.plot(angles_4, pd_4, c='C1', lw=2, label='unit 4')

    ax1.plot(np.deg2rad(angles_3), np.mean([pd_3, pd_4], axis=0), c='k', lw=3, alpha=0.7)
    ax2.plot(np.deg2rad(angles_4), np.mean([pd_3, pd_4], axis=0), c='k', lw=3, alpha=0.7)

    # plt.legend()

    for ax in [ax1, ax2]:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        ax.set_xlim(-np.pi/6, np.pi/6)
        ax.set_ylim(-0.01, 1.01)
        ax.set_yticks([])

    ax1.set_title("unit 3")
    ax2.set_title("unit 4")
    # ax1.set_ylabel("photodiode (relative) response")
    # ax2.set_xlabel("direction of light with respect to photodiode orientation\n(minus = left, plus = right)")

    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join("..", "plots", "photodiode-response-distribution.svg"))
