import image_processing as imp

import skylight as skl
import numpy as np
import rosbag

import shutil
import os
import re

S2I = ["135", "045", "090", "000"]

response_dist_base = os.path.abspath(os.path.join(os.getcwd(), '..', 'sensor_response_distribution'))


def rosbag_to_dict(bag):
    """
    Extract the relevant messages from a rosbag file and store them in a
    dictionary. The dictionary has one key per topic, and each topic contains
    all message data recorded on that topic. Pol op preferred angles are also
    included for completeness.

    :param bag: a rosbag (note not a filename, the actual bag object)
    :returns: a dictionary containing the recorded data
    """
    # Simplistic but works
    po = [[],[],[],[],[],[],[],[]]
    yaw = []

    # For pol-ops, topics are pol_op_X where x is from 0 to 7
    topics_of_interest = ["pol_op_{}".format(x) for x in range(8)]
    topics_of_interest.append("yaw")

    # These recordings also contain (blurred) image data (topic 'frames')
    # and full IMU data (topic 'odom').
    for topic,msg,t in bag.read_messages(topics=topics_of_interest):
        if "pol_op_" in topic:
            idx = int(topic.split("_")[2])
            po[idx].append(msg.data)
        elif "yaw" in topic:
            yaw.append(msg.data)

    # Using this construction:
    # po[x] -> recorded data for pol-op unit x
    # po[x][y] -> pol-op x at time y
    # po[x][y][z] -> pol-op x, time y, photodiode z
    # Absolute value is taken as some photodiodes return negative values. Using
    # the absolute values gives the correct sensor output.
    po = np.abs(po).tolist()

    # Azimuths for each pol-op unit. po_az[i] corresponds
    # to po[i].
    po_az = np.radians([0, 90, 180, 270, 45, 135, 225, 315]).tolist()

    data_dictionary = dict()
    for idx in range(8):
        data_dictionary["pol_op_{}".format(idx)] = list(po[idx])

    data_dictionary["azimuths"] = po_az
    data_dictionary["yaw"] = yaw

    return data_dictionary


def read_bagfile(session, images_dir):
    # timestamp = datetime.strptime(session.split(os.path.sep)[-1], '%H-%M__%A_%d-%m-%y')
    # category = session.split(os.path.sep)[-3].replace("_data", "")
    collection = session.split(os.path.sep)[-4].replace("_data", "")

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
                  "I135": [], "I045": [], "I090": [], "I000": [], "device": [], "rotation": [],
                  "location": [], "timestamp": [], "longitude": [], "latitude": [],
                  "collection": [], "session": []}

    image_name = image_file.replace('.jpg', '')
    obs = imp.get_observer_from_file_name(image_name)
    sun_predict = imp.extract_sun_vector_from_name(image_name)
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
                data_frame[f"I{S2I[i]}"].extend(pol[unit, :, i])
            data_frame["device"].extend([unit] * imu.shape[1])
            data_frame["rotation"].extend([key.split('.')[0]] * imu.shape[1])
            # data_frame["category"].extend([category] * imu.shape[1])
            data_frame["location"].extend([obs.city] * imu.shape[1])
            data_frame["timestamp"].extend([obs.date] * imu.shape[1])
            data_frame["longitude"].extend([obs.lon] * imu.shape[1])
            data_frame["latitude"].extend([obs.lat] * imu.shape[1])
            data_frame["collection"].extend([collection] * imu.shape[1])
            data_frame["session"].extend([image_file] * imu.shape[1])

    return data_frame


def read_pd_bagfile(session=None):
    if session is None:
        session = response_dist_base

    os.chdir(session)
    map_file = [x for x in os.listdir() if ".txt" in x][0]
    recordings = [x for x in os.listdir() if ".bag" in x]

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
