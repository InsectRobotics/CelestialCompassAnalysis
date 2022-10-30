"""
rosbag_to_json.py

Takes a rosbag as an input file, extracts relevant topics (all pol_op data
and yaw), packs the data into a dictionary and outputs a json file.

Requires rosbag and numpy.
"""
import rosbag
import numpy as np
import json
import argparse

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


def rosbag_to_json(infile, outfile,verbose=True):
    """
    Given a rosbag file as input, create a dictionary containing all
    recorded messages for the relevant topics then store this in json
    format
    """
    if verbose:
        print("Input: {}".format(infile))
        print("Output: {}".format(outfile))

    bag = rosbag.Bag(infile)
    data_dictionary = rosbag_to_dict(bag)

    with open(outfile, "w") as f:
        json.dump(data_dictionary, f)

    if verbose:
        print("Topics saved: {}".format(list(data_dictionary.keys())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example, read from pol sensor bagfile."
    )

    parser.add_argument("-if", dest='infile', type=str, required=True,
                        help="Input filepath")
    parser.add_argument("-of", dest='outfile', type=str, required=False,
                         help="Output filepath"
    )

    args = parser.parse_args()

    infile = args.infile
    outfile = args.outfile if args.outfile != None else "out.json"

    rosbag_to_json(infile, outfile)
