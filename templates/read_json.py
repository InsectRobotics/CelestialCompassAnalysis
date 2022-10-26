"""
read_json.py

Example for reading from the json files produced by rosbag_to_json.py.
No ROS packages are needed to run this.
"""

import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example, read data from file produced by rosbag_to_json.py."
    )

    parser.add_argument("-if", dest='infile', type=str, required=True,
                        help="Input filepath")

    args = parser.parse_args()
    infile = args.infile

    with open(infile) as f:
        data_dictionary = json.load(f)

    print(data_dictionary.keys())

    ### Do stuff, e.g. decode sensor data...
