"""
translate_all.py

Brittle script to translate all the rosbag data in the data directory into
json using rosbag_to_json.py. The directory structure is assumed so this will
break if anything changes.
"""

import os
import shutil
from rosbag_to_json import rosbag_to_json

copy_all = True

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(),'../sardinia_data')
    out_base_dir = os.path.join(os.getcwd(),'../json_data')

    # Make the json directory if it doesn't exist. Files will
    # be overwritten.
    if not os.path.exists(out_base_dir):
        os.mkdir(out_base_dir)

    # Change into data directory - tod_data, canopy_data
    datasets = os.listdir(data_dir)
    for s in datasets:
        os.chdir(os.path.join(data_dir, s))
        days = os.listdir(os.getcwd())
        for d in days:
            os.chdir(os.path.join(data_dir, s, d))
            rec_sessions = os.listdir(os.getcwd())
            for r in rec_sessions:
                session_path = os.path.join(data_dir, s, d, r)
                if not os.path.isdir(session_path):
                    continue

                os.chdir(session_path)
                bagfiles = [
                    x for x in os.listdir(os.getcwd()) if ".bag" in x
                ]



                # Mirror structure
                output_path = os.path.join(
                    out_base_dir, s, d, r
                )
                if copy_all:
                    # Most likely jpegs or notes
                    other_files = [
                        x for x in os.listdir(os.getcwd()) if ".bag" not in x
                    ]
                    for of in other_files:
                        shutil.copy(of, os.path.join(output_path, of))

                # Create output directory
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                for bag in bagfiles:
                    infile = os.path.join(session_path, bag)
                    outfile = os.path.join(
                        output_path, "{}.json".format(bag.split(".")[0])
                    )

                    rosbag_to_json(infile, outfile, verbose=False)

