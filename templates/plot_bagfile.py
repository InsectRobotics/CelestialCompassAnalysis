"""
plot_json.py

Produce a Figure from a recording session where each recording is stored as a
rosbag file.
"""
import re

from rosbag_to_json import rosbag_to_dict
from datetime import datetime

import plot_from_dict as pfd
import models as md

import rosbag
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import skylight
import pytz
import re


def plot_bagfile(session, outfile=None, mode="polarisation", ring_size=8):
    """
    Produce a figure along with a local output filename. For the output filename
    the caller must specify any preceeding directory structure, otherwise the
    output file will be relative to the calling directory.

    :param session: A directory containing bagfiles from a recording session.
    :param outfile: A specified output filename. If None, the timestamped
                    sky image filename will be used. If there is no image, then
                    the lowest level session directory name will be used. If
                    specified this must contain the file extension (which must be
                    supported by matplotlib).
    :returns: A Figure and the local filename.
    """

    #
    # Pull in data
    #
    if not os.path.isdir(session):
        print("-s must specify a directory")
        sys.exit()

    timestamp = datetime.strptime(session.split(os.path.sep)[-1], '%H-%M__%A_%d-%m-%y')
    o = create_sardinia_observer(date=timestamp)
    print(o)

    polarisation = "polarisation" in mode.lower()
    if "intensity" in mode.lower():
        details = re.match(r'[\w]+\+([\d]*)[\w]+', mode.lower())
        i_str = details.group(1)
        intensity = 10 ** (-float(i_str))
    else:
        intensity = False

    calling_dir = os.getcwd()
    os.chdir(session)
    imagefile = [x for x in os.listdir() if ".jpg" in x]
    recordings = [x for x in os.listdir() if ".bag" in x]

    if len(recordings) == 0:
        print("There are no rosbag files in the specified directory.")
        sys.exit()

    if len(imagefile) == 0:
        print("Warning: missing sky image file from the session directory.")
        imagefile = None
    elif len(imagefile) > 1:
        print("Warning: multiple image files in directory.")
    else:
        imagefile = imagefile[0]
        # If outfile not specified, set to session spec
        if outfile == None:
            outfile = f"{imagefile.split('.')[0]}-{mode}-{ring_size:02d}.pdf"

    # No output file specified and no/multiple image files
    # Use session directory name as pdf name
    if outfile == None:
        session_path = os.path.abspath(session.split(os.path.sep))
        outfile = f"{session_path[len(session_path) - 2]}-{mode}-u{ring_size:02d}.pdf"


    #
    # Read in relevant rosbag data and pack into dictionary.
    #
    # Pull all recordings into dictionary
    full_data = dict()
    for r in recordings:
        bag = rosbag.Bag(r)
        rec_data = rosbag_to_dict(bag)
        full_data[r] = rec_data

    full_data["image_filename"] = imagefile

    fig, mse = pfd.produce_plot(full_data, polarisation=polarisation, intensity=intensity,
                                ring_size=ring_size, nb_samples=360,
                                # observer=o
                                )
    return fig, outfile, mse


def create_sardinia_observer(date=None):
    date = pytz.timezone('Europe/Rome').localize(date)
    return skylight.Observer(lon=8.440184, lat=39.258648, date=date, degrees=True)


def create_vryburg_observer(date=None):
    date = pytz.timezone('Africa/Johannesburg').localize(date)
    return skylight.Observer(lon=24.327144, lat=-26.398643, date=date, degrees=True)


def create_bela_bela_observer(date=None):
    date = pytz.timezone('Africa/Johannesburg').localize(date)
    return skylight.Observer(lon=27.918972, lat=-24.714872, date=date, degrees=True)


if __name__ == "__main__":
    calling_directory = os.getcwd()
    parser = argparse.ArgumentParser(
        description="Produce a recording-session plot from json."
    )

    parser.add_argument("-s", dest="session", type=str, required=True,
                        help="The recording session directory."
    )
    parser.add_argument("-o", dest="output", type=str, required=False,
                        help="Desired output file (specify matplotlib-supported format. PDF default."
    )

    args = parser.parse_args()

    session = os.path.abspath(args.session)
    outfile = args.output

    fig = None
    unit_angles = np.deg2rad([
        [45.0, -45.0, 90.0, 0.0],
        # [-45.0, 45.0, 90.0, 0.0],
        # [45.0, -45.0, 0.0, 90.0],
        # [-45.0, 45.0, 0.0, 90.0]
    ])
    for ua in unit_angles:
        print("\t".join([f"{a:.0f}" for a in np.rad2deg(ua)]), end="\t")
        md.UNIT_ANGLES[:] = ua
        md.update_globals()
        fig, outfile, error = plot_bagfile(session, outfile=outfile, mode="polarisation-only")
        # fig, outfile = plot_bagfile(session, outfile=outfile, mode="eigenvector")
        outfile = outfile.replace(".pdf", ".png")

        # Save files relative to calling directory unless absolute
        # path is specified.
        # os.chdir(calling_directory)
        # fig.savefig(outfile, bbox_inches="tight")
        plt.show()

