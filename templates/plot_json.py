"""
plot_json.py

Produce a Figure from a recording session where each recording is stored as a
json file.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os
import sys
from scipy.stats import circmean

import plot_from_dict as pfd

def plot_json(session, outfile=None):
        """
    Produce a figure along with a local output filename. For the output filename
    the caller must specify any preceeding directory structure, otherwise the
    output file will be relative to the calling directory.

    :param session: A directory containing json files, translated from a
                    recording session.
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

    calling_dir = os.getcwd()
    os.chdir(session)
    imagefile = [x for x in os.listdir() if ".jpg" in x]
    recordings = [x for x in os.listdir() if ".json" in x]

    if len(recordings) == 0:
        print("There are no json files in the specified directory.")
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
            outfile = imagefile.split(".")[0] + ".pdf"

    # No output file specified and no/multiple image files
    # Use session directory name as pdf name
    if outfile == None:
        session_path = session.split("/")
        print(session_path)
        outfile = session_path[len(session_path) - 2] + ".pdf"

    # Pull all recordings into dictionary
    full_data = dict()
    for r in recordings:
        with open(r) as f:
            rec_data = json.load(f)

        full_data[r] = rec_data

    full_data["image_filename"] = imagefile

    fig = pfd.produce_plot(full_data)
    return fig, outfile

if __name__=="__main__":
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

    session = args.session
    outfile = args.output

    fig, outfile = plot_json(session, outfile=outfile)

    # Save files relative to calling directory unless absolute
    # path is specified.
    os.chdir(calling_directory)
    fig.savefig(outfile, bbox_inches="tight")
    plt.show()

