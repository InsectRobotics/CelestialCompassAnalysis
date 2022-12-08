"""
plot_all.py

Brittle script to produce all plots from the tod and canopy recordings. The
directory structure is assumed so this will break if anything changes.
"""

import os
import shutil
import plot_bagfile as pb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), '../sardinia_data')
    out_base_dir = os.path.join(os.getcwd(), '../plots')

    # Make plot directory if it doesn't exist. Existing files will
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

            # Plots are split by experiment and day
            # e.g. ../plots/tod_data/Friday_13-05-22/x.pdf
            output_file_base = os.path.join(out_base_dir, s,d)
            if not os.path.exists(output_file_base):
                os.makedirs(output_file_base)
            rec_sessions = os.listdir(os.getcwd())
            for r in rec_sessions:
                session_path = os.path.join(data_dir, s, d, r)
                if not os.path.isdir(session_path):
                    continue

                fig, outfile = pb.plot_bagfile(session_path)
                outpath = os.path.join(output_file_base,outfile)

                print(outpath)
                fig.savefig(outpath, bbox_inches="tight")
                plt.close(fig)

