from create_csv import read_csv_dataset, eigenvectors, MAX_INT, compare
from circstats import circ_norm

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_base = os.path.abspath(os.path.join(os.getcwd(), '..'))
csv_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'csv'))
clean_path = os.path.join(csv_dir, "clean_all_units.csv")
out_base = os.path.abspath(os.path.join(os.getcwd(), '..', 'plots', 'alternative'))
if not os.path.exists(out_base):
    os.makedirs(out_base)

# data_dirs = ['sardinia_data', 'south_africa_data']
data_dirs = ['south_africa_data']

clean_df = pd.read_csv(clean_path)

# session = "Monday_14-11-22_08-31-21_SAST"
session = "Monday_21-11-22_06-00-11_SAST"
# session = "Sunday_27-11-22_15-46-11_SAST"

dfs = clean_df[clean_df["session"] == session]

df_000 = dfs[dfs["unit_type"] == "S000"]
df_045 = dfs[dfs["unit_type"] == "S045"]
df_090 = dfs[dfs["unit_type"] == "S090"]
df_135 = dfs[dfs["unit_type"] == "S135"]

recordings = np.unique(dfs["recording"])

nb_samples = 8
dev_direct = np.linspace(0, 2 * np.pi, nb_samples, endpoint=False)

for rec in recordings:
    dfr_000 = df_000[df_000["recording"] == rec]
    dfr_045 = df_045[df_045["recording"] == rec]
    dfr_090 = df_090[df_090["recording"] == rec]
    dfr_135 = df_135[df_135["recording"] == rec]

    s1 = dfr_000["response"].to_numpy()
    s2 = dfr_045["response"].to_numpy()
    s3 = dfr_090["response"].to_numpy()
    s4 = dfr_135["response"].to_numpy()

    sun = dfr_000["sun_azimuth"].to_numpy().mean()

    for r in range(0, 360, 360):

        # implement and test the eigenvectors algorithm
        x = circ_norm(np.linspace(0, 2 * np.pi, 360, endpoint=False) + np.deg2rad(r))
        ix = np.argsort(x)
        xi = x[ix]
        s1i = s1[ix]
        s2i = s2[ix]
        s3i = s3[ix]
        s4i = s4[ix]
        pol_prefs = circ_norm(np.linspace(0, 2 * np.pi, nb_samples, endpoint=False))

        s1i_samples = np.interp(pol_prefs, xi, s1i)
        s2i_samples = np.interp(pol_prefs, xi, s2i)
        s3i_samples = np.interp(pol_prefs, xi, s3i)
        s4i_samples = np.interp(pol_prefs, xi, s4i)

        est_sun, phi, d, p = eigenvectors(s1i_samples, s2i_samples, s3i_samples, s4i_samples,
                                          pol_prefs=pol_prefs, verbose=True)

        e, = np.rad2deg(compare(est_sun, np.deg2rad(sun + r)))
        # if abs(e) > 90:
        #     est_sun += np.pi  # break for the 180 ambiguity
        #     p *= (-1)
        #     e, = np.rad2deg(compare(est_sun, np.deg2rad(sun + r)))

        print(f"sun: {sun + r:.2f}, error: {e:.2f}")
        # ----------------------------------------------

        fig = plt.figure(rec, figsize=(2.5, 2))

        ax1 = plt.subplot(111, polar=True)
        ax1.set_theta_direction(-1)
        ax1.set_theta_zero_location("N")

        azi = pol_prefs
        ele = np.full_like(pol_prefs, np.pi / 4)

        plt.plot([np.deg2rad(sun + r)] * 2, [-np.pi/2, np.pi/2], 'g', lw=3)
        plt.plot([est_sun] * 2, [-np.pi/2, np.pi/2], 'k:')
        plt.quiver(azi, ele, p[0], p[1], scale=5)
        plt.scatter(azi, ele, c=phi, s=5, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)

        plt.yticks([])
        plt.xticks([])
        plt.ylim(-np.pi/2, np.pi/2)
        plt.colorbar()

        # ax2 = plt.subplot(122, polar=True)
        # ax2.set_theta_direction(-1)
        # ax2.set_theta_zero_location("N")
        #
        # plt.plot(azi, d, 'k.')
        # plt.plot([np.deg2rad(sun + r)] * 2, [-.1, 1], 'g')
        # plt.plot([est_sun] * 2, [-.1, 1], 'r')
        #
        # plt.yticks([])
        # plt.xticks([])
        # plt.ylim(-.1, 1)

        fig.savefig(os.path.join(out_base, f"{session}-r{rec:02d}-eig.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(out_base, f"{session}-r{rec:02d}-eig.png"), bbox_inches="tight")

        plt.tight_layout()
        plt.show()
