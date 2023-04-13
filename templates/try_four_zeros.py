from create_csv import four_zeros, compare, pol2sol, sol2angle
from circstats import circ_norm

from scipy.optimize import fsolve

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

data_dirs = ['sardinia_data', 'south_africa_data']

clean_df = pd.read_csv(clean_path)

# session = "Friday_13-05-22_18-00-12_CEST"
# session = "Friday_13-05-22_20-30-23_CEST"
# session = "Monday_16-05-22_11-19-06_CEST"

# session = "Monday_14-11-22_08-31-21_SAST"
# session = "Monday_21-11-22_06-00-11_SAST"
session = "Sunday_27-11-22_15-46-11_SAST"

dfs = clean_df[clean_df["session"] == session]
dfp = dfs[dfs["unit_type"] == "POL"]

recordings = np.unique(dfp["recording"])
print(dfp.columns)

nb_samples = 8

for rec in recordings:
    dfr = dfp[dfp["recording"] == rec]

    p = dfr["response"].to_numpy()
    s = dfr["sun_azimuth"].to_numpy().mean()

    # implement and test four zeros
    x = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ix = np.argsort(x)
    xi = x[ix]
    pi = p[ix]
    pol_prefs = circ_norm(np.linspace(0, 2 * np.pi, nb_samples, endpoint=False), 2 * np.pi, 0)
    # pol_prefs = (np.linspace(0, 2 * np.pi, nb_samples, endpoint=False) + np.deg2rad(r)) % (2 * np.pi)

    p_samples = np.interp(pol_prefs, xi, pi)
    alpha, y, z0, z1, z2 = four_zeros(p_samples, pol_prefs, verbose=True)

    sol = pol2sol(-p_samples, pol_prefs, pol_prefs)
    ang, sig = sol2angle(sol)
    print(f"SOL: {np.rad2deg(ang):.2f} ({sig:.2f})")

    r0, r1, r2 = np.abs([z0, z1, z2])
    a0, a1, a2 = np.angle([z0, z1, z2])

    p_z0 = r0 * np.cos(a0 - np.linspace(0, 0 * np.pi, 360, endpoint=False))
    p_z1 = r1 * np.cos(a1 - np.linspace(0, 2 * np.pi, 360, endpoint=False))
    p_z2 = r2 * np.cos(a2 - np.linspace(0, 4 * np.pi, 360, endpoint=False))

    e, = np.rad2deg(compare(alpha, np.deg2rad(s)))
    print(f"sun: {s:.2f}, error: {e:.2f}")

    fig = plt.figure(rec, figsize=(2, 2))

    ax1 = plt.subplot(111, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")

    plt.plot(xi, pi - p_z0, color='gray')
    # plt.plot(xi, p_z1, color='C0')
    # plt.plot(xi, p_z2, color='C1')
    plt.plot([np.deg2rad(s)] * 2, [-1, 1], 'g', lw=3)
    plt.plot([alpha % (2 * np.pi)] * 2, [-1, 1], 'k:')
    plt.plot(xi, p_z1 + p_z2, color='k')
    # plt.plot([ang % (2 * np.pi)] * 2, [-1, 1], 'C0:')
    # plt.plot(np.rad2deg([(a2 / 2) % (2 * np.pi)] * 2), [-.2, .2], 'k:')
    plt.plot(y, np.zeros(4), 'ro')
    plt.ylim([-1, 1])
    plt.yticks([0], [""])
    plt.xticks([])

    p_z1_n = (p_z1 - p_z1.min()) / (p_z1.max() - p_z1.min())
    p_z2_n = (p_z2 - p_z2.min()) / (p_z2.max() - p_z2.min())
    p_sun = p_z2_n - p_z1_n
    p_samples = np.interp(pol_prefs, xi, p_sun)
    z_sun = np.sum(p_samples * np.exp(1j * pol_prefs))

    # ax2 = plt.subplot(122, polar=True)
    # ax2.set_theta_direction(-1)
    # ax2.set_theta_zero_location("N")
    #
    # plt.plot(xi, p_z1_n, color='C0')
    # plt.plot(xi, p_z2_n, color='C1')
    # plt.plot(xi, p_sun, color='k')
    # plt.plot([alpha % (2 * np.pi)] * 2, [-.1, 1], 'C1:')
    # plt.plot([ang % (2 * np.pi)] * 2, [-1, 1], 'C0:')
    # plt.plot([np.angle(z_sun) % (2 * np.pi)] * 2, [-1, .5], 'k')
    # plt.plot(np.deg2rad(s), 0, 'go')
    # plt.ylim([-1, 1])
    # plt.yticks([0], [""])
    # plt.xticks([])

    fig.savefig(os.path.join(out_base, f"{session}-r{rec:02d}-fz.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(out_base, f"{session}-r{rec:02d}-fz.png"), bbox_inches="tight")

    plt.tight_layout()
    plt.show()

