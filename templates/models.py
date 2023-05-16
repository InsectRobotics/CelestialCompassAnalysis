import circstats as cs

import scipy.optimize as so
import numpy as np

UNIT_ANGLES = np.deg2rad([45, 135, 90, 0], dtype='float64')  # -45, +45, 0, 90
POL_PREFS = np.deg2rad([0, 90, 180, 270, 45, 135, 225, 315], dtype='float64')


def pol2sol(pol, sol_prefs, pol_prefs):
    n_sol = len(sol_prefs)
    n_pol = len(pol_prefs)

    sol = np.sum((n_sol / n_pol) * np.cos(pol_prefs[:, None] - sol_prefs[None, :]) * pol[:, None], axis=0)

    return sol


def sol2angle(sol):
    z = np.sum(sol * np.exp(1j * np.linspace(0, 2 * np.pi, len(sol), endpoint=False)))

    angle = np.angle(z)
    confidence = np.abs(z)

    return angle, confidence


def four_zeros(po, pol_prefs, x0=None, verbose=False):
    z0 = 1 / len(pol_prefs) * np.sum(po * np.exp(0j * pol_prefs))
    z1 = 2 / len(pol_prefs) * np.sum(po * np.exp(1j * pol_prefs))
    z2 = 2 / len(pol_prefs) * np.sum(po * np.exp(2j * pol_prefs))

    r0, r1, r2 = np.abs([z0, z1, z2])
    a0, a1, a2 = np.angle([z0, z1, z2]) % (2 * np.pi)

    if verbose:
        print(f"z0: {r0:.2f} exp(i {np.rad2deg(a0):.2f}), "
              f"z1: {r1:.2f} exp(i {np.rad2deg(a1):.2f}), "
              f"z2: {r2:.2f} exp(i {np.rad2deg(a2):.2f})")

    if x0 is None:
        x0 = np.sort((a2 / 2 + np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4) % (2 * np.pi))

    f = lambda a: r1 * np.cos(a1 - a) + r2 * np.cos(a2 - 2 * a)
    df = lambda a: r1 * np.sin(a1 - a) + 2 * r2 * np.sin(a2 - 2 * a)

    # solve the equation to find the 4 zeros
    sol = so.newton(f, x0, fprime=df)

    # sort the solution
    y = np.sort(np.array(sol) % (2 * np.pi))

    # # 4 zeros algorithm
    # s = np.mean(y)
    # i, m = 0, 0
    # if not np.isclose(y[1] - y[0], y[3] - y[2]):
    #     i = 1
    #     m = 1
    # if y[i % 4] - y[(i + 3) % 4] < y[(i + 2) % 4] - y[(i + 1) % 4]:
    #     m += 2
    #
    # angle = (s + m * np.pi/2) % (2 * np.pi)

    # our alternative
    dy = abs(cs.circ_norm(np.diff(y, append=y[0] + 2 * np.pi)))
    i = np.argmin(dy)
    angle = cs.circ_norm(y[i] + 0.5 * dy[i])

    res = [angle]

    if verbose:
        print(f"predicted solar azimuth: {np.rad2deg(angle):.2f}, "
              f"zeros: ", *[f"{np.rad2deg(yi):.2f}" for yi in y])
        res += [y, z0, z1, z2]

    return tuple(res)


def eigenvectors(s000, s045, s090, s135, pol_prefs, verbose=False):
    phi_1, phi_2, phi_3, phi_4 = np.deg2rad([0, 45, 90, 135])  # in local frame

    # Equation (4), Zhao et al. (2016)
    t1 = s000 / s090
    t2 = s045 / s135

    # Equations (5)
    phi = cs.circ_norm(0.5 * np.arctan2(
        (t2 - 1) * np.cos(2 * phi_1) + (1 - t1) * np.cos(2 * phi_2) +
        (t1 - t1 * t2) * np.cos(2 * phi_3) + (t1 * t2 - t2) * np.cos(2 * phi_4),
        (1 - t2) * np.sin(2 * phi_1) + (t1 - 1) * np.sin(2 * phi_2) +
        (t1 * t2 - t1) * np.sin(2 * phi_3) + (t2 - t1 * t2) * np.sin(2 * phi_4)
    ))
    d = (t1 - 1) / (np.cos(2 * phi - 2 * phi_1) - t1 * np.cos(2 * phi - 2 * phi_3))

    # Sturzl (2017)
    azi = pol_prefs  # the azimuth of the device
    ele = np.full_like(pol_prefs, np.pi/4)  # the elevation of the device

    # alpha = phi - azi  # AOP in global frame (original, from paper)
    # e_par = np.array([-np.cos(azi) * np.sin(ele), -np.sin(azi) * np.sin(ele), np.cos(ele)])  # parallel to e-vector
    # e_per = np.array([-np.sin(azi), np.cos(azi), np.zeros_like(azi)])  # perpendicular to e-vector
    # p = -np.cos(alpha) * e_par + np.sin(alpha) * e_per

    alpha = azi + phi  # AOP in global frame (my change)
    # p = np.array([np.sin(azi), np.cos(azi)])  # this points outwards (towards the facing direction)
    # p = np.array([np.cos(-azi), np.sin(-azi)])  # this points 90 deg from facing direction
    p = -np.array([np.sin(alpha), np.cos(alpha)])  # this points 90 deg from facing direction (inwards)

    # # compute eigenvectors
    # eig = p.dot(p.T)
    # eig[~np.isfinite(eig)] = 0.  # ensure that there are not infinites or NaNs
    # w, v = np.linalg.eig(eig)
    # iv = np.argmax(w)
    # vi = v[iv]

    # compute mean vector (equivalent)
    vi = np.mean(d * p, axis=1)

    z_sun = vi[1] + 1j * vi[0]
    res = [np.angle(z_sun)]

    if verbose:
        res += [phi, d, d * p]

    return tuple(res)
