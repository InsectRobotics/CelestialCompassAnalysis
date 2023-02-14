from scipy.optimize import lsq_linear

import numpy as np

UNIT_ANGLES = np.deg2rad([45, 135, 90, 0], dtype='float64')  # -45, +45, 0, 90
POL_PREFS = np.deg2rad([0, 90, 180, 270, 45, 135, 225, 315], dtype='float64')

A: np.ndarray = None
A_inv: np.ndarray = A


def decode_sky_compass(po, n_sol=8, pol_prefs=POL_PREFS, polarisation=True, intensity=False):
    """
    Sky compass decoding routine from Gkanias et al. (2019). This has been
    implemeneted from scratch using the equations presented as a demonstration
    that everything can be programmed from scratch without calling out to
    external libraries or FFT implementations.

    :param po: Pol-op unit responses
    :param n_sol: The number of SOL neurons in use
    :param pol_prefs: The preferred angles of each POL neuron. This is determined
                      by the hardware configuration at the time the recording was
                      taken.
    :returns: A list of azimuthal angles with a list of associated confidence
              in each.
    """
    # po structure
    # po[p] Unit p
    # po[p][t] readings for unit p at time t
    # po[p][t][d] reading for diode d of unit p at time t
    units, diodes = po.shape

    sol_prefs = np.linspace(0, 2 * np.pi, n_sol, endpoint=False)

    # ensure that the response is in [0, 1]
    # the highest observed value was 32768 (Thursday 17:20)
    response = np.clip(abs(po) / 33000., 0., 1.)
    if not polarisation and not bool(intensity):
        angle, sigma = pol2eig(response, pol_prefs)
    elif polarisation and not bool(intensity):
        angle, sigma = pol2sol(response, sol_prefs, pol_prefs, unit_transform=unit2pol)
    elif not polarisation and bool(intensity):
        angle, sigma = pol2sol(response, sol_prefs + np.pi, pol_prefs, unit_transform=unit2int)
    else:
        a_pol, c_pol = pol2sol(response, sol_prefs, pol_prefs, unit_transform=unit2pol)
        a_int, c_int = pol2sol(response, sol_prefs + np.pi, pol_prefs, unit_transform=unit2int)
        c_int /= len(pol_prefs)
        s_pol = 1. / (c_pol + np.finfo(float).eps) ** 2
        s_int = 1. / (float(intensity) * c_int + np.finfo(float).eps) ** 2

        angle = (s_pol * a_int + s_int * a_pol) / (s_pol + s_int)
        sigma = (s_pol * s_int) / (s_pol + s_int)

    return angle, sigma


def pol2sol(po, sol_prefs, pol_prefs, unit_transform):
    units, diodes = po.shape
    n_sol = len(sol_prefs)
    n_pol = len(pol_prefs)

    # Get photodiode responses for each unit
    r_pol = np.array([unit_transform(po[x]) for x in range(units)])

    # Init for sum
    r_sol = np.zeros(n_sol)
    R = 0
    for z in range(n_sol):
        for j in range(n_pol):  # Compute SOL responses (Eq. (2))
            aj = pol_prefs[j] - np.pi / 2
            r_sol[z] += (n_sol / n_pol) * np.sin(aj - sol_prefs[z]) * r_pol[j]

        # Accumulate R (Eq. (3))
        R += r_sol[z] * np.exp(-1j * 2 * np.pi * (z - 1) / n_sol)

    a = np.real(R)
    b = np.imag(R)

    # Compute argument and magnitude of complex conjugate of R (Eq. (4))
    angle = np.arctan2(-b, a)
    confidence = np.sqrt(a ** 2 + b ** 2)

    return angle, confidence


def unit2pol(unit_response, log=True):
    r_h, r_v = unit2vh(unit_response, log)
    # return r_h - r_v
    return (r_h - r_v) / (r_h + r_v + np.finfo(float).eps)


def unit2int(unit_response, log=True):
    r_h, r_v = unit2vh(unit_response, log)
    r = (r_h + r_v) / 2
    # r = np.sqrt(r_h**2 + r_v**2)
    return r


def unit2vh(unit_response, log=True):
    # default: -45, +45, 0, 90
    s = act(abs(unit_response) + np.finfo(float).eps, log)
    s_v = np.squeeze(s[np.isclose(UNIT_ANGLES, 0)])
    s_h = np.squeeze(s[np.isclose(UNIT_ANGLES, np.pi/2)])

    return s_h, s_v


def act(s, log=True):
    """
    Photoreceptor activation function. Gkanias et al. (2019) uses square
    root activation, however log activation appears to generate a slightly
    more stable angular representation.

    :param s: The input to the photoreceptor.
    :param log: Set True to use Log activation rather than square root.
    :return: The firing rate of the photoreceptor given input s.
    """
    if not log:
        return np.sqrt(s)

    return np.log(s)


def pol2eig(po, pol_prefs):

    pol_prefs = pol_prefs[:-1]
    units, diodes = po.shape
    units -= 1

    # Get photodiode responses for each unit
    phi, d = np.array([unit2eig(po[x], pol_prefs[x]) for x in range(units)]).T

    pe = np.array([np.cos(phi), np.sin(phi), np.zeros_like(phi)])

    alpha = pol_prefs
    gamma = np.full_like(alpha, np.deg2rad(45))

    e = np.zeros_like(pe)
    for i, a, g in zip(range(len(alpha)), alpha, gamma):
        c1 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype='float32')
        c2 = np.array([[np.cos(g), 0, np.sin(g)], [0, 1, 0], [-np.sin(g), 0, np.cos(g)]], dtype='float32')
        c = np.dot(c1, c2)

        e[:, i] = c.dot(pe[:, i])

    eig = e.dot(e.T)

    eigenvalues, eigenvectors = np.linalg.eigh(eig)

    i_star = np.argmin(eigenvalues)
    s_1, s_2, s_3 = eigenvectors[i_star]

    # print(eigenvalues)
    # print(eigenvectors)

    # import sys
    # sys.exit()

    # Compute argument and magnitude of complex conjugate of R (Eq. (4))
    angle = np.arctan2(s_2, s_1)
    confidence = 1.0

    return angle, confidence


def unit2eig(unit_response, pref_angle=0):
    # q1, q2, q3 = np.linalg.inv(a.T.dot(a)).dot(a.T).dot(unit_response)
    q1, q2, q3 = update_globals(unit_response)
    # q1 = i_0 - i_90
    # q2 = i_45 - i_135
    # q3 = (i_0 + i_45 + i_90 + i_135) / 2

    phi = 0.5 * np.arctan2(q2, q1) - pref_angle
    d = np.sqrt(np.square(q1) + np.square(q2)) / q3

    return phi, d


def update_globals(f=None):
    global A, A_inv

    A = np.round(np.array([np.cos(2 * UNIT_ANGLES), np.sin(2 * UNIT_ANGLES), np.ones_like(UNIT_ANGLES)]).T, decimals=2)

    if f is None:
        # A_inv = np.linalg.inv(A.T.dot(A)).dot(A.T)
        A_inv = np.linalg.pinv(A)
        return None
    else:
        res = lsq_linear(A, f)
        if res.x is not None:
            return res.x
        else:
            A_inv = np.linalg.pinv(A)
            return np.dot(A_inv, f)


update_globals()


if __name__ == "__main__":
    print(np.rad2deg(UNIT_ANGLES))
    phi_i, d_i = unit2eig([0.25, 0.5, 1, 0])
    print(f"phi = {np.rad2deg(phi_i):.2f}, d = {d_i:.2f}")
