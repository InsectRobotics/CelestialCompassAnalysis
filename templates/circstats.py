from scipy.stats import circstd

import numpy as np


def circ_mean(samples, high=np.pi, low=-np.pi, axis=None):
    c = to_complex(samples)
    return circ_norm(np.angle(np.nanmean(c, axis=axis)), high, low)


def circ_r(samples, bias=None, axis=None):
    c = to_complex(samples)
    r = abs(np.nanmean(c, axis=axis))
    if bias is not None:
        r *= bias / (2 * np.sin(bias / 2))
    return r


def circ_var(samples, bias=None, axis=None):
    r = circ_r(samples, bias=bias, axis=axis)
    var = -np.log(r)
    var[np.isclose(r, 0)] = 1 - r[np.isclose(r, 0)]  # linear variance
    return var


def circ_std(samples, bias=None, axis=None):
    s = circ_var(samples, bias, axis)
    return np.sqrt(2 * s)
    # return circstd(samples, axis=axis, nan_policy='omit')


def circ_quantiles(samples, quantiles=None, high=np.pi, low=-np.pi, axis=None):

    if quantiles is None:
        quantiles = [0.25, 0.50, 0.75]

    mu = circ_mean(samples, axis=None)
    samples = circ_norm(samples, mu + np.pi, mu - np.pi)

    q = np.nanquantile(samples, quantiles, axis=axis)

    return circ_norm(q, high, low)


def circ_median(samples, high=np.pi, low=-np.pi, axis=None):
    return circ_quantiles(samples, [0.5], high, low, axis)


# def circ_median(samples, high=np.pi, low=-np.pi, axis=None):
#     c = to_complex(samples)
#     return circ_norm(np.angle(np.nanmedian(c, axis=axis)), high, low)
#
#
# def circ_quantiles_custom(samples, quantiles=None, high=np.pi, low=-np.pi, axis=None):
#
#     if quantiles is None:
#         quantiles = [0.25, 0.50, 0.75]
#     if not isinstance(quantiles, list):
#         quantiles = [quantiles]
#
#     if axis is None:
#         samples = np.reshape(samples, -1)
#     elif not isinstance(axis, Iterable):
#         if axis < 0:
#             axis = np.arange(samples.ndim)[axis]
#
#         keep = set(range(samples.ndim)) - {axis}
#         samples = np.transpose(samples, (axis,) + tuple(keep))
#     else:
#         keep = set(range(samples.ndim)) - set(axis)
#         samples = np.transpose(samples, tuple(axis) + tuple(keep))
#
#         shape = samples.shape[-len(keep)]
#
#         samples = np.reshape(samples, (-1,) + tuple(shape))
#
#     n = samples.shape[0]
#
#     d = circ_dist(samples[None, ...], samples[:, None, ...])
#
#     m1 = np.sum(d > 0, axis=1)
#     m2 = np.sum(d < 0, axis=1)
#
#     # calculate the mean of the samples
#     md_mean = circ_mean(samples, axis=0)
#
#     dm = m2 - m1
#     idx = np.argsort(dm, axis=0)
#     md = []
#     for q in quantiles:
#         qq = q * (n - 1)
#
#         frac = qq % 1
#         md.append((1 - frac) * np.take_along_axis(samples, idx[int(qq)][None, ...], axis=0)[0])
#
#         if not np.isclose(frac, 0):
#             # take the average of the two sides
#             md[-1] += frac * np.take_along_axis(samples, idx[int(qq) + 1][None, ...], axis=0)[0]
#
#         # shift the mean towards the quantile
#         md_mean_q = md_mean + np.pi * (q - 0.5)
#
#         # if opposite side of the diameter is closer to the mean, use it as the median instead
#         flip = abs(circ_dist(md_mean_q, md[-1])) > abs(circ_dist(md_mean_q, md[-1] + np.pi))
#         md[-1][flip] += np.pi
#
#     return circ_norm(np.asanyarray(md), high, low)


def to_complex(angle):
    return np.exp(1j * angle)


def circ_dist(a, b, high=np.pi, low=-np.pi):
    return circ_norm(a - b, high, low)


def circ_norm(samples, high=np.pi, low=-np.pi):
    return (samples - low) % (high - low) + low
