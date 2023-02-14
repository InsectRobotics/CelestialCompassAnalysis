from skimage.filters.thresholding import threshold_otsu, threshold_minimum, threshold_li, threshold_isodata
from skimage.filters.thresholding import threshold_mean, threshold_yen, threshold_triangle
from skimage import measure
from datetime import datetime

import skylight as skl
import numpy as np
import imutils
import pytz
import cv2
import os

scale_percent = 15

thresholds = [threshold_isodata, threshold_li, threshold_mean, threshold_minimum, threshold_otsu,
              threshold_triangle, threshold_yen]

coordinates = {
    "Sardinia": {"lon": 8.440184, "lat": 39.258648, "timezone": 'Europe/Rome'},
    "Vryburg": {"lon": 24.327144, "lat": -26.398643, "timezone": 'Africa/Johannesburg'},
    "Bela Bela": {"lon": 27.918972, "lat": -24.714872, "timezone": 'Africa/Johannesburg'},
    None: {"lon": None, "lat": None, "timezone": None}
}

sky_centre_images = {
    "Saturday_14-05-22_09-00-11_CEST": [datetime(2022, 5, 11, 1), datetime(2022, 5, 20, 23)],
    "Friday_11-11-22_16-01-02_SAST": [datetime(2022, 11, 10, 1), datetime(2022, 11, 13, 23)],
    "Sunday_20-11-22_14-28-16_SAST": [datetime(2022, 11, 14, 1), datetime(2022, 11, 29, 23)],
}

# Saturday_14-05-22_09-00-11_CEST, {'centre': array(1378.+1639.j), 'radius': array(1965, dtype=uint32)}
#   Friday_11-11-22_16-01-02_SAST, {'centre': array(1214.+1806.j), 'radius': array(1861, dtype=uint32)}
#   Sunday_20-11-22_14-28-16_SAST, {'centre': array(1272.+1682.j), 'radius': array(1404, dtype=uint32)}
sky_centres = {
    pytz.timezone(coordinates["Sardinia"]["timezone"]).localize(datetime(2022, 5, 20, 23)): {
        'centre': 1345.+1589.j, 'radius': 2000},
    pytz.timezone(coordinates["Vryburg"]["timezone"]).localize(datetime(2022, 11, 13, 23)): {
        'centre': 1214.+1806.j, 'radius': 2000},
    pytz.timezone(coordinates["Bela Bela"]["timezone"]).localize(datetime(2022, 11, 29, 23)): {
        'centre': 1272.+1682.j, 'radius': 1504},
}

sun_centres = {  # threshold algorithm, exposure, hand-picked x, y, POL, INT
    'Friday_13-05-22_18-00-12_CEST': [threshold_minimum, 8, 87, 1733, -25.12],
    'Friday_13-05-22_18-30-11_CEST': [threshold_minimum, 8, 100, 1900, -28.91],
    'Friday_13-05-22_19-00-23_CEST': [threshold_minimum, 8, 107, 2093, -29.73],
    'Friday_13-05-22_19-30-09_CEST': [threshold_minimum, 8, 107, 2240, -31.84],
    'Friday_13-05-22_20-00-18_CEST': [threshold_minimum, 8, 100, 2307, -33.57],
    'Friday_13-05-22_20-30-23_CEST': [threshold_minimum, 8, 107, 2333, -22.56],
    'Friday_13-05-22_21-00-22_CEST': [threshold_minimum, 8, 40, 2400, -22.56],
    'Monday_16-05-22_11-19-06_CEST': [threshold_yen, 2, 2293, 1073, -59.24, -47.95],
    'Saturday_14-05-22_06-32-27_CEST': [threshold_minimum, 8, 3147, 2280, -57.70, -51.25],
    'Saturday_14-05-22_07-00-12_CEST': [threshold_minimum, 8, 3160, 1927, -45.86, -40.32],
    'Saturday_14-05-22_07-30-13_CEST': [threshold_otsu, 8, 3160, 1787, -41.08, -37.95],
    'Saturday_14-05-22_08-00-13_CEST': [threshold_otsu, 8, 3160, 1620, -41.88, -37.68],
    'Saturday_14-05-22_08-30-18_CEST': [threshold_minimum, 8, 3120, 1467, -43.74, -38.54],
    'Saturday_14-05-22_09-00-11_CEST': [threshold_minimum, 8, 3060, 1293, -39.88, -34.12],
    'Sunday_15-05-22_17-05-48_CEST': [threshold_yen, 2, 333, 1360, -37.98, -1.57],
    'Sunday_15-05-22_17-41-05_CEST': [threshold_minimum, 8, 227, 1487, -46.87, -72.68],
    'Thursday_12-05-22_09-08-12_CEST': [threshold_minimum, 8, 3067, 1633, -64.79, -59.31],
    'Thursday_12-05-22_10-00-48_CEST': [threshold_minimum, 8, 2813, 1300, -47.63, -43.00],
    'Thursday_12-05-22_11-01-03_CEST': [threshold_minimum, 8, 2433, 980, -39.99, -35.86],
    'Thursday_12-05-22_12-03-06_CEST': [threshold_minimum, 8, 1993, 920, -30.87, -30.86],
    'Thursday_12-05-22_13-00-12_CEST': [threshold_minimum, 8, 1647, 887, -21.32, -21.29],
    'Thursday_12-05-22_14-09-27_CEST': [threshold_minimum, 8, 1260, 927, -30.51, -31.19],
    'Thursday_12-05-22_15-09-27_CEST': [threshold_otsu, 8, 887, 1053, -24.74, -25.29],
    'Thursday_12-05-22_16-01-54_CEST': [threshold_minimum, 8, 600, 1167, -31.09, -27.52],
    'Thursday_12-05-22_17-01-44_CEST': [threshold_minimum, 8, 273, 1407, -31.54, -28.04],
    'Thursday_12-05-22_18-00-11_CEST': [threshold_minimum, 8, 87, 1547, -40.49, -36.33],
    'Thursday_12-05-22_19-00-20_CEST': [threshold_minimum, 8, 47, 1660, -45.60, -39.45],
    'Thursday_12-05-22_20-02-32_CEST': [threshold_minimum, 8, 67, 2253, -33.28, -23.99],
    'Thursday_12-05-22_21-03-00_CEST': [threshold_triangle, 2, 53, 2427, 11.31, 18.43],
    'Thursday_19-05-22_11-30-21_CEST': [threshold_otsu, 8, 2200, 973, -45.80, -42.40],
    'Thursday_19-05-22_12-00-16_CEST': [threshold_minimum, 8, 2060, 980, -48.80, -45.75],
    'Thursday_19-05-22_12-30-11_CEST': [threshold_minimum, 8, 1847, 940, -38.62, -35.76],
    'Thursday_19-05-22_13-00-10_CEST': [threshold_minimum, 8, 1673, 933, -37.83, -35.51],
    'Thursday_19-05-22_13-30-25_CEST': [threshold_minimum, 8, 1540, 933, -42.41, -40.42],
    'Thursday_19-05-22_14-00-15_CEST': [threshold_minimum, 8, 1347, 960, -37.63, -35.15],
    'Thursday_19-05-22_14-30-14_CEST': [threshold_minimum, 8, 1173, 987, -36.32, -32.97],
    'Thursday_19-05-22_15-00-14_CEST': [threshold_minimum, 8, 987, 1040, -37.11, -32.11],
    'Thursday_19-05-22_16-33-48_CEST': [threshold_triangle, 2, 533, 1213, -46.74, -53.71],
    'Thursday_19-05-22_16-54-50_CEST': [threshold_minimum, 8, 433, 1113, -61.51, 82.38],
    'Thursday_19-05-22_17-20-24_CEST': [threshold_minimum, 8, 260, 1473, -61.88, -29.14],
    'Friday_11-11-22_14-25-46_SAST': [threshold_yen, 8, 1200, 1153, 168.28, -11.55],
    'Friday_11-11-22_16-01-02_SAST': [threshold_triangle, 2, 660, 1533, -86.12, -168.99],
    'Friday_18-11-22_09-22-15_SAST': [threshold_minimum, 8, 2180, 1040, 3.13, -13.00],
    'Friday_18-11-22_09-34-45_SAST': [threshold_minimum, 8, 2153, 1113, -15.44, -17.38],
    'Monday_14-11-22_07-35-27_SAST': [threshold_minimum, 8, 2627, 920, -17.15, -23.41],
    'Monday_14-11-22_08-31-21_SAST': [threshold_minimum, 8, 2460, 1113, -21.59, -28.34],
    'Monday_14-11-22_09-31-28_SAST': [threshold_minimum, 8, 2100, 1287, -24.81, -32.33],
    'Monday_14-11-22_10-33-11_SAST': [threshold_yen, 2, 2020, 1320, -17.53, -18.86],
    'Monday_14-11-22_11-30-27_SAST': [threshold_minimum, 2, 1793, 1400, -46.30, -56.05],
    'Monday_14-11-22_12-32-42_SAST': [threshold_li, 8, 1573, 1460, -29.57, -29.90],
    'Monday_14-11-22_13-31-22_SAST': [threshold_minimum, 2, 1333, 1513, -11.74, -0.63],
    'Monday_14-11-22_14-30-10_SAST': [threshold_minimum, 2, 1120, 1473, -20.45, -16.96],
    'Monday_14-11-22_15-32-43_SAST': [threshold_triangle, 2, 827, 1567, 0.73, -5.26],
    'Monday_14-11-22_16-31-03_SAST': [threshold_minimum, 8, 680, 1487, 7.18, 0.00],
    'Monday_14-11-22_17-30-59_SAST': [threshold_minimum, 8, 527, 1267, -13.22, -8.27],
    'Monday_14-11-22_18-30-49_SAST': [threshold_minimum, 8, 467, 973, -0.11, 3.59],
    'Monday_14-11-22_19-30-02_SAST': [threshold_minimum, 2, 353, 740, 56.31, 110.93],
    'Monday_21-11-22_05-31-37_SAST': [threshold_minimum, 8, 2927, 833, -104.83, 159.21],
    'Monday_21-11-22_06-00-11_SAST': [threshold_minimum, 8, 2933, 813, -36.07, -35.81],
    'Monday_21-11-22_06-31-27_SAST': [threshold_minimum, 8, 2893, 807, -50.52, -109.34],
    'Monday_21-11-22_07-01-53_SAST': [threshold_minimum, 8, 2833, 973, -39.28, -90.12],
    'Monday_21-11-22_08-01-37_SAST': [threshold_minimum, 8, 2720, 1047, 21.10, -9.02],
    'Monday_21-11-22_14-26-58_SAST': [threshold_minimum, 8, 1047, 1440, -14.71, -17.35],
    'Monday_28-11-22_08-29-06_SAST': [threshold_isodata, 8, 2540, 993, -30.16, -145.29],
    'Monday_28-11-22_08-48-02_SAST': [threshold_isodata, 8, 2433, 1207, -19.45, 12.50],
    'Monday_28-11-22_13-36-03_SAST': [threshold_isodata, 8, 1307, 1420, -28.38, 34.67],
    'Saturday_12-11-22_10-20-48_SAST': [threshold_minimum, 8, 2533, 1320, -41.26, -37.79],
    'Saturday_12-11-22_11-18-20_SAST': [threshold_minimum, 8, 2240, 1407, -148.09, 7.17],
    'Saturday_12-11-22_13-35-07_SAST': [threshold_minimum, 8, 1407, 1233, -51.37, -55.77],
    'Saturday_12-11-22_14-31-01_SAST': [threshold_minimum, 8, 960, 1280, -44.19, -41.82],
    'Saturday_12-11-22_15-23-06_SAST': [threshold_minimum, 8, 740, 1140, -50.35, 105.74],
    'Saturday_12-11-22_16-16-00_SAST': [threshold_minimum, 8, 500, 940, -61.26, -44.55],
    'Saturday_26-11-22_10-17-03_SAST': [threshold_minimum, 8, 2120, 1160, -56.31, -11.31],
    'Saturday_26-11-22_14-03-34_SAST': [threshold_triangle, 2, 1120, 1427, -36.51, 16.25],
    'Saturday_26-11-22_16-02-49_SAST': [threshold_minimum, 8, 687, 1320, -3.36, -14.33],
    'Sunday_13-11-22_08-01-27_SAST': [threshold_minimum, 8, 3160, 1040, -45.39, -45.66],
    'Sunday_13-11-22_09-00-04_SAST': [threshold_minimum, 8, 3000, 1233, -48.78, -48.07],
    'Sunday_13-11-22_10-00-06_SAST': [threshold_minimum, 8, 2053, 333, 37.08, 33.52],
    'Sunday_13-11-22_11-00-09_SAST': [threshold_otsu, 2, 2313, 1400, -51.00, -61.96],
    'Sunday_13-11-22_12-00-05_SAST': [threshold_otsu, 8, 1827, 1327, -99.50, -99.27],
    'Sunday_13-11-22_13-00-06_SAST': [threshold_minimum, 8, 1653, 1293, -50.33, -59.78],
    'Sunday_13-11-22_14-00-04_SAST': [threshold_otsu, 8, 1400, 1293, -43.19, -35.15],
    'Sunday_13-11-22_15-00-05_SAST': [threshold_otsu, 8, 933, 1253, -34.41, -42.07],
    'Sunday_13-11-22_16-00-03_SAST': [threshold_minimum, 8, 627, 1160, -50.94, -37.95],
    'Sunday_13-11-22_17-00-05_SAST': [threshold_minimum, 8, 347, 333, -64.79, -63.60],
    'Sunday_20-11-22_14-28-16_SAST': [threshold_minimum, 8, 1047, 1440, -27.02, -15.07],
    'Sunday_20-11-22_14-36-46_SAST': [threshold_minimum, 8, 1027, 1447, -25.83, -19.02],
    'Sunday_20-11-22_14-45-36_SAST': [threshold_minimum, 8, 1040, 1627, 5.40, 4.45],
    'Sunday_27-11-22_15-14-02_SAST': [threshold_yen, 2, 807, 1553, -7.88, -27.42],
    'Sunday_27-11-22_15-30-38_SAST': [threshold_minimum, 8, 680, 1340, -7.45, -46.16],
    'Sunday_27-11-22_15-46-11_SAST': [threshold_minimum, 8, 680, 1360, -3.92, -42.44],
    'Sunday_27-11-22_16-02-54_SAST': [threshold_minimum, 8, 613, 1300, 6.20, -44.35],
    'Tuesday_22-11-22_10-23-15_SAST': [threshold_minimum, 8, 2513, 1227, -119.19, 68.54],
    'Tuesday_22-11-22_13-07-52_SAST': [threshold_minimum, 8, 1413, 1413, -6.39, -6.39],
    'Tuesday_22-11-22_14-16-59_SAST': [threshold_minimum, 8, 1113, 1407, -13.84, -9.24],
    'Wednesday_23-11-22_11-00-00_SAST': [threshold_minimum, 8, 2153, 1253, -20.13, -174.81],
    'Wednesday_23-11-22_12-01-04_SAST': [threshold_minimum, 8, 1673, 1347, 37.68, 3.59],
    'Wednesday_23-11-22_13-01-47_SAST': [threshold_minimum, 8, 1380, 1360, -30.07, -29.83],
    'Wednesday_23-11-22_15-33-11_SAST': [threshold_minimum, 8, 2607, 1320, -30.91, -27.12],
}

continue_exploring = False


def expose(x, sigma=1.0):
    default_exp = 0.0
    # default_exp = np.median(x, axis=(0, 1)) / 255  # 0.5
    y = np.exp(-np.square(x / 255. - default_exp) / (2 * np.square(sigma)))
    if y.ndim > 2:
        y = np.prod(y, axis=-1)
    return (y - y.min()) / (y.max() - y.min())


def extract_sun_vector(image, approx_xy=None, time=None, fig_name=None, show=False):
    global continue_exploring

    img_sun = read_image(image)

    height, width, _ = img_sun.shape
    dim = (int(width * scale_percent / 100), int(height * scale_percent / 100))

    img_2 = np.clip(np.power(img_sun[..., 2] / 255, 2) * 255, 0, 255).astype('uint8')
    img_8 = np.clip(np.power(img_sun[..., 2] / 255, 8) * 255, 0, 255).astype('uint8')

    # cv2.imshow("Image exposure", cv2.resize(cv2.hconcat([img_2, img_8]), (dim[0] * 2, dim[1]), interpolation=cv2.INTER_AREA))

    xy_sun = {
        "xy": [],
        "threshold_name": [],
        "threshold_value": [],
        "exposure": []
    }

    threshold = threshold_otsu
    exposure = 8
    img_temp = img_8

    print(fig_name, end=",   \t")
    if fig_name == "" and not continue_exploring:
        continue_exploring = True
    elif fig_name in sun_centres and not continue_exploring:
        threshold = sun_centres[fig_name][0]
        exposure = sun_centres[fig_name][1]
        x_sun = sun_centres[fig_name][2]
        y_sun = sun_centres[fig_name][3]
        # if exposure == 2:
        #     img_temp = img_2
        # else:
        #     img_temp = img_8
        xy_sun["threshold_name"].append(threshold.__name__)
        xy_sun["threshold_value"].append(-1)
        xy_sun["exposure"].append(exposure)
        xy_sun["xy"].append(y_sun + 1j * x_sun)

    # if not continue_exploring:
    #     theta = threshold(img_temp)
    #
    #     img_t_exp = (img_temp > theta).astype('uint8') * 255
    #
    #     xy_sun["threshold_name"].append(threshold.__name__)
    #     xy_sun["threshold_value"].append(theta)
    #     xy_sun["exposure"].append(exposure)
    #
    #     c_sun = find_sun2(img_t_exp, img_temp)
    #
    #     xy_sun["xy"].append(c_sun)
    # else:
    if continue_exploring:
        for threshold in thresholds:
            img_t = {}
            for exposure, img_temp in [(2, img_2), (8, img_8)]:
                try:
                    theta = threshold(img_temp)
                except (RuntimeError, ValueError):
                    print(f"Warning: threshold '{threshold.__name__}' could not operate the image "
                          f"with exposure: {exposure}.")
                    continue

                img_t_exp = (img_temp > theta).astype('uint8') * 255

                c_sun = find_sun2(img_t_exp, img_temp)

                if c_sun is None:
                    continue

                xy_sun["xy"].append(c_sun)
                xy_sun["threshold_name"].append(threshold.__name__)
                xy_sun["threshold_value"].append(theta)
                xy_sun["exposure"].append(exposure)

                img_t_exp = cv2.cvtColor(img_t_exp, cv2.COLOR_GRAY2BGR)
                cv2.circle(img_t_exp, (int(np.imag(c_sun)), int(np.real(c_sun))), 20, (0, 0, 255), 3)
                # xy_sun["xy"].append(find_sun(img_t[exposure]))

                img_t[exposure] = img_t_exp

            cv2.imshow(f"Image {threshold.__name__}",
                       cv2.resize(cv2.hconcat([img_t[k] for k in img_t]),
                                  (dim[0] * len(img_t.keys()), dim[1]), interpolation=cv2.INTER_AREA))

            cv2.waitKey(1)

    c_sky, c_radius = [], []
    for ki, k in enumerate(sky_centres):
        if time < k:
            c_sky = sky_centres[k]["centre"]
            c_radius = sky_centres[k]["radius"]
            break

    # evaluate the predictions
    if approx_xy is not None:
        nb_xys = len(xy_sun["xy"])

        errors = []
        for i in range(nb_xys):
            xy_pred = xy_sun["xy"][i]
            error_local = abs((approx_xy - np.angle(xy_pred - c_sky) + np.pi) % (2 * np.pi) - np.pi)
            errors.append(error_local)

        error_min_i = np.argmin(errors)
        a_predi = (np.angle(xy_sun['xy'][error_min_i] - c_sky) + np.pi) % (2 * np.pi) - np.pi
        a_model = (approx_xy + np.pi) % (2 * np.pi) - np.pi
        print(f"Best fit: {xy_sun['threshold_name'][error_min_i]} ({xy_sun['threshold_value'][error_min_i]:.2f}), "
              f"exposure = {xy_sun['exposure'][error_min_i]}, "
              f"prediction = {np.rad2deg(a_predi):.2f}, model = {np.rad2deg(a_model):.2f}, "
              f"x = {np.imag(xy_sun['xy'][error_min_i]):.0f}, y = {np.real(xy_sun['xy'][error_min_i]):.0f}")

        c_sun = xy_sun['xy'][error_min_i]
    else:
        default_threshold = 'threshold_minimum'
        default_exposure = 8
        index, value = np.where(np.all([np.array(xy_sun['threshold']) == default_threshold,
                                np.array(xy_sun['exposure']) == default_exposure], axis=0))

        print(index, value)
        c_sun = xy_sun["xy"][index][:, 1] + 1j * xy_sun["xy"][index][:, 0]

    # for x, y, s, ch in xy_sun:
        # cv2.circle(image, (x, y), int(s / 10), (0, 0, 255), 3)
        # i = np.square(np.arange(2 * r) - r)
        # m = np.zeros((2 * r, 2 * r), dtype='uint8')
        # m[(i[:, None] + i[None, :]) < r ** 2] = 1
        # print(f", {s} ({r:.0f}, {m.sum():.0f}, {s / m.sum():.2f})", end="")
        # print(f", {s} ({r:.0f}, {2 * np.pi * r:.0f})", end="")
        # print(f", {s} ({r:.0f}, {np.sqrt(np.pi * np.square(r)):.0f})", end="")
        # print(f", {s} ({ch}, {ch / s:.2f})", end="")
    # print()

    # draw centre and sky-area on image
    draw_marks(image, c_sun, c_sky, c_radius)

    cv2.imshow("Image value", cv2.resize(image, dim, interpolation=cv2.INTER_AREA))

    if show:
        if fig_name is None:
            fig_name = "Sky Image"
        cv2.imshow(fig_name, cv2.resize(image, dim, interpolation=cv2.INTER_AREA))

    cv2.waitKey(int(not continue_exploring))

    return c_sun - c_sky


def extract_sun_vector_from_name(file_name, draw=None):

    x_sun = sun_centres[file_name][2]
    y_sun = sun_centres[file_name][3]
    c_sun = np.array([y_sun + 1j * x_sun])

    observer = get_observer_from_file_name(file_name)

    c_sky, c_radius = 0., 0.
    for ki, k in enumerate(sky_centres):
        if observer.date < k:
            c_sky = sky_centres[k]["centre"]
            c_radius = sky_centres[k]["radius"]
            break

    # rotate = np.deg2rad(sun_centres[file_name][5])
    # print(f"Sun position (org): {np.imag(np.squeeze(c_sun)):.0f}, {np.real(np.squeeze(c_sun)):.0f}", end=" | ")
    # c_sun = (c_sun - c_sky) * np.exp(-1j * rotate) + c_sky
    # print(f"(rot): {np.imag(np.squeeze(c_sun)):.0f}, {np.real(np.squeeze(c_sun)):.0f}")

    if draw is not None:
        draw_marks(draw, c_sun, c_sky, c_radius)

    return c_sun - c_sky


def extract_sky_centres():
    for k in sky_centre_images:
        csv_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'csv'))

        if "CEST" in k:
            img_dir = os.path.join(csv_dir, "sardinia_data")
        else:
            img_dir = os.path.join(csv_dir, "south_africa_data")

        img_sky_list = []
        for img_sky in os.listdir(img_dir):
            if ".jpg" not in img_sky:
                continue

            time = img_sky.replace("_CEST.jpg", "").replace("_SAST.jpg", "")
            time = datetime.strptime(time, '%A_%d-%m-%y_%H-%M-%S')

            if not sky_centre_images[k][0] < time < sky_centre_images[k][1]:
                continue

            img_sky_list.append(read_image(os.path.join(img_dir, img_sky), grayscale=True, red=False, green=False))
        img_sky = np.array(np.median(img_sky_list, axis=0), dtype='uint8')

        height, width = img_sky.shape
        dim = (int(width * scale_percent / 100), int(height * scale_percent / 100))

        cv2.imshow(f"Mean {k}", cv2.resize(img_sky, dim, interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)

        xy_sky = [[], []]
        ts = 40
        # ts = 70
        while len(xy_sky) > 1 and ts > 0:
            img_tsh = threshold_image(img_sky, threshold=ts, blob_size=(10, 1))
            img_tsh = connected_component_analysis(img_tsh)

            xy_sky = find_sky(img_tsh)[:1]

            img_tsh = cv2.cvtColor(img_tsh, cv2.COLOR_GRAY2BGR)

            cv2.circle(img_tsh, (int(xy_sky[0, 0]), int(xy_sky[0, 1])), int(xy_sky[0, 2]), (0, 255, 0), 3)
            cv2.circle(img_tsh, (int(xy_sky[0, 0]), int(xy_sky[0, 1])), 10, (0, 255, 0), 3)

            cv2.imshow(f"THR {k}", cv2.resize(img_tsh, dim, interpolation=cv2.INTER_AREA))
            cv2.waitKey(0)

            ts -= 1

        if len(xy_sky) == 1:
            sky_centres[sky_centre_images[k][1]] = {
                "centre": np.squeeze(xy_sky[:, 1] + 1j * xy_sky[:, 0]),
                "radius": np.squeeze(xy_sky[:, 2])
            }
            print(k)
            print(sky_centres[sky_centre_images[k][1]])


def read_image(image_path, grayscale=False, red=True, green=True, blue=True, blur=False):
    """
    Load the image, convert it to grayscale, and blur it
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path.copy()

    if grayscale:
        if not red:
            image[:, :, 2] = 0  # remove red
        if not green:
            image[:, :, 1] = 0  # remove green
        if not blue:
            image[:, :, 0] = 0  # remove blue
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 3. / (float(red) + float(green) + float(blue)),
                         dtype='uint8')
    if blur:
        image = cv2.GaussianBlur(image, (11, 11), 0)

    return image


def threshold_image(image, threshold_percent=0.03, remove_small_blobs=True, blob_size=10):
    """
    Threshold the image to reveal light regions in the image.

    This operation takes any pixel value p >= threshold and sets it to 255 (white).
    Pixel values < threshold are set to 0 (black).
    """
    if image.ndim > 2:
        thresh = np.array(np.mean(image, axis=-1), dtype='uint8')
    else:
        thresh = image.copy()

    thresh_r = thresh.reshape(-1)
    t_sort = np.argsort(thresh_r.copy())[::-1]

    # print(np.diff(thresh_r[t_sort]))
    thresh_r[t_sort[:int(threshold_percent * thresh_r.size)]] = 255
    thresh_r[t_sort[int(threshold_percent * thresh_r.size):]] = 0

    if remove_small_blobs:
        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        if isinstance(blob_size, tuple):
            erode, dilate = blob_size
        else:
            erode, dilate = int(20 * blob_size), int(2 * blob_size)
        thresh = cv2.erode(thresh, None, iterations=erode)
        thresh = cv2.dilate(thresh, None, iterations=dilate)

    return thresh


def connected_component_analysis(image, large_threshold=300):
    """
    Perform connected component analysis on the image,
    then initialise a mask to store only the "large" components.
    """
    labels = measure.label(image, background=0, connectivity=2)
    mask = np.zeros(image.shape, dtype="uint8")
    print(len(labels))

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the number of pixels
        label_mask = np.zeros(image.shape, dtype="uint8")
        label_mask[labels == label] = 255
        nb_pixels = cv2.countNonZero(label_mask)

        # if the number of pixels in the component is sufficiently large,
        # then add it to the mask of "large blobs"
        if nb_pixels > large_threshold:
            mask = cv2.add(mask, label_mask)

    return mask


def find_sun(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    # cnts = contours.sort_contours(cnts)[0]

    # loop over the contours
    if len(cnts) > 1:
        print(f"Warning: multiple ({len(cnts)}) sun blobs found.")
    elif len(cnts) < 1:
        print(f"Warning: no sun blobs found.")

    c_size = [len(c) for c in cnts]
    i_sort = np.argsort(c_size)[::-1]
    xy_sun = np.zeros((len(cnts), 4), dtype='uint32')
    for j, i in enumerate(i_sort):
        c = cnts[i]
        x, y = [], []
        for kp in c:
            x.append(kp[0][0])
            y.append(kp[0][1])
        x, y = int(np.nanmedian(x)), int(np.nanmedian(y))
        ch = cv2.convexHull(c, hull=True, returnPoints=True)
        cp = poly_perimeter(c)
        hp = poly_perimeter(ch)

        xy_sun[j] = [x, y, cp, hp]

    return xy_sun


def find_sun2(mask, weight=None):
    y_c, x_c = np.where(mask > 128)
    w_c = weight[mask > 128]

    if len(y_c) < 1 or len(x_c) < 1:
        return None

    if weight is not None:
        x_sun = np.sum(x_c * w_c) / (np.sum(w_c) + np.finfo(float).eps)
        y_sun = np.sum(y_c * w_c) / (np.sum(w_c) + np.finfo(float).eps)
    else:
        x_sun = np.median(x_c)
        y_sun = np.median(y_c)

    return y_sun + 1j * x_sun


def find_sky(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    # cnts = contours.sort_contours(cnts)[0]

    # loop over the contours
    if len(cnts) > 1:
        print(f"Warning: multiple ({len(cnts)}) sky blobs found.")
    elif len(cnts) < 1:
        print(f"Warning: no sky blobs found.")

    c_size = [len(c) for c in cnts]
    i_sort = np.argsort(c_size)[::-1]
    xy_sky = np.zeros((len(cnts), 3), dtype='uint32')
    for j, i in enumerate(i_sort):
        c = cnts[i]
        (x, y), radius = cv2.minEnclosingCircle(c)
        xy_sky[j] = [int(x), int(y), int(radius)]

    return xy_sky


def poly_perimeter(points):
    points = np.array(points)
    return np.squeeze(np.sum(np.sqrt(np.sum(np.square(np.diff(points, axis=0, append=points[:1])), axis=-1)), axis=0))


def get_observer_from_file_name(filename):
    timestamp = datetime.strptime(filename.replace("_CEST", "").replace("_SAST", ""), '%A_%d-%m-%y_%H-%M-%S')

    place = None
    if "CEST" in filename:
        place = "Sardinia"
    elif "SAST" in filename:
        if timestamp < datetime(2022, 11, 16):
            place = "Vryburg"
        else:
            place = "Bela Bela"

    if place is not None:
        local_time = pytz.timezone(coordinates[place]["timezone"]).localize(timestamp)
        return skl.ephemeris.Observer(lon=coordinates[place]["lon"], lat=coordinates[place]["lat"],
                                      date=local_time, city=place, degrees=True)
    else:
        return skl.ephemeris.Observer(lon=coordinates[place]["lon"], lat=coordinates[place]["lat"],
                                      date=timestamp, degrees=True)


def draw_marks(image, sun_centre, sky_centre, sky_radius):
    # draw centre and sky-area on image
    cv2.circle(image, (int(np.imag(sky_centre)), int(np.real(sky_centre))), int(sky_radius), (0, 255, 0), 3)
    cv2.circle(image, (int(np.imag(sky_centre)), int(np.real(sky_centre))), 20, (0, 255, 0), 3)
    cv2.circle(image, (int(np.imag(sun_centre)), int(np.real(sun_centre))), 20, (0, 0, 255), 3)


if len(sky_centres) < 1:
    extract_sky_centres()


if __name__ == "__main__":

    data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'csv'))

    datasets = ["sardinia_data", "south_africa_data"]
    # datasets = ["sardinia_data"]

    for ds in datasets:

        ds_dir = os.path.join(data_dir, ds)

        datasets = os.listdir(data_dir)
        img_files = [x for x in os.listdir(ds_dir) if ".jpg" in x]
        for i_file in img_files:
            i_name = i_file.replace(".jpg", "")
            img_path = os.path.join(ds_dir, i_file)
            # print(img_path)
            # img_path = os.path.join(ds_dir, "Friday_13-05-22_18-00-12_CEST.jpg")
            img = read_image(img_path)

            obs = get_observer_from_file_name(i_name)

            h, w, channels = img.shape
            d = (int(w * scale_percent / 100), int(h * scale_percent / 100))
            # cv2.imshow(i_name, cv2.resize(img, d, interpolation=cv2.INTER_AREA))
            # cv2.waitKey(1)
            sun_model = skl.Sun(obs)
            sun_pred = extract_sun_vector_from_name(i_name, draw=img)

            # print(obs)
            cv2.imshow("Sky image", cv2.resize(img, d, interpolation=cv2.INTER_AREA))
            cv2.waitKey(1)

            # cv2.waitKey(1)
            sun_dir = os.path.join(data_dir, "sun")
            if not os.path.exists(sun_dir):
                os.mkdir(sun_dir)
            cv2.imwrite(os.path.join(sun_dir, i_file), img)

            azi_model = (np.rad2deg(sun_model.az) + 180) % 360 - 180
            azi_predict = np.rad2deg(np.squeeze(np.angle(sun_pred)))
            print(f"{i_file},   \t", end="")
            print(f"model: {azi_model:.2f},   \tpredicted: {azi_predict:.2f},   \t"
                  f"difference: {azi_model - azi_predict:.2f}")

