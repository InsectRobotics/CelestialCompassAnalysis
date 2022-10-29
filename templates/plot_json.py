import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json
import argparse
import os
from scipy.stats import circmean

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

def synchronise(d, zero=True, unwrap=True):
    """
    Zero each angle series and adjust values so wraps plot nicely.

    :param d: A series of angles (a single recording)
    :param zero: Shift series so angles start from zero
    :param unwrap: Unwrap angles which jump over 0/2pi boundary.
    :returns: The adjusted array.
    """
    d = np.array(d)
    if zero:
        d = d - d[0]
    d = d % (2*np.pi)
    if unwrap:
        for i in range(len(d) - 1):
            if (d[i] < 3*np.pi/4) and (d[i+1] > 7*np.pi/4):
                d[i+1] = d[i+1] - 2*np.pi
            elif (d[i] > 7*np.pi/4) and (d[i+1] < 3*np.pi/4):
                d[i+1] = d[i+1] + 2*np.pi

    return d

def decode_sky_compass(po,
                       n_sol=8,
                       pol_prefs=np.radians([0,90,180,270,45,135,225,315])
):
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
    units, duration, diodes = po.shape

    n_pol = len(pol_prefs)
    sol_prefs = np.linspace(0,2*np.pi - (2*np.pi / n_sol),n_sol)

    angular_outputs = []
    confidence_outputs = []

    for t in range(duration): # For each timestep
        # Get photodiode responses for each unit
        unit_responses = [ po[x][t] for x in range(units) ]
        ss = [ (s_h, s_v) for [_, _, s_v, s_h] in unit_responses ]
        rates = [ (act(h), act(v)) for (h, v) in ss ]
        r_pol = [ (r_h - r_v) / (r_h + r_v) for (r_h, r_v) in rates ]

        # Init for sum
        r_sol = np.zeros(n_sol)
        R = 0
        for z in range(n_sol):
            for j in range(n_pol): # Compute SOL responses (Eq. (2))
                aj = pol_prefs[j] - np.pi/2
                r_sol[z] += (n_sol/n_pol) * np.sin(aj - sol_prefs[z]) * r_pol[j]

            # Accumulate R (Eq. (3))
            R += r_sol[z]*np.exp(-1j*2*np.pi*(z - 1) / n_sol)

        a = np.real(R)
        b = np.imag(R)

        # Compute argument and magnitude of complex conjugate of R (Eq. (4))
        angle = np.arctan2(-b,a)
        confidence = np.sqrt(a**2 + b**2)
        angular_outputs.append(angle)
        confidence_outputs.append(confidence)

    return angular_outputs, confidence_outputs


def produce_plot(session, outfile=None):

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

    mosaic = [["image","skycompass"],
              ["image","imu_yaw"],
              ["image","averages"]]

    fig, ax = plt.subplot_mosaic(mosaic, figsize=(10,4))

    #
    # Plot image if available
    #
    if imagefile != None:
        print(imagefile)
        if outfile == None:
            outfile = imagefile.split(".")[0] + ".pdf"

        img = mpimg.imread(imagefile)
        ax["image"].imshow(img)
        ax["image"].set_title(imagefile.split(".")[0].replace("_", " "))
    else:
        if outfile == None:
            outfile = "out.pdf"
        ax["image"].set_title("No image found")

    max_length = 0
    min_length = 0
    first = True
    corrected_skycompass_table = []
    corrected_imu_table = []
    for r in recordings:
        with open(r) as f:
            data = json.load(f)

        # Extract data
        imu = synchronise(np.array(data["yaw"]))
        pol_op_keys = ["pol_op_{}".format(x) for x in range(8)]
        po = []
        for k in pol_op_keys:
            po.append(data[k])

        po = np.array(po)
        pol_sensor_angle = synchronise(np.array(decode_sky_compass(po)[0]))

        if max_length < len(imu):
            max_length = len(imu)

        if first:
            first = False
            min_length = len(imu)

        if min_length > len(imu):
            min_length = len(imu)

        # Plot cleaned angles
        ax["skycompass"].plot(pol_sensor_angle)
        ax["imu_yaw"].plot(imu)

        corrected_skycompass_table.append(pol_sensor_angle)
        corrected_imu_table.append(imu)

    # Compute mean angle over all reacordings at each timestep.
    sc_means = []
    imu_means = []
    for t in range(min_length):
        sc_angles = [x[t] for x in corrected_skycompass_table]
        imu_angles = [x[t] for x in corrected_imu_table]
        scmean = circmean(sc_angles, nan_policy='omit')
        imumean = circmean(imu_angles, nan_policy='omit')
        sc_means.append(scmean)
        imu_means.append(imumean)

    # Plot (clean) averages
    ax["averages"].plot(synchronise(sc_means))
    ax["averages"].plot(synchronise(imu_means))

    # Add text
    # ax["skycompass"].text(0,5,"Sky Compass")
    # ax["imu_yaw"].set_ylim([-padding, 2*np.pi + padding])
    # ax["averages"].set_ylim([-padding, 2*np.pi + padding])

    ax["skycompass"].set_title("Sky Compass")
    ax["imu_yaw"].set_title("IMU (Yaw)")
    ax["averages"].set_title("Circular average")
    fig.tight_layout()

    # General formatting
    ax["image"].set_xticks([])
    ax["image"].set_yticks([])
    padding = 0.7
    ax["skycompass"].set_ylim([-padding, 2*np.pi + padding])
    ax["imu_yaw"].set_ylim([-padding, 2*np.pi + padding])
    ax["averages"].set_ylim([-padding, 2*np.pi + padding])
    x_axis_padding = 20
    ax["skycompass"].set_xlim([-x_axis_padding, max_length + x_axis_padding])
    ax["imu_yaw"].set_xlim([-x_axis_padding, max_length + x_axis_padding])
    ax["averages"].set_xlim([-x_axis_padding, max_length + x_axis_padding])
    ax["skycompass"].set_xticks([])
    ax["imu_yaw"].set_xticks([])

    os.chdir(calling_dir)
    fig.savefig(outfile, bbox_inches="tight")
#    plt.show()

if __name__=="__main__":
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

    produce_plot(session, outfile=outfile)

