import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json
import argparse
import os

def act(s):
    """
    Photoreceptor activation function.

    :param s: The input to the photoreceptor.
    :return: The firing rate of the photoreceptor given input s.
    """
    if s == 0:
        return 0
    else:
        return np.log(s)
#    return np.sqrt(s)

def synchronise(d, zero=True, unwrap=True):
    d = np.array([ x if x >= 0 else x + 2*np.pi for x in d])
    if zero:
        d = d - d[0]
    if unwrap:
        for i in range(len(d) - 1):
            if (d[i] < np.pi/4) and (d[i+1] > 7*np.pi/4):
                d[i+1] = d[i+1] - 2*np.pi
            elif (d[i] > 7*np.pi/4) and (d[i+1] < np.pi/4):
                d[i+1] = d[i+1] + 2*np.pi

    return d

def decode_sky_compass(po,
                       n_sol=8,
                       pol_prefs=np.radians([0,90,180,270,45,135,225,315]),
):
    # po[p] Unit p
    # po[p][t] readings for unit p at time t
    # po[p][t][d] reading for diode d of unit p at time t
    units, duration, diodes = po.shape # Know the shape dimension in advance

    n_pol = len(pol_prefs)
    sol_prefs = np.linspace(0,2*np.pi - (2*np.pi / n_sol),n_sol)

    angular_outputs = []
    confidence_outputs = []

    for t in range(duration): # For each timestep
        # Get photodiode responses for each unit
        unit_responses = [po[x][t] for x in range(units)]
        ss = [ (s_h, s_v) for [s_v, s_h, _, _] in unit_responses ]
        rates = [ (act(h), act(v)) for (h, v) in ss ]
        r_pol = [ (r_h - r_v) / (r_h + r_v) for (r_h, r_v) in rates ]
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
        angle = np.arctan2(-b,a)
        confidence = np.sqrt(a**2 + b**2)
        angular_outputs.append(angle)
        confidence_outputs.append(confidence)

    return angular_outputs, confidence_outputs

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

    #
    # Pull in data
    #
    if not os.path.isdir(session):
        print("-s must specify a directory")
        sys.exit()

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

    fig, ax = plt.subplot_mosaic(mosaic)

    #
    # Plot image if available
    #
    if imagefile != None:
        print(imagefile)
        img = mpimg.imread(imagefile)
        ax["image"].imshow(img)
        ax["image"].set_title(imagefile.split(".")[0])
    else:
        ax["image"].set_title("No image found")

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


        # # Shift data to start at 0
        # offset = data["yaw"][0]
        # data["yaw"] = data["yaw"] - offset
        # pol_offset = pol_sensor_angle[0]
        # pol_sensor_angle = pol_sensor_angle - pol_offset

        # # Add 2pi to angles less than 0 to move into 0,360
        # data["yaw"] = [ x if x >= 0 else x + 2*np.pi for x in data["yaw"]]
        # pol_sensor_angle = [
        #     x if x >= 0 else x + 2*np.pi for x in pol_sensor_angle
        # ]

        # # Check for jumps across 0/360 and correct for the sake of plotting
        # for i in range(len(data["yaw"]) - 1):
        #     # Jumps from >0 to <360
        #     if data["yaw"][i] < np.pi/4 and data["yaw"][i + 1] > 7*np.pi/4:
        #         data["yaw"][i + 1] = data["yaw"][i + 1] - 2*np.pi

        #     # Jumps from <360 to >0
        #     if data["yaw"][i] > 7*np.pi/4 and data["yaw"][i + 1] < np.pi/4:
        #         data["yaw"][i + 1] = data["yaw"][i + 1] + 2*np.pi

        #     # Jumps from >0 to <360
        #     if pol_sensor_angle[i] < np.pi/4 and pol_sensor_angle[i + 1] > 7*np.pi/4:
        #         pol_sensor_angle[i + 1] = pol_sensor_angle[i + 1] - 2*np.pi

        #     # Jumps from <360 to >0
        #     if pol_sensor_angle[i] > 7*np.pi/4 and pol_sensor_angle[i + 1] < np.pi/4:
        #         pol_sensor_angle[i + 1] = pol_sensor_angle[i + 1] + 2*np.pi


        ax["skycompass"].plot(pol_sensor_angle)

        # Plot cleaned angles
        ax["imu_yaw"].plot(imu)

    # General formatting
    ax["image"].set_xticks([])
    ax["image"].set_yticks([])

    plt.show()

