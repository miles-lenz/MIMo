"""..."""

from mimoGrowth.constants import AGE_GROUPS, MEASUREMENTS
from mimoGrowth.growth import Growth
from mimoGrowth.mujoco.motor import calc_motor_gear
import re
import argparse
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import xml.etree.ElementTree as ElementTree

# todo: adjust style of plots to be more similar


def approximation():
    """..."""

    # ? add polynomial fit again for comparison
    # ? add MSE and LOOCV

    # Pick a measurement from the constants.py file and add
    # the correct description of this measurement. (see docstring in constants.py)
    measurement = MEASUREMENTS["head"][0]
    body_part_descr = "Head Circumference"

    # Create evenly spaced samples based on min and max age.
    # This will be used to predict measurements between original data points.
    age_samples = np.linspace(min(AGE_GROUPS), max(AGE_GROUPS), 100)

    # Approximate a cubic spline function based on the age groups and the
    # body part measurement. Then, predict values between the original data
    # using this function.
    func = CubicSpline(AGE_GROUPS, measurement)
    prediction = func(age_samples)

    # Plot the original data and the approximated function.
    plt.plot(AGE_GROUPS, measurement, "o", label="Original Data", color="black")
    plt.plot(age_samples, prediction, label="Approximated Cubic Spline Function")
    plt.title(f"Predicting {body_part_descr} by Age with Cubic Spline Approximation", fontsize=16)
    plt.xlabel("Age (Months)", fontsize=12)
    plt.ylabel(f"{body_part_descr} (cm)", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()


def density():
    """..."""

    # Load the MIMo model.
    model = ElementTree.parse("mimoEnv/assets/mimo/MIMo_model.xml").getroot()

    # Store geom names and their density to plot them later.
    names, densities = [], []

    # Iterate over all geoms.
    for geom in model.findall(".//geom"):

        # Get mass and size.
        mass = float(geom.attrib.pop("mass"))
        size = [float(num) for num in re.sub(r"\s+", " ", geom.attrib["size"]).split(" ")]

        # Calculate density based on the type of the geom.
        type_ = geom.attrib["type"]
        if type_ == "box":
            density = mass / (np.prod(size) * 8)
        elif type_ == "capsule":
            density = mass / (np.pi * size[0] ** 2 * size[1] * 2 + (4 / 3) * np.pi * size[0] ** 3)
        elif type_ == "cylinder":
            density = mass / (np.pi * size[0] ** 2 * size[1] * 2)
        elif type_ == "sphere":
            density = mass / ((4 / 3) * np.pi * size[0] ** 3)

        # Store name and density.
        names.append(geom.attrib["name"].replace("geom:", ""))
        densities.append(int(density))

    # Plot the densities.
    bars = plt.bar(names, densities, color="skyblue", edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, height + 0.05, f"{height}",
            ha="center", va="bottom", fontsize=10
        )
    plt.title("Density of All Geoms", fontsize=16)
    plt.xlabel("Geom", fontsize=12, labelpad=10)
    plt.ylabel("Density (kg/m³)", fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def compare_csa_vol():
    """..."""

    # Load the model.
    model = mujoco.MjModel.from_xml_path("mimoEnv/assets/growth.xml")
    data = mujoco.MjData(model)

    # Get the geom sizes for a specific age.
    geoms = Growth(model, data).calc_growth_params(1)["geom"]

    # Calculate CSA and volume values based on the geoms.
    csa_vals = calc_motor_gear(geoms, model, use_csa=True)
    vol_vals = calc_motor_gear(geoms, model, use_csa=False)

    # Format CSA and volume values for plotting.
    motors, gears = [], {"CSA": [], "Volume": []}
    for motor in csa_vals.keys():
        motor_format = motor.replace("act:", "").replace("right_", "").replace("left_", "")
        if motor_format not in motors:
            motors.append(motor_format)
            gears["CSA"].append(np.round(csa_vals[motor]["gear"][0], 3))
            gears["Volume"].append(np.round(vol_vals[motor]["gear"][0], 3))

    # Plot the differenes between CSA and volume.
    x = np.arange(len(motors))
    width, multiplier = 0.25, 0
    _, ax = plt.subplots(layout="constrained")
    for attr, meas in gears.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, meas, width, label=attr, edgecolor="black")
        ax.bar_label(rects, padding=3, fontsize=10)
        multiplier += 1
    plt.title("Comparison of Predicted Gear Values Based on CSA vs. Volume", fontsize=16)
    plt.ylabel("Gear Value", fontsize=12)
    plt.xlabel("Motor", fontsize=12)
    ax.set_xticks(x + width, motors)
    plt.xticks(rotation=90, fontsize=10)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # Create a mapping from keywords to functions.
    func_map = {
        "approximation": approximation,
        "density": density,
        "csa_vol": compare_csa_vol
    }

    # Create a parser that allows to pass the name of the function to execute in the terminal.
    parser = argparse.ArgumentParser(description="Run functions from the terminal.")
    parser.add_argument("function", choices=["approx", "density", "csa_vol"], help="The function to call.")

    # Call the specified function.
    func_map[parser.parse_args().function]()
