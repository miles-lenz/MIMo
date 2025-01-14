"""..."""

from mimoGrowth.constants import AGE_GROUPS, MEASUREMENTS
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import xml.etree.ElementTree as ElementTree


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


if __name__ == "__main__":

    # Create a mapping from keywords to functions.
    func_map = {
        "density": density,
        "approximation": approximation
    }

    # Create a parser that allows to pass the name of the function to execute in the terminal.
    parser = argparse.ArgumentParser(description="Run functions from the terminal.")
    parser.add_argument("function", choices=["density", "approximation"], help="The function to call.")

    # Call the specified function.
    func_map[parser.parse_args().function]()
