"""..."""

from mimoGrowth.constants import AGE_GROUPS, MEASUREMENTS
from mimoGrowth.growth import Growth
from mimoGrowth.mujoco.motor import calc_motor_gear
import re
import argparse
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ElementTree

# todo: make plots more pretty


def approximation():
    """..."""

    # ? add MSE and LOOCV

    # Pick a measurement from the constants.py file and add
    # the correct description of this measurement.
    measurement = MEASUREMENTS["upper_leg"][0]
    body_part_descr = "Mid Thigh Circumference"

    # Create evenly spaced samples based on min and max age.
    # This will be used to predict measurements between original data points.
    age_samples = np.linspace(min(AGE_GROUPS), max(AGE_GROUPS), 100)

    # Approximate different frunctions based on the age groups and the body
    # part measurement. Then, predict values between the original data
    # using these functions.
    # Use comments to pick the functions you want to see.
    predictions = {
        # "Linear Spline": interp1d(AGE_GROUPS, measurement, kind="linear")(age_samples),
        # "Quadratic Spline": interp1d(AGE_GROUPS, measurement, kind="quadratic")(age_samples),
        "Cubic Spline": interp1d(AGE_GROUPS, measurement, kind="cubic")(age_samples),
        # "Polynomial Fit (deg=1)": np.polyval(np.polyfit(AGE_GROUPS, measurement, deg=1), age_samples),
        "Polynomial Fit (deg=3)": np.polyval(np.polyfit(AGE_GROUPS, measurement, deg=3), age_samples),
        # "Polynomial Fit (deg=5)": np.polyval(np.polyfit(AGE_GROUPS, measurement, deg=5), age_samples),
    }

    # Plot the original data and the approximated function(s).
    plt.plot(AGE_GROUPS, measurement, "ko", label="Original Data")
    for title, pred in predictions.items():
        plt.plot(age_samples, pred, label=title)
    plt.title(f"Predicting {body_part_descr} by Age")
    plt.xlabel("Age (Months)")
    plt.ylabel(f"{body_part_descr} (cm)")
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
    plt.bar(names, densities, edgecolor="k")
    plt.title("Density of Every Geom")
    plt.xlabel("Geom")
    plt.ylabel("Density (kg/m³)")
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def strength():
    """..."""

    # Load the model.
    model = mujoco.MjModel.from_xml_path("mimoEnv/assets/growth.xml")
    data = mujoco.MjData(model)

    # Create an evenly spaced interval for the ages.
    ages = np.linspace(1, 21.5, 100)

    # Iterate over all ages.
    csa_avgs, vol_avgs = [], []
    for age in ages:

        # Get the geom sizes for a specific age.
        geoms = Growth(model, data).calc_growth_params(age)["geom"]

        # Calculate CSA and volume values based on the geoms.
        csa_vals = calc_motor_gear(geoms, model, use_csa=True)
        vol_vals = calc_motor_gear(geoms, model, use_csa=False)

        # Copmute the average gear value either based on CSA or volume.
        avg_csa = np.mean([csa_vals[key]["gear"][0] for key in csa_vals.keys()])
        avg_vol = np.mean([vol_vals[key]["gear"][0] for key in vol_vals.keys()])

        # Store the averages.
        csa_avgs.append(avg_csa)
        vol_avgs.append(avg_vol)

    # Plot the average gear values based on the age.
    plt.plot(ages, csa_avgs, label="CSA")
    plt.plot(ages, vol_avgs, label="Volume")
    plt.title("Average Gear Value by Age")
    plt.xlabel("Age (months)")
    plt.ylabel("Average Gear Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # Create a mapping from keywords to functions.
    func_map = {
        "approximation": approximation,
        "density": density,
        "strength": strength,
    }

    # Create a parser that allows to pass the name of the function to execute in the terminal.
    parser = argparse.ArgumentParser(description="Run functions from the terminal.")
    parser.add_argument("function", choices=func_map.keys(), help="The function to call.")

    # Call the specified function.
    func_map[parser.parse_args().function]()
