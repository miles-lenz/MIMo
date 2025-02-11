"""..."""

import mujoco.viewer
from mimoGrowth.constants import AGE_GROUPS, MEASUREMENTS
from mimoGrowth.growth import adjust_mimo_to_age, delete_growth_scene
from mimoGrowth.elements.motor_handler import calc_motor_gear
import re
import argparse
import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText as AT
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ElementTree

# todo: Improve style and consistency of plots.


def approximation():
    """..."""

    # Idea: Add MSE and LOOCV.

    # Pick a measurement from the constants.py file and add
    # the correct description of this measurement.
    measurement = MEASUREMENTS["upper_leg"][0]
    body_part_descr = "Mid Thigh Circumference"

    # Create evenly spaced samples based on min and max age.
    # This will be used to predict measurements between original data points.
    age_samples = np.linspace(min(AGE_GROUPS), max(AGE_GROUPS), 100)

    # Approximate different functions based on the age groups and the body
    # part measurement. Then, predict values between the original data
    # using these functions.
    # Use comments to pick the functions you want to see.
    predictions = {
        # "Linear Spline": interp1d(
        #     AGE_GROUPS, measurement, kind="linear")(age_samples),
        # "Quadratic Spline": interp1d(
        #     AGE_GROUPS, measurement, kind="quadratic")(age_samples),
        "Cubic Spline": interp1d(
            AGE_GROUPS, measurement, kind="cubic")(age_samples),
        # "Polynomial Fit (deg=1)": np.polyval(
        #     np.polyfit(AGE_GROUPS, measurement, deg=1), age_samples),
        "Polynomial Fit (deg=3)": np.polyval(
            np.polyfit(AGE_GROUPS, measurement, deg=3), age_samples),
        # "Polynomial Fit (deg=5)": np.polyval(
        #     np.polyfit(AGE_GROUPS, measurement, deg=5), age_samples),
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
        size = re.sub(r"\s+", " ", geom.attrib["size"]).split(" ")
        size = [float(num) for num in size]

        # Calculate density based on the type of the geom.
        type_ = geom.attrib["type"]
        if type_ == "box":
            density = mass / (np.prod(size) * 8)
        elif type_ == "capsule":
            vol_cylinder = np.pi * size[0] ** 2 * size[1] * 2
            density = mass / (vol_cylinder + (4 / 3) * np.pi * size[0] ** 3)
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

    # ! This function is broken at the moment. !

    # Load the model.
    model = mujoco.MjModel.from_xml_path("mimoEnv/assets/growth.xml")

    # Create an evenly spaced interval for the ages.
    ages = np.linspace(1, 21.5, 100)

    # Iterate over all ages.
    csa_avg, vol_avg = [], []
    for age in ages:

        # todo: Get calculated geom values for the current age.
        # I can get the from elements.geom_handler.py
        geoms = None

        # Calculate CSA and volume values based on the geoms.
        csa = calc_motor_gear(geoms, model, use_csa=True)
        vol = calc_motor_gear(geoms, model, use_csa=False)

        # Compute the average gear value either based on CSA or volume.
        avg_csa = np.mean([csa[key]["gear"][0] for key in csa.keys()])
        avg_vol = np.mean([vol[key]["gear"][0] for key in vol.keys()])

        # Store the averages.
        csa_avg.append(avg_csa)
        vol_avg.append(avg_vol)

    # Plot the average gear values based on the age.
    plt.plot(ages, csa_avg, label="CSA")
    plt.plot(ages, vol_avg, label="Volume")
    plt.title("Average Gear Value by Age")
    plt.xlabel("Age (months)")
    plt.ylabel("Average Gear Value")
    plt.legend()
    plt.show()


def growth_comparison():
    """..."""

    # todo: Add fourth plot (weight-for-age).

    # IDEAS
    # - Mark some key data points.

    # todo: Improve the graphs.
    # - smooth paper functions (maybe by rounding within func)
    # - redo the GPT functions above

    # ...
    main_path = "mimoGrowth_temp/data/"
    table_paths = {
        "weight": [
            "WHO-Boys-Weight-for-age.csv",
            "WHO-Girls-Weight-for-age.csv"
        ],
        "height": [
            "WHO-Boys-Length-for-age.csv",
            "WHO-Girls-Length-for-age.csv"
        ],
        "head_circum": [
            "WHO-Boys-Head-Circumference-for-age.csv",
            "WHO-Girls-Head-Circumference-for-age.csv"
        ],
    }

    # ...
    data = {
        "mimo": {
            "weight": [],
            "height": [],
            "head_circum": [],
        },
        "WHO": {}
    }

    # ...
    for param, paths in table_paths.items():
        cols = []
        for path in paths:
            df = pd.read_csv(main_path + path)
            cols.append(df.M)
        data["WHO"][param] = np.mean(cols, 0)[1:-2]

    # ...
    ages_mimo = np.linspace(1, 21.5, 42)
    ages_WHO = df.Month[1:-2]

    # Iterate over all ages.
    for i, age in enumerate(ages_mimo):

        # Print the progress.
        print(f"{(i / len(ages_mimo) * 100):.2f}%", end="\r")

        # Create the growth scene.
        growth_scene = adjust_mimo_to_age("mimoEnv/assets/growth.xml", age)

        # Load the model.
        mj_model = mujoco.MjModel.from_xml_path(growth_scene)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        # Store the weight of MIMo.
        weight = mj_model.body("hip").subtreemass[0]
        data["mimo"]["weight"].append(weight)

        # Store the height of MIMo.
        head_pos = mj_data.geom("head").xpos
        head_size = mj_model.geom("head").size
        height_head = head_pos[2] + head_size[0]
        foot_size = mj_model.geom("geom:left_foot3").size
        height_foot = mj_data.geom("geom:left_foot3").xpos[2] - foot_size[0]
        height = (height_head - height_foot) * 100
        data["mimo"]["height"].append(height)

        # Store the head circumference of MIMo.
        head_circum = mj_model.geom("head").size[0] * 2 * np.pi * 100
        data["mimo"]["head_circum"].append(head_circum)

        # Delete the scene.
        delete_growth_scene(growth_scene)

    # Print the progress.
    print("100.0%")

    # Function for creating a subplot.
    def create_subplot(title, y_label, y_mimo, y_WHO):
        plt.plot(ages_WHO, y_WHO, label="WHO Data", linestyle="--")
        plt.plot(ages_mimo, y_mimo, label="MIMo")
        plt.title(title)
        plt.xlabel("Age (months)")
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True, alpha=0.5)
        print(y_mimo)
        print(y_WHO)
        corr = np.corrcoef(y_mimo[::2], y_WHO[:-1])[0, 1]
        text_box = AT(
            f"$r$ = {corr:.3f}",
            frameon=True, loc=4, pad=0.5,
            prop={"alpha": 1, "fontsize": 12}
        )
        plt.gca().add_artist(text_box)

    # Plot the weight.
    plt.subplot(2, 1, 1)
    create_subplot(
        "Weight", "Weight (kg)",
        data["mimo"]["weight"],
        data["WHO"]["weight"],
    )

    # Plot the height.
    plt.subplot(2, 2, 3)
    create_subplot(
        "Height", "Height (cm)",
        data["mimo"]["height"],
        data["WHO"]["height"],
    )

    # Plot the head circumference.
    plt.subplot(2, 2, 4)
    create_subplot(
        "Head Circumference",
        "Head Circumference (cm)",
        data["mimo"]["head_circum"],
        data["WHO"]["head_circum"],
    )

    # Show the final graph.
    plt.suptitle(
        'Comparison of Growth Parameters between MIMo and Real Infant Data',
        fontsize=16, fontweight="bold"
    )
    plt.show()


if __name__ == "__main__":

    # Create a mapping from keywords to functions.
    func_map = {
        "approximation": approximation,
        "density": density,
        "strength": strength,
        "growth_comparison": growth_comparison,
    }

    # Create a parser that allows to pass the name of the function
    # to execute in the terminal.
    parser = argparse.ArgumentParser(
        description="Run functions from the terminal."
    )
    parser.add_argument(
        "function",
        choices=func_map.keys(),
        help="The function to call."
    )

    # Call the specified function.
    func_map[parser.parse_args().function]()
