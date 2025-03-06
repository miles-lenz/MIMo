""" This module contains different plotting functions. """

import mujoco.viewer
from mimoGrowth.constants import AGE_GROUPS
from mimoGrowth.growth import adjust_mimo_to_age, delete_growth_scene
from mimoGrowth.utils import load_measurements, store_base_values, \
    approximate_growth_functions
from mimoGrowth.utils import growth_function as func
import argparse
from collections import defaultdict
import os
import re
import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import cv2


def save_plot():
    """
    This function saves the current plot in the
    directory of this script.
    The file will be called 'plot.pdf'.
    """

    dirname = os.path.dirname(__file__)
    plt.title("")
    plt.savefig(os.path.join(dirname, "plot.pdf"))


def growth_function(
        body_part: str = "head_circumference", save: bool = False) -> None:
    """
    This function plots different growth functions with their
    associated original data points.

    Arguments:
        body_part (str): The body part the growth function belongs to.
            Default is 'head_circumference'.
        save (bool): If the plot should be saved instead of shown.
    """

    age_samples = np.linspace(0, 24, 100)

    measurements = load_measurements()
    params = approximate_growth_functions(measurements)[body_part]
    pred = func(age_samples, *params)

    y_true = measurements[body_part]["mean"]
    y_pred = func(AGE_GROUPS, *params)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    plt.plot(age_samples, pred, label="MIMo")
    plt.errorbar(
        AGE_GROUPS[:-1], measurements[body_part]["mean"][:-1],
        measurements[body_part]["std"][:-1],
        fmt="o", label="Original Data"
    )

    plt.title(f"Predicting Growth of {body_part} by Age")
    plt.xlabel("Age (Months)")
    plt.ylabel("Size (Centimeter)")

    plt.legend()
    save_plot() if save else plt.show()


def multiple_functions(save: bool = False) -> None:
    """
    This function plots multiple growth functions to compare them.
    Just modify the below variable to select different functions.

    Arguments:
        save (bool): If the plot should be saved instead of shown.
    """

    body_parts_to_plot = [
        "ankle_circumference",
        "foot_length",
        "hip_breadth",
        "mid_thigh_circumference",
        "rump_knee_length",
        "shoulder_elbow_length",
    ]

    measurements = load_measurements()
    age_samples = np.linspace(0, 24, 100)

    for body_part in body_parts_to_plot:

        params = approximate_growth_functions(measurements)[body_part]
        pred = func(age_samples, *params)

        plt.plot(age_samples, pred, label=body_part.replace("_", " ").title())

    plt.title("Comparing Growth of Various Body Parts by Age")
    plt.xlabel("Age (Months)")
    plt.ylabel("Size (Centimeter)")

    plt.legend()
    save_plot() if save else plt.show()


def density(save: bool = False) -> None:
    """
    This functions plots the density of each geom.

    Note that identical geoms (e.g left_eye and right_eye) are only
    plotted once since they have the same density.

    Arguments:
        save (bool): If the plot should be saved instead of shown.
    """

    base_values = store_base_values("mimoEnv/assets/growth.xml")

    names, densities = [], []

    for name, attributes in base_values["geom"].items():
        name = re.sub(r"geom:|left_|right_", "", name)
        if name not in names:
            names.append(name)
            densities.append(attributes["density"])

    plt.bar(names, densities, zorder=3, edgecolor="k")
    plt.title("Density of Every Geom")
    plt.xlabel("Geom")
    plt.ylabel("Density (kg/m³)")

    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.15)

    save_plot() if save else plt.show()


def comparison(metric: str = "height", save: bool = False) -> None:
    """
    This function will compare a growth parameter of MIMo and
    real infants from WHO growth charts.

    Use this link for more information:
    https://www.cdc.gov/growthcharts/who-growth-charts.htm

    Arguments:
        metric (str): Selects which data will be compared. Can be one of the
            following: ['height', 'weight', 'head_circumference']
        save (bool): If the plot should be saved instead of shown.
    """

    data = {"mimo": defaultdict(list), "WHO": {}}

    path = "mimoGrowth_temp/growth_charts/"
    for dirpath, _, filenames in os.walk(path):

        if filenames == []:
            continue

        dfs = []
        for path in filenames:
            full_path = os.path.join(dirpath, path)
            dfs.append(pd.read_csv(full_path))
        df = sum(dfs) / len(dfs)

        key = dirpath.split("/")[-1]
        data["WHO"][key] = df

    age_mimo = np.linspace(0, 24, 50)
    age_who = list(range(0, 25))

    for i, age in enumerate(age_mimo):

        print(f"{(i / len(age_mimo) * 100):.2f}%", end="\r")

        growth_scene = adjust_mimo_to_age(
            age, "mimoEnv/assets/growth.xml", False)

        mj_model = mujoco.MjModel.from_xml_path(growth_scene)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        weight = mj_model.body("hip").subtreemass[0]
        data["mimo"]["weight"].append(weight)

        head_pos = mj_data.geom("head").xpos
        head_size = mj_model.geom("head").size
        height_head = head_pos[2] + head_size[0]
        foot_size = mj_model.geom("geom:left_foot3").size
        height_foot = mj_data.geom("geom:left_foot3").xpos[2] - foot_size[0]
        height = (height_head - height_foot) * 100
        data["mimo"]["height"].append(height)

        head_circum = mj_model.geom("head").size[0] * 2 * np.pi * 100
        data["mimo"]["head_circumference"].append(head_circum)

        delete_growth_scene(growth_scene)

    print("100.0%")

    plt.plot(age_mimo, data["mimo"][metric], label="MIMo")
    plt.errorbar(
        age_who, data["WHO"][metric]["M"].tolist(),
        data["WHO"][metric]["M"] * data["WHO"][metric]["S"],
        linestyle="--", label="Mean with Standard Deviation"
    )
    plt.fill_between(
        age_who, data["WHO"][metric]["5th"], data["WHO"][metric]["95th"],
        color='gray', alpha=0.3,
        label="5th - 95th Percentile"
    )
    plt.fill_between(
        age_who, data["WHO"][metric]["10th"], data["WHO"][metric]["90th"],
        color='gray', alpha=0.4,
        label="10th - 90th Percentile"
    )

    y_label = {
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "head_circumference": "Head Circumference (cm)"
    }[metric]

    plt.title("Comparing MIMo to Real Infants")
    plt.xlabel("Age (months)")
    plt.ylabel(y_label)

    plt.legend()
    save_plot() if save else plt.show()


def video_to_image(
        path: str, count_images: int = 3, overlay: bool = False,
        alpha: float = 0.5, save: bool = False) -> None:

    count_images = int(count_images)

    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, count_images, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if overlay:

        base_frame = frames[0].astype(float)
        for frame in frames[1:]:
            base_frame = cv2.addWeighted(
                base_frame, 1 - alpha, frame.astype(float), alpha, 0)

        plt.imshow(base_frame.astype(np.uint8))
        plt.axis("off")
        save_plot() if save else plt.show()

    else:

        _, axes = plt.subplots(1, len(frames), figsize=(12, 4))
        for ax, frame in zip(axes, frames):
            ax.imshow(frame)
            ax.axis("off")

        plt.tight_layout()
        save_plot() if save else plt.show()


if __name__ == "__main__":

    func_map = {
        "growth_function": growth_function,
        "multiple_functions": multiple_functions,
        "density": density,
        "comparison": comparison,
        "video_to_image": video_to_image,
    }

    parser = argparse.ArgumentParser(
        description="Run functions from the terminal."
    )
    parser.add_argument(
        "function",
        choices=func_map.keys(),
        help="The function to call."
    )
    parser.add_argument(
        "kwargs",
        nargs=argparse.REMAINDER,
        help="Additional keyword arguments."
    )

    kwargs = {}
    for param in parser.parse_args().kwargs:
        key, value = param.split("=")
        kwargs[key] = value

    func_map[parser.parse_args().function](**kwargs)
