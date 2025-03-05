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
import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_squared_error, r2_score
import cv2

# This will save the plots instead of opening the viewer.
SAVE_PLOTS = False
SAVE_DIR_PATH = "your/path/here"


def save_plot(name: str = "plot") -> None:
    """
    This function saves the current plot.

    Arguments:
        name (str): The name of the saved plot. Default is 'plot'.
    """

    plt.title("")

    config = {
        "bbox_inches": "tight",
        "pad_inches": 0.1,
    }
    full_path = os.path.join(SAVE_DIR_PATH, name + ".pdf")

    plt.savefig(full_path, **config)


def growth_function(body_part: str = "head_circumference") -> None:
    """
    This function plots different growth functions with their
    associated original data points.

    Arguments:
        body_part (str): The body part the growth function belongs to.
            Default is 'head_circumference'.
    """

    age_samples = np.linspace(0, 24, 100)

    measurements = load_measurements()
    params = approximate_growth_functions(measurements)[body_part]
    pred = func(age_samples, *params)

    y_true = measurements[body_part]["mean"]
    y_pred = func(AGE_GROUPS, *params)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    text_box = AnchoredText(
        f"{'R²':<4} = {r2:.3f}\n{'RMSE':<4} = {rmse:.3f}",
        frameon=True, loc=4, pad=0.5,
        prop={"fontsize": 12, "family": "monospace"}
    )
    plt.setp(text_box.patch, facecolor='white', alpha=1)
    plt.gca().add_artist(text_box)

    plt.errorbar(
        AGE_GROUPS[:-1], measurements[body_part]["mean"][:-1],
        measurements[body_part]["std"][:-1],
        fmt="o", color="black", capsize=5,
        label="Original Data"
    )
    plt.plot(age_samples, pred, color="darkorange", label="MIMo")

    plt.title(f"Predicting Growth of {body_part} by Age")
    plt.xlabel("Age (Months)")
    plt.ylabel("Size (Centimeter)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.5)

    if SAVE_PLOTS:
        save_plot(f"growth_function_{body_part}")
    else:
        plt.show()


def multiple_functions() -> None:
    """
    This function plots multiple growth functions to compare them.
    Just modify the below variable to select different functions.
    """

    body_parts_to_plot = [
        "head_circumference",
        "mid_thigh_circumference",
        "foot_breadth",
        "knee_sole_length",
        "hip_breadth",
        "hand_length",
        "calf_circumference"
    ]

    measurements = load_measurements()
    age_samples = np.linspace(0, 24, 100)

    colors = [plt.get_cmap("tab10")(i) for i in range(10)]

    for i, body_part in enumerate(body_parts_to_plot):
        params = approximate_growth_functions(measurements)[body_part]
        pred = func(age_samples, *params)
        plt.errorbar(
            AGE_GROUPS[:-1], measurements[body_part]["mean"][:-1],
            measurements[body_part]["std"][:-1],
            fmt="o", color=colors[i], capsize=5,
            label=body_part.replace("_", " ").title()
        )
        plt.plot(age_samples, pred, color=colors[i])

    plt.title("Comparing Growth of Various Body Parts by Age")
    plt.xlabel("Age (Months)")
    plt.ylabel("Size (Centimeter)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    if SAVE_PLOTS:
        save_plot("multiple_functions")
    else:
        plt.show()


def density() -> None:
    """
    This functions plots the density of each geom.
    """

    base_values = store_base_values("mimoEnv/assets/growth.xml")

    names, densities = [], []

    for name, attributes in base_values["geom"].items():
        names.append(name.replace("geom:", ""))
        densities.append(attributes["density"])

    plt.bar(names, densities, edgecolor="k")
    plt.title("Density of Every Geom")
    plt.xlabel("Geom")
    plt.ylabel("Density (kg/m³)")
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)
    if SAVE_PLOTS:
        save_plot("density")
    else:
        plt.show()


def comparison() -> None:
    """
    This function will compare weight, height and head circumference
    of MIMo to real infant data from the WHO.

    Use this link for more information:
    https://www.cdc.gov/growthcharts/who-growth-charts.htm
    """

    USE_SUBPLOTS = False

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

    def create_subplot(key, y_label):
        plt.plot(
            age_mimo, data["mimo"][key],
            color="darkorange",
            label="MIMo"
        )
        plt.errorbar(
            age_who, data["WHO"][key]["M"].tolist(),
            data["WHO"][key]["M"] * data["WHO"][key]["S"],
            color="black", linestyle="--", capsize=3,
            label="Mean with Standard Deviation"
        )
        plt.fill_between(
            age_who, data["WHO"][key]["5th"], data["WHO"][key]["95th"],
            color='gray', alpha=0.3,
            label="5th - 95th Percentile"
        )
        plt.fill_between(
            age_who, data["WHO"][key]["10th"], data["WHO"][key]["90th"],
            color='gray', alpha=0.5,
            label="10th - 90th Percentile"
        )
        plt.xlabel("Age (months)")
        plt.ylabel(y_label)
        plt.title(key.replace("_", " ").title())
        plt.grid(True, alpha=0.5)
        plt.legend()

    if USE_SUBPLOTS:

        plt.subplot(2, 1, 1)
        create_subplot("weight", "Weight (kg)")

        plt.subplot(2, 2, 3)
        create_subplot("height", "Height (cm)")

        plt.subplot(2, 2, 4)
        create_subplot("head_circumference", "Weight (kg)")

        title = "Comparison of Parameters between MIMo and Growth Charts"
        plt.suptitle(title, fontsize=16, fontweight="bold")

        if SAVE_PLOTS:
            save_plot("comparison_WHO")
        else:
            plt.show()

    else:

        index = 0
        key = ["weight", "height", "head_circumference"][index]
        y_label = ["Weight (kg)", "Height (cm)", "Circumference (cm)"][index]

        plt.subplot(1, 1, 1)
        create_subplot(key, y_label)

        title = f"Comparison of {y_label.split(" (")[0]} between MIMo "
        title += "and Real Infant Data"
        plt.suptitle(title, fontsize=16, fontweight="bold")

        if SAVE_PLOTS:
            save_plot("comparison_WHO")
        else:
            plt.show()


def video_to_images() -> None:
    """
    This function converts a video to a sequence of images.

    The path to the video and the number of images in the sequence can
    be adjusted below.
    """

    video_path = "path/to/video.mp4"
    num_images = 4

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_images, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    num_frames = len(frames)
    _, axes = plt.subplots(1, num_frames, figsize=(12, 4))

    for ax, frame, step in zip(axes, frames, range(1, num_frames + 1)):
        ax.imshow(frame)
        ax.axis("off")

    plt.tight_layout()

    if SAVE_PLOTS:
        save_plot("video_to_images")
    else:
        plt.show()


if __name__ == "__main__":

    func_map = {
        "growth_function": growth_function,
        "multiple_functions": multiple_functions,
        "density": density,
        "comparison": comparison,
        "video_to_images": video_to_images,
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
