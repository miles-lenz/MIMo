"""..."""

from mimoGrowth.constants import MEASUREMENT_TYPES, MEASUREMENTS, AGE_GROUPS
from collections import defaultdict
import re
import os
from scipy.interpolate import CubicSpline
import numpy as np
import xml.etree.ElementTree as ET


def approximate_functions() -> dict:
    """..."""

    functions = defaultdict(list)

    # Iterate over all body parts and their associated measurements.
    for body_part, measurements in MEASUREMENTS.items():

        # Iterate over all measurements for the current body part.
        # Note that some body parts have multiple measurements e.g. the
        # upper arm has a circumference and a length.
        for meas in measurements:

            # Approximate and store a cubic spline function based on the
            # current measurements and the corresponding age groups.
            func = CubicSpline(AGE_GROUPS, meas)
            functions[body_part].append(func)

    return functions


def store_original_values(path_scene: str) -> None:
    """..."""

    # Find the model and meta path within the scene.
    xml_scene = ET.parse(path_scene).getroot()
    paths = [inc.attrib["file"] for inc in xml_scene.findall(".//include")]

    # Load the model and the meta file.
    dir_name = os.path.dirname(path_scene) + "/"
    model = ET.parse(dir_name + [path for path in paths if "model" in path][0])
    meta = ET.parse(dir_name + [path for path in paths if "meta" in path][0])

    # Keep a dictionary to store all values.
    og_values = {"geom": {}, "motor": {}}

    # Iterate over all geoms.
    for geom in model.getroot().findall(".//geom"):

        # Get size, mass and type and convert them to the
        # appropriate datatypes.
        size = re.sub(r"\s+", " ", geom.attrib["size"]).strip()
        size = np.array(size.split(" "), dtype=float)
        mass = float(geom.attrib["mass"])
        type_ = geom.attrib["type"]

        # Compute the volume based on the type.
        if type_ == "sphere":
            vol = (4 / 3) * np.pi * size[0] ** 3
        elif type_ == "capsule":
            vol = (4 / 3) * np.pi * size[0] ** 3
            vol += np.pi * size[0] ** 2 * size[1] * 2
        elif type_ == "box":
            vol = np.prod(size) * 8

        # Calculate the density.
        density = mass / vol

        # Store all values.
        og_values["geom"][geom.attrib["name"]] = {
            "size": size, "mass": mass, "type": type_,
            "vol": vol, "density": density
        }

    # Iterate over all motors.
    for motor in meta.getroot().find("actuator").findall("motor"):

        # Get and convert the gear value.
        gear = float(motor.attrib["gear"])

        # Store all values.
        og_values["motor"][motor.attrib["name"]] = {"gear": gear}

    return og_values


def prepare_size_for_mujoco(size: list, body_part: str) -> np.array:
    """..."""

    # Convert to meters and use a numpy array to make calculations easier.
    size = np.array(size) / 100

    # Derive radius from circumference or split lengths in half since this
    # is what MuJoCo expects.
    for i, size_type in enumerate(MEASUREMENT_TYPES[body_part]):
        size[i] /= 2 * np.pi if size_type == "circ" else 2

    # For some body parts we need to subtract the radius from the length
    # since MuJoCo expects the half-length only of the cylinder part.
    if body_part in ["upper_arm", "lower_arm", "upper_leg"]:
        size[1] -= size[0]
    elif body_part == "lower_leg":
        size[2] -= size[0] / 2 + size[1] / 2

    # For the torso we need to duplicate the size by five
    # since the whole torso is made up of five capsules.
    # Each capsule will be tweaked a little by the ratio.
    if body_part == "torso":
        size = np.repeat(size, 5)

    return size
