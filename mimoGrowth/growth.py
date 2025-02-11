"""
This module is the entry point for adjusting the age of MIMo.

The basic workflow looks like this:
- Use the `adjust_mimo_to_age` function to create a temporary duplicate of the
provided scene where the growth parameters are updated to the given age.
- Use the returned path to load the model.
- Delete the temporary scene with the `delete_growth_scene` function.

It is assumed that every MuJoCo scene has two <include> elements.
One that links to the meta file of MIMo and another one that links
to the actual model file. Is is important the the words *meta* and
*model* are within the file names.

The following functions should not be called directly since they will
be used by other functions:
- `calc_growth_params`
- `create_new_growth_scene`

Example Code:
```
# Set the age of MIMo and the path to the MuJoCo scene.
AGE, SCENE = 2, "path/to/the/scene.xml"

# Create a duplicate of your scene that
# includes MIMo with the specified age.
growth_scene = adjust_mimo_to_age(scene, age)

# Do something with the new scene.
model = mujoco.MjModel.from_xml_path(growth_scene)
data = mujoco.MjData(model)

# Delete this temporary growth scene.
delete_growth_scene(growth_scene)
```
"""

from mimoGrowth.constants import RATIOS_MIMO_GEOMS
from mimoGrowth.elements import geom_handler, body_handler, motor_handler
import mimoGrowth.utils as utils
import os
import xml.etree.ElementTree as ET
import numpy as np


def adjust_mimo_to_age(path_scene: str, age: float,
                       use_csa: bool = True) -> str:
    """
    This function creates a temporary duplicate of the provided scene
    where the growth parameters of MIMo are adjusted to the given age.

    Arguments:
        path_scene (str): The path to the MuJoCo scene.
        age (float): The age of MIMo. Possible values are between 1 and 21.5.

    Returns:
        str: The path to the new growth scene. Use this path to load the model.

    Raises:
        FileNotFoundError: If the scene path is invalid.
        ValueError: If the age is not within the valid interval.
    """

    # Raise an error if the path is invalid.
    if not os.path.exists(path_scene):
        raise FileNotFoundError(f"The path '{path_scene}' does not exist.")

    # Raise an error if the age parameter is invalid.
    if age < 1 or age > 21.5:
        warning = f"The Age'{age}' is invalid. Must be between 1 and 21.5"
        raise ValueError(warning)

    # Calculate all growth parameters that need to be changed in order
    # to correctly simulate the growth at the given age.
    growth_params = calc_growth_params(path_scene, age, use_csa)

    # Create a new scene that contains the updated version of MIMo.
    new_path = create_new_growth_scene(path_scene, growth_params)

    return new_path


def delete_growth_scene(path_scene: str) -> None:
    """
    This function deletes the temporary growth scene and all
    associated files like the model and meta file.

    Arguments:
        path_scene (str): Path to the growth scene which will be deleted.
    """

    # Get the paths to the MIMo model and meta file.
    xml_scene = ET.parse(path_scene).getroot()
    paths = [inc.attrib["file"] for inc in xml_scene.findall(".//include")]

    # Store all paths (including the scene path) in a list.
    paths = [os.path.join(os.path.dirname(path_scene), path) for path in paths]
    paths.append(path_scene)

    # Delete scene, model and meta file.
    for path in paths:
        os.remove(path)


def calc_growth_params(path_scene: str, age: float,
                       use_csa: bool = True) -> dict:
    """
    This function calculates and returns all relevant growth parameters.
    TNamely, this includes:
    - Position, size and mass of geoms.
    - Position of bodies.
    - Gear values of motors.

    Arguments:
        path_scene (str): The path to the MuJoCo scene.
        age (float): The age of MIMo.

    Returns:
        dict: All relevant growth parameters.
    """

    # Store all relevant parameters so they can be used later.
    params = {"geom": {}, "body": {}, "motor": {}}

    # Approximate growth functions for every body part.
    growth_functions = utils.approximate_functions()

    # Store relevant values from the original MIMo model.
    # They will be used for calculations later on.
    og_vals = utils.store_original_values(path_scene)

    # Iterate over all body parts and their associated growth
    # functions. Approximate the size(s) for each body part using
    # these functions.
    sizes = {}
    for body_part, functions in growth_functions.items():

        # Iterate over all growth functions for the current
        # body part and store the estimated size.
        size = []
        for growth_func in functions:
            approx_size = growth_func(age)
            size.append(approx_size)

        # Prepare the size for MuJoco and apply a ratio in order to
        # maintain the little tweaks/changes from the original MIMo model.
        size = utils.prepare_size_for_mujoco(size, body_part)
        size *= RATIOS_MIMO_GEOMS[body_part]

        # Store the size.
        sizes[body_part] = size

    # Calculate size, position and mass for all geoms based on the
    # estimated body sizes from the measurements.
    params["geom"] = geom_handler.calc_geom_params(sizes, og_vals)

    # Calculate position for all bodies based on the
    # size/position of geoms.
    params["body"] = body_handler.calc_body_params(params["geom"])

    # Calculate the gear values for all motors based on the CSA
    # or volume of the body parts.
    params["motor"] = motor_handler.calc_motor_params(
        params["geom"], og_vals, use_csa)

    return params


def create_new_growth_scene(path_scene: str, growth_params: dict) -> None:
    """
    This function will create duplicates of the provided scene and
    the model and meta files of MIMo. Within these duplicates, MIMo
    will have been adjusted to the specified age.

    These new files use the same name with the additional suffix '_temp' and
    will be stored in the same folders as the original files..

    Arguments:
        path_scene (str): The path to the MuJoCo scene.
        growth_params (dict): The growth parameters.
    """

    # Parse the scene.
    xml_scene = ET.parse(path_scene)

    # Get the <include> elements for the model and the meta file.
    includes = xml_scene.getroot().findall(".//include")
    include_model = [i for i in includes if "model" in i.attrib["file"]][0]
    include_meta = [i for i in includes if "meta" in i.attrib["file"]][0]

    # Get the path to model and meta file.
    dir_name = os.path.dirname(path_scene)
    path_model = os.path.join(dir_name, include_model.attrib["file"])
    path_meta = os.path.join(dir_name, include_meta.attrib["file"])

    # Parse the model and meta file.
    xml_model, xml_meta = ET.parse(path_model), ET.parse(path_meta)

    # Iterate over all <geom> elements in the model.
    for geom in xml_model.getroot().findall(".//geom"):

        name = geom.attrib["name"]

        size = growth_params["geom"][name]["size"]
        geom.attrib["size"] = " ".join(np.array(size, dtype=str))

        pos = growth_params["geom"][name]["pos"]
        geom.attrib["pos"] = " ".join(np.array(pos, dtype=str))

        mass = growth_params["geom"][name]["mass"]
        geom.attrib["mass"] = str(mass)

    # Iterate over all <body> elements in the model.
    for body in xml_model.getroot().findall(".//body"):

        name = body.attrib["name"]

        pos = growth_params["body"][name]["pos"]
        body.attrib["pos"] = " ".join(np.array(pos, dtype=str))

    # Iterate over all <motor> elements within the
    # <actuator> element in the meta file.
    for motor in xml_meta.getroot().find("actuator").findall(".//motor"):

        name = motor.attrib["name"]

        gear = growth_params["motor"][name]["gear"]
        motor.attrib["gear"] = str(gear)

    # Define a helper function that creates a new temporary path.
    def temp_path(path: str) -> str:
        return path.replace(".xml", "_temp.xml")

    # Save the updated model and meta file.
    xml_model.write(temp_path(path_model))
    xml_meta.write(temp_path(path_meta))

    # Update the path of the <include> elements in the scene.
    include_model.attrib["file"] = temp_path(include_model.attrib["file"])
    include_meta.attrib["file"] = temp_path(include_meta.attrib["file"])

    # Save the updated scene.
    path_new_scene = temp_path(path_scene)
    xml_scene.write(path_new_scene)

    return path_new_scene
