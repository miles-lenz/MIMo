"""..."""

from mimoGrowth.growth import Growth
import mimoEnv.utils as mimo_utils
import time
import argparse
import os
import copy
import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET


def adjust_height(model, data):
    """
    This function adjust the height of MIMo so that he will
    stand correctly on the ground.
    """

    # Calculate height based on leg and foot position/size.
    height = sum([
        -model.body("left_upper_leg").pos[2],
        -model.body("left_lower_leg").pos[2],
        -model.body("left_foot").pos[2],
        model.geom("geom:left_foot2").size[2]
    ])

    # Update the height.
    model.body("hip").pos = [0, 0, height]
    mujoco.mj_forward(model, data)


def growth():
    """..."""

    # Use a state to pause and reset the growth of MIMo.
    state = {"paused": True, "reset": False}

    # Declare which keys can pause/reset.
    def key_callback(keycode):
        if keycode == 32:  # space
            state["paused"] = not state["paused"]
        elif keycode == 341:  # strg
            state["reset"] = True

    # Load the model.
    model = mujoco.MjModel.from_xml_path("mimoEnv/assets/growth.xml")
    data = mujoco.MjData(model)

    # Add growth to the model and set the starting age to one month.
    age_months = 1
    model_with_growth = Growth(model, data)
    model_with_growth.adjust_mimo_to_age(age_months)
    adjust_height(model, data)

    # Specify which stages of age the reference cube should display.
    # Use only integers in the range from 1 to 21. Otherwise, this will result in an error.
    AGES_ON_CUBE = [1, 3, 6, 9, 12, 15, 18, 21]

    # Store the materials for the age cube.
    mat_age_cube = {}
    for age in AGES_ON_CUBE:
        mat_age_cube[age] = model.material(f"age_{age}").id

    # Launch the MuJoCo viewer.
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():

            # Perform a mujoco step and sycn the viewer.
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Let the simulation sleep. This controls how fast MIMo grows.
            time.sleep(0.04)

            # Reset the model if wanted.
            if state["reset"]:
                age_months = 1
                model_with_growth.adjust_mimo_to_age(age_months)
                adjust_height(model, data)
                state["reset"], state["paused"] = False, True
                model.geom("ref_age").matid = mat_age_cube[1]
                continue

            # Stop growing after MIMo has reached the maximum age.
            if state["paused"] or age_months >= 21.5:
                continue

            # Let the model grow.
            age_months = np.round(age_months + 0.05, 2)
            model_with_growth.adjust_mimo_to_age(age_months)
            adjust_height(model, data)

            # Adjust the material of the age cube if the next age is reached.
            if age_months in mat_age_cube.keys():
                model.geom("ref_age").matid = mat_age_cube[age_months]


def multiple_mimos():
    """
    Note that this function may affect the original behavior of MIMo.
    Therefore, this method is intended only for asthetic purposes and
    NOT for any RL experiments.
    """

    # Declare the ages for the different version of MIMo
    # and the order in which they should appear.
    AGES = [1, 8, 21.5, 15, 3]

    # Decleare some necessary paths.
    PATH_SCENE_OG = "mimoEnv/assets/growth.xml"
    PATH_SCENE_TEMP = "mimoEnv/assets/multiple_mimos.xml"

    # Keep track of temporary files so they can be
    # deleted afterwards.
    temp_files = [PATH_SCENE_TEMP]

    # Load the scene and select certain tags.
    scene = ET.parse(PATH_SCENE_OG).getroot()
    sc_worldbody = scene.find("worldbody")
    sc_body = sc_worldbody.find("body[@name='mimo_location']")
    sc_include_meta = scene.find("include")
    sc_light = sc_worldbody.findall("light")[1]

    # Store the original joint range values.
    joint_ranges = {}
    model_og = ET.parse("mimoEnv/assets/mimo/MIMo_model.xml").getroot()
    for elem in model_og.iter():
        if elem.tag == "joint":
            joint_ranges[elem.attrib["name"]] = elem.attrib["range"]

    # Iterate over all ages.
    for i, age in enumerate(AGES):

        # Load the model and let it grow to the current age.
        model = mujoco.MjModel.from_xml_path(PATH_SCENE_OG)
        data = mujoco.MjData(model)
        Growth(model, data).adjust_mimo_to_age(age)
        adjust_height(model, data)

        # Save the model temporary.
        path_model = f"mimoEnv/assets/mimo/MIMo_model_{i}.xml"
        mujoco.mj_saveLastXML(path_model, model)
        temp_files.append(path_model)

        # Load the model as an XML file and extract the main <body> element.
        model = ET.parse(path_model).getroot()
        mo_body = model.find("worldbody").find("body[@name='mimo_location']").find("body")

        # Readjust the joint range values.
        for elem in mo_body.iter():
            if elem.tag == "joint":
                elem.attrib["range"] = joint_ranges[elem.attrib["name"]]

        # Create a new mujoco element and add the body.
        mo_mujoco = ET.Element("mujoco")
        mo_mujoco.set("model", "MIMo")
        mo_mujoco.append(mo_body)

        # Change any name to avoid duplicates.
        for elem in mo_mujoco.iter():
            if "name" in elem.attrib.keys():
                elem.attrib["name"] = f"{elem.attrib['name']}_{i}"

        # Save the mujoco element as an XML file.
        ET.ElementTree(mo_mujoco).write(path_model, encoding="utf-8", xml_declaration=True)

        # Load the meta file.
        meta_file = ET.parse("mimoEnv/assets/mimo/MIMo_meta.xml").getroot()

        # Change any name in the meta file to avoid duplicates.
        for elem in meta_file.iter():
            for key, val in elem.attrib.items():
                if key in ["name", "joint1", "joint", "site", "geom1", "geom2", "body1", "body2"] and elem.tag not in ["material", "texture"]:
                    elem.attrib[key] = f"{val}_{i}"

        # Remove some duplicatess.
        if i > 0:
            meta_file.remove(meta_file.find("default"))
            meta_file.remove(meta_file.find("asset"))

        # Save a temporary meta file.
        meta_temp_path = f"mimoEnv/assets/mimo/MIMo_meta_{i}.xml"
        ET.ElementTree(meta_file).write(meta_temp_path, encoding="utf-8", xml_declaration=True)
        temp_files.append(meta_temp_path)

        # Copy the scene body and change name and position.
        new_sc_body = copy.deepcopy(sc_body)
        new_sc_body.set("name", f"{i}")
        new_sc_body.set("pos", f"0 {i * 0.4} 0")  # next to each other
        # new_sc_body.set("pos", f"{i * 0.2} 0 0")  # in front of each other

        # Change the name of the joint and the path of the include.
        new_sc_body.find("freejoint").set("name", f"mimo_location_{i}")
        new_sc_body.find("include").set("file", f"mimo/MIMo_model_{i}.xml")

        # Add a light.
        new_sc_light = copy.deepcopy(sc_light)
        new_sc_light.set("target", f"upper_body_{i}")
        new_sc_light.set("diffuse", "0.17 0.17 0.17")

        # Add body and light to the overall scene.
        sc_worldbody.append(new_sc_body)
        sc_worldbody.append(new_sc_light)

        # Add an include for the meta file.
        new_sc_include_meta = copy.deepcopy(sc_include_meta)
        new_sc_include_meta.set("file", f"mimo/MIMo_meta_{i}.xml")
        scene.append(new_sc_include_meta)

    # Remove the <include> element, the <body> for MIMo and the light.
    scene.remove(sc_include_meta)
    sc_worldbody.remove(sc_body)
    sc_worldbody.remove(sc_light)

    # Save the temporary scene.
    ET.ElementTree(scene).write(PATH_SCENE_TEMP, encoding="utf-8", xml_declaration=True)

    # Load the model from the temporary scene.
    model = mujoco.MjModel.from_xml_path(PATH_SCENE_TEMP)
    data = mujoco.MjData(model)

    # Remove any temporary files.
    for file_ in temp_files:
        if os.path.exists(file_):
            os.remove(file_)

    # Hide the growth references.
    model.body("growth_references").pos = [0, 0, -2]

    # Launch the MuJoCo viewer.
    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            pass


def position(age: str, pos: str = None, passive: str = "False"):
    """..."""

    # Convert arguments to correct types.
    age, passive = float(age), eval(passive)

    # Load the model.
    model = mujoco.MjModel.from_xml_path("mimoEnv/assets/roll_over.xml")
    data = mujoco.MjData(model)

    # Try to hide the growth references.
    try:
        model.body("growth_references").pos = [0, 0, -5]
    except KeyError:
        pass

    # Let MIMo grow and adjust standing height if no position was given.
    Growth(model, data).adjust_mimo_to_age(age)
    if not pos:
        adjust_height(model, data)

    # Change MIMo to the specified position.
    if pos == "prone":
        model.body("hip").quat = [0, -0.7071068, 0, 0.7071068]
        model.body("hip").pos = [0, 0, 0.1]
        for _ in range(100):
            mujoco.mj_step(model, data)
    elif pos == "supine":
        model.body("hip").quat = [0, 0.7071068, 0, 0.7071068]
        model.body("hip").pos = [0, 0, 0.1]
        for _ in range(100):
            mujoco.mj_step(model, data)
    elif pos == "roll_over":
        model.body("hip").quat = [0, 0.7071068, 0, 0.7071068]
        model.body("hip").pos = [0, 0, 0.1]
        for _ in range(100):
            mujoco.mj_step(model, data)
        mimo_utils.set_joint_qpos(model, data, "robot:right_hip1", [-2.3])
        mimo_utils.set_joint_qpos(model, data, "robot:right_knee", [-2.3])
        mimo_utils.set_joint_qpos(model, data, "robot:right_shoulder_horizontal", [1.4])
        mimo_utils.set_joint_qpos(model, data, "robot:right_shoulder_ad_ab", [0.3])
        mimo_utils.set_joint_qpos(model, data, "robot:right_shoulder_rotation", [0.4])
        mimo_utils.set_joint_qpos(model, data, "robot:right_elbow", [-1.4])
        mimo_utils.set_joint_qpos(model, data, "robot:head_swivel", [0.8])

    # === USE THE SPACE BELOW FOR DEBUGGING ===
    # ...
    # =========================================

    # Load an active or passive launcher.
    launch = mujoco.viewer.launch_passive if passive else mujoco.viewer.launch

    # Launch the MuJoCo viewer.
    with launch(model, data) as viewer:
        while viewer.is_running():
            pass


if __name__ == "__main__":

    # Create a mapping from keywords to functions.
    func_map = {
        "growth": growth,
        "multiple_mimos": multiple_mimos,
        "position": position,
    }

    # Create a parser that allows to select the function to execute with additional arguments.
    parser = argparse.ArgumentParser(description="Run functions from the terminal.")
    parser.add_argument("function", choices=func_map.keys(), help="The function to call.")
    parser.add_argument("kwargs", nargs=argparse.REMAINDER, help="Additional keyword arguments.")

    # Store the passed arguments so they can be passed to a function.
    kwargs = {}
    for param in parser.parse_args().kwargs:
        key, value = param.split("=")
        kwargs[key] = value

    # Call the specified function.
    func_map[parser.parse_args().function](**kwargs)
