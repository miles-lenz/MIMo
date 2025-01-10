"""..."""

from mimoGrowth.growth import Growth
import time
import os
import copy
import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET


def adjust_height(model):
    """
    This function adjust the height of MIMo so that he will
    stand correctly on the ground.
    """

    height = sum([
        -model.body("left_upper_leg").pos[2],
        -model.body("left_lower_leg").pos[2],
        -model.body("left_foot").pos[2],
        model.geom("geom:left_foot2").size[2]
    ])

    model.body("hip").pos = [0, 0, height]
    mujoco.mj_forward(model, mujoco.MjData(model))


def growing():
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
    model_with_growth = Growth(model)
    model_with_growth.adjust_mimo_to_age(age_months)
    adjust_height(model)

    # Launch the MuJoCo viewer.
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():

            # Perform a mujoco step and sycn the viewer.
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Let the simulation sleep. This controls how fast MIMo grows.
            time.sleep(0.03)

            # Reset the model if wanted.
            if state["reset"]:
                age_months = 1
                model_with_growth.adjust_mimo_to_age(age_months)
                adjust_height(model)
                state["reset"], state["paused"] = False, True
                continue

            # Stop growing after MIMo has reached the maximum age.
            if state["paused"] or age_months >= 21.5:
                continue

            # Let the model grow.
            age_months = np.round(age_months + 0.1, 1)
            model_with_growth.adjust_mimo_to_age(age_months)
            adjust_height(model)


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
        Growth(model).adjust_mimo_to_age(age)
        adjust_height(model)

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

        # Change the path.
        new_sc_include_model = new_sc_body.find("include")
        new_sc_include_model.set("file", f"mimo/MIMo_model_{i}.xml")

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

    # Launch the MuJoCo viewer.
    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            pass


if __name__ == "__main__":
    # test_standup()
    growing()
    # multiple_mimos()
