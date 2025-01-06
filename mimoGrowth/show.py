"""..."""

from mimoGrowth.growth import Growth
import time
import os
import copy
import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET


def growing(physics=False, active=False):
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

    # Launch the active MuJoCo viewer if specified.
    if active:
        with mujoco.viewer.launch(model, data) as viewer:
            while viewer.is_running():
                pass
        return

    # Launch the passive MuJoCo viewer.
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():

            step_start = time.time()

            # Perform a step in the MuJoCo viewer.
            if physics:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()

            # Let the simulation sleep either by the optimal or manual timestep.
            if physics:
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            else:
                time.sleep(0.025)

            # Reset the model if wanted.
            if state["reset"]:
                age_months = 1
                model_with_growth.adjust_mimo_to_age(age_months)
                state["reset"], state["paused"] = False, True
                continue

            # Stop growing after MIMo has reached the maximum age.
            if state["paused"] or age_months >= 21.5:
                continue

            # Let the model grow.
            age_months = np.round(age_months + 0.1, 1)
            model_with_growth.adjust_mimo_to_age(age_months)


def multiple_mimos():
    """..."""

    # Set the ages you want to see in the order you
    # want to see them.
    AGES = [1, 8, 21.5, 15, 3]

    # Declare some necessary paths.
    PATH_SCENE = "mimoEnv/assets/growth.xml"
    PATH_NEW_SCENE = "mimoEnv/assets/multiple_mimos.xml"
    PATH_META = "mimoEnv/assets/mimo/MIMo_meta.xml"
    PATH_NEW_META = "mimoEnv/assets/mimo/MIMo_meta_modified.xml"

    # Keep track of newly created files.
    new_files = [PATH_NEW_SCENE]

    # Iterate over all ages.
    for i, age in enumerate(AGES):

        # Load the model and let it grow to the current age.
        model = mujoco.MjModel.from_xml_path(PATH_SCENE)
        Growth(model).adjust_mimo_to_age(age)

        # Save the model.
        path_model = f"mimoEnv/assets/mimo/MIMo_model_{i}.xml"
        mujoco.mj_saveLastXML(path_model, model)
        new_files.append(path_model)

        # Load the model as an XML file.
        model = ET.parse(path_model).getroot()
        body = model.find("worldbody").find("body[@name='mimo_location']").find("body")

        # Create a new mujoco root and add the body.
        new_root = ET.Element("mujoco")
        new_root.set("model", "MIMo")
        new_root.append(body)

        # Change the attribute names to avoid duplicates.
        for elem in new_root.iter():
            if "name" in elem.attrib:
                elem.attrib["name"] = f"{elem.attrib['name']}_{i}"

        # Save the modified XML file.
        new_tree = ET.ElementTree(new_root)
        new_tree.write(path_model, encoding="utf-8", xml_declaration=True)

    # Load the scene and change the path to the meta file.
    scene = ET.parse(PATH_SCENE).getroot()
    scene.find("include").set("file", "mimo/MIMo_meta_modified.xml")

    # Get the MIMo body element.
    worldbody = scene.find("worldbody")
    body = worldbody.find("body[@name='mimo_location']")

    # Get the light element.
    light_element = worldbody.findall("light")[1]

    # Iterate over all different aged versions of MIMo.
    for i, age in enumerate(AGES):

        # Copy the body and change name and position.
        new_body = copy.deepcopy(body)
        new_body.set("name", f"{body.attrib['name']}_{i}")
        new_body.set("pos", f"0 {i * 0.4} 0")

        # Change the path.
        include = new_body.find("include")
        include.set("file", f"mimo/MIMo_model_{i}.xml")

        # Add a light.
        new_light = copy.deepcopy(light_element)
        new_light.set("target", f"upper_body_{i}")
        new_light.set("diffuse", "0.17 0.17 0.17")

        # Add body and light to the overall scene.
        worldbody.append(new_body)
        worldbody.append(new_light)

    # Remove original body and light.
    worldbody.remove(body)
    worldbody.remove(light_element)

    # Save the scene.
    scene = ET.ElementTree(scene)
    scene.write(PATH_NEW_SCENE, encoding="utf-8", xml_declaration=True)

    # Remove not needed meta elements.
    tree = ET.parse(PATH_META)
    root = tree.getroot()
    for elem in list(root):
        if elem.tag not in ['asset', 'default']:
            root.remove(elem)

    # Save the new meta file.
    tree.write(PATH_NEW_META, encoding="utf-8", xml_declaration=True)
    new_files.append(PATH_NEW_META)

    # Load the model.
    model = mujoco.MjModel.from_xml_path(PATH_NEW_SCENE)
    data = mujoco.MjData(model)

    # Remove any temporary files.
    for file_ in new_files:
        if os.path.exists(file_):
            os.remove(file_)

    # Launch the MuJoCo viewer.
    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            pass


if __name__ == "__main__":
    growing(physics=False, active=False)
    # multiple_mimos()
