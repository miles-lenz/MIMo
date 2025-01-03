"""..."""

from mimoBody.body import adjust_body

import time
import os
import copy

import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET


def growing():

    PHYSICS = False

    state = {"paused": True, "reset": False}

    def key_callback(keycode):
        if keycode == 32:  # space
            state["paused"] = not state["paused"]
        elif keycode == 341:  # strg
            state["reset"] = True

    model = mujoco.MjModel.from_xml_path("mimoEnv/assets/physical_growth.xml")
    data = mujoco.MjData(model)

    age_months = 1
    adjust_body(age_months, model)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():

            step_start = time.time()

            if PHYSICS:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()

            if PHYSICS:
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            else:
                time.sleep(0.025)

            if state["reset"]:
                age_months = 1
                adjust_body(age_months, model)
                state["reset"], state["paused"] = False, True
                continue

            if state["paused"] or age_months >= 21.5:
                continue

            age_months = np.round(age_months + 0.1, 1)
            adjust_body(age_months, model)


def multiple_mimos():

    AGES = [1, 8, 21.5, 15, 3]

    PATH_SCENE = "mimoEnv/assets/physical_growth.xml"
    PATH_NEW_SCENE = "mimoEnv/assets/multiple_mimos.xml"
    PATH_META = "mimoEnv/assets/mimo/MIMo_meta.xml"
    PATH_NEW_META = "mimoEnv/assets/mimo/MIMo_meta_modified.xml"

    new_files = [PATH_NEW_SCENE]

    for i, age in enumerate(AGES):

        model = mujoco.MjModel.from_xml_path(PATH_SCENE)
        adjust_body(age, model)

        path_model = f"mimoEnv/assets/mimo/MIMo_model_{i}.xml"
        mujoco.mj_saveLastXML(path_model, model)
        new_files.append(path_model)

        model = ET.parse(path_model).getroot()
        body = model.find("worldbody").find("body[@name='mimo_location']").find("body")

        new_root = ET.Element("mujoco")
        new_root.set("model", "MIMo")
        new_root.append(body)

        for elem in new_root.iter():
            if "name" in elem.attrib:
                elem.attrib["name"] = f"{elem.attrib['name']}_{i}"

        new_tree = ET.ElementTree(new_root)
        new_tree.write(path_model, encoding="utf-8", xml_declaration=True)

    scene = ET.parse(PATH_SCENE).getroot()
    scene.find("include").set("file", "mimo/MIMo_meta_modified.xml")

    worldbody = scene.find("worldbody")
    body = worldbody.find("body[@name='mimo_location']")

    light_element = worldbody.findall("light")[1]

    for i, age in enumerate(AGES):

        new_body = copy.deepcopy(body)
        new_body.set("name", f"{body.attrib['name']}_{i}")
        new_body.set("pos", f"0 {i * 0.4} 0")

        include = new_body.find("include")
        include.set("file", f"mimo/MIMo_model_{i}.xml")

        new_light = copy.deepcopy(light_element)
        new_light.set("target", f"upper_body_{i}")
        new_light.set("diffuse", "0.17 0.17 0.17")

        worldbody.append(new_body)
        worldbody.append(new_light)

    worldbody.remove(body)
    worldbody.remove(light_element)

    scene = ET.ElementTree(scene)
    scene.write(PATH_NEW_SCENE, encoding="utf-8", xml_declaration=True)

    tree = ET.parse(PATH_META)
    root = tree.getroot()

    for elem in list(root):
        if elem.tag not in ['asset', 'default']:
            root.remove(elem)

    tree.write(PATH_NEW_META, encoding="utf-8", xml_declaration=True)
    new_files.append(PATH_NEW_META)

    model = mujoco.MjModel.from_xml_path(PATH_NEW_SCENE)
    data = mujoco.MjData(model)

    for file_ in new_files:
        if os.path.exists(file_):
            os.remove(file_)

    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            pass


if __name__ == "__main__":
    growing()
    # multiple_mimos()
