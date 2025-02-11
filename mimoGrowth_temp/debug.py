import numpy as np
import mujoco

# todo: Get growth params from growth.py and call this function.


def debug(growth_params):
    """
    This method is used to compare my calculations to the original model.
    Use this only if the calculations were made with an age of 17.5 months.
    Otherwise, everything will be off since we would compare different ages.
    """

    model = mujoco.MjModel.from_xml_path("mimoEnv/assets/growth.xml")

    any_issue = False
    is_close_config = {"atol": 1e-5, "rtol": 1e-4}

    for geom_name, params in growth_params["geom"].items():

        og_size = model.geom_size[model.geom(geom_name).id]
        size_is_close = np.isclose(og_size, params["size"], **is_close_config)
        if not all(size_is_close):
            print(geom_name, "size", og_size, params["size"])
            any_issue = True

        og_pos = model.geom(geom_name).pos
        pos_is_close = np.isclose(og_pos, params["pos"], **is_close_config)
        if not all(pos_is_close):
            print(geom_name, "pos", og_pos, params["pos"])
            any_issue = True

    for body_name, params in growth_params["body"].items():
        og_pos = model.body(body_name).pos
        pos_is_close = np.isclose(og_pos, params["pos"], **is_close_config)
        if not all(pos_is_close):
            print(body_name, "pos", og_pos, params["pos"])
            any_issue = True

    for motor_name, params in growth_params["motor"].items():
        og_gear = model.actuator_gear[model.actuator(motor_name).id]
        gear_is_close = np.isclose(og_gear, params["gear"])
        if not all(gear_is_close):
            print(motor_name, "gear", og_gear, params["gear"])
            any_issue = True

    if not any_issue:
        print("Everything seems right!")
