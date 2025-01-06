import numpy as np


def debug(growth_params, model_og):
    """
    This method is used to compare my calculations to the original model.
    Use this only if the calculations were made with an age of 17.5 months.
    Otherwise, everything will be off since we would compare different ages.
    """

    any_issue = False

    for geom_name, params in growth_params["geom"].items():

        og_size = model_og.geom_size[model_og.geom(geom_name).id]
        size_is_close = np.isclose(og_size, params["size"], atol=1e-5, rtol=1e-4)
        if not all(size_is_close):
            print(geom_name, "size", og_size, params["size"])
            any_issue = True

        og_pos = model_og.geom(geom_name).pos
        pos_is_close = np.isclose(og_pos, params["pos"], atol=1e-5, rtol=1e-4)
        if not all(pos_is_close):
            print(geom_name, "pos", og_pos, params["pos"])
            any_issue = True

    for body_name, params in growth_params["body"].items():
        og_pos = model_og.body(body_name).pos
        pos_is_close = np.isclose(og_pos, params["pos"], atol=1e-5, rtol=1e-4)
        if not all(pos_is_close):
            print(body_name, "pos", og_pos, params["pos"])
            any_issue = True

    print("motor amount", len(growth_params["motor"].keys()))
    for motor_name, params in growth_params["motor"].items():
        og_gear = model_og.actuator_gear[model_og.actuator(motor_name).id]
        gear_is_close = np.isclose(og_gear, params["gear"])
        if not all(gear_is_close):
            print(motor_name, "gear", og_gear, params["gear"])
            any_issue = True

    if not any_issue:
        print("Everything seems right!")
