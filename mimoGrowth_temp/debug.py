from mimoGrowth.growth import calc_growth_params
import numpy as np
import mujoco


def debug():
    """
    This function can be used to check if the growth calculations are correct.
    The main idea is that the calculations should exactly match the values from
    the original model if the initial MIMo age is used.

    Therefore, this function will load the original model and calculate growth
    parameters at the same age and will then compare all values.

    If something is off, it will be printed in the console.
    """

    # Calculate the growth parameters at the original age.
    scene = "mimoEnv/assets/growth.xml"
    growth_params = calc_growth_params(scene, 17.5)

    # Load the original model.
    model = mujoco.MjModel.from_xml_path("mimoEnv/assets/growth.xml")

    # Set some initial variables.
    any_issue = False
    is_close_config = {"atol": 1e-5, "rtol": 1e-4}

    # Iterate over geoms.
    for geom_name, params in growth_params["geom"].items():

        # Check if the sizes match.
        og_size = model.geom_size[model.geom(geom_name).id]
        size_is_close = np.isclose(og_size, params["size"], **is_close_config)
        if not all(size_is_close):
            print(geom_name, "size", og_size, params["size"])
            any_issue = True

        # Check if the positions match.
        og_pos = model.geom(geom_name).pos
        pos_is_close = np.isclose(og_pos, params["pos"], **is_close_config)
        if not all(pos_is_close):
            print(geom_name, "pos", og_pos, params["pos"])
            any_issue = True

        # Check if the mass values match.
        og_mass = model.geom(geom_name).mass
        mass_is_close = np.isclose(og_mass, params["mass"], **is_close_config)
        if not all(mass_is_close):
            print(geom_name, "mass", og_mass, params["mass"])
            any_issue = True

    # Iterate over bodies.
    for body_name, params in growth_params["body"].items():

        # Check if the positions match.
        og_pos = model.body(body_name).pos
        pos_is_close = np.isclose(og_pos, params["pos"], **is_close_config)
        if not all(pos_is_close):
            print(body_name, "pos", og_pos, params["pos"])
            any_issue = True

    # Iterate over motors.
    for motor_name, params in growth_params["motor"].items():

        # Check if the gear values match.
        og_gear = model.actuator_gear[model.actuator(motor_name).id][0]
        gear_is_close = np.isclose(og_gear, params["gear"])
        if not gear_is_close:
            print(motor_name, "gear", og_gear, params["gear"])
            any_issue = True

    # If there was not an issue, print that too.
    if not any_issue:
        print("Everything seems right!")


if __name__ == "__main__":
    debug()
