"""
This module manages all calculations related to MuJoCo motors.

The main function, `calc_motor_params`, returns all relevant parameters.

Other functions can be used to retrieve specific parameters as needed.
"""

from mimoGrowth.constants import MAPPING_MOTOR
from mimoGrowth.utils import calc_volume


def calc_motor_gear(params_geoms: dict, base_values: dict) -> dict:
    """
    This function will calculate the gear of every motor based on
    the given geom parameters and base values from the original model.

    The strength is currently computed based on the volume of the nearest limb.
    The exact mapping can be found in the constants.py file.

    If you want to use another approach to compute strength (like CSA), there
    are some comments on how do this below.

    Arguments:
        params_geoms (dict): All relevant geom parameters.
        base_values (dict): Relevant values from the original MIMo.

    Returns:
        dict: The gear value of every motor.
    """

    gears = {}
    for geom, motors in MAPPING_MOTOR.items():

        type_ = base_values["geom"][geom]["type"]
        size = params_geoms[geom]["size"]

        vol = calc_volume(size, type_)
        base_vol = base_values["geom"][geom]["vol"]

        # The below code is a demonstration on how to use another
        # base to compute strength. This example uses CSA.

        # Compute csa based on size and type of the geom.
        # csa = ...

        # Get the base CSA value. Keep in mind to update the
        # function that store these base values.
        # base_csa = ...

        # Finally, change 'vol' and 'base_vol' variables below.

        for motor in motors:

            # Depending on the model, some motors might
            # be intentionally omitted.
            try:
                base_gear = base_values["motor"][motor]["gear"]
            except KeyError:
                continue

            ratio = base_gear / base_vol  # base_csa
            gear = ratio * vol  # csa

            gears[motor] = {"gear": gear}
            if "right" in motor:
                gears[motor.replace("right", "left")] = {"gear": gear}

    return gears


def calc_motor_params(params_geoms: dict, base_values: dict) -> dict:
    """
    This function calculates all relevant motor parameters based on the
    geom parameters for the given age and base values of the original MIMo.

    Arguments:
        params_geoms (dict): All relevant geom parameters.
        base_values (dict): Relevant values from the original MIMo.

    Returns:
        dict: All relevant motor parameters. Can be accessed via motor name.
    """

    motor_gears = calc_motor_gear(params_geoms, base_values)

    return motor_gears
