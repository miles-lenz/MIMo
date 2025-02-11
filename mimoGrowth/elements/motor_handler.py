"""..."""

from mimoGrowth.constants import MAPPING_MOTOR
import numpy as np


def calc_motor_gear(geoms: dict, og_vals: dict, use_csa: bool = True) -> dict:
    """..."""

    # Iterate over geoms and their corresponding motors.
    gears = {}
    for geom, motors in MAPPING_MOTOR.items():

        # Get the size of the current and the original model.
        size = geoms[geom]["size"]
        size_og = og_vals["geom"][geom]["size"]

        # Get the type of the geom.
        type_ = og_vals["geom"][geom]["type"]

        # Calculate the CSA and volume with the current and original size.
        # The formula will change depending on the type of geom.
        if type_ in ["sphere", "capsule"]:
            csa = np.pi * size[0] ** 2
            csa_og = np.pi * size_og[0] ** 2
            vol = (4 / 3) * np.pi * size[0] ** 3
            vol_og = (4 / 3) * np.pi * size_og[0] ** 3
            if type_ == "capsule":
                vol += np.pi * size[0] ** 2 * size[1] * 2
                vol_og += np.pi * size_og[0] ** 2 * size_og[1] * 2

        elif type_ == "box":
            csa, csa_og = size[0] * size[1] * 4, size_og[0] * size_og[1] * 4
            vol, vol_og = np.prod(size) * 8, np.prod(size_og) * 8

        # Calculate the gear value for each motor.
        for motor in motors:

            # Get the original gear value.
            gear_og = og_vals["motor"][motor]["gear"]

            # Calculate the ratio based on the original gear
            # and CSA/volume values. Then, calculate the gear value for the
            # current age based on the CSA/volume value and the ratio.
            if use_csa:
                ratio = gear_og / csa_og
                gear = csa * ratio
            else:
                ratio = gear_og / vol_og
                gear = vol * ratio

            # Store the gear value. If the motor is specifically for a 'right'
            # body part, store it also for its 'left' counterpart.
            gears[motor] = {"gear": gear}
            if "right" in motor:
                gears[motor.replace("right", "left")] = {"gear": gear}

    return gears


def calc_motor_params(geoms: dict, og_vals: dict) -> dict:
    """..."""

    # Calculate all motor gears based on the calculated
    # geoms and the original model.
    motor_gears = calc_motor_gear(geoms, og_vals)

    return motor_gears
