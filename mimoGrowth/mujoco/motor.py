"""..."""

from mimoGrowth.constants import MAPPING_MOTOR
from mujoco import MjModel
import numpy as np


def calc_motor_gear(geoms: dict, model_og: MjModel, use_csa: bool = True) -> dict:
    """..."""

    # Iterate over geoms and their corresponding motors.
    gears = {}
    for geom, motors in MAPPING_MOTOR.items():

        # Get the size of the current and the original model.
        size = geoms[geom]["size"]
        size_og = model_og.geom_size[model_og.geom(geom).id]

        # Get the type of the geom.
        type_ = model_og.geom_type[model_og.geom(geom).id]

        # Calculate the CSA and volume with the current and original size.
        # The formula will change depending on the type of geom.
        if type_ in [2, 3]:  # sphere/capsule
            csa, csa_og = np.pi * size[0] ** 2, np.pi * size_og[0] ** 2
            vol, vol_og = (4 / 3) * np.pi * size[0] ** 3, (4 / 3) * np.pi * size_og[0] ** 3
            if type_ == 3:  # capsule:
                vol += np.pi * size[0] ** 2 * size[1] * 2
                vol_og += np.pi * size_og[0] ** 2 * size_og[1] * 2

        elif type_ == 6:  # box
            csa, csa_og = size[0] * size[1] * 4, size_og[0] * size_og[1] * 4
            vol, vol_og = np.prod(size) * 8, np.prod(size_og) * 8

        # Calculate the gear value for each motor.
        for motor in motors:

            # Get the original gear value if the motor is present in the model.
            try:
                gear_og = model_og.actuator_gear[model_og.actuator(motor).id]
            except KeyError:
                continue

            # Calculate the ratio based on the original gear and CSA/volume values.
            # Then, calculate the gear value for the current age based on the CSA/volume value and the ratio.
            if use_csa:
                ratio = gear_og / csa_og
                gear = csa * ratio
            else:
                ratio = gear_og / vol_og
                gear = vol * ratio

            # Store the gear value. If the motor is specifically for a 'right' body part,
            # store it also for its 'left' counterpart.
            gears[motor] = {"gear": gear}
            if "right" in motor:
                gears[motor.replace("right", "left")] = {"gear": gear}

    return gears
