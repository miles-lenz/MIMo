"""..."""

from mimoGrowth.constants import MAPPING_MOTOR
from mujoco import MjModel
import numpy as np


def calc_motor_gear(geoms: dict, model_og: MjModel) -> dict:
    """..."""

    # Iterate over geoms and their corresponding motors.
    gears = {}
    for geom, motors in MAPPING_MOTOR.items():

        # Get the size of the current and the original model.
        size = geoms[geom]["size"]
        size_og = model_og.geom_size[model_og.geom(geom).id]

        # Compute the CSA with the current and the original size.
        # The formular will change depending on the type of geom.
        if geom in ["geom:right_hand1", "geom:right_foot2"]:  # box
            csa = size[0] * size[1]
            csa_og = size_og[0] * size_og[1]
        else:  # sphere/capsule
            csa = np.pi * size[0] ** 2
            csa_og = np.pi * size_og[0] ** 2

        # Calculate the gear value for each motor.
        for motor in motors:

            # Get the original gear value.
            gear_og = model_og.actuator_gear[model_og.actuator(motor).id]

            # Calculate the ratio based on the original gear and CSA values.
            ratio = gear_og / csa_og

            # Calculate the gear value for the current age based on the CSA value and the ratio.
            gear = csa * ratio

            # Store the gear value. If the motor is specifically for a 'right' body part,
            # store it also for its 'left' counterpart.
            gears[motor] = {"gear": gear}
            if "right" in motor:
                gears[motor.replace("right", "left")] = {"gear": gear}

    return gears
