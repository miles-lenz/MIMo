"""..."""

from mimoGrowth.constants import MEASUREMENT_TYPES, MEASUREMENTS, AGE_GROUPS
from collections import defaultdict
from scipy.interpolate import CubicSpline
import numpy as np


def approximate_functions() -> dict:
    """..."""

    funcs = defaultdict(list)

    # Iterate over all body parts and their associated measurements.
    for body_part, measurements in MEASUREMENTS.items():

        # Iterate over all measuemrents for the current body part.
        # Note that some body parts have multiple measurements e.g. the
        # upper arm has a circumference and a length.
        for meas in measurements:

            # Approximate and store a cubic spline function based on the
            # current measurements and the corresponding age groups.
            func = CubicSpline(AGE_GROUPS, meas)
            funcs[body_part].append(func)

    return funcs


def prepare_size_for_mujoco(size: list, body_part: str) -> np.array:
    """..."""

    # Convert to meters and use a numpy array to make calculations easier.
    size = np.array(size) / 100

    # Derive radius from circumference or split lengths in half since this
    # is what MuJoCo expects.
    for i, size_type in enumerate(MEASUREMENT_TYPES[body_part]):
        size[i] /= 2 * np.pi if size_type == "circ" else 2

    # For some body parts we need to subtract the radius from the length
    # since MuJoCo expects the half-length only of the cylinder part.
    if body_part in ["upper_arm", "lower_arm", "upper_leg"]:
        size[1] -= size[0]
    elif body_part == "lower_leg":
        size[2] -= size[0] / 2 + size[1] / 2

    # For the torso we need to duplicate the size by five
    # since the whole torso is made up of five capsules.
    # Each capsule will be tweaked a little by the ratio.
    if body_part == "torso":
        size = np.repeat(size, 5)

    return size
