"""..."""

from mimoBody.constants import AGE_MONTHS, MEASUREMENTS, RATIOS_MIMO_GEOMS, GEOM_MAPPING
import mimoBody.utils as utils
from scipy.interpolate import CubicSpline
from mujoco import MjModel

# import numpy as np


def adjust_body(body_params: dict, model: MjModel) -> None:
    """..."""

    # Iterate over all geoms.
    for geom, params in body_params["geoms"].items():

        # Get the geom id.
        geom_id = model.geom(geom).id

        # * DEBUG *
        # if not all(np.isclose(model.geom_size[geom_id], params["size"])):
        #     print(f"Geom Size does not match for {geom}")
        #     print(model.geom_size[geom_id], params["size"])
        # if not all(np.isclose(model.geom(geom).pos, params["pos"])):
        #     print(f"Geom Position does not match for {geom}")
        #     print(model.geom(geom).pos, params["pos"])

        # Adjust size and position of the geom.
        model.geom_size[geom_id] = params["size"]
        model.geom(geom).pos = params["pos"]

    # Iterate over all bodies and adjust their position.
    for body, pos in body_params["bodies"].items():
        model.body(body).pos = pos

        # * DEBUG *
        # if not all(np.isclose(model.body(body).pos, pos)):
        #     print(f"Body Position does not match for {body}")
        #     print(model.body(body).pos, pos)


def get_body_params(age: float) -> dict:
    """..."""

    # Raise a value error if the age parameter is
    # not within the valid interval.
    if age < 1 or age > 21.5:
        raise ValueError(f"Invalid value for age: {age}. Must be between 1 and 21.5")

    # Start by calculating size and position for every geom
    # by iterating over all measurements.
    params = {"geoms": {}}
    for body_part, measurements in MEASUREMENTS.items():

        # Compute the size(s) for the current body part and age by
        # approximating a cubic spline function using the concrete
        # measurements for different ages from the website.
        size = []
        for meas in measurements:
            func = CubicSpline(AGE_MONTHS, meas)
            size.append(func(age))

        # Prepare the size for MuJoco usage and apply a ratio in order to
        # maintain the little tweaks/changes from the original MIMo model.
        size = utils.prepare_size_for_mujoco(size, body_part)
        size *= RATIOS_MIMO_GEOMS[body_part]

        # ...
        foot_height = None
        if body_part == "lower_leg":
            foot_height = params["geoms"]["geom:left_foot2"]["size"][2]

        # Compute size and positions vectors for the geoms.
        vectors = utils.create_geom_vectors(size, body_part, foot_height)

        # Map the vectors to the correct geom names as defined
        # in the mapping.
        for i, geom in enumerate(GEOM_MAPPING[body_part]):
            keys = geom if isinstance(geom, tuple) else [geom]
            for key in keys:
                params["geoms"][key] = vectors[i]

    # Create position vectors for all bodies based on the
    # size/position of geoms.
    params["bodies"] = utils.create_body_vectors(params["geoms"])

    # Round all vectors and add add a padding so that they
    # are ready for the MuJoCo model.
    utils.pad_and_round_vectors(params)

    return params
