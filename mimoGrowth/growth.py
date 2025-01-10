"""..."""

from mimoGrowth.constants import RATIOS_MIMO_GEOMS
from mimoGrowth.mujoco import geom, body, motor
# from mimoGrowth.debug import debug
import mimoGrowth.utils as utils
from mujoco import MjModel
import mujoco


class Growth:

    def __init__(self, model: MjModel) -> None:
        self.model = model
        self.model_og = model
        self.funcs = utils.approximate_functions()

    def apply_params_to_model(self, growth_params) -> None:
        """..."""

        # *** temp ***
        # debug(growth_params, self.model_og)

        # Iterate over all geoms and adjust their size/position.
        for geom_name, params in growth_params["geom"].items():
            self.model.geom_size[self.model.geom(geom_name).id] = params["size"]
            self.model.geom(geom_name).pos = params["pos"]

        # Iterate over all bodies and adjust their position.
        for body_name, params in growth_params["body"].items():
            self.model.body(body_name).pos = params["pos"]

        # Iterate over all motors and adjust their gear.
        for motor_name, params in growth_params["motor"].items():
            self.model.actuator_gear[self.model.actuator(motor_name).id] = params["gear"]

        # Update the model state.
        mujoco.mj_forward(self.model, mujoco.MjData(self.model))

    def calc_growth_params(self, age: float) -> dict:
        """..."""

        # Store all relevant parameters so they can be
        # applied to the model later.
        params = {"geom": {}, "body": {}, "motor": {}}

        # Iterate over all body parts and their associated growth
        # functions. Approximate the size(s) for each body part using
        # these functions.
        sizes = {}
        for body_part, funcs in self.funcs.items():

            # Iterate over all growth functions for the current
            # body part and store the estimated size.
            size = []
            for growth_func in funcs:
                approx_size = growth_func(age)
                size.append(approx_size)

            # Prepare the size for MuJoco and apply a ratio in order to
            # maintain the little tweaks/changes from the original MIMo model.
            size = utils.prepare_size_for_mujoco(size, body_part)
            size *= RATIOS_MIMO_GEOMS[body_part]

            # Store the size.
            sizes[body_part] = size

        # Calculate size and position for all geoms based on the
        # estimated body sizes from the measurements.
        params["geom"] = geom.calc_geom_params(sizes)

        # Calculate position vectors for all bodies based on the
        # size/position of geoms.
        params["body"] = body.calc_body_positions(params["geom"])

        # Calculate the correct gear values for all motors based on the CSA
        # of the body parts.
        params["motor"] = motor.calc_motor_gear(params["geom"], self.model_og)

        return params

    def adjust_mimo_to_age(self, age: float) -> None:
        """..."""

        # Raise an error if the age parameter is invalid.
        if age < 1 or age > 21.5:
            raise ValueError(f"Invalid age: {age}. Must be between 1 and 21.5")

        # Calculate all parameters that need to be changed in order
        # to correctly simulate the growth at the given age.
        growth_params = self.calc_growth_params(age)

        # Apply the calculated paramters to the actual model of MIMo.
        self.apply_params_to_model(growth_params)
