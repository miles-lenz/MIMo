from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, \
    DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS
from mimoActuation.actuation import SpringDamperModel
import mujoco
import numpy as np
import os

TASK = ["BELLY_TO_BACK", "BACK_TO_BELLY"][0]

SCENE_PATH = os.path.join(SCENE_DIRECTORY, "roll_over.xml")


class MIMoRollOverEnv(MIMoEnv):

    def __init__(self,
                 model_path=SCENE_PATH,
                 initial_qpos=None,
                 frame_skip=2,
                 age=None,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 actuation_model=SpringDamperModel,
                 **kwargs):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         age=age,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=False,
                         done_active=False,
                         **kwargs)

        # Bring MIMo into start position.
        self.model.body("hip").pos = [0, 0, 0.2]
        if TASK == "BELLY_TO_BACK":
            self.model.body("hip").quat = [0, -0.7071068, 0, 0.7071068]
        else:
            self.model.body("hip").quat = [0, 0.7071068, 0, 0.7071068]
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self.init_position = self.data.qpos.copy()

    def is_success(self, achieved_goal, desired_goal):
        """..."""

        return achieved_goal >= desired_goal

    def is_failure(self, achieved_goal, desired_goal):
        """..."""

        return False

    def is_truncated(self):
        """..."""

        return False

    def reset_model(self):
        """..."""

        self.set_state(self.init_qpos, self.init_qvel)

        qpos = self.init_position.copy()
        qpos[7:] = qpos[7:] + self.np_random.uniform(
            low=-0.01, high=0.01, size=len(qpos[7:]))

        qvel = np.zeros(self.data.qvel.shape)

        self.set_state(qpos, qvel)

        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        mujoco.mj_step(self.model, self.data, nstep=100)

        return self._get_obs()

    def sample_goal(self):
        """..."""

        return 0.9

    def get_achieved_goal(self):
        """..."""

        # Calculate the normalized hip rotation around the y-axis.
        # This value will be 0 if MIMo lies on the back and it will be
        # 1 when MIMo lies on the belly.
        xmat = self.data.body("hip").xmat.reshape(3, 3)
        angle = np.arctan2(
            -xmat[2, 0], np.sqrt(xmat[2, 1] ** 2 + xmat[2, 2] ** 2))
        angle *= (180 / np.pi)
        angle_norm = (angle - (-90)) / (90 - (-90))

        # Invert the normalized angle depending on the task.
        if TASK == "BELLY_TO_BACK":
            angle_norm = 1 - angle_norm

        return angle_norm

    def compute_reward(self, achieved_goal, desired_goal, info):

        # Use the hip rotation as the main reward.
        reward = achieved_goal  # [0, 1]

        # Penalize excessive use of force.
        quad_ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()  # [0, 0.44]
        reward -= quad_ctrl_cost

        return reward
