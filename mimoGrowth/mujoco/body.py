"""..."""

from mimoGrowth.constants import RATIOS_MIMO_BODIES as ratios
import numpy as np


def calc_body_positions(geoms: dict) -> dict:
    """..."""

    # Create some shortcuts to make the next part more readable.
    g_lb, g_cb = geoms["lb"], geoms["cb"]
    g_ub1, g_ub3 = geoms["ub1"], geoms["ub3"]
    g_head, g_hand = geoms["head"], geoms["geom:right_hand1"]
    g_u_arm, g_l_arm = geoms["left_uarm1"], geoms["left_larm"]
    g_u_leg, g_l_leg1 = geoms["geom:right_upper_leg1"], geoms["geom:left_lower_leg1"]
    g_l_leg2, g_foot = geoms["geom:left_lower_leg2"], geoms["geom:left_foot3"]

    # Calculate all body positions based on the geom sizes and/or positions.
    hip = [0, 0, g_u_leg["size"][1] * 2 + g_l_leg1["size"][1] * 2 + g_l_leg2["size"][1] * 2 + g_foot["size"][1] + g_ub3["size"][0]]
    lower_body = [0.002, 0, (g_lb["size"][0] + g_cb["size"][0]) * ratios["lower_body"]]
    upper_body = [-0.002, 0, (g_cb["size"][0] + g_ub1["size"][0]) * ratios["upper_body"]]
    eye = ratios["eye"] * g_head["size"][0]
    head = [0, 0, (g_ub3["pos"][2] + g_ub3["size"][0]) * ratios["head"]]
    upper_arm = [-0.005, (np.sum(g_ub3["size"]) + g_u_arm["size"][0]) * ratios["upper_arm"][0], g_ub3["pos"][2] * ratios["upper_arm"][1]]
    lower_arm = [0, 0, (g_u_arm["size"][0] + 2 * g_u_arm["size"][1] - g_l_arm["size"][0]) * ratios["lower_arm"]]
    hand = [0, 0.007, (g_l_arm["size"][0] + 2 * g_l_arm["size"][1]) * ratios["hand"]]
    fingers = [0, 0, g_hand["size"][2] * 2]
    upper_leg = [0.005, (np.sum(g_lb["size"]) - g_u_leg["size"][0]) * ratios["upper_leg"], -.007]
    lower_leg = [0, 0, -(g_u_leg["size"][0] + 2 * g_u_leg["size"][1] - g_l_leg1["size"][0]) * ratios["lower_leg"]]
    foot = [0, 0, (g_l_leg1["pos"][2] + g_l_leg2["pos"][2]) * ratios["foot"]]
    toes = [g_foot["pos"][0], 0, 0]

    # Store all calculated position with their correct body names.
    # Sometimes it is necessary to change the sign within a vector since some
    # body parts need to be left and others should be on the right.
    positions = [
        (["hip"], hip),
        (["lower_body"], lower_body),
        (["upper_body"], upper_body),
        (["head"], head),
        (["left_eye"], eye), (["right_eye"], eye * np.array([1, -1, 1])),
        (["left_upper_arm"], upper_arm), (["right_upper_arm"], upper_arm * np.array([1, -1, 1])),
        (["right_lower_arm", "left_lower_arm"], lower_arm),
        (["left_hand"], hand), (["right_hand"], hand * np.array([1, -1, 1])),
        (["right_fingers", "left_fingers"], fingers),
        (["right_upper_leg"], upper_leg * np.array([1, -1, 1])), (["left_upper_leg"], upper_leg),
        (["right_lower_leg", "left_lower_leg"], lower_leg),
        (["right_foot", "left_foot"], foot),
        (["right_toes", "left_toes"], toes),
    ]

    # Create a dictionary to be consistent with how the data is stored.
    positions_dict = {}
    for body_names, pos in positions:
        for name in body_names:
            positions_dict[name] = {"pos": pos}

    return positions_dict
