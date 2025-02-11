"""..."""

from mimoGrowth.constants import RATIOS_DERIVED as ratios
from mimoGrowth.constants import MAPPING_GEOM
import numpy as np


# Use a small constant to subtract from some geom vectors
# so that the individual parts won't have a visual overlap.
# This value is from the original MIMo model.
EPSILON = 0.0001


def calc_geom_sizes(sizes: dict, extras: dict):
    """"..."""

    # Unpack the needed extra values.
    h_hand, len_hand, len_fingers = extras["hand"]
    h_foot, len_foot, len_toes = extras["foot"]
    len_lower_leg1, len_lower_leg2 = extras["lower_leg"]

    # Calculate all geom sizes based on the measurement sizes and extras.
    geom_sizes = {
        "head": [
            sizes["head"],  # head
            sizes["head"] * ratios["eye"]  # eye
        ],
        "upper_arm": [sizes["upper_arm"]],
        "lower_arm": [sizes["lower_arm"]],
        "hand": [
            # hand1
            [sizes["hand"][2], h_hand, len_hand],
            # hand2
            [h_hand + EPSILON * 2, sizes["hand"][2] - EPSILON * 3, 0],
            # fingers1
            [sizes["hand"][1], h_hand, len_fingers],
            # fingers2
            [h_hand + EPSILON * 2, sizes["hand"][1] + EPSILON * 2, 0],
        ],
        "torso": [],
        "upper_leg": [sizes["upper_leg"]],
        "lower_leg": [
            [sizes["lower_leg"][0], len_lower_leg1],  # lower_leg1
            [sizes["lower_leg"][1], len_lower_leg2]  # lower_leg2
        ],
        "foot": [
            [sizes["foot"][1] - EPSILON, h_foot - EPSILON],  # foot1
            [len_foot, sizes["foot"][1], h_foot],  # foot2
            [h_foot - EPSILON, sizes["foot"][1] - EPSILON * 2],  # foot3
            [len_toes, sizes["foot"][1] - EPSILON, h_foot - EPSILON],  # toes1
            [h_foot, sizes["foot"][1]]  # toes2
        ]
    }

    # Add sizes for the torso parts.
    for i in range(5):
        size, ratio = sizes["torso"][i], ratios["torso"][i]
        vec = [size * ratio, size * (1 - ratio)]
        geom_sizes["torso"].append(vec)

    # Add a padding to the vectors so that they
    # are prepared for MuJoCo.
    for name, vectors in geom_sizes.items():
        for i, vec in enumerate(vectors):
            geom_sizes[name][i] = np.pad(vec, (0, 3 - len(vec)))

    return geom_sizes


def calc_geom_positions(sizes: dict, extras: dict):
    """"..."""

    # Unpack the needed extra values.
    h_hand, len_hand, len_fingers = extras["hand"]
    _, len_foot, len_toes = extras["foot"]
    len_lower_leg1, len_lower_leg2 = extras["lower_leg"]

    # Calculate the geom position based on size and extras.
    positions = {
        "head": [
            [0.01, 0, sizes["head"][0]],  # head
            [0, 0, 0]  # eye
        ],
        "upper_arm": [[0, 0, sizes["upper_arm"][1]]],
        "lower_arm": [[0, 0, sizes["lower_arm"][1]]],
        "hand": [
            [h_hand / 2, 0, len_hand],  # hand1
            [h_hand / 2, 0, len_hand * 2],  # hand2
            [0, 0, len_fingers],  # fingers1
            [0, 0, len_fingers * 2]  # fingers2
        ],
        "torso": [
            [-0.002, 0, 0.005],  # lb
            [0.005, 0, -0.008],  # cb
            [0.007, 0, -0.032],  # ub1
            [0.004, 0, 0.03],  # ub2
            [0, 0, 0.09],  # ub3
        ],
        "upper_leg": [[0, 0, -sizes["upper_leg"][1] * ratios["upper_leg"]]],
        "lower_leg": [
            # lower_leg1
            [0, 0, -len_lower_leg1],
            # lower_leg2
            [
                0,
                0,
                (-len_lower_leg1 * 2 - len_lower_leg2 - sizes["lower_leg"][1])
                * ratios["lower_leg"][1]
            ]
        ],
        "foot": [
            # foot1
            [-len_foot * ratios["foot"][3], 0, 0],
            # foot2
            [len_foot - len_foot * ratios["foot"][3], 0, 0],
            # foot3
            [len_foot + (len_foot - len_foot * ratios["foot"][3]), 0, 0],
            # toes1
            [len_toes, 0, 0],
            # toes2
            [len_toes * 2, 0, 0]
        ]
    }

    return positions


def calc_extras(sizes: dict) -> dict:
    """..."""

    extras = {}

    # ===== HAND =====

    # Unpack the sizes for the hand.
    l_hand, hand_breadth, _ = sizes["hand"]

    # Compute the hand height based on geometric mean of hand length and
    # half-breadth since there are no direct height measurements
    # for the hand available.
    h_hand = np.sqrt(l_hand * hand_breadth) * ratios["hand"][0]

    # ...
    l_hand -= (h_hand + EPSILON * 2) / 2

    # Divide the overall length between the fingers and the actual 'hand'.
    len_hand = l_hand * ratios["hand"][1]
    len_fingers = l_hand * (1 - ratios["hand"][1])

    # Store the extras for the hand.
    extras["hand"] = [h_hand, len_hand, len_fingers]

    # ===== FOOT =====

    # Compute the foot height based on geometric mean of foot length and
    # foot width since there are no direct height measurements
    # for the foot available.
    h_foot = np.sqrt(np.prod(sizes["foot"])) * ratios["foot"][0]

    # Divide the overall foot length between the toes and the actual 'foot'.
    len_foot = sizes["foot"][0] * ratios["foot"][1]
    len_toes = sizes["foot"][0] * ratios["foot"][2]

    # Store the extras for the foot.
    extras["foot"] = [h_foot, len_foot, len_toes]

    # ===== LOWER LEG =====

    # Get the length of the lower leg.
    len_lower_leg = sizes["lower_leg"][2]

    # Subtract the foot height from the length since the measurements
    # from the website include the foot but in the MIMo model we handle
    # lower leg and foot separately.
    len_lower_leg -= h_foot

    # Divide the overall lower leg length into the two parts
    # the MIMo model uses.
    len_lower_leg1 = len_lower_leg * ratios["lower_leg"][0]
    len_lower_leg2 = len_lower_leg * (1 - ratios["lower_leg"][0])

    # Store the extras for the lower leg.
    extras["lower_leg"] = [len_lower_leg1, len_lower_leg2]

    return extras


def calc_geom_masses(params, og_vals):
    """..."""

    # Iterate over all geom parameters.
    for geom_name, attributes in params.items():

        # Get type and size of the current geom.
        type_ = og_vals["geom"][geom_name]["type"]
        size = attributes["size"]

        # Calculate the volume based on the type.
        if type_ == "sphere":
            vol = (4 / 3) * np.pi * size[0] ** 3
        elif type_ == "capsule":
            vol = (4 / 3) * np.pi * size[0] ** 3
            vol += np.pi * size[0] ** 2 * size[1] * 2
        elif type_ == "box":
            vol = np.prod(size) * 8

        # Calculate and store the mass.
        attributes["mass"] = vol * og_vals["geom"][geom_name]["density"]


def calc_geom_params(sizes: dict, original_values: dict) -> dict:
    """..."""

    # Calculate some extra values that are not
    # within the measurement sizes e.g. the foot height.
    # These value are needed in order to correctly update
    # all geom sizes and positions.
    extras = calc_extras(sizes)

    # Calculate all geom sizes and positions based on the
    # estimated measurement sizes for the given age and the extras.
    geom_sizes = calc_geom_sizes(sizes, extras)
    geom_pos = calc_geom_positions(sizes, extras)

    # Map the size and position vectors to the correct geom names
    # as defined in the mapping.
    params = {}
    for body_part, geom_names in MAPPING_GEOM.items():
        for i, name in enumerate(geom_names):
            keys = name if isinstance(name, tuple) else [name]
            for key in keys:
                size, pos = geom_sizes[body_part][i], geom_pos[body_part][i]
                params[key] = {"size": size, "pos": pos}

    # Calculate and store the mass of geoms.
    calc_geom_masses(params, original_values)

    return params
