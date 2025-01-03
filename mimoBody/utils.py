"""..."""

from mimoBody.constants import MEASUREMENT_TYPES, RATIOS_MIMO_BODIES, RATIOS_DERIVED
import numpy as np


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


def pad_and_round_vectors(params: dict) -> None:
    """..."""

    # Define a little helper function that pads and rounds
    # a given vector.
    def pad_and_round(vec):
        return np.pad(np.round(vec, 4), (0, 3 - len(vec)))  # todo: think about removing round? or maybe round to place like in original MIMo

    # Pad and round all size and position vectors of the geoms.
    for geom, vectors in params["geoms"].items():
        for attr, vec in vectors.items():
            params["geoms"][geom][attr] = pad_and_round(vec)

    # Pad and round all position vectors of the bodies.
    for body, vec in params["bodies"].items():
        params["bodies"][body] = pad_and_round(vec)


# todo: cleanup
def create_geom_vectors(size: list, body_part: str, foot_height: float = None) -> list:
    """..."""

    if body_part == "head":

        vectors = [
            {"size": size, "pos": [0.01, 0, size[0]]},  # head
            {"size": size * RATIOS_DERIVED["eye"], "pos": [0, 0, 0]},  # eye
        ]

    elif body_part in ["upper_arm", "lower_arm"]:

        vectors = [
            {"size": size, "pos": [0, 0, size[1]]},
        ]

    elif body_part == "hand":

        vectors = create_hand_vectors(size)

    elif body_part == "torso":

        positions = [
            [-0.002, 0, 0.005],  # lb
            [0.005, 0, -0.008],  # cb
            [0.007, 0, -0.032],  # ub1
            [0.004, 0, 0.03],  # ub2
            [0, 0, 0.09],  # ub3
        ]

        vectors = []
        for i in range(5):
            vectors.append({
                "size": [size[i] * RATIOS_DERIVED["torso"][i], size[i] * (1 - RATIOS_DERIVED["torso"][i])],
                "pos": positions[i]
            })

    elif body_part == "upper_leg":

        vectors = [
            {"size": size, "pos": [0, 0, -size[1] * RATIOS_DERIVED["upper_leg"]]},
        ]

    elif body_part == "lower_leg":

        rad_calf, rad_ankle, length = size

        # ...
        length -= foot_height

        # ...
        len_leg1 = length * RATIOS_DERIVED["lower_leg"][0]
        len_leg2 = length * (1 - RATIOS_DERIVED["lower_leg"][0])

        vectors = [
            {"size": [rad_calf, len_leg1], "pos": [0, 0, -len_leg1]},  # lower_leg1
            {"size": [rad_ankle, len_leg2], "pos": [0, 0, (-len_leg1 * 2 - len_leg2 - rad_ankle) * RATIOS_DERIVED["lower_leg"][1]]},  # lower_leg2
        ]

    elif body_part == "foot":

        vectors = create_foot_vectors(size)

    else:
        raise ValueError(f"Invalid value for body_part: {body_part}")

    return vectors


# todo: cleanup
def create_hand_vectors(size: list) -> list:
    """..."""

    length, hand_breadth, fist_breadth = size

    # Compute the height based on geometric mean of hand length and half-breadth since
    # there are no direct height measurements for the hand available.
    height = np.sqrt(length * hand_breadth) * RATIOS_DERIVED["hand"][0]

    # Use a small constant to subtract from some geoms so that
    # the individual parts won't have a visual overlap.
    # This value is from the original MIMo model.
    EPSILON = 0.0001

    # todo: check how I handled the overlap again

    # ...
    length -= (height + EPSILON * 2) / 2

    # ...
    len_hand = length * RATIOS_DERIVED["hand"][1]
    len_fingers = length * (1 - RATIOS_DERIVED["hand"][1])

    vectors = [
        {
            "size": [fist_breadth, height, len_hand],
            "pos": [height / 2, 0, len_hand]
        },  # hand1
        {
            "size": [height + EPSILON * 2, fist_breadth - EPSILON * 3, 0],
            "pos": [height / 2, 0, len_hand * 2]
        },  # hand2
        {
            "size": [hand_breadth, height, len_fingers],
            "pos": [0, 0, len_fingers]
        },  # fingers1
        {
            "size": [height + EPSILON * 2, hand_breadth + EPSILON * 2, 0],
            "pos": [0, 0, len_fingers * 2]
        },  # fingers2
    ]

    return vectors


# todo: cleanup
def create_foot_vectors(size: list) -> list:
    """..."""

    length, width = size

    # ...
    height = np.sqrt(length * width) * RATIOS_DERIVED["foot"][0]

    # Use a small constant to subtract from some geoms so that
    # the individual parts won't have a visual overlap.
    # This value is from the original MIMo model.
    EPSILON = 0.0001

    # ...
    len_foot = length * RATIOS_DERIVED["foot"][1]
    len_toes = length * RATIOS_DERIVED["foot"][2]

    vectors = [
        {"size": [width - EPSILON, height - EPSILON], "pos": [-len_foot * RATIOS_DERIVED["foot"][3], 0, 0]},  # foot1
        {"size": [len_foot, width, height], "pos": [len_foot - len_foot * RATIOS_DERIVED["foot"][3], 0, 0]},  # foot2
        {"size": [height - EPSILON, width - EPSILON * 2], "pos": [len_foot + (len_foot - len_foot * RATIOS_DERIVED["foot"][3]), 0, 0]},  # foot3
        {"size": [len_toes, width - EPSILON, height - EPSILON], "pos": [len_toes, 0, 0]},  # toes1
        {"size": [height, width], "pos": [len_toes * 2, 0, 0]},  # toes2
    ]

    return vectors


# todo: cleanup
def create_body_vectors(geoms: dict) -> dict:
    """..."""

    ub3 = geoms["ub3"]
    head = geoms["head"]
    u_arm = geoms["left_uarm1"]
    l_arm = geoms["left_larm"]
    hand = geoms["geom:right_hand1"]
    u_leg = geoms["geom:right_upper_leg1"]
    l_leg1 = geoms["geom:left_lower_leg1"]
    l_leg2 = geoms["geom:left_lower_leg2"]
    foot = geoms["geom:left_foot3"]

    vectors = {
        "hip": [0, 0, u_leg["size"][1] * 2 + l_leg1["size"][1] * 2 + l_leg2["size"][1] * 2 + foot["size"][1] + ub3["size"][0]],
        "lower_body": [0.002, 0, (geoms["lb"]["size"][0] + geoms["cb"]["size"][0]) * RATIOS_MIMO_BODIES["lower_body"]],
        "upper_body": [-0.002, 0, (geoms["cb"]["size"][0] + geoms["ub1"]["size"][0]) * RATIOS_MIMO_BODIES["upper_body"]],
        "head": [0, 0, (ub3["pos"][2] + ub3["size"][0]) * RATIOS_MIMO_BODIES["head"]],
        "left_eye": RATIOS_MIMO_BODIES["eye"] * head["size"][0],
        "right_eye": RATIOS_MIMO_BODIES["eye"] * head["size"][0] * np.array([1, -1, 1]),
        "right_upper_arm": [
            -0.005,
            -(np.sum(ub3["size"]) + u_arm["size"][0]) * RATIOS_MIMO_BODIES["upper_arm"][0],
            ub3["pos"][2] * RATIOS_MIMO_BODIES["upper_arm"][1]
        ],
        "right_lower_arm": [0, 0, (u_arm["size"][0] + 2 * u_arm["size"][1] - l_arm["size"][0]) * RATIOS_MIMO_BODIES["lower_arm"]],
        "right_hand": [0, -0.007, (l_arm["size"][0] + 2 * l_arm["size"][1]) * RATIOS_MIMO_BODIES["hand"]],
        "right_fingers": [0, 0, hand["size"][2] * 2],
        "left_upper_arm": [
            -0.005,
            (np.sum(ub3["size"]) + u_arm["size"][0]) * RATIOS_MIMO_BODIES["upper_arm"][0],
            ub3["pos"][2] * RATIOS_MIMO_BODIES["upper_arm"][1]
        ],
        "left_lower_arm": [0, 0, (u_arm["size"][0] + 2 * u_arm["size"][1] - l_arm["size"][0]) * RATIOS_MIMO_BODIES["lower_arm"]],
        "left_hand": [0, 0.007, (l_arm["size"][0] + 2 * l_arm["size"][1]) * RATIOS_MIMO_BODIES["hand"]],
        "left_fingers": [0, 0, hand["size"][2] * 2],
        "right_upper_leg": [0.005, -(np.sum(geoms["lb"]["size"]) - u_leg["size"][0]) * RATIOS_MIMO_BODIES["upper_leg"], -.007],
        "right_lower_leg": [0, 0, -(u_leg["size"][0] + 2 * u_leg["size"][1] - l_leg1["size"][0]) * RATIOS_MIMO_BODIES["lower_leg"]],
        "right_foot": [0, 0, (l_leg1["pos"][2] + l_leg2["pos"][2]) * RATIOS_MIMO_BODIES["foot"]],
        "right_toes": [foot["pos"][0], 0, 0],
        "left_upper_leg": [0.005, (np.sum(geoms["lb"]["size"]) - u_leg["size"][0]) * RATIOS_MIMO_BODIES["upper_leg"], -.007],
        "left_lower_leg": [0, 0, -(u_leg["size"][0] + 2 * u_leg["size"][1] - l_leg1["size"][0]) * RATIOS_MIMO_BODIES["lower_leg"]],
        "left_foot": [0, 0, (l_leg1["pos"][2] + l_leg2["pos"][2]) * RATIOS_MIMO_BODIES["foot"]],
        "left_toes": [foot["pos"][0], 0, 0],
    }

    return vectors
