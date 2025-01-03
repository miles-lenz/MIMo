"""
The measurement values can be found on the following website: https://math.nist.gov/~SRessler/anthrokids/
All values are provided in centimeter.

The following shows
- head      : Head Circumference
- upper_arm : [Upper Arm Circumference, Shoulder Elbow Length]
- lower_arm : [Forearm Circumference, Elbow Hand Length - Hand Length]
- hand      : [Hand Length, Hand Breadth, Maximum Fist Breadth]
- torso     : Hip Breadth
- upper_leg : [Mid Thigh Circumference, Rump Knee Length]
- lower_leg : [Calf Circumference, Ankle Circumference, Knee Sole Length]
- foot      : [Foot Length, Foot Breadth]

"""

import numpy as np


# Store the mean value for the different age groups from the website.
# They will be needed to approximate a function that can predict sizes by age.
AGE_MONTHS = [1, 3, 7, 10, 13.5, 17.5, 21.5]

# Store relevant measurements from the website and categorize them by body parts.
# The docstring provides detailed information on how to find these exact values.
MEASUREMENTS = {
    "head": [[38.5, 41.7, 43.9, 45.5, 46.6, 46.8, 47.8]],
    "upper_arm": [
        [11.8, 13.0, 14.0, 14.8, 14.5, 14.7, 15.0],
        [10.9, 12.3, 13.1, 14.5, 14.9, 15.4, 16.2]
    ],
    "lower_arm": [
        [11.8, 13.1, 14.0, 14.3, 14.5, 14.5, 14.8],
        [14.9 - 6.8, 16.6 - 7.4, 18.0 - 8.0, 19.6 - 8.9, 19.9 - 9.2, 20.7 - 9.3, 21.5 - 9.5]
    ],
    "hand": [
        [6.8, 7.4, 8.0, 8.9, 9.2, 9.3, 9.5],
        [3.7, 4.1, 4.2, 4.5, 4.6, 4.6, 4.7],
        [4.2, 4.6, 4.9, 5.1, 5.3, 5.5, 5.5]
    ],
    "torso": [[13.2, 14.3, 15.9, 16.6, 16.9, 17.1, 17.1]],
    "upper_leg": [
        [16.9, 20.7, 21.2, 23.2, 23.4, 24.4, 24.7],
        [13.9, 15.9, 17.2, 19.2, 19.9, 21.3, 22.6]
    ],
    "foot": [
        [8.2, 9.1, 10.0, 10.9, 11.7, 11.9, 12.5],
        [3.6, 4.0, 4.2, 4.7, 4.9, 5.0, 5.2]
    ],
    "lower_leg": [
        [13.7, 15.6, 16.9, 18.1, 18.1, 18.4, 19.0],
        [10.2, 11.6, 12.3, 12.9, 13.2, 13.3, 13.6],
        [14.9, 16.5, 17.9, 19.8, 20.8, 21.6, 23.0]
    ],
}

# Store the type of all measurements. This will be useful when the measurements
# are prepared for usage in the MuJoCo model.
MEASUREMENT_TYPES = {
    "head": ["circ"],
    "upper_arm": ["circ", "len"],
    "lower_arm": ["circ", "len"],
    "hand": ["len", "len", "len"],
    "torso": ["len"],
    "upper_leg": ["circ", "len"],
    "lower_leg": ["circ", "circ", "len"],
    "foot": ["len", "len"],
}

# Store ratios that describe the difference between measurements and the original
# MIMo model. These ratios will be used to maintain all the small tweaks that
# were made by hand along any age.
RATIOS_MIMO_GEOMS = {
    "head": [(0.0735 * 200 * np.pi) / 46.8],  # circumference model / circumference measurement
    "upper_arm": [
        (0.024 * 200 * np.pi) / 14.7,  # circumference model / circumference measurement
        (0.0536 * 2) / ((15.4 - (2 * (14.7 / (2 * np.pi)))) / 100),  # middle-length model / middle-length measurement
    ],
    "lower_arm": [
        (0.023 * 200 * np.pi) / 14.5,  # circumference model / circumference measurement
        (0.037 * 2) / (((20.7 - 9.3) - (2 * (14.5 / (2 * np.pi)))) / 100),  # middle-length model / middle-length measurement
    ],
    "hand": [
        ((0.0208 * 2 + 0.0207 * 2 + 0.0102) * 100) / 9.3,  # length model / length measurement
        (0.0228 * 2 * 100) / 4.6,  # hand breadth model / hand breadth measurement
        (0.0281 * 2 * 100) / 5.5,  # fist breadth model / fist breadth measurement
    ],
    "torso": [
        # breadth model / breadth measurement
        ((0.048 * 2 + 0.043 * 2) * 100) / 17.1,  # lb
        ((0.053 * 2 + 0.035 * 2) * 100) / 17.1,  # cb
        ((0.052 * 2 + 0.035 * 2) * 100) / 17.1,  # ub1
        ((0.048 * 2 + 0.039 * 2) * 100) / 17.1,  # ub2
        ((0.041 * 2 + 0.047 * 2) * 100) / 17.1   # ub3
    ],
    "upper_leg": [
        (0.037 * 200 * np.pi) / 24.4,  # circumference model / circumference measurement
        (0.0625 * 2) / ((21.3 - (2 * (24.4 / (2 * np.pi)))) / 100),  # middle-length model / middle-length measurement
    ],
    "lower_leg": [
        (0.029 * 200 * np.pi) / 18.4,  # circumference calve model / circumference calve measurement
        (0.021 * 200 * np.pi) / 13.3,  # circumference ankle model / circumference ankle measurement
        (((0.02 + 0.029 + 0.044 * 2 + 0.021 + 0.028 * 2) * 100)) / 21.6,  # length model (with foot) / length measurement
    ],
    "foot": [
        (0.0249 + 0.035 * 2 + 0.007 * 2 + 0.01) * 100 / 11.9,  # length model / length measurement
        (0.025 * 2 * 100) / 5,  # breadth model / breadth measurement
    ]
}

# Store ratios that describe the difference between the body positions from the
# original model and the computed position based on other body parts. These ratios
# will be used to maintain all the small tweaks that were made by hand along any age.
RATIOS_MIMO_BODIES = {
    "head": 0.135 / 0.131,  # model pos / calculated pos
    "eye": np.array([0.07, 0.0245, 0.067375]) / 0.0735,  # model pos / model head circumference
    "upper_arm": [
        0.105 / 0.112,  # model y-pos / calculated y-pos
        0.093 / 0.09  # model z-pos / calculated z-pos
    ],
    "lower_arm": 0.1076 / 0.1082,  # ...
    "hand": 0.087 / 0.097,  # ...
    "lower_body": 0.076 / 0.101,  # ...
    "upper_body": 0.091 / 0.105,  # ...
    "upper_leg": 0.051 / 0.054,  # ...
    "lower_leg": 0.135 / 0.133,  # ...
    "foot": 0.177 / 0.178,  # ...
}

# Use ratios between different body parts from the original model to infer
# sizes for which there are no direct measurements on the website.
RATIOS_DERIVED = {
    "eye": 0.01125 / 0.0735,  # radius eye / radius head
    "hand": [
        0.01 / np.sqrt((0.0932 / 2) * 0.0228),  # height model / geometric mean of half-length and half-breadth of hand
        0.0208 / (0.0208 + 0.0207),  # hand1-to-fingers1 length ratio
    ],
    "torso": [
        # radius-to-length ratio
        0.048 / (0.048 + 0.043),  # lb
        0.053 / (0.053 + 0.035),  # cb
        0.052 / (0.052 + 0.035),  # ub1
        0.048 / (0.048 + 0.039),  # ub2
        0.041 / (0.041 + 0.047),  # ub3
    ],
    "upper_leg": 0.0645 / 0.0625,  # model pos / model size
    "lower_leg": [
        0.044 / (0.044 + 0.028),  # lower_leg_1-to-lower_leg_2 ratio
        0.134 / 0.137,  # model pos / calculated pos
    ],
    "foot": [
        0.01 / np.sqrt((0.1189 / 2) * 0.025),  # half-height model / geometric mean of half-width and half-breadth
        (0.035 * 2) / (0.0249 + 0.035 * 2 + 0.007 * 2 + 0.01),  # foot2-to-length ratio
        (0.007 * 2) / (0.0249 + 0.035 * 2 + 0.007 * 2 + 0.01),  # toes1-to-length ratio
        0.016 / 0.035,  # ...
    ]
}

# Map the keywords that are used in the measurements to the actual geom
# names they are intended for.
GEOM_MAPPING = {
    "head": ["head", ("geom:left_eye1", "geom:right_eye1")],
    "upper_arm": [("left_uarm1", "right_uarm1")],
    "lower_arm": [("left_larm", "right_larm")],
    "hand": [
        ("geom:right_hand1", "geom:left_hand1"),
        ("geom:right_hand2", "geom:left_hand2"),
        ("geom:right_fingers1", "geom:left_fingers1"),
        ("geom:right_fingers2", "geom:left_fingers2"),
    ],
    "torso": ["lb", "cb", "ub1", "ub2", "ub3"],
    "upper_leg": [("geom:left_upper_leg1", "geom:right_upper_leg1")],
    "lower_leg": [
        ("geom:left_lower_leg1", "geom:right_lower_leg1"),
        ("geom:left_lower_leg2", "geom:right_lower_leg2"),
    ],
    "foot": [
        ("geom:left_foot1", "geom:right_foot1"),
        ("geom:left_foot2", "geom:right_foot2"),
        ("geom:left_foot3", "geom:right_foot3"),
        ("geom:left_toes1", "geom:right_toes1"),
        ("geom:left_toes2", "geom:right_toes2"),
    ]
}
