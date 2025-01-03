"""..."""


import re
import numpy as np
import xml.etree.ElementTree as ElementTree


def calc_density():

    tree = ElementTree.parse('mimoEnv/assets/mimo/MIMo_model.xml')
    root = tree.getroot()

    for geom in root.findall(".//geom"):

        mass = float(geom.attrib.pop("mass"))
        size = [float(num) for num in re.sub(r'\s+', ' ', geom.attrib["size"]).split(" ")]

        type_ = geom.attrib["type"]
        if type_ == "box":
            density = mass / np.prod(size)
        elif type_ == "capsule":
            density = mass / (np.pi * size[0] ** 2 * size[1] * 2 + (4 / 3) * np.pi * size[0] ** 3)
        elif type_ == "cylinder":
            density = mass / (np.pi * size[0] ** 2 * size[1] * 2)
        elif type_ == "sphere":
            density = mass / ((4 / 3) * np.pi * size[0] ** 3)

        geom.set("density", f'{density:.4f}')

    tree.write('mimoEnv/assets/mimo/MIMo_model_density.xml', encoding='utf-8', xml_declaration=True)
