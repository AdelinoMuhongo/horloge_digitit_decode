import matplotlib.pyplot as plt
import utils
import numpy as np
from datetime import time


def extact_data(region):
    found_objects = []
    for i, rect in enumerate(utils.find_contours(region)):
        toto = region[rect[0][0]:rect[1][0], rect[0][1]:rect[1][1]]
        result = utils.compare_digits(toto)

        if result[0] == -1:
            result[0] = 0  # Pour ne pas generer d'exceptions durant la conversion en entier

        found_objects.append(result[0])

    return found_objects


if __name__ == '__main__':
    img = plt.imread("../images/horloge1.jpg")

    gray = utils.luminance(img)

    binary = (gray >= 130).astype(np.uint8)
    binary = utils.median_filter(binary, 3)
    regions = utils.regions_of_interest(binary)

    temperature = 'NOTSET'
    temps = 'NOTSET'
    humidity = 'NOTSET'
    for region_name in ['temp', 'time', 'humidity']:
        region = regions[region_name]
        found_digits = extact_data(region)
        
        if region_name == 'temp':
            temperature = int("".join(map(str, found_digits[:2])))
        elif region_name == 'time':
            h = int("".join(map(str, found_digits[:2])))
            m = int("".join(map(str, found_digits[-2:])))
            temps = time(h, m)
        else:
            humidity = int("".join(map(str, found_digits[:2])))

    print(f"Temps : {temps}")
    print(f"Temperature : {temperature} °C")
    print(f"Humidité : {humidity} %")