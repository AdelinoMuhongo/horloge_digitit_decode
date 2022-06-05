"""
Create by david.yama-yama
on 05/06/2022

Description
"""

import numpy as np
import utils
from datetime import time
import logging

_logger = logging.getLogger(__name__)


class ClockDigiCoder:

    def __init__(self, clock_image):
        self._img = clock_image

        self._initialize(clock_image)

    def decode(self):
        _logger.info("###### Decodage et extraction des données #######\n  ")
        self._set_time()
        self._set_humidity()
        self._set_temperature()

    @property
    def temperature(self):
        return self._temperature

    @property
    def humidity(self):
        return self._humidity

    @property
    def time(self):
        return self._time

    # ## PRIVATES
    # -----------
    def _initialize(self, clock_image):
        _logger.info('######## Pre-traitement ########\n')

        # Extraction de la luminance : canal V (HSV)
        _logger.info('Niveau gris : extraction de la luminance (canal V : HSV) ')
        gray = utils.luminance(clock_image)

        # Recherche et application d'un seuil
        _logger.info("Segmentation de l'image: recherche et application d'un seuil automatique")
        thresh = utils.threshold(gray)
        binary = (gray >= thresh).astype(np.uint8)

        # Application d'un filtre median pour eliminer
        # pour lisser l'image et eliminer quelques parasites
        _logger.info("Filtrage et definition de la région d'intérêt\n")
        self._binary_img = utils.median_filter(binary, 3)
        self._regions = utils.regions_of_interest(self._binary_img)

    def _extact_data(self, region_name):
        region = self._regions[region_name]
        found_digits = []
        for i, rect in enumerate(utils.find_contours(region)):
            toto = region[rect[0][0]:rect[1][0], rect[0][1]:rect[1][1]]
            result = utils.compare_digits(toto)

            if result[0] == -1:
                result[0] = 0  # Pour ne pas generer d'exceptions durant la conversion en entier
                _logger.warning(f"Aucune correspondnace trouvée pour le digit numero {i} de la region '{region_name}")
            else:
                _logger.info(f"{result[1]} % de correspondance trouvé pour le digit {i} de la region '{region_name}")

            found_digits.append(result[0])

        return found_digits

    def _set_time(self):
        found_digits = self._extact_data('time')
        h = int("".join(map(str, found_digits[:2])))
        m = int("".join(map(str, found_digits[-2:])))
        self._time = time(h, m)

    def _set_temperature(self):
        found_digits = self._extact_data('temp')
        self._temperature = int("".join(map(str, found_digits[:2])))

    def _set_humidity(self):
        found_digits = self._extact_data('humidity')
        self._humidity = int("".join(map(str, found_digits[:2])))
