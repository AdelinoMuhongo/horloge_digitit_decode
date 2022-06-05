"""
Create by david.yama-yama
on 04/06/2022
"""

import matplotlib.pyplot as plt
import logging
from digicoder import ClockDigiCoder

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    img = plt.imread("../images/horloge2.jpg")

    cdc = ClockDigiCoder(img)
    cdc.decode()

    print()
    _logger.info("####### Resultats #######\n")
    _logger.info(f"Heure : {cdc.time}")
    _logger.info(f"Temperature : {cdc.temperature} °C")
    _logger.info(f"Humidité : {cdc.humidity} %")
