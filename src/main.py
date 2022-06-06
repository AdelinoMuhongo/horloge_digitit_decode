"""
Create by david.yama-yama
on 04/06/2022
"""

import matplotlib.pyplot as plt
import logging
from digicoder import ClockDigiCoder

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    img = plt.imread("../images/horloge2.jpg")

    cdc = ClockDigiCoder(img)
    cdc.decode()

    logger.info("####### Resultats #######\n")
    logger.info(f"Heure : {cdc.time}")
    logger.info(f"Temperature : {cdc.temperature} °C")
    logger.info(f"Humidité : {cdc.humidity} %")
