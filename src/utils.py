"""
TP Algo et Prog

@auteur : YAMA YAMA Kibala David
INSA Rouen PERFNII - IPN02
cree le 01/06/2022

Script contenant toutes les fonctions utiles pour le traitement de l'image
"""

import numpy as np
import const


# #################### FONCTIONS PRE-TAITEMENT #############################
# --------------------------------------------------------------------------

def luminance(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]

    return np.maximum(np.maximum(r, g), b).astype(np.uint8)


def _compute_otsu_criteria(binary_img, thresh):
    # Creation d'une image seuil
    thresh_img = np.zeros(binary_img.shape)
    thresh_img[binary_img >= thresh] = 1

    nb_pixels1 = np.count_nonzero(thresh_img)
    w1 = nb_pixels1 / binary_img.size  # Proportion des pixels = 1
    w0 = 1 - w1  # Proportion des pixels = 0

    # les valeurs nulles ne seront pas consideré dans le calcul du meilleur seuil
    if w1 == 0 or w0 == 0:
        return np.inf

    val_pixels1 = binary_img[thresh_img == 1]  # Extraction des pixels = 1
    val_pixels0 = binary_img[thresh_img == 0]  # Extraction des pixels = 0

    # Recherche des variances
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return w0 * var0 + w1 * var1


def threshold(img):
    """
    Cette fonction permet de trouver le niveau de seuil qui génère la meilleure
    classification des pixels d'une image, en utilisant la methode de 'Otsu'

    :param img: tableau numpy representant l'image

    ref : https://en.wikipedia.org/wiki/Otsu%27s_method

    """
    threshold_range = range(np.max(img) + 1)
    criterias = [_compute_otsu_criteria(img, th) for th in threshold_range]

    # best threshold is the one minimizing the Otsu criteria
    return threshold_range[np.argmin(criterias)]


def median_filter(gray, ksize=11):
    """
    Retourne une image avec un filtre median applique dessus

    :param gray: tableau numpy representant l'image en niveau gris
    :param ksize: la taille du mask
    """
    # ajout d'un offset pour delimiter les bords (le filtre n'est pas applique sur les bords)
    offset = int(ksize / 2)
    median_img = np.zeros_like(gray)
    for i in range(offset, gray.shape[0] - offset):
        for j in range(offset, gray.shape[1] - offset):
            m_values = np.ravel(gray[i - offset: i + offset + 1, j - offset: j + offset + 1])
            median = np.uint8(np.median(m_values))
            # median = np.sort(kernel)[np.uint8(np.divide((np.multiply(ksize, ksize)), 2) + 1)]
            median_img[i, j] = median
    return median_img

def _morpho_dilatation(binary_img, radius=2):
    """
    Realise une dilataion sur une image binaire

    ref: https://en.wikipedia.org/wiki/Dilation_(morphology)
    """

    row, col = binary_img.shape
    result = np.copy(binary_img)

    for i in range(radius, row - radius):
        for j in range(radius, col - radius):
            m = binary_img[i - radius: i + radius + 1, j - radius:j + radius + 1]
            result[i, j] = np.amax(m)
    return result


# #################### FONCTIONS DE TRANSFORMATION GEOMETRIQUE #############
# --------------------------------------------------------------------------

def _resize(binary_img, new_size):
    h, w = binary_img.shape[:2]

    new_height, new_width = new_size
    x_scale = new_width / (w - 1)
    y_scale = new_height / (h - 1)

    out_img = np.zeros((new_height, new_width))

    for i in range(new_height - 1):
        for j in range(new_width - 1):
            out_img[i + 1, j + 1] = binary_img[1 + int(i / y_scale), 1 + int(j / x_scale)]

    return out_img.astype(np.uint8)


def _zoom(im, zoom):
    return np.kron(im, np.ones((zoom, zoom))).astype(np.uint8)


def _crop(im, row, col, height, width):
    return im[row:row + height, col:col + width]


# #################### FONCTIONS RECHERCHES CONTOURS ET DETECTION D'OJETS ###
# --------------------------------------------------------------------------

def _rect_bounded_area(points):
    row = min([i for i, _ in points])
    col = min([i for _, i in points])
    height = max([i for i, _ in points]) - row
    width = max([i for _, i in points]) - col

    return row, col, height, width


def _extract_temp_humidity_region(cropped):
    """
      Extrait la region temps sur l'image
      x0 = 65% largeur  et xn = 95% largeur
      de la largeur de l'image pre-decoupé
    """
    _, width = cropped.shape
    w0 = int(width * 70 / 100)
    wn = int(width * 93 / 100)
    return cropped[:, w0:wn]


def _extract_time_region(cropped):
    """
    Extrait la region temps sur l'image
    x0 = 0  et xn = 54% de la largeur
    """
    _, width = cropped.shape
    w = int(width * 54 / 100)
    return cropped[:, :w]


def regions_of_interest(binary_img, foreground_val=1):
    """ Permet d'extraire les regions utiles qui devront être traitées """

    regions = {'time': None, 'temp': None, 'humidity': None}
    fg_pixels = list(zip(*np.where(binary_img == foreground_val)))
    row, col, height, width = _rect_bounded_area(fg_pixels)
    cropped = _crop(binary_img, row, col, height, width)
    regions['time'] = np.pad(_morpho_dilatation(_extract_time_region(cropped)), 1, constant_values=0)
    th_region = _morpho_dilatation(_extract_temp_humidity_region(cropped))
    height, _ = th_region.shape
    h = int(height * 50 / 100)
    regions['humidity'] = np.pad(th_region[:h, :], 1, constant_values=0)
    regions['temp'] = np.pad(th_region[h:, :], 1, constant_values=0)

    return regions


def _find_start_point(binary_img, foreground_val):
    """
    Permet de trouver le premier pixel sur l'image
    en partant de gauche à droite, et de haut vers le bas
    """
    height, width = binary_img.shape

    for col in range(width):
        for row in range(height):
            if binary_img[row, col] == foreground_val:
                return row, col


def _moore_neighbor(central_pixel):
    """
    Permet de trouver les Pn (n = 1 à 8) voisins d'un pixel central

    ref : https://en.wikipedia.org/wiki/Moore_neighborhood
    """

    row, col = central_pixel
    return ((row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
            (row, col + 1), (row + 1, col + 1), (row + 1, col),
            (row + 1, col - 1), (row, col - 1))


def _next_neighbor(central_pixel, neighbor):
    neighbors = _moore_neighbor(central_pixel)
    idx = neighbors.index(neighbor)
    idx = (idx + 1) % 8

    return neighbors[idx]


def _find_obj_contour(binary_img, foreground_val=1):
    """
    Retourne les points representants le contour de l'objet trouvé

    ref: https://en.wikipedia.org/wiki/Moore_neighborhood
    """
    contour_pixels = []
    start_point = _find_start_point(binary_img, foreground_val)
    if start_point:
        contour_pixels.append(start_point)
        p = start_point
        prev = p[0] - 1, p[1]
        c = _next_neighbor(p, prev)
        count = 0
        while c != start_point:

            if binary_img[c[0], c[1]] == foreground_val:
                contour_pixels.append(c)
                p = c
                c = prev
            else:
                prev = c
                c = _next_neighbor(p, c)

        return contour_pixels


def _delete_object(binary_img, area):
    """
    Supprime un objet detecté sur l'image en
    transformant sa surface délimité en noir
    """

    y0, x0, height, width = area
    xn, yn = x0 + width + 1, y0 + height + 1

    binary_img[y0: yn, x0:xn] = 0

    return (y0, x0), (yn, xn)


def find_contours(imx, foreground_val=1):
    im = np.copy(imx)
    rect_contours = []
    empty = False
    while not empty:
        contour = _find_obj_contour(im, foreground_val)
        if contour:
            if len(contour) >= 8:
                pass
            area = _rect_bounded_area(contour)
            r = _delete_object(im, area)
            rect_contours.append(r)

        if contour is None:
            empty = True

    return rect_contours


def compare_digits(binary_img):
    """
    Fonction permettant de comparer les chiffres extrait de l'image aux modèles

    :return: le chiffre avec le meilleur pourcentage de correspondance
    """

    best_matching_percent = -1
    matched_digit = -1
    digits = [v[1:-1, 1:-1] for _, v in const.DIGITS.items()]
    for i, digit in enumerate(digits):

        # realise un zoom puis un resize pour na pas trop mdoifier la structure
        digit = _zoom(digit, binary_img.shape[1])
        digit = _resize(digit, binary_img.shape[:2])

        # permet de conserver ce qui n'est pas commun entre les deux images
        xor_diff = np.bitwise_xor(binary_img, digit)

        # permet de garder que la region d'interet
        bitwise = np.bitwise_and(digit, xor_diff)

        # le resultat est donné par la plus grosse perte de pixel blanc
        before = np.sum(digit == 1)
        current_matching_percent = 100 - (np.sum(bitwise == 1) / before * 100)
        if best_matching_percent < current_matching_percent:
            best_matching_percent = current_matching_percent
            matched_digit = i
    return [matched_digit, best_matching_percent]



