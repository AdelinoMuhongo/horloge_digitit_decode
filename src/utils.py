"""
TP Algo et Prog

@auteur : YAMA YAMA Kibala David
INSA Rouen PERFNII - IPN02
cree le 01/06/2022

Script contenant toutes les fonctions utiles pour le traitement de l'image

Points clés :   - Dilatation morphologique
                - Moore-neighbor algorithm
                - Espace couleur HSV
                - Filtre médian
                - Opérations booléennes ( bitwise XOR and AND)

"""

import numpy as np
import const


def luminance(im):
    """ Permet d'extraire la luminance d'une image :  canal V (HSV) """
    r, g, b = im[..., 0], im[..., 1], im[..., 2]

    return np.maximum(np.maximum(r, g), b).astype(np.uint8)


# ## REDUCTION DES BRUIS, LISSAGE ET TRANSFORMATION  MORPHOLOGIQUE
# ----------------------------------------------------------------

def median_filter(gray, ksize=5, padding_mode='constant'):
    """
    Retourne une image avec un filtre médian appliqué dessus

    :param gray: tableau numpy representant l'image en niveau gris
    :param ksize: la taille du mask
    :param padding_mode : mode de remplissage des extremités (cfr: numpy.pad)
    """

    if ksize % 2 == 0 or ksize == 1:
        raise ValueError('La taille du masque doit être positive et impaire')

    height, width = gray.shape[:2]
    offset = int(ksize / 2)

    if padding_mode == 'constant':
        img_padded = np.pad(gray, offset, constant_values=0)
    elif padding_mode == 'edge':
        img_padded = np.pad(gray, offset, mode='edge')
    else:
        raise ValueError("Le mode de remplissage doit etre 'constant' or 'edge'")

    median_img = np.zeros_like(gray)
    for i in range(height):
        for j in range(width):
            m_values = img_padded[i:i + ksize, j:j + ksize].flatten()
            median_img[i, j] = np.sort(m_values)[(ksize * ksize) // 2]
    return median_img


def _morpho_dilatation(binary_img, radius=2):
    """
    Réalise une dilatation morphologique sur une image binaire

    ref: https://en.wikipedia.org/wiki/Dilation_(morphology)
    """

    row, col = binary_img.shape
    result = np.copy(binary_img)

    for i in range(radius, row - radius):
        for j in range(radius, col - radius):
            m = binary_img[i - radius: i + radius + 1, j - radius:j + radius + 1]
            result[i, j] = np.amax(m)
    return result


# ## TRANSFORMATIONS GEOMETRIQUES ET EXTRACTION DES ZONES UTILES
# --------------------------------------------------------------

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


def _rect_bounded_area(points):
    row = min([i for i, _ in points])
    col = min([i for _, i in points])
    height = max([i for i, _ in points]) - row
    width = max([i for _, i in points]) - col

    return row, col, height, width


def _extract_temp_humidity_region(cropped):
    """
    Extrait la region temps sur l'image x0 = 70% largeur
    et xn = 93% largeur de la largeur de l'image pre-decoupé
    """
    _, width = cropped.shape
    x0 = int(width * 70 / 100)
    xn = int(width * 93 / 100)
    return cropped[:, x0:xn]


def _extract_time_region(cropped):
    """
    Extrait la region temps sur l'image
    x0 = 0  et xn = 54% de la largeur
    """
    _, width = cropped.shape
    xn = int(width * 54 / 100)
    return cropped[:, :xn]


def regions_of_interest(binary_img, foreground_val=1):
    """ Permet d'extraire les regions utiles qui devront être traitées """

    regions = {'time': None, 'temp': None, 'humidity': None}

    # Definition et rognage de la region d'interêt de l'image
    fg_pixels = list(zip(*np.where(binary_img == foreground_val)))
    row, col, height, width = _rect_bounded_area(fg_pixels)
    cropped = _crop(binary_img, row, col, height, width)

    # On extrait la region 'temps', on applique une dilatation (pour bien souder les objets uniques)
    # et on applique un padding, pour ne pas avoir des objets a l'extremite
    # (utile pour l'algorithme de recherche des contours )
    regions['time'] = np.pad(_morpho_dilatation(_extract_time_region(cropped)), 1, constant_values=0)

    # region temperature - humidite
    th_region = _morpho_dilatation(_extract_temp_humidity_region(cropped))

    height, _ = th_region.shape
    h = int(height * 50 / 100)
    regions['humidity'] = np.pad(th_region[:h, :], 1, constant_values=0)
    regions['temp'] = np.pad(th_region[h:, :], 1, constant_values=0)

    return regions


# ## RECHERCHE ET IDENTIFICATION DES OBJETS (CHIFFRES)
# ----------------------------------------------------

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
    """
    Permet de trouver le prochain pixel voisin
    en suivant le sens horaire
    """
    neighbors = _moore_neighbor(central_pixel)
    idx = neighbors.index(neighbor)
    idx = (idx + 1) % 8

    return neighbors[idx]


def _find_obj_contour(binary_img, foreground_val=1):
    """
    Retourne les points représentant le contour de l'objet trouvé

    ref: https://en.wikipedia.org/wiki/Moore_neighborhood
    """
    contour_pixels = []
    start_point = _find_start_point(binary_img, foreground_val)
    if start_point:
        contour_pixels.append(start_point)
        p = start_point
        prev = p[0] - 1, p[1]
        c = _next_neighbor(p, prev)
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
    Supprime un objet detecté sur l'image
    en transformant sa surface en noir
    """

    y0, x0, height, width = area
    xn, yn = x0 + width + 1, y0 + height + 1

    # On supprime l'objet qui a deja ete detecte sur l'image
    binary_img[y0: yn, x0:xn] = 0

    return (y0, x0), (yn, xn)


def find_contours(binary_img, foreground_val=1):
    """
    Permet de rechercher tous les objets sur l'image,
    representés par des contours en rectangle
    """

    im = np.copy(binary_img)

    # Liste contenant tous les contours (definis par un carré) des objets détectés
    rect_contours = []
    empty = False
    while not empty:
        contour = _find_obj_contour(im, foreground_val)
        if contour:
            area = _rect_bounded_area(contour)
            # On retire l'objet qui a déjà été détecté
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

        # le résultat est donné par la plus grosse perte de pixel blanc
        before = np.sum(digit == 1)
        current_matching_percent = 100 - (np.sum(bitwise == 1) / before * 100)
        if best_matching_percent < current_matching_percent:
            best_matching_percent = current_matching_percent
            matched_digit = i
    return [matched_digit, best_matching_percent]
