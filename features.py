"""
TP : Diagnostic Rouille Polysora sur Feuilles de Maïs à Madagascar
PARTIE 1 : Feature Engineering — Du Pixel aux Caractéristiques

Auteur : Rakotoarimalala Tsinjo Tony
"""

import cv2
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1 : pct_rouille — Pourcentage de pixels "rouille" (HSV masking)
# ─────────────────────────────────────────────────────────────────────────────
def extraire_pct_rouille(img_bgr):
    """
    Convertit l'image en HSV et crée un masque pour les teintes
    'rouille' (marrons / oranges / jaunes).
    
    La rouille polysora se manifeste par des pustules de couleur
    orange à brun-orangé → plage HSV : H ∈ [5°, 30°], S > 50%, V > 50%.
    
    Returns
    -------
    float : proportion de pixels rouille (0.0 à 1.0)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Plage HSV pour les teintes orange/brun/rouille
    # En OpenCV : H ∈ [0, 179], S et V ∈ [0, 255]
    lower_rouille = np.array([5,  80, 80],  dtype=np.uint8)
    upper_rouille = np.array([30, 255, 255], dtype=np.uint8)

    masque = cv2.inRange(hsv, lower_rouille, upper_rouille)
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    pixels_rouille = np.count_nonzero(masque)

    return pixels_rouille / total_pixels


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2 : rugosite — Variance du gradient de Sobel
# ─────────────────────────────────────────────────────────────────────────────
def extraire_rugosite(img_bgr):
    """
    Applique un filtre de Sobel sur le canal niveaux de gris de l'image.
    
    Les pustules de rouille créent des irrégularités de surface qui se
    traduisent par des gradients d'intensité élevés. La variance du gradient
    capture cette hétérogénéité texturale.
    
    Returns
    -------
    float : variance de la magnitude du gradient de Sobel (normalisée)
    """
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Gradients horizontaux et verticaux (kernel 3×3)
    grad_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude du gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # La variance mesure la dispersion : élevée → surface irrégulière
    return float(np.var(magnitude))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3 : ratio_saturation — Saturation moyenne de la zone verte
# ─────────────────────────────────────────────────────────────────────────────
def extraire_ratio_saturation(img_bgr):
    """
    Intuition agronomique : Une feuille saine de maïs présente un vert
    franc, saturé et homogène (chlorophylle abondante). Lorsque la rouille
    s'installe, la chlorophylle se dégrade (chlorose) et la saturation
    globale de la zone verte diminue au profit des teintes marron/beige.

    On extrait la saturation moyenne dans l'espace HSV uniquement sur les
    pixels "verts" (H ∈ [35°, 85°]) pour rester robuste aux pustules.
    
    Returns
    -------
    float : saturation moyenne des pixels verts (0.0 à 1.0)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Masque des pixels verts (feuille elle-même)
    lower_vert = np.array([35, 30, 30], dtype=np.uint8)
    upper_vert = np.array([85, 255, 255], dtype=np.uint8)
    masque_vert = cv2.inRange(hsv, lower_vert, upper_vert)

    if np.count_nonzero(masque_vert) == 0:
        # Cas dégénéré : pas de pixel vert détecté
        return 0.0

    # Canal S ∈ [0, 255] → normalisé en [0, 1]
    saturation_canal = hsv[:, :, 1].astype(np.float32) / 255.0
    saturation_verts = saturation_canal[masque_vert > 0]

    return float(np.mean(saturation_verts))


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION COMPLÈTE SUR UN DOSSIER
# ─────────────────────────────────────────────────────────────────────────────
def extraire_features_dossier(dossier, label):
    """
    Parcourt un dossier d'images et retourne une liste de dictionnaires
    avec les features de chaque image.
    """
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    lignes = []

    fichiers = sorted([
        f for f in os.listdir(dossier)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    for nom_fichier in fichiers:
        chemin = os.path.join(dossier, nom_fichier)
        img = cv2.imread(chemin)

        if img is None:
            print(f"  [AVERTISSEMENT] Impossible de lire : {chemin}")
            continue

        pct_rouille     = extraire_pct_rouille(img)
        rugosite        = extraire_rugosite(img)
        ratio_sat       = extraire_ratio_saturation(img)

        lignes.append({
            'ID_Image'        : nom_fichier,
            'pct_rouille'     : round(pct_rouille, 6),
            'rugosite'        : round(rugosite, 4),
            'ratio_saturation': round(ratio_sat, 6),
            'label_malade'    : label
        })

    return lignes


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DOSSIER_SAINES  = 'dataset/saines'
    DOSSIER_MALADES = 'dataset/malades'
    FICHIER_SORTIE  = 'features.csv'

    print("=" * 60)
    print("  EXTRACTION DE FEATURES — ROUILLE POLYSORA (MAÏS)")
    print("=" * 60)

    print(f"\n[1/2] Traitement des feuilles SAINES  ({DOSSIER_SAINES})...")
    data_saines  = extraire_features_dossier(DOSSIER_SAINES,  label=0)
    print(f"      → {len(data_saines)} images traitées")

    print(f"\n[2/2] Traitement des feuilles MALADES ({DOSSIER_MALADES})...")
    data_malades = extraire_features_dossier(DOSSIER_MALADES, label=1)
    print(f"      → {len(data_malades)} images traitées")

    # Assemblage du DataFrame
    df = pd.DataFrame(data_saines + data_malades)
    df.to_csv(FICHIER_SORTIE, index=False)

    print(f"\n{'─'*60}")
    print(f"  DataFrame sauvegardé → {FICHIER_SORTIE}")
    print(f"  Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"{'─'*60}")
    print("\nAperçu des données :")
    print(df.head(10).to_string(index=False))

    print("\nStatistiques descriptives :")
    print(df.groupby('label_malade')[['pct_rouille','rugosite','ratio_saturation']].mean().round(4))
    print("\nDistribution des classes :")
    print(df['label_malade'].value_counts().rename({0:'Saines', 1:'Malades'}))
