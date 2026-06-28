"""
TP : Diagnostic Rouille Polysora sur Feuilles de Maïs à Madagascar
PARTIE 2 : Algorithme Max-Minority — Indice de pureté personnalisé

Auteur : Rakotoarimalala Tsinjo Tony
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUE DE PURETÉ : Max-Minority
# ─────────────────────────────────────────────────────────────────────────────
def purity_max_minority(y):
    """
    Calcule la pureté d'un nœud selon la métrique Max-Minority.

    P(t) = max(n_c / N) pour c ∈ {0, 1}

    Cas limites :
      - Nœud vide       → pureté 0 (nœud inexploitable)
      - Nœud pur à 100% → pureté 1.0

    Parameters
    ----------
    y : array-like de 0/1 (labels du nœud)

    Returns
    -------
    float : pureté ∈ [0.5, 1.0]
    """
    y = np.asarray(y)
    N = len(y)
    if N == 0:
        return 0.0
    n1 = np.sum(y == 1)
    n0 = N - n1
    return max(n0, n1) / N


# ─────────────────────────────────────────────────────────────────────────────
# RECHERCHE DU MEILLEUR SPLIT (brute-force optimisé)
# ─────────────────────────────────────────────────────────────────────────────
def trouver_meilleur_split(X_column, y):
    """
    Recherche par balayage le seuil s qui maximise la pureté pondérée
    du split pour une variable continue.

    Algorithme :
      1. Trier les valeurs de X_column de façon croissante.
      2. Pour chaque seuil candidat (milieu entre deux valeurs consécutives
         uniques) séparer les données en Gauche (≤ s) et Droite (> s).
      3. Calculer la pureté pondérée :
         P_split = |G|/N × P(G) + |D|/N × P(D)
      4. Retourner le seuil qui maximise P_split.

    Parameters
    ----------
    X_column : array-like (N,)  — valeurs d'une seule feature
    y        : array-like (N,)  — labels correspondants (0 ou 1)

    Returns
    -------
    meilleur_seuil : float | None
    meilleure_purete : float     (P_split maximale trouvée)
    """
    X_column = np.asarray(X_column, dtype=np.float64)
    y        = np.asarray(y)
    N        = len(y)

    if N < 2:
        return None, purity_max_minority(y)

    # ── Étape 1 : trier par X_column ──────────────────────────────────────
    ordre      = np.argsort(X_column)
    X_trié     = X_column[ordre]
    y_trié     = y[ordre]

    # ── Étape 2 : initialisation ───────────────────────────────────────────
    meilleur_seuil  = None
    meilleure_purete = -np.inf

    # ── Étape 3 : balayage des seuils candidats ────────────────────────────
    valeurs_uniques = np.unique(X_trié)

    if len(valeurs_uniques) < 2:
        # Toutes les valeurs sont identiques → impossible de splitter
        return None, purity_max_minority(y)

    for i in range(len(valeurs_uniques) - 1):
        # Seuil candidat = milieu entre deux valeurs consécutives uniques
        seuil = (valeurs_uniques[i] + valeurs_uniques[i + 1]) / 2.0

        masque_gauche = X_trié <= seuil
        y_G = y_trié[masque_gauche]
        y_D = y_trié[~masque_gauche]

        taille_G = len(y_G)
        taille_D = len(y_D)

        # Pureté pondérée du split
        P_split = (taille_G / N) * purity_max_minority(y_G) + \
                  (taille_D / N) * purity_max_minority(y_D)

        if P_split > meilleure_purete:
            meilleure_purete = P_split
            meilleur_seuil   = seuil

    # ── Étape 4 : retour ──────────────────────────────────────────────────
    return meilleur_seuil, meilleure_purete


# ─────────────────────────────────────────────────────────────────────────────
# DÉMONSTRATION ISOLÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  TEST DE LA MÉTRIQUE MAX-MINORITY")
    print("=" * 60)

    # Cas 1 : nœud parfaitement mélangé (50/50)
    y_mixte = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    print(f"\nNœud 50/50     : P(t) = {purity_max_minority(y_mixte):.2f}  (attendu 0.50)")

    # Cas 2 : nœud à 90% sain
    y_90 = np.array([0]*9 + [1]*1)
    print(f"Nœud 90% sain  : P(t) = {purity_max_minority(y_90):.2f}  (attendu 0.90)")

    # Cas 3 : nœud pur
    y_pur = np.array([1]*10)
    print(f"Nœud pur (100%): P(t) = {purity_max_minority(y_pur):.2f}  (attendu 1.00)")

    print("\n" + "─" * 60)
    print("  TEST DE trouver_meilleur_split")
    print("─" * 60)

    # Exemple : pct_rouille faible → sain, élevé → malade
    np.random.seed(0)
    X_test = np.concatenate([
        np.random.uniform(0.0, 0.02, 20),   # feuilles saines
        np.random.uniform(0.05, 0.20, 20),  # feuilles malades
    ])
    y_test = np.array([0]*20 + [1]*20)

    seuil, purete = trouver_meilleur_split(X_test, y_test)
    print(f"\nVariable  : pct_rouille (simulé)")
    print(f"Meilleur seuil   s = {seuil:.4f}")
    print(f"Pureté pondérée  P = {purete:.4f}")
    print(f"\nInterprétation : si pct_rouille ≤ {seuil:.4f} → probablement SAINE")
    print(f"                 si pct_rouille >  {seuil:.4f} → probablement MALADE")
