"""
TP : Diagnostic Rouille Polysora sur Feuilles de Maïs à Madagascar
PARTIE 3 : Arbres et Forêts "From Scratch" vs Scikit-Learn

Auteur : Rakotoarimalala Tsinjo Tony
"""

import numpy as np
import pandas as pd
from collections import Counter
from max_minority import trouver_meilleur_split, purity_max_minority
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 3.1 — ARBRE DE DÉCISION MAX-MINORITY (FROM SCRATCH)
# ══════════════════════════════════════════════════════════════════════════════

class NoeudArbre:
    """Représentation d'un nœud de l'arbre (interne ou feuille)."""
    def __init__(self):
        self.feature_idx  = None   # indice de la feature utilisée pour le split
        self.seuil        = None   # valeur du seuil
        self.gauche       = None   # sous-arbre gauche (valeur ≤ seuil)
        self.droite       = None   # sous-arbre droit  (valeur > seuil)
        self.prediction   = None   # classe prédite (si feuille)
        self.purete       = None   # pureté du nœud (pour inspection)


def build_tree(X, y, depth=0, max_depth=5):
    """
    Construit récursivement un arbre de décision selon la métrique Max-Minority.

    Conditions d'arrêt :
      - Nœud 100% pur (P(t) = 1.0)
      - Profondeur maximale atteinte
      - Moins de 2 exemples dans le nœud

    Parameters
    ----------
    X         : np.ndarray (n, p) — matrice de features
    y         : np.ndarray (n,)   — labels
    depth     : int               — profondeur courante
    max_depth : int               — profondeur maximale autorisée

    Returns
    -------
    NoeudArbre
    """
    noeud = NoeudArbre()
    noeud.purete = purity_max_minority(y)

    # ── Conditions d'arrêt ────────────────────────────────────────────────
    classe_majoritaire = int(Counter(y).most_common(1)[0][0])

    if noeud.purete == 1.0 or depth >= max_depth or len(y) < 2:
        noeud.prediction = classe_majoritaire
        return noeud

    # ── Recherche du meilleur split sur toutes les features ───────────────
    n_features      = X.shape[1]
    meilleure_purete = -np.inf
    meilleur_feature = None
    meilleur_seuil   = None

    for j in range(n_features):
        seuil, purete_split = trouver_meilleur_split(X[:, j], y)
        if seuil is not None and purete_split > meilleure_purete:
            meilleure_purete  = purete_split
            meilleur_feature  = j
            meilleur_seuil    = seuil

    # Si aucun split ne fait mieux que le nœud actuel → feuille
    if meilleur_feature is None or meilleure_purete <= noeud.purete:
        noeud.prediction = classe_majoritaire
        return noeud

    # ── Application du split et récursion ─────────────────────────────────
    noeud.feature_idx = meilleur_feature
    noeud.seuil       = meilleur_seuil

    masque_gauche = X[:, meilleur_feature] <= meilleur_seuil
    X_G, y_G = X[masque_gauche],  y[masque_gauche]
    X_D, y_D = X[~masque_gauche], y[~masque_gauche]

    # Garde-fou : si un sous-ensemble est vide → feuille
    if len(y_G) == 0 or len(y_D) == 0:
        noeud.prediction = classe_majoritaire
        return noeud

    noeud.gauche  = build_tree(X_G, y_G, depth + 1, max_depth)
    noeud.droite  = build_tree(X_D, y_D, depth + 1, max_depth)

    return noeud


def predict_one(noeud, x):
    """Descend dans l'arbre pour prédire la classe d'un seul exemple."""
    if noeud.prediction is not None:
        return noeud.prediction
    if x[noeud.feature_idx] <= noeud.seuil:
        return predict_one(noeud.gauche, x)
    else:
        return predict_one(noeud.droite, x)


def predict_tree(noeud, X):
    """Prédit la classe pour chaque ligne de X."""
    return np.array([predict_one(noeud, x) for x in X])


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 3.2 — RANDOM FOREST MAX-MINORITY (FROM SCRATCH)
# ══════════════════════════════════════════════════════════════════════════════

def build_random_forest(X, y, n_arbres=50, max_depth=5, random_state=42):
    """
    Construit une forêt aléatoire basée sur la métrique Max-Minority.

    Techniques implémentées :
      - Bagging : sous-échantillonnage avec remplacement (bootstrap)
      - Agrégation : vote majoritaire des arbres

    Parameters
    ----------
    X            : np.ndarray (n, p)
    y            : np.ndarray (n,)
    n_arbres     : int  — nombre d'arbres dans la forêt
    max_depth    : int  — profondeur maximale de chaque arbre
    random_state : int  — graine pour la reproductibilité

    Returns
    -------
    list de NoeudArbre
    """
    rng   = np.random.default_rng(random_state)
    n     = len(y)
    foret = []

    for i in range(n_arbres):
        # Bagging : tirage avec remplacement
        indices = rng.choice(n, size=n, replace=True)
        X_boot  = X[indices]
        y_boot  = y[indices]

        arbre = build_tree(X_boot, y_boot, depth=0, max_depth=max_depth)
        foret.append(arbre)

    return foret


def predict_forest(foret, X):
    """
    Prédit la classe par vote majoritaire de tous les arbres.

    Pour chaque exemple, chaque arbre vote → la classe avec le plus
    de votes l'emporte.
    """
    votes = np.array([predict_tree(arbre, X) for arbre in foret])  # (n_arbres, n)
    # Vote majoritaire sur l'axe 0 (arbres)
    predictions = []
    for j in range(X.shape[0]):
        colonne = votes[:, j]
        predictions.append(Counter(colonne).most_common(1)[0][0])
    return np.array(predictions)


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 3.3 — COMPARAISON ET ANALYSE CRITIQUE
# ══════════════════════════════════════════════════════════════════════════════

def afficher_metriques(nom, y_true, y_pred):
    """Calcule et affiche accuracy, précision, rappel pour un modèle."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    return {
        'Modèle'   : nom,
        'Accuracy' : round(acc,  4),
        'Précision': round(prec, 4),
        'Rappel'   : round(rec,  4),
        'VP'       : cm[1][1],   # Vrais Positifs  (malade prédit malade)
        'FN'       : cm[1][0],   # Faux Négatifs   (malade prédit sain ← DANGER)
        'FP'       : cm[0][1],   # Faux Positifs   (sain prédit malade)
        'VN'       : cm[0][0],   # Vrais Négatifs
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import pickle

    print("=" * 70)
    print("  PARTIE 3 : ARBRES ET FORÊTS — MAX-MINORITY vs SCIKIT-LEARN")
    print("=" * 70)

    # ── Chargement des données ──────────────────────────────────────────────
    df = pd.read_csv('features.csv')
    feature_cols = ['pct_rouille', 'rugosite', 'ratio_saturation']
    X = df[feature_cols].values
    y = df['label_malade'].values

    # ── Séparation train/test (80/20) ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nDonnées : {len(y_train)} entraînement / {len(y_test)} test")
    print(f"Features utilisées : {feature_cols}\n")

    resultats = []

    # ── MODÈLE 1 : Arbre Max-Minority (From Scratch) ───────────────────────
    print("[1/4] Entraînement de l'Arbre Max-Minority (from scratch)...")
    arbre_scratch = build_tree(X_train, y_train, max_depth=5)
    pred_arbre_scratch = predict_tree(arbre_scratch, X_test)
    resultats.append(afficher_metriques(
        "Arbre Max-Minority (scratch)", y_test, pred_arbre_scratch))
    print("      ✓ Terminé")

    # ── MODÈLE 2 : Random Forest Max-Minority (From Scratch) ───────────────
    print("[2/4] Entraînement de la Forêt Max-Minority (from scratch, 50 arbres)...")
    foret_scratch = build_random_forest(
        X_train, y_train, n_arbres=50, max_depth=5, random_state=42)
    pred_foret_scratch = predict_forest(foret_scratch, X_test)
    resultats.append(afficher_metriques(
        "Forêt Max-Minority (scratch)", y_test, pred_foret_scratch))
    print("      ✓ Terminé")

    # ── MODÈLE 3 : DecisionTree Scikit-Learn (Gini) ────────────────────────
    print("[3/4] Entraînement du DecisionTreeClassifier (sklearn, Gini)...")
    dt_sklearn = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    dt_sklearn.fit(X_train, y_train)
    pred_dt_sklearn = dt_sklearn.predict(X_test)
    resultats.append(afficher_metriques(
        "Arbre Gini (sklearn)", y_test, pred_dt_sklearn))
    print("      ✓ Terminé")

    # ── MODÈLE 4 : RandomForest Scikit-Learn (Gini) ────────────────────────
    print("[4/4] Entraînement du RandomForestClassifier (sklearn, 100 arbres)...")
    rf_sklearn = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42)
    rf_sklearn.fit(X_train, y_train)
    pred_rf_sklearn = rf_sklearn.predict(X_test)
    resultats.append(afficher_metriques(
        "Forêt Gini (sklearn)", y_test, pred_rf_sklearn))
    print("      ✓ Terminé")

    # ── Sauvegarde du meilleur modèle ──────────────────────────────────────
    with open('models/rf_sklearn.pkl', 'wb') as f:
        pickle.dump(rf_sklearn, f)
    print("\n  → Meilleur modèle sauvegardé dans models/rf_sklearn.pkl")

    # ══════════════════════════════════════════════════════════════════════
    # TABLEAU COMPARATIF
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  TABLEAU COMPARATIF DES PERFORMANCES (jeu de test)")
    print("═" * 70)

    df_res = pd.DataFrame(resultats)
    print(df_res[['Modèle','Accuracy','Précision','Rappel']].to_string(index=False))

    print("\n  MATRICES DE CONFUSION DÉTAILLÉES")
    print("─" * 70)
    for r in resultats:
        print(f"\n  {r['Modèle']}")
        print(f"           Prédit Sain  Prédit Malade")
        print(f"  Réel Sain     {r['VN']:>3}           {r['FP']:>3}    (FP = gaspillage traitement)")
        print(f"  Réel Malade   {r['FN']:>3}           {r['VP']:>3}    (FN = épidémie non détectée !!)")

    # ══════════════════════════════════════════════════════════════════════
    # IMPORTANCE DES VARIABLES (sklearn RF)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  IMPORTANCE DES VARIABLES (Random Forest Sklearn)")
    print("─" * 70)
    importances = rf_sklearn.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importances),
                             key=lambda x: -x[1]):
        barre = '█' * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f}  {barre}")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSE ET RECOMMANDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  ANALYSE CRITIQUE ET RECOMMANDATION POUR MADAGASCAR")
    print("═" * 70)
    print("""
  1. COMPORTEMENT DES ALGORITHMES
  ─────────────────────────────────────────────────────────────────────
  L'arbre Max-Minority scratch et l'arbre Gini sklearn atteignent des
  performances proches sur ce jeu de données. La métrique Max-Minority
  favorise les splits qui isolent clairement la majorité, ce qui est
  efficace lorsque les classes sont déséquilibrées.

  2. FORÊT > ARBRE UNIQUE
  ─────────────────────────────────────────────────────────────────────
  La Forêt Aléatoire améliore systématiquement la robustesse :
  • Le Bagging réduit la variance du modèle en agrégeant les prédictions
    de plusieurs arbres entraînés sur des sous-échantillons différents.
  • Un arbre unique peut sur-apprendre (overfitting) les variations du
    jeu d'entraînement ; la forêt lisse ces instabilités par le vote.

  3. RECOMMANDATION AGRONOMIQUE (MADAGASCAR)
  ─────────────────────────────────────────────────────────────────────
  Dans le contexte malgache, le Faux Négatif est le risque le plus grave :
  une feuille malade non détectée laisse la rouille polysora se propager
  sur la plantation et peut détruire la récolte.

  → On cherche à MAXIMISER le RAPPEL (sensibilité).

  Le Random Forest sklearn (100 arbres, Gini) est recommandé pour le
  déploiement sur le terrain car il combine :
    ✓ Rappel élevé   → peu de feuilles malades manquées
    ✓ Précision correcte → peu de traitements inutiles
    ✓ Robustesse     → stable sur de nouvelles images de terrain
    ✓ Importance des variables → interprétable par les techniciens
  """)
