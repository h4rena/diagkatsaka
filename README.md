# DiagMaïs — Détection de la Rouille Polysora sur Feuilles de Maïs

Système d'aide au diagnostic pour techniciens agricoles à Madagascar.
Classifie les feuilles de maïs en **Saine (0)** ou **Malade (1)** à partir de photos.

---

## Contexte du TP

La sécurité alimentaire à Madagascar dépend fortement de la culture du maïs.
Les plantations sont régulièrement menacées par des maladies fongiques, notamment
la **Rouille Polysora** (*Puccinia polysora*), particulièrement dévastatrice dans
les zones chaudes et humides de l'île.

Ce projet implémente une chaîne complète de traitement de données :

1. **Extraction de features** depuis des images brutes (traitement d'images)
2. **Implémentation "from scratch"** d'un arbre de décision basé sur une métrique
   personnalisée de pureté : **Max-Minority**
3. **Entraînement et comparaison** de 4 modèles (scratch vs scikit-learn)
4. **Déploiement** d'une application web Streamlit pour le diagnostic terrain

---

## Structure du projet

```
Katsaka/
├── .streamlit/
│   └── config.toml          # Thème Streamlit (couleurs)
├── dataset/
│   ├── saines/               # Photos de feuilles saines (label 0)
│   └── malades/              # Photos de feuilles avec rouille (label 1)
├── models/
│   └── rf_sklearn.pkl        # Modèle Random Forest sauvegardé (pickle)
├── uploads/                  # Images téléversées via l'app (créé à l'exécution)
├── app.py                    # Application web Streamlit (Partie 4)
├── partie1_features.py       # Feature engineering (Partie 1)
├── partie2_max_minority.py   # Algorithme Max-Minority (Partie 2)
├── partie3_modeles.py        # Arbres et forêts (Partie 3)
├── max_minority.py           # Duplicate de partie2 (importé par modeles.py)
├── modeles.py                # Version standalone de partie3
├── features.csv              # DataFrame des features extraites (généré)
├── requirements.txt          # Dépendances Python
├── venv/                     # Environnement virtuel
└── README.md                 # Ce fichier
```

## Fichiers principaux

| Fichier | Rôle |
|---|---|
| `partie1_features.py` | Parcourt `dataset/`, extrait 3 features par image, génère `features.csv` |
| `partie2_max_minority.py` | Définit la métrique de pureté Max-Minority et la recherche de meilleur split |
| `partie3_modeles.py` | Construit et compare 4 modèles (scratch + sklearn) |
| `modeles.py` | Version standalone (importe `max_minority.py`) |
| `app.py` | Application Streamlit web avec upload et historique |
| `max_minority.py` | Copie de `partie2_max_minority.py` utilisée par `modeles.py` |

---

## Prérequis

- **Python 3.12+**
- **pip** (gestionnaire de paquets)

### Dépendances

```
streamlit>=1.28
opencv-python>=4.8
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
pillow>=10.0
```

---

## Installation

```bash
# 1. Cloner ou se placer dans le dossier du projet
cd /home/hatsugoki/MrTsinjo/Katsaka

# 2. Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install streamlit opencv-python numpy pandas scikit-learn pillow

# 4. (Optionnel) Vérifier l'installation
pip list
```

---

## Dataset

### Source

Les images proviennent du dataset **CornLeafDiseaseCollection** (Mendeley Data,
DOI: 10.17632/w56xxnykcc). Il contient 1 079 images de feuilles de maïs collectées
dans trois cantons de Manabí, Équateur.

### Catégories utilisées

| Dossier source | Classe | Label | Nombre d'images |
|---|---|---|---|
| `common_rust/` | Rouille (Southern rust) | 1 | 200 |
| `healthy/` | Saine | 0 | 232 |

### Organisation

```bash
mkdir -p dataset/saines dataset/malades
cp /chemin/vers/CornLeafDiseaseCollection/healthy/*  dataset/saines/
cp /chemin/vers/CornLeafDiseaseCollection/common_rust/* dataset/malades/
```

Les autres catégories du dataset source (`asphalt_stain`, `bipolaris`,
`stenocarpella`) ne sont pas utilisées car elles correspondent à d'autres
maladies que la rouille.

---

## Utilisation — Pipeline complet

### Étape 1 : Extraire les features (Partie 1)

```bash
source venv/bin/activate
python partie1_features.py
```

Ce script parcourt `dataset/saines/` et `dataset/malades/`, extrait pour chaque
image les 3 features suivantes et génère `features.csv`.

**Sortie :** `features.csv` — 432 lignes × 5 colonnes :
`[ID_Image | pct_rouille | rugosite | ratio_saturation | label_malade]`

### Étape 2 : Entraîner les modèles (Parties 2 et 3)

```bash
python partie3_modeles.py
```

Ou avec la version standalone :

```bash
python modeles.py
```

Le script :

1. Charge `features.csv`
2. Sépare les données : 80% entraînement / 20% test (stratifié)
3. Construit 4 modèles
4. Affiche un tableau comparatif (Accuracy, Précision, Rappel)
5. Affiche les matrices de confusion
6. Sauvegarde le meilleur modèle dans `models/rf_sklearn.pkl`

**Sortie :** `models/rf_sklearn.pkl` — modèle Random Forest sklearn (100 arbres)

### Étape 3 : Lancer l'application web (Partie 4)

```bash
streamlit run app.py
```

Ouvrir l'URL affichée (généralement `http://localhost:8501`).

---

## Partie 1 — Feature Engineering (3 features)

### Feature 1 : `pct_rouille` — Pourcentage de pixels rouille

**Méthode :** Masquage HSV

1. Conversion de l'image BGR → HSV
2. Définition d'une plage de teintes "rouille" :
   - Teinte H ∈ [5°, 30°] (orange à brun)
   - Saturation S ≥ 80/255
   - Valeur V ≥ 80/255
3. Comptage des pixels dans cette plage
4. Normalisation : `pct_rouille = pixels_rouille / total_pixels`

**Intuition agronomique :** Les pustules de rouille polysora sont de couleur
orange à brun-orangé, facilement isolables dans l'espace HSV.

### Feature 2 : `rugosite` — Variance du gradient de Sobel

**Méthode :** Filtre de Sobel + variance

1. Conversion de l'image BGR → niveaux de gris
2. Application du filtre de Sobel (kernel 3×3) sur les axes X et Y
3. Calcul de la magnitude du gradient : `sqrt(Gx² + Gy²)`
4. Calcul de la variance de la magnitude

**Intuition agronomique :** Les pustules de rouille créent des irrégularités
de surface (texture rugueuse) qui se traduisent par des gradients d'intensité
élevés. La variance capture cette hétérogénéité texturale.

### Feature 3 : `ratio_saturation` — Saturation moyenne de la zone verte

**Méthode :** Masquage HSV + moyenne de saturation

1. Conversion BGR → HSV
2. Création d'un masque des pixels "verts" : H ∈ [35°, 85°]
3. Extraction du canal S (saturation) sur les pixels verts uniquement
4. Normalisation S ∈ [0, 255] → [0, 1]
5. Calcul de la moyenne

**Intuition agronomique :** Une feuille saine présente un vert franc et saturé
(chlorophylle abondante). La rouille dégrade la chlorophylle (chlorose),
diminuant la saturation globale de la zone verte.

---

## Partie 2 — Algorithme Max-Minority

### Métrique de pureté

Pour un nœud *t* contenant *N* individus, la pureté *P(t)* est définie par la
proportion de la classe majoritaire :

```
P(t) = max(n_c / N) pour c ∈ {0, 1}
```

**Exemples :**

| Composition | P(t) |
|---|---|
| 90% saines / 10% malades | 0.90 |
| 50% saines / 50% malades | 0.50 |
| 100% saines | 1.00 |

### Algorithme de recherche du meilleur split

Pour une variable continue, le seuil optimal est trouvé par balayage :

1. **Trier** les valeurs de X par ordre croissant
2. **Pour chaque seuil candidat** s (milieu entre deux valeurs consécutives
   uniques) :
   - Séparer les données en Gauche (≤ s) et Droite (> s)
   - Calculer la pureté pondérée du split :
     ```
     P_split = |G|/N × P(G) + |D|/N × P(D)
     ```
3. **Retenir** le seuil s qui maximise P_split

### Fonctions implémentées

| Fonction | Rôle |
|---|---|
| `purity_max_minority(y)` | Calcule P(t) pour un ensemble de labels |
| `trouver_meilleur_split(X_column, y)` | Trouve le seuil optimal pour une feature |

---

## Partie 3 — Modèles et performances

### Les 4 modèles comparés

| # | Modèle | Critère | Origine |
|---|---|---|---|
| 1 | Arbre Max-Minority | Max-Minority | From scratch |
| 2 | Forêt Max-Minority (50 arbres) | Max-Minority | From scratch |
| 3 | DecisionTreeClassifier | Gini | scikit-learn |
| 4 | RandomForestClassifier (100 arbres) | Gini | scikit-learn |

### Résultats (jeu de test — 87 images)

```
                      Modèle  Accuracy  Précision  Rappel
Arbre Max-Minority (scratch)    0.9195     0.9024   0.9250
Forêt Max-Minority (scratch)    0.9425     0.9070   0.9750
        Arbre Gini (sklearn)    0.9195     0.8667   0.9750
        Forêt Gini (sklearn)    0.9080     0.8478   0.9750
```

### Matrices de confusion

**Arbre Max-Minority (scratch) :**
```
           Prédit Sain  Prédit Malade
Réel Sain      43             4    (FP = gaspillage)
Réel Malade     3            37    (FN = épidémie !)
```

**Forêt Max-Minority (scratch) :**
```
           Prédit Sain  Prédit Malade
Réel Sain      43             4    (FP = gaspillage)
Réel Malade     1            39    (FN = épidémie !)
```

**Arbre Gini (sklearn) :**
```
           Prédit Sain  Prédit Malade
Réel Sain      41             6    (FP = gaspillage)
Réel Malade     1            39    (FN = épidémie !)
```

**Forêt Gini (sklearn) :**
```
           Prédit Sain  Prédit Malade
Réel Sain      40             7    (FP = gaspillage)
Réel Malade     1            39    (FN = épidémie !)
```

### Importance des variables (Random Forest sklearn)

| Feature | Importance |
|---|---|
| `rugosite` | **0.472** (47.2%) |
| `pct_rouille` | **0.433** (43.3%) |
| `ratio_saturation` | **0.095** (9.5%) |

→ La rugosité de surface et le pourcentage de pixels rouille sont les deux
features les plus discriminantes.

### Analyse critique

**Comportement :**
- L'arbre Max-Minority scratch et l'arbre Gini sklearn atteignent des
  performances proches. Max-Minority favorise les splits qui isolent la
  majorité, efficace sur classes déséquilibrées.
- La Forêt Aléatoire (scratch ou sklearn) améliore systématiquement la
  robustesse par rapport à un arbre unique :
  - Le Bagging réduit la variance
  - Le vote majoritaire lisse les instabilités

**Recommandation pour Madagascar :**

Dans le contexte malgache, le **Faux Négatif** (feuille malade classée saine)
est le risque le plus grave : une épidémie non détectée peut détruire la
récolte. On cherche donc à **maximiser le Rappel**.

Le **Random Forest sklearn** (100 arbres, Gini) est recommandé pour le
déploiement terrain :
- Rappel élevé (0.975) → peu de feuilles malades manquées
- Précision correcte → peu de traitements inutiles
- Robustesse → stable sur de nouvelles images
- Importance des variables → interprétable par les techniciens

---

## Partie 4 — Application Web Streamlit

### Fonctionnalités

**Onglet 1 : Analyse d'image**
- Upload d'une photo (.jpg, .jpeg, .png)
- Affichage de l'image téléversée
- Extraction en temps réel des 3 features
- Prédiction avec probabilité
- Message visuel : vert (saine) ou rouge (malade)
- Recommandation agronomique

**Onglet 2 : Historique des détections**
- Galerie des photos précédemment analysées
- Miniatures avec diagnostic (Saine/Malade)
- Métriques récapitulatives (total, saines, malades)
- Horodatage de chaque analyse

**Sidebar :**
- Informations sur le modèle (Random Forest, features)
- Contexte sur la Rouille Polysora
- Bouton pour vider l'historique

### Lancement

```bash
streamlit run app.py
```

L'application est accessible à l'adresse : `http://localhost:8501`

### Capture d'écran

*[L'interface affiche un en-tête vert "DiagMaïs", une zone d'upload,
une zone de résultat colorée, et une galerie d'historique en bas.]*

---

## Dépannage

### Erreur : `No such file or directory: 'models/rf_sklearn.pkl'`

```bash
mkdir -p models
python partie3_modeles.py
```

### Erreur : `Port 8501 is not available`

```bash
streamlit run app.py --server.port 8502
```

### Erreur : `ModuleNotFoundError` (opencv, sklearn, etc.)

```bash
source venv/bin/activate
pip install opencv-python scikit-learn
```

### Le fichier `features.csv` n'existe pas

```bash
python partie1_features.py
```

### L'application n'affiche pas les images

Vérifier que le dossier `uploads/` existe et est accessible :
```bash
mkdir -p uploads
```

---

## Références

- **Dataset :** Murillo Parraga, L. M. & Silva Villafuerte, C. A. (2025).
  *CornLeafDiseaseCollection*. Mendeley Data. DOI: 10.17632/w56xxnykcc
- **OpenCV :** Documentation du masquage HSV et filtre de Sobel
- **scikit-learn :** DecisionTreeClassifier, RandomForestClassifier
- **Streamlit :** Documentation de déploiement d'applications web

---

## Licence

Projet réalisé dans le cadre d'un TP universitaire.
