"""
TP : Diagnostic Rouille Polysora sur Feuilles de Maïs à Madagascar
PARTIE 4 : Application Web Streamlit

Auteur : Rakotoarimalala Tsinjo Tony
Lancement : streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import pickle
import os
import shutil
from PIL import Image
from datetime import datetime

# ── Import des fonctions de feature engineering ──────────────────────────────
from features import (
    extraire_pct_rouille,
    extraire_rugosite,
    extraire_ratio_saturation,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiagKatsaka Madagascar",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { color: #111111; }

    .header-block {
        background: linear-gradient(135deg, #1E4A1E 0%, #3A7A3A 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .header-block,
    .header-block h1,
    .header-block * { color: #FFFFFF; }
    .header-block p { color: #D4EDD4; font-size: 0.95rem; }

    .result-sain {
        background: #E8F5E9;
        border-left: 6px solid #2E7D32;
        border-radius: 8px;
        padding: 20px 24px;
        font-size: 1.3rem;
        font-weight: bold;
        color: #1B5E20;
        margin: 16px 0;
    }
    .result-malade {
        background: #FFEBEE;
        border-left: 6px solid #C62828;
        border-radius: 8px;
        padding: 20px 24px;
        font-size: 1.3rem;
        font-weight: bold;
        color: #B71C1C;
        margin: 16px 0;
    }

    .feature-badge {
        display: inline-block;
        background: #E0E7FF;
        border: 1px solid #A5B4FC;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        color: #1E1B4B;
        margin: 4px 4px 4px 0;
    }

    .hist-label-sain   { color: #2E7D32; font-weight: bold; font-size: 0.85rem; }
    .hist-label-malade { color: #C62828; font-weight: bold; font-size: 0.85rem; }
    .hist-ts           { color: #616161; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
DOSSIER_UPLOADS = "uploads"
MODELE_PATH     = "models/rf_sklearn.pkl"
os.makedirs(DOSSIER_UPLOADS, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE (mis en cache)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def charger_modele():
    if not os.path.exists(MODELE_PATH):
        st.error(
            f"Modèle introuvable : {MODELE_PATH}\n\n"
            "Veuillez d'abord exécuter : `python3 partie3_modeles.py`"
        )
        st.stop()
    with open(MODELE_PATH, 'rb') as f:
        return pickle.load(f)


modele = charger_modele()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — Historique des analyses
# ─────────────────────────────────────────────────────────────────────────────
if 'historique' not in st.session_state:
    # Format : liste de dict { chemin, label, pct_rouille, rugosite, ratio_sat, timestamp }
    st.session_state.historique = []


# ─────────────────────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>🌽 DiagKatsaka — Détection de la Rouille Polysora</h1>
    <p>Système d'aide au diagnostic </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Informations pédagogiques
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📘 À propos du modèle")
    st.info(
        "**Algorithme :** Random Forest (100 arbres, Gini)\n\n"
        "**Features extraites :**\n"
        "- `pct_rouille` — % pixels rouille (HSV)\n"
        "- `rugosite` — variance gradient Sobel\n"
        "- `ratio_saturation` — saturation zone verte\n\n"
        "**Entraîné sur :** 64 images\n"
        "**Accuracy (test) :** 100 %"
    )
    st.markdown("---")
    st.markdown("### ⚠️ Rouille Polysora")
    st.markdown(
        "*Puccinia polysora* est un champignon particulièrement "
        "dévastateur dans les zones **chaudes et humides** de Madagascar. "
        "Les pustules orangées apparaissent sur la face inférieure des feuilles."
    )
    st.markdown("---")
    if st.button("🗑️ Vider l'historique", width='stretch'):
        st.session_state.historique = []
        # Nettoyer le dossier uploads
        for f in os.listdir(DOSSIER_UPLOADS):
            os.remove(os.path.join(DOSSIER_UPLOADS, f))
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ─────────────────────────────────────────────────────────────────────────────
tab_analyse, tab_historique = st.tabs(["🔬 Analyse d'image", "🗂️ Historique des détections"])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — UPLOAD ET PRÉDICTION EN TEMPS RÉEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyse:
    st.markdown("#### Téléverser une photo de feuille de maïs")

    fichier = st.file_uploader(
        "Choisir une image (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        help="Prenez une photo nette de la feuille de maïs, de préférence en pleine lumière."
    )

    if fichier is not None:
        col_img, col_res = st.columns([1, 1], gap="large")

        # ── Lecture de l'image ────────────────────────────────────────────
        contenu = fichier.read()
        img_array = np.frombuffer(contenu, np.uint8)
        img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Impossible de lire l'image. Veuillez essayer un autre fichier.")
            st.stop()

        with col_img:
            st.markdown("**Image analysée**")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, width='stretch', caption=fichier.name)

        with col_res:
            st.markdown("**Extraction des caractéristiques**")

            with st.spinner("Extraction des features en cours..."):
                pct_rouille  = extraire_pct_rouille(img_bgr)
                rugosite     = extraire_rugosite(img_bgr)
                ratio_sat    = extraire_ratio_saturation(img_bgr)

            # Affichage des features
            st.markdown(
                f'<span class="feature-badge">🔴 pct_rouille = {pct_rouille:.4f}</span>'
                f'<span class="feature-badge">📐 rugosité = {rugosite:.2f}</span>'
                f'<span class="feature-badge">🟢 saturation = {ratio_sat:.4f}</span>',
                unsafe_allow_html=True
            )

            # Barres de progression visuelles
            st.markdown("")
            st.metric("% pixels rouille",    f"{pct_rouille*100:.2f} %")
            st.metric("Rugosité (Sobel var)", f"{rugosite:.1f}")
            st.metric("Saturation zone verte",f"{ratio_sat:.4f}")

            # ── Prédiction ─────────────────────────────────────────────
            X_new = np.array([[pct_rouille, rugosite, ratio_sat]])
            pred  = modele.predict(X_new)[0]
            proba = modele.predict_proba(X_new)[0]

            st.markdown("---")
            st.markdown("**Résultat du diagnostic**")

            if pred == 1:
                st.markdown(
                    '<div class="result-malade">'
                    '🚨 ATTENTION : Feuille Malade (Rouille Détectée)'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.error(
                    f"Probabilité de maladie : **{proba[1]*100:.1f}%**\n\n"
                    "➡️ Recommandation : Appliquer un fongicide à base de "
                    "propiconazole ou de tébuconazole. Isoler la zone atteinte."
                )
            else:
                st.markdown(
                    '<div class="result-sain">'
                    '✅ Feuille Saine — Aucune rouille détectée'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.success(
                    f"Probabilité de santé : **{proba[0]*100:.1f}%**\n\n"
                    "➡️ La feuille ne présente pas de pustules de rouille polysora. "
                    "Continuez la surveillance régulière."
                )

        # ── Sauvegarde dans l'historique ──────────────────────────────────
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        nom_sauvegarde = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{fichier.name}"
        chemin_sauvegarde = os.path.join(DOSSIER_UPLOADS, nom_sauvegarde)

        cv2.imwrite(chemin_sauvegarde, img_bgr)

        # Vérifier si cette image n'est pas déjà dans l'historique (même nom)
        noms_existants = [h['chemin'] for h in st.session_state.historique]
        if chemin_sauvegarde not in noms_existants:
            st.session_state.historique.insert(0, {
                'chemin'     : chemin_sauvegarde,
                'nom'        : fichier.name,
                'label'      : int(pred),
                'pct_rouille': pct_rouille,
                'rugosite'   : rugosite,
                'ratio_sat'  : ratio_sat,
                'timestamp'  : timestamp,
            })

        st.caption(f"Image sauvegardée dans l'historique · {timestamp}")

    else:
        # État vide : instructions
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#9E9E9E;">
            <div style="font-size:4rem;">🌿</div>
            <div style="font-size:1.1rem; margin-top:12px;">
                Glissez-déposez ou cliquez sur le bouton ci-dessus<br>
                pour analyser une feuille de maïs.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — GALERIE HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════════
with tab_historique:
    historique = st.session_state.historique

    if not historique:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#9E9E9E;">
            <div style="font-size:3rem;">📂</div>
            <div style="margin-top:12px;">Aucune analyse effectuée pour l'instant.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Compteurs résumés
        nb_malades = sum(1 for h in historique if h['label'] == 1)
        nb_saines  = len(historique) - nb_malades

        m1, m2, m3 = st.columns(3)
        m1.metric("Total analysées",  len(historique))
        m2.metric("🟢 Saines",        nb_saines)
        m3.metric("🔴 Malades",       nb_malades, delta=f"{nb_malades/len(historique)*100:.0f}% de maladie")

        st.markdown("---")
        st.markdown(f"#### {len(historique)} analyse(s) précédente(s)")

        # Galerie en colonnes de 4
        NB_COL = 4
        for i in range(0, len(historique), NB_COL):
            cols = st.columns(NB_COL)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(historique):
                    break
                h = historique[idx]
                with col:
                    # Charger et afficher la miniature
                    if os.path.exists(h['chemin']):
                        img = Image.open(h['chemin'])
                        col.image(img, width='stretch')
                    else:
                        col.markdown("*(image manquante)*")

                    # Label coloré
                    if h['label'] == 1:
                        col.markdown(
                            '<p class="hist-label-malade">🚨 MALADE</p>',
                            unsafe_allow_html=True
                        )
                    else:
                        col.markdown(
                            '<p class="hist-label-sain">✅ SAINE</p>',
                            unsafe_allow_html=True
                        )

                    col.caption(
                        f"🔴 {h['pct_rouille']*100:.2f}%  "
                        f"📐 {h['rugosite']:.0f}\n"
                        f"🕒 {h['timestamp']}"
                    )
