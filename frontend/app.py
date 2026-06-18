"""Frontend Streamlit : démonstrateur de classification cardiaque.

Lancement : `uv run streamlit run frontend/app.py`
Variables d'environnement : API_URL, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import mlflow
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "heart-disease-classification")
MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", "http://127.0.0.1:5000")

DEFAULT_VALUES: dict[str, float | str] = {
    "BMI": 27.34,
    "PhysicalHealth": 0.0,
    "MentalHealth": 0.0,
    "SleepTime": 7.0,
    "Smoking": "No",
    "AlcoholDrinking": "No",
    "Stroke": "No",
    "DiffWalking": "No",
    "Sex": "Female",
    "AgeCategory": "55-69",
    "Race": "White",
    "Diabetic": "No",
    "PhysicalActivity": "Yes",
    "GenHealth": "Very good",
    "Asthma": "No",
    "KidneyDisease": "No",
    "SkinCancer": "No",
}

PAGES: dict[str, str] = {
    "accueil": "🏠 Accueil",
    "rapide": "⚡ Estimation rapide",
    "complet": "📋 Estimation complète",
    "metrics": "📈 Métriques MLflow",
    "history": "📊 Historique",
}

DIABETIC_OPTIONS = ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"]
GEN_HEALTH_OPTIONS = ["Poor", "Fair", "Good", "Very good", "Excellent"]

# ---------------------------------------------------------------------------
# Page config & styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Prédiction cardiaque",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a8f 50%, #3d8b6e 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; font-size: 2rem; margin: 0 0 0.5rem 0; }
    .hero p { color: rgba(255,255,255,0.88); margin: 0; font-size: 1.05rem; }
    .info-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        background: #f8fafc;
        height: 100%;
    }
    .info-card h3 { margin-top: 0; color: #1e3a5f; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "accueil"

PAGE_KEYS = list(PAGES.keys())

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🫀 Navigation")
    selected_page = st.radio(
        "Aller à",
        options=PAGE_KEYS,
        format_func=lambda k: PAGES[k],
        index=PAGE_KEYS.index(st.session_state.nav_page),
    )
    st.session_state.nav_page = selected_page
    st.divider()
    st.markdown("### ⚙️ Configuration")
    api_url = st.text_input("URL de l'API", value=API_URL)
    mlflow_uri = st.text_input("MLflow tracking URI", value=MLFLOW_TRACKING_URI)
    mlflow_ui = st.text_input("MLflow UI (lien)", value=MLFLOW_UI_URL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_yes_no(value: str) -> str:
    return "Oui" if value == "Yes" else "Non"


def fmt_sex(value: str) -> str:
    return "Homme" if value == "Male" else "Femme"


def fmt_diabetic(value: str) -> str:
    return {
        "Yes": "Oui",
        "No": "Non",
        "No, borderline diabetes": "Non, limite",
        "Yes (during pregnancy)": "Oui (grossesse)",
    }[value]


def fmt_gen_health(value: str) -> str:
    return {
        "Poor": "Mauvaise",
        "Fair": "Passable",
        "Good": "Bonne",
        "Very good": "Très bonne",
        "Excellent": "Excellente",
    }[value]


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Insuffisance pondérale"
    if bmi < 25:
        return "Poids normal"
    if bmi < 30:
        return "Surpoids"
    return "Obésité"


def render_bmi_section(key: str, default: float = 27.3) -> float:
    """Calculateur IMC en temps réel (hors formulaire pour mise à jour live)."""
    st.markdown("**IMC (indice de masse corporelle)**")
    method = st.radio(
        "Comment renseigner votre IMC ?",
        ["Calculer à partir du poids et de la taille", "Je connais déjà mon IMC"],
        horizontal=True,
        key=f"{key}_bmi_method",
    )
    if method.startswith("Calculer"):
        c1, c2 = st.columns(2)
        with c1:
            weight = st.number_input(
                "Poids (kg)",
                min_value=30.0,
                max_value=300.0,
                value=70.0,
                step=0.5,
                key=f"{key}_weight",
            )
        with c2:
            height = st.number_input(
                "Taille (cm)",
                min_value=100.0,
                max_value=250.0,
                value=170.0,
                step=0.5,
                key=f"{key}_height",
            )
        bmi = round(weight / (height / 100) ** 2, 1)
        st.info(f"📐 Votre IMC estimé : **{bmi}** — {bmi_category(bmi)}")
        with st.expander("Comment est calculé l'IMC ?"):
            st.markdown(
                "L'IMC se calcule ainsi : **poids (kg) ÷ taille (m)²**\n\n"
                f"Exemple : {weight} ÷ ({height / 100:.2f})² = **{bmi}**"
            )
        return bmi
    return st.number_input(
        "IMC",
        min_value=10.0,
        max_value=60.0,
        value=default,
        step=0.1,
        key=f"{key}_bmi_direct",
    )


def build_payload(**kwargs: float | str) -> dict[str, float | str]:
    payload = dict(DEFAULT_VALUES)
    payload.update(kwargs)
    return payload


def call_predict(payload: dict[str, float | str]) -> dict | None:
    try:
        response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"Appel à l'API impossible : {exc}")
        return None


def display_result(result: dict, mode: str) -> None:
    proba = result["probability"]
    prediction = result["prediction"]
    is_risk = prediction == 1
    risk_pct = proba * 100

    label = "Risque élevé" if is_risk else "Risque faible"
    emoji = "⚠️" if is_risk else "✅"
    mode_label = "rapide (valeurs par défaut pour les critères secondaires)" if mode == "rapide" else "complet"

    st.markdown("### Résultat de la prédiction")

    if is_risk:
        st.error(f"{emoji} **{label}** — probabilité estimée : **{risk_pct:.1f} %**")
    else:
        st.success(f"{emoji} **{label}** — probabilité estimée : **{risk_pct:.1f} %**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diagnostic", label)
    with col2:
        st.metric("Probabilité", f"{risk_pct:.1f} %")
    with col3:
        st.metric("Classe prédite", "Maladie cardiaque" if prediction == 1 else "Pas de maladie")

    st.progress(min(risk_pct / 100, 1.0))
    st.caption(f"Prédiction effectuée en mode **{mode_label}**.")

    if mode == "rapide":
        st.warning(
            "Pour une estimation plus fine, utilisez l'onglet **Estimation complète** "
            "dans la barre latérale."
        )


@st.cache_data(ttl=60, show_spinner=False)
def load_mlflow_runs(tracking_uri: str, experiment_name: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.model_family != ''",
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )

    rows: list[dict] = []
    seen: set[str] = set()
    for run in runs:
        family = run.data.tags.get("model_family", run.info.run_name or "")
        if not family or family in seen:
            continue
        metrics = run.data.metrics
        if not any(k in metrics for k in ("roc_auc", "test_roc_auc", "cv_roc_auc")):
            continue
        seen.add(family)
        rows.append(
            {
                "Modèle": family.replace("_", " ").title(),
                "ROC-AUC (test)": metrics.get("roc_auc") or metrics.get("test_roc_auc"),
                "ROC-AUC (CV)": metrics.get("cv_roc_auc"),
                "F1": metrics.get("f1"),
                "Précision": metrics.get("precision"),
                "Rappel": metrics.get("recall"),
                "Run": run.info.run_name,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def load_evaluation_metrics(tracking_uri: str, experiment_name: str) -> dict[str, float] | None:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.run_name = 'evaluate' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return dict(runs[0].data.metrics)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

page = st.session_state.nav_page

if page == "accueil":
    st.markdown(
        """
        <div class="hero">
            <h1>🫀 Détection du risque cardiaque</h1>
            <p>Outil pédagogique de classification binaire basé sur des données de santé publique.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Contexte du projet")
    st.markdown(
        """
        Ce démonstrateur s'appuie sur le jeu de données **Heart Disease** (BRFSS 2020, CDC) :
        des milliers d'individus décrits par leur profil de santé (âge, IMC, tabac, diabète,
        activité physique, etc.) et une cible binaire indiquant la présence ou non d'une
        maladie cardiaque.

        Le pipeline ML comprend :
        - **Préparation** des données et encodage des variables catégorielles
        - **Entraînement** de modèles (Random Forest, XGBoost, LightGBM) avec suivi **MLflow**
        - **API FastAPI** pour servir le meilleur modèle en production
        - **Interface Streamlit** (cette application) pour tester des scénarios
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="info-card"><h3>⚡ Estimation rapide</h3>'
            "<p>7 critères clés : âge, sexe, tabac, IMC, diabète, activité physique, "
            "santé générale. Idéal pour un premier aperçu.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="info-card"><h3>📋 Estimation complète</h3>'
            "<p>Les 17 variables du modèle pour une prédiction plus personnalisée "
            "et potentiellement plus précise.</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="info-card"><h3>📈 Métriques MLflow</h3>'
            "<p>Consultez les performances des modèles entraînés : ROC-AUC, F1, "
            "précision, rappel.</p></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Commencer une estimation")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("⚡ Lancer une estimation rapide", type="primary", use_container_width=True):
            st.session_state.nav_page = "rapide"
            st.rerun()
    with b2:
        if st.button("📋 Lancer une estimation complète", use_container_width=True):
            st.session_state.nav_page = "complet"
            st.rerun()

    st.info(
        "⚠️ **Avertissement** : cet outil est à visée pédagogique. "
        "Il ne remplace en aucun cas un avis médical professionnel."
    )

elif page == "rapide":
    st.markdown(
        """
        <div class="hero">
            <h1>⚡ Estimation rapide</h1>
            <p>Renseignez les facteurs principaux — les autres critères seront estimés automatiquement.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bmi = render_bmi_section("rapide", default=27.3)

    with st.form("predict_form_rapide"):
        col_a, col_b = st.columns(2)
        with col_a:
            age_category = st.selectbox("Tranche d'âge", ["18-39", "40-54", "55-69", "70+"], index=2)
            sex = st.selectbox("Sexe", ["Male", "Female"], index=1, format_func=fmt_sex)
            smoking = st.selectbox("Tabagisme", ["Yes", "No"], format_func=fmt_yes_no)
        with col_b:
            diabetic = st.selectbox("Diabète", DIABETIC_OPTIONS, format_func=fmt_diabetic)
            physical_activity = st.selectbox("Activité physique régulière", ["Yes", "No"], format_func=fmt_yes_no)
            gen_health = st.selectbox("Santé générale perçue", GEN_HEALTH_OPTIONS, index=3, format_func=fmt_gen_health)

        submitted = st.form_submit_button("Lancer la prédiction rapide", type="primary", use_container_width=True)

    if submitted:
        payload = build_payload(
            AgeCategory=age_category,
            Sex=sex,
            Smoking=smoking,
            BMI=bmi,
            Diabetic=diabetic,
            PhysicalActivity=physical_activity,
            GenHealth=gen_health,
        )
        result = call_predict(payload)
        if result:
            display_result(result, "rapide")

elif page == "complet":
    st.markdown(
        """
        <div class="hero">
            <h1>📋 Estimation complète</h1>
            <p>Renseignez tous les critères pour affiner la prédiction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bmi = render_bmi_section("complet", default=27.3)

    with st.form("predict_form_complet"):
        st.markdown("**Autres mesures numériques**")
        c2, c3, c4 = st.columns(3)
        with c2:
            physical_health = st.number_input("Santé physique (jours/mois)", min_value=0.0, max_value=30.0, value=0.0)
        with c3:
            mental_health = st.number_input("Santé mentale (jours/mois)", min_value=0.0, max_value=30.0, value=0.0)
        with c4:
            sleep_time = st.number_input("Heures de sommeil", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

        st.markdown("**Mode de vie & antécédents**")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            smoking = st.selectbox("Tabagisme", ["Yes", "No"], format_func=fmt_yes_no)
            alcohol = st.selectbox("Consommation d'alcool", ["Yes", "No"], format_func=fmt_yes_no)
        with c6:
            stroke = st.selectbox("AVC antérieur", ["Yes", "No"], format_func=fmt_yes_no)
            diff_walking = st.selectbox("Difficulté à marcher", ["Yes", "No"], format_func=fmt_yes_no)
        with c7:
            physical_activity = st.selectbox("Activité physique", ["Yes", "No"], format_func=fmt_yes_no)
            diabetic = st.selectbox("Diabète", DIABETIC_OPTIONS, format_func=fmt_diabetic)
        with c8:
            gen_health = st.selectbox("Santé générale", GEN_HEALTH_OPTIONS, index=3, format_func=fmt_gen_health)

        st.markdown("**Profil démographique & pathologies**")
        c9, c10, c11, c12 = st.columns(4)
        with c9:
            sex = st.selectbox("Sexe", ["Male", "Female"], format_func=fmt_sex)
            age_category = st.selectbox("Tranche d'âge", ["18-39", "40-54", "55-69", "70+"], index=2)
        with c10:
            race = st.selectbox(
                "Origine ethnique",
                ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other", "Hispanic"],
            )
        with c11:
            asthma = st.selectbox("Asthme", ["Yes", "No"], format_func=fmt_yes_no)
            kidney = st.selectbox("Maladie rénale", ["Yes", "No"], format_func=fmt_yes_no)
        with c12:
            skin_cancer = st.selectbox("Cancer de la peau", ["Yes", "No"], format_func=fmt_yes_no)

        submitted = st.form_submit_button("Lancer la prédiction complète", type="primary", use_container_width=True)

    if submitted:
        payload = build_payload(
            BMI=bmi,
            PhysicalHealth=physical_health,
            MentalHealth=mental_health,
            SleepTime=sleep_time,
            Smoking=smoking,
            AlcoholDrinking=alcohol,
            Stroke=stroke,
            DiffWalking=diff_walking,
            Sex=sex,
            AgeCategory=age_category,
            Race=race,
            Diabetic=diabetic,
            PhysicalActivity=physical_activity,
            GenHealth=gen_health,
            Asthma=asthma,
            KidneyDisease=kidney,
            SkinCancer=skin_cancer,
        )
        result = call_predict(payload)
        if result:
            display_result(result, "complet")

elif page == "metrics":
    st.markdown(
        """
        <div class="hero">
            <h1>📈 Métriques MLflow</h1>
            <p>Performances des modèles entraînés et suivis dans l'expérience MLflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.link_button("Ouvrir l'interface MLflow complète ↗", mlflow_ui)

    try:
        models_df = load_mlflow_runs(mlflow_uri, MLFLOW_EXPERIMENT)
        eval_metrics = load_evaluation_metrics(mlflow_uri, MLFLOW_EXPERIMENT)
    except Exception as exc:
        st.error(f"Impossible de lire MLflow ({mlflow_uri}) : {exc}")
        st.info(
            "Lancez MLflow avec `make mlflow` ou `docker compose up -d mlflow`, "
            "puis vérifiez l'URI dans la barre latérale."
        )
    else:
        if models_df.empty:
            st.warning(
                f"Aucun run trouvé dans l'expérience **{MLFLOW_EXPERIMENT}**. "
                "Entraînez des modèles avec `make train-models` ou `make train-optuna`."
            )
        else:
            st.subheader("Comparaison des modèles")
            display_df = models_df.drop(columns=["Run"], errors="ignore")
            st.dataframe(
                display_df.style.format(
                    {
                        "ROC-AUC (test)": "{:.3f}",
                        "ROC-AUC (CV)": "{:.3f}",
                        "F1": "{:.3f}",
                        "Précision": "{:.3f}",
                        "Rappel": "{:.3f}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )

            chart_df = models_df.set_index("Modèle")[["ROC-AUC (test)", "F1", "Précision", "Rappel"]].dropna(how="all")
            if not chart_df.empty:
                st.subheader("Visualisation")
                st.bar_chart(chart_df)

        if eval_metrics:
            st.subheader("Dernière évaluation (jeu de test)")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("ROC-AUC", f"{eval_metrics.get('roc_auc', 0):.3f}")
            with m2:
                st.metric("F1-score", f"{eval_metrics.get('f1_score', 0):.3f}")
            with m3:
                st.metric("Précision", f"{eval_metrics.get('precision_score', 0):.3f}")
            with m4:
                st.metric("Rappel", f"{eval_metrics.get('recall_score', 0):.3f}")

            st.caption(
                f"Échantillons évalués : {int(eval_metrics.get('example_count', 0)):,} · "
                f"Exactitude : {eval_metrics.get('accuracy_score', 0):.1%}"
            )
        else:
            st.info("Aucune évaluation `evaluate` terminée. Lancez `make evaluate` pour en générer une.")

elif page == "history":
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Historique</h1>
            <p>Journal des prédictions effectuées via l'API.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        response = httpx.get(f"{api_url}/predictions", timeout=10.0)
        response.raise_for_status()
        rows = response.json()
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Aucune prédiction enregistrée pour le moment.")
    except httpx.HTTPError:
        st.info(
            "Aucun journal de prévisions disponible. "
            "Ajoutez un endpoint `GET /predictions` à l'API pour activer cette fonctionnalité."
        )
