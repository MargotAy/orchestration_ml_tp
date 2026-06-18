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
import altair as alt
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

PAGE_LABELS: dict[str, str] = {
    "accueil": "Accueil",
    "rapide": "Estimation rapide",
    "complet": "Estimation complète",
    "metrics": "Métriques",
    "history": "Historique",
}

PAGE_ICONS: dict[str, str] = {
    "accueil": "🏠",
    "rapide": "⚡",
    "complet": "📋",
    "metrics": "📈",
    "history": "📊",
}

PAGE_KEYS = list(PAGE_LABELS.keys())

AUTHOR_NAME = "Margot AYAZ"
AUTHOR_CLASS = "5IABD2"
PROFESSOR_NAME = "Lewis HOUNKPEVI"
GITHUB_REPO = "https://github.com/MargotAy/orchestration_ml_tp"
KAGGLE_DATASET = (
    "https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data"
)

# ---------------------------------------------------------------------------
# Page config & styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CardioPredict",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --accent: #e11d48;
        --accent-dark: #be123c;
        --accent-soft: #fff1f2;
        --accent-glow: #fecdd3;
        --text-muted: #78716c;
        --border: rgba(244, 63, 94, 0.32);
        --surface: rgba(255, 255, 255, 0.82);
    }
    .stApp {
        background: linear-gradient(
            165deg,
            #fff5f5 0%,
            #ffffff 35%,
            #fff8f8 70%,
            #ffffff 100%
        );
    }
    section[data-testid="stSidebar"] { display: none; }
    section[data-testid="stMain"] > div { max-width: 100%; }
    .main .block-container {
        padding-top: 1.25rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 100%;
    }
    .top-brand {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--accent-dark);
        margin-bottom: 0.25rem;
    }
    .author-badge {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--accent-dark);
        background: var(--accent-soft);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        margin-bottom: 1.25rem;
    }
    .context-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        margin: 1rem 0 1.5rem 0;
        font-size: 0.92rem;
        line-height: 1.65;
        color: #44403c;
    }
    .context-box a { color: var(--accent-dark); text-decoration: none; font-weight: 500; }
    .context-box a:hover { text-decoration: underline; }
    /* Navigation horizontale (style onglets) */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.65rem;
        margin-bottom: 1.25rem;
    }
    div[data-testid="stRadio"] label {
        background: transparent !important;
        padding: 0.55rem 1rem !important;
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        color: #57534e !important;
        font-weight: 500;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: var(--accent-soft) !important;
        color: var(--accent-dark) !important;
        border-color: rgba(244, 63, 94, 0.22) !important;
    }
    /* Cartes cliquables de l'accueil */
    .feature-cards [data-testid="stButton"] button {
        min-height: 11rem;
        white-space: pre-wrap;
        text-align: left;
        padding: 1.2rem 1.3rem !important;
        line-height: 1.55;
        font-size: 0.875rem !important;
        background: rgba(255, 255, 255, 0.92) !important;
        border: 1px solid rgba(244, 63, 94, 0.28) !important;
        color: #44403c !important;
        border-radius: 12px !important;
        transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
    }
    .feature-cards [data-testid="stButton"] button:hover {
        border-color: var(--accent) !important;
        background: var(--accent-soft) !important;
        box-shadow: 0 4px 14px rgba(225, 29, 72, 0.08);
        color: #1c1917 !important;
    }
    .feature-cards [data-testid="stButton"] button p {
        font-size: 0.875rem !important;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .page-header { margin-bottom: 1.75rem; }
    .page-header .page-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 1.75rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0 0 0.35rem 0;
        line-height: 1.2;
        color: #1c1917;
    }
    .page-header h1 {
        font-size: 1.75rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0 0 0.35rem 0;
        line-height: 1.2;
    }
    .page-header p {
        color: var(--text-muted);
        margin: 0;
        font-size: 1rem;
        line-height: 1.5;
    }
    .page-header .accent-bar {
        width: 2.5rem;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent-glow));
        border-radius: 2px;
        margin-bottom: 1rem;
    }
    .features {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.75rem 0;
    }
    @media (max-width: 768px) {
        .features { grid-template-columns: 1fr; }
    }
    .feature {
        display: flex;
        flex-direction: column;
        min-height: 11rem;
        padding: 1.25rem 1.35rem;
        border: 1px solid rgba(244, 63, 94, 0.28);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.92);
    }
    .feature-top {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.75rem;
    }
    .feature-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 2.1rem;
        height: 2.1rem;
        font-size: 1.1rem;
        border-radius: 9px;
        background: var(--accent-soft);
        flex-shrink: 0;
    }
    .feature-tag {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--accent-dark);
        margin-bottom: 0;
    }
    .feature h3 {
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        line-height: 1.3;
        color: #1c1917;
    }
    .feature p {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin: 0;
        line-height: 1.55;
        flex-grow: 1;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: var(--text-muted);
        border-left: 3px solid var(--accent-glow);
        padding: 0.5rem 0 0.5rem 1rem;
        margin-top: 2rem;
    }
    .intro-text {
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.65;
        margin: 0 0 1.25rem 0;
    }
    .cta-text {
        font-size: 1rem;
        line-height: 1.6;
        margin: 0.5rem 0 1rem 0;
        color: #44403c;
    }
    /* Lisibilité : texte sombre sur fond clair (thème Streamlit forcé en light) */
    .stApp, .stApp [data-testid="stAppViewContainer"] {
        color: #1c1917;
    }
    .main p, .main li, .main label, .main h1, .main h2, .main h3, .main h4,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #44403c;
    }
    .stTabs [data-baseweb="tab"] {
        color: #57534e !important;
    }
    div[data-testid="stMetricValue"] {
        color: #1c1917 !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #78716c !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border-color: rgba(244, 63, 94, 0.25) !important;
    }
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid var(--border);
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary span {
        color: #44403c !important;
    }
    .stCaption, .stCaption p, small {
        color: #78716c !important;
    }
    /* Boutons primaires (Lancer la prédiction) */
    button[kind="primary"],
    [data-testid="stBaseButton-primary"] button,
    [data-testid="stFormSubmitButton"] button {
        background-color: #be123c !important;
        color: #ffffff !important;
        border: 1px solid #be123c !important;
        font-weight: 600;
    }
    button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #9f1239 !important;
        border-color: #9f1239 !important;
        color: #ffffff !important;
    }
    button[kind="primary"] p,
    [data-testid="stBaseButton-primary"] button p,
    [data-testid="stFormSubmitButton"] button p {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DIABETIC_OPTIONS = ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"]
GEN_HEALTH_OPTIONS = ["Poor", "Fair", "Good", "Very good", "Excellent"]

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


def page_header(title: str, subtitle: str, icon: str = "") -> None:
    title_html = (
        f'<div class="page-title"><span>{icon}</span>{title}</div>'
        if icon
        else f"<h1>{title}</h1>"
    )
    st.markdown(
        f"""
        <div class="page-header">
            <div class="accent-bar"></div>
            {title_html}
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def intro_text(text: str) -> None:
    st.markdown(f'<p class="intro-text">{text}</p>', unsafe_allow_html=True)


FEATURE_CARDS: list[tuple[str, str, str, str]] = [
    (
        "rapide",
        "⚡ Estimation rapide",
        "7 critères essentiels : âge, sexe, tabac, IMC, diabète, activité physique et santé générale.",
    ),
    (
        "complet",
        "📋 Estimation complète",
        "Les 17 variables du modèle pour une prédiction plus personnalisée et affinée.",
    ),
    (
        "metrics",
        "📈 Métriques MLflow",
        "Performances des modèles entraînés : ROC-AUC, F1, précision et rappel.",
    ),
]


def render_feature_cards() -> None:
    """Cartes cliquables de l'accueil → changent l'onglet actif."""
    st.markdown('<div class="feature-cards">', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (page_key, title, description) in zip(cols, FEATURE_CARDS, strict=True):
        with col:
            label = f"{title}\n\n{description}\n\n→ Accéder"
            if st.button(label, key=f"nav_card_{page_key}", use_container_width=True, type="secondary"):
                st.session_state.nav_page = page_key
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


METRIC_DEFINITIONS: list[dict[str, str | float]] = [
    {
        "key": "roc_auc",
        "label": "ROC-AUC",
        "definition": (
            "Aire sous la courbe ROC : mesure la capacité du modèle à **distinguer** "
            "les patients avec maladie cardiaque de ceux sans, quel que soit le seuil de décision."
        ),
        "interpretation": (
            "**Quand c'est bon ?** ≥ 0,80 : excellente discrimination. "
            "Entre 0,70 et 0,80 : correct. En dessous de 0,70 : le modèle peine à séparer les classes."
        ),
        "good": 0.80,
        "ok": 0.70,
    },
    {
        "key": "f1_score",
        "label": "F1-score",
        "definition": (
            "Moyenne harmonique de la **précision** et du **rappel**. "
            "Synthèse utile quand les classes sont déséquilibrées (peu de malades dans les données)."
        ),
        "interpretation": (
            "**Quand c'est bon ?** ≥ 0,80 : bon équilibre global. "
            "Entre 0,70 et 0,80 : acceptable. En dessous de 0,70 : performances insuffisantes."
        ),
        "good": 0.80,
        "ok": 0.70,
    },
    {
        "key": "precision_score",
        "label": "Précision",
        "definition": (
            "Parmi les patients prédits **à risque**, quelle proportion l'est **réellement** ? "
            "Une précision élevée limite les **faux positifs** (alertes inutiles)."
        ),
        "interpretation": (
            "**Quand c'est bon ?** ≥ 0,75 : peu de fausses alertes. "
            "Entre 0,65 et 0,75 : modéré. En dessous de 0,65 : trop de patients sains classés à risque."
        ),
        "good": 0.75,
        "ok": 0.65,
    },
    {
        "key": "recall_score",
        "label": "Rappel",
        "definition": (
            "Parmi les patients **réellement malades**, quelle proportion le modèle **détecte** ? "
            "Un rappel élevé limite les **faux négatifs** — crucial en contexte médical."
        ),
        "interpretation": (
            "**Quand c'est bon ?** ≥ 0,80 : la majorité des malades sont repérés. "
            "Entre 0,70 et 0,80 : correct. En dessous de 0,70 : trop de malades passent inaperçus."
        ),
        "good": 0.80,
        "ok": 0.70,
    },
]


def metric_verdict(value: float, good: float, ok: float) -> tuple[str, str]:
    if value >= good:
        return "Bon", "success"
    if value >= ok:
        return "Correct", "warning"
    return "À améliorer", "error"


def render_metric_explainer(metric_key: str, value: float) -> None:
    info = next(m for m in METRIC_DEFINITIONS if m["key"] == metric_key)
    label = str(info["label"])
    verdict, level = metric_verdict(value, float(info["good"]), float(info["ok"]))
    message = f"**{label}** — {value:.3f} · *{verdict}*"

    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)

    st.markdown(str(info["definition"]))
    st.caption(str(info["interpretation"]))


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
    with st.container(border=True):
        st.markdown("**⚖️ Indice de masse corporelle**")
        st.caption(
            "Vous ne connaissez pas votre IMC ? On vous aide : entrez votre poids et votre taille, "
            "ou saisissez-le directement si vous l'avez déjà."
        )
        method = st.radio(
            "Comment renseigner votre IMC ?",
            ["Calculer (poids & taille)", "Saisir directement"],
            horizontal=True,
            key=f"{key}_bmi_method",
            label_visibility="collapsed",
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
            st.caption(f"IMC estimé : **{bmi}** · {bmi_category(bmi)}")
            with st.expander("Formule"):
                st.markdown(f"`{weight} ÷ ({height / 100:.2f})² = {bmi}`")
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


def call_predict(api_url: str, payload: dict[str, float | str]) -> dict | None:
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
    mode_label = "rapide" if mode == "rapide" else "complet"

    with st.container(border=True):
        st.markdown("**🔮 Résultat**")
        if is_risk:
            st.error(f"⚠️ **{label}** — probabilité estimée : **{risk_pct:.1f} %**")
        else:
            st.success(f"✅ **{label}** — probabilité estimée : **{risk_pct:.1f} %**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Diagnostic", label)
        with col2:
            st.metric("Probabilité", f"{risk_pct:.1f} %")
        with col3:
            st.metric("Prédiction", "Positive" if prediction == 1 else "Négative")

        st.progress(min(risk_pct / 100, 1.0))
        st.caption(f"Mode {mode_label}")

        if mode == "rapide":
            st.caption("Pour plus de précision, passez à l'onglet **Estimation complète** en haut de page.")


def render_metrics_chart(chart_df: pd.DataFrame) -> None:
    """Graphique en barres au style clair, cohérent avec le thème cardiaque."""
    df_long = chart_df.reset_index().melt(id_vars="Modèle", var_name="Métrique", value_name="Score")
    chart = (
        alt.Chart(df_long)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Modèle:N", title=None, axis=alt.Axis(labelColor="#44403c")),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Métrique:N",
                scale=alt.Scale(range=["#be123c", "#e11d48", "#fb7185", "#fda4af"]),
                legend=alt.Legend(labelColor="#44403c", titleColor="#44403c"),
            ),
            xOffset="Métrique:N",
        )
        .properties(height=320)
        .configure_view(fill="#fff8f8", stroke="#fecdd3")
        .configure_axis(gridColor="#fecdd3", domainColor="#fecdd3", labelColor="#44403c", titleColor="#57534e")
        .configure_title(color="#1c1917")
    )
    st.altair_chart(chart, use_container_width=True)


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
# Layout & pages
# ---------------------------------------------------------------------------

st.markdown('<div class="top-brand">🫀 CardioPredict</div>', unsafe_allow_html=True)

with st.expander("⚙️ Configuration technique", expanded=False):
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        api_url = st.text_input("URL de l'API", value=API_URL)
    with cfg2:
        mlflow_uri = st.text_input("MLflow tracking URI", value=MLFLOW_TRACKING_URI)
    with cfg3:
        mlflow_ui = st.text_input("MLflow UI", value=MLFLOW_UI_URL)

if "nav_page" not in st.session_state:
    st.session_state.nav_page = PAGE_KEYS[0]

st.radio(
    "Navigation",
    options=PAGE_KEYS,
    format_func=lambda k: f"{PAGE_ICONS[k]} {PAGE_LABELS[k]}",
    horizontal=True,
    label_visibility="collapsed",
    key="nav_page",
)

page = st.session_state.nav_page

if page == "accueil":
    page_header(
        "CardioPredict",
        "Estimation du risque de maladie cardiaque à partir d'indicateurs de santé.",
        icon="🫀",
    )

    st.markdown(
        f'<span class="author-badge">👤 {AUTHOR_NAME} · {AUTHOR_CLASS} · 2026</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="context-box">
        <strong>Contexte du projet</strong><br><br>
        Cette application a été réalisée dans le cadre du cours de <strong>{PROFESSOR_NAME}</strong>.
        Elle illustre un pipeline MLOps complet : préparation des données, entraînement de modèles,
        suivi via MLflow, API FastAPI et interface Streamlit.<br><br>
        📂 <strong>Dépôt GitHub :</strong>
        <a href="{GITHUB_REPO}" target="_blank">{GITHUB_REPO}</a><br>
        📊 <strong>Jeu de données (Kaggle) :</strong>
        <a href="{KAGGLE_DATASET}" target="_blank">Personal Key Indicators of Heart Disease (BRFSS 2020)</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Application pédagogique de **classification binaire** : un modèle estime la probabilité
        de maladie cardiaque à partir du profil de santé d'un patient (âge, IMC, tabac, diabète,
        activité physique, etc.).
        """
    )

    st.markdown(
        '<p class="cta-text">'
        "<strong>Envie de voir la prédiction en action ?</strong> "
        "Cliquez sur une carte ci-dessous pour accéder directement à la section concernée."
        "</p>",
        unsafe_allow_html=True,
    )

    render_feature_cards()

    with st.expander("🔬 À propos du pipeline", expanded=False):
        st.markdown(
            """
            - 🧹 Préparation et encodage des variables
            - 🤖 Entraînement (Random Forest, XGBoost, LightGBM) avec MLflow
            - 🚀 API FastAPI pour l'inférence en production
            - 🖥️ Interface Streamlit pour tester des scénarios
            """
        )

    st.markdown(
        '<p class="disclaimer">⚕️ Outil à visée pédagogique — ne remplace pas un avis médical professionnel.</p>',
        unsafe_allow_html=True,
    )

elif page == "rapide":
    page_header(
        "Estimation rapide",
        "Les critères secondaires sont complétés par des valeurs typiques de la population.",
        icon="⚡",
    )

    intro_text(
        "Idéal pour commencer : répondez à quelques questions essentielles et obtenez "
        "une première estimation en quelques secondes. Les critères non renseignés sont "
        "complétés automatiquement avec des valeurs typiques."
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

        submitted = st.form_submit_button("Lancer la prédiction", type="primary", use_container_width=True)

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
        result = call_predict(api_url, payload)
        if result:
            display_result(result, "rapide")

elif page == "complet":
    page_header(
        "Estimation complète",
        "Renseignez l'ensemble des critères pour affiner la prédiction.",
        icon="📋",
    )

    intro_text(
        "Pour une prédiction plus précise, renseignez l'ensemble des critères ci-dessous. "
        "Chaque information compte : plus votre profil est détaillé, plus le résultat "
        "reflète votre situation."
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

        submitted = st.form_submit_button("Lancer la prédiction", type="primary", use_container_width=True)

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
        result = call_predict(api_url, payload)
        if result:
            display_result(result, "complet")

elif page == "metrics":
    page_header(
        "Métriques MLflow",
        "Performances des modèles entraînés.",
        icon="📈",
    )

    intro_text(
        "Ces indicateurs mesurent la qualité du modèle sur un jeu de test. "
        "Plus une métrique est élevée (proche de 1), meilleure est la performance — "
        "avec des nuances selon l'indicateur, détaillées ci-dessous."
    )

    st.link_button("📊 Ouvrir MLflow", mlflow_ui)

    try:
        models_df = load_mlflow_runs(mlflow_uri, MLFLOW_EXPERIMENT)
        eval_metrics = load_evaluation_metrics(mlflow_uri, MLFLOW_EXPERIMENT)
    except Exception as exc:
        st.error(f"Impossible de lire MLflow ({mlflow_uri}) : {exc}")
        st.info(
            "Lancez MLflow avec `make mlflow` ou `docker compose up -d mlflow`, "
            "puis vérifiez l'URI dans la configuration technique."
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
                render_metrics_chart(chart_df)

        if eval_metrics:
            st.subheader("Dernière évaluation (jeu de test)")
            m1, m2, m3, m4 = st.columns(4)
            roc = float(eval_metrics.get("roc_auc", 0))
            f1 = float(eval_metrics.get("f1_score", 0))
            prec = float(eval_metrics.get("precision_score", 0))
            rec = float(eval_metrics.get("recall_score", 0))

            with m1:
                st.metric("ROC-AUC", f"{roc:.3f}")
            with m2:
                st.metric("F1-score", f"{f1:.3f}")
            with m3:
                st.metric("Précision", f"{prec:.3f}")
            with m4:
                st.metric("Rappel", f"{rec:.3f}")

            st.caption(
                f"Échantillons évalués : {int(eval_metrics.get('example_count', 0)):,} · "
                f"Exactitude : {eval_metrics.get('accuracy_score', 0):.1%}"
            )

            st.subheader("Comprendre les résultats")
            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                with st.container(border=True):
                    render_metric_explainer("roc_auc", roc)
            with row1_c2:
                with st.container(border=True):
                    render_metric_explainer("f1_score", f1)

            row2_c1, row2_c2 = st.columns(2)
            with row2_c1:
                with st.container(border=True):
                    render_metric_explainer("precision_score", prec)
            with row2_c2:
                with st.container(border=True):
                    render_metric_explainer("recall_score", rec)

            with st.expander("📖 Glossaire rapide", expanded=False):
                for info in METRIC_DEFINITIONS:
                    st.markdown(f"**{info['label']}** — {info['definition']}")
                    st.caption(str(info["interpretation"]))
                    st.divider()
        else:
            st.info("Aucune évaluation `evaluate` terminée. Lancez `make evaluate` pour en générer une.")

elif page == "history":
    page_header(
        "Historique",
        "Journal des prédictions effectuées via l'API.",
        icon="📊",
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
