
from pathlib import Path
from textwrap import dedent
import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from biomed.data import load_cohort, get_X_y
from biomed.modeling import load_model, load_metrics
from biomed.omics import (
    OMICS,
    omics_columns,
    group_summary,
    pathway_scores,
    target_prioritization,
)
from biomed.literature import search_pubmed, keywords, summarize
from biomed.explainability import generate_shap


st.set_page_config(
    page_title="OncoOmics Research Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# IMPORTANT:
# Remove leading indentation from every HTML block.
# This prevents Streamlit from displaying HTML as code.

def render_html(content):
    """Render HTML directly instead of passing it through Markdown."""
    st.html(dedent(content).strip())


# ============================================================
# 2. LOAD PROJECT DATA
# ============================================================

@st.cache_data
def load_cohort_data():
    return load_cohort()


@st.cache_resource
def load_trained_model():
    return load_model()


@st.cache_data
def load_experiment_metrics():
    return load_metrics()


df = load_cohort_data()
X, y, features = get_X_y(df)

MODEL = load_trained_model()
METRICS = load_experiment_metrics()
TEST = METRICS["test"]

IMPORTANCE_PATH = ROOT / "reports" / "permutation_importance.csv"

if not IMPORTANCE_PATH.exists():
    st.error(
        "Permutation importance file is missing. "
        "Run the project training pipeline first."
    )
    st.stop()

importance = pd.read_csv(IMPORTANCE_PATH)


# ============================================================
# 3. GLOBAL CSS
# ============================================================

render_html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --background: #080f1b;
    --panel: #101b2b;
    --panel-light: #142238;
    --border: #263b54;
    --text: #e8f0fa;
    --muted: #8fa4bb;
    --teal: #29c4b4;
    --blue: #438cf5;
    --green: #53d6a3;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(8,15,27,.96);
}

.block-container {
    max-width: 1900px;
    padding: 1rem 1.2rem 2rem;
}

#MainMenu, footer {
    visibility: hidden;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: #edf5ff !important;
}

p, li, label {
    color: #b6c6d8;
}

[data-testid="stSidebar"] {
    background: #080f1b;
    border-right: 1px solid #20334b;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #101b2b;
    border: 1px solid #263b54;
    border-radius: 8px;
    padding: 7px 9px;
    margin: 4px 0;
    font-size: 12px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #123c4b;
    border-color: #29c4b4;
}

.stTextInput input,
.stTextArea textarea,
[data-baseweb="select"] > div {
    background: #0c1726 !important;
    color: #e8f0fa !important;
    border-color: #2b4059 !important;
    border-radius: 8px !important;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(100deg,#176e83,#168d87);
    color: white;
    border: 1px solid #248e91;
    border-radius: 8px;
    font-weight: 700;
    min-height: 38px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #53d6a3;
    color: white;
}

[data-testid="stMetric"] {
    background: #101d2e;
    border: 1px solid #263b54;
    border-radius: 10px;
    padding: 12px;
}

[data-testid="stMetricLabel"] {
    color: #91a7be !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: .08em;
}

[data-testid="stMetricValue"] {
    color: #eaf4ff !important;
    font-family: Manrope;
    font-size: 23px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: #0c1726;
    padding: 5px;
    border-radius: 9px;
}

.stTabs [data-baseweb="tab"] {
    background: #142238;
    border-radius: 6px;
    padding: 8px 10px;
    color: #a9bdd1;
    font-size: 12px;
}

.stTabs [aria-selected="true"] {
    background: #155267 !important;
    color: white !important;
}

[data-testid="stExpander"] {
    background: #101d2e;
    border: 1px solid #263b54;
    border-radius: 8px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #263b54;
    border-radius: 8px;
}

[data-testid="stAlert"] {
    background: #142238;
    border: 1px solid #30445d;
}

.hero {
    background: linear-gradient(
        115deg,
        #102b3a,
        #145d68 58%,
        #168c8a
    );
    padding: 28px;
    border: 1px solid #286b78;
    border-radius: 14px;
    margin-bottom: 15px;
    box-shadow: 0 12px 35px rgba(0,0,0,.2);
}

.hero-eyebrow {
    font-size: 10px;
    letter-spacing: 2px;
    color: #9ce9dc;
    font-weight: 800;
}

.hero h1 {
    font-size: 28px;
    margin: 8px 0;
    color: white !important;
}

.hero p {
    font-size: 13px;
    color: #d6e8ec;
    margin: 0;
}

.kicker {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #60d2c3;
    font-weight: 800;
}

.panel-title {
    font-family: Manrope;
    font-size: 15px;
    font-weight: 800;
    color: #eaf2fb;
    margin-bottom: 7px;
}

.chat-user {
    background: #14283e;
    border: 1px solid #29425d;
    border-radius: 9px;
    padding: 12px;
    color: #dce9f6;
    font-size: 12px;
    line-height: 1.7;
    margin: 10px 0;
}

.chat-ai {
    background: #0e1a29;
    border-left: 3px solid #29c4b4;
    border-radius: 5px;
    padding: 13px;
    color: #c7d5e4;
    font-size: 12px;
    line-height: 1.7;
    margin: 10px 0;
}

.pill {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 20px;
    background: #123b42;
    color: #75dfca;
    font-size: 9px;
    font-weight: 700;
}

.recent-card {
    background: #101b2b;
    border: 1px solid #263b54;
    border-radius: 8px;
    padding: 10px;
    margin: 8px 0;
}

.footer-note {
    font-size: 10px;
    color: #71869d;
    border-top: 1px solid #20334b;
    padding-top: 12px;
    margin-top: 18px;
}

</style>
""")


# ============================================================
# 4. SIDEBAR
# ============================================================

PAGES = [
    "Multi-Omics Analysis",
    "Pathway Analysis",
    "Target Prioritization",
    "Cohort Explorer",
    "Literature Intelligence",
    "Explainable AI",
    "Model Evaluation"
]

with st.sidebar:

    render_html("""
    <div style="font-size:27px">🧬</div>
    <div style="font:800 19px Manrope;color:#f3f7fc">
        OncoOmics <span style="color:#39c8b7">RI</span>
    </div>
    <div style="font-size:9px;color:#8fa4bb;letter-spacing:1px">
        MULTI-OMICS RESEARCH STUDIO
    </div>
    """)

    st.write("")

    if st.button(
        "＋ Create research workspace",
        use_container_width=True
    ):
        for key in [
            "pubmed_records",
            "text_summary",
            "shap_status"
        ]:
            st.session_state.pop(key, None)

        st.rerun()

    search = st.text_input(
        "⌕ Search workspace history",
        placeholder="Search analyses..."
    )

    page = st.radio(
        "WORKSPACE",
        PAGES,
        key="page_navigation"
    )

    st.markdown("---")

    render_html("""
    <div class="kicker">RECENT RESEARCH</div>

    <div class="recent-card">
        <b style="font-size:11px">Ovarian cancer cohort</b>
        <div style="font-size:10px;color:#8fa4bb">
            Multi-omics analysis
        </div>
    </div>

    <div class="recent-card">
        <b style="font-size:11px">Pathway activity study</b>
        <div style="font-size:10px;color:#8fa4bb">
            Pathway analysis
        </div>
    </div>

    <div class="recent-card">
        <b style="font-size:11px">Candidate target ranking</b>
        <div style="font-size:10px;color:#8fa4bb">
            Target prioritization
        </div>
    </div>

    <div class="recent-card">
        <b style="font-size:11px">PubMed evidence review</b>
        <div style="font-size:10px;color:#8fa4bb">
            Literature intelligence
        </div>
    </div>
    """)

    st.markdown("---")

    render_html("""
    <div class="kicker">DATASET</div>
    <div style="font-size:11px;margin-top:7px">
        Synthetic ovarian multi-omics cohort
    </div>
    <div style="font-size:10px;color:#8fa4bb">
        RNA · Protein · Metabolite · Immune
    </div>
    """)


# ============================================================
# 5. MAIN HEADER
# ============================================================

render_html(f"""
<div class="hero">

    <div class="hero-eyebrow">
        BIOMEDICAL AI / RESEARCH WORKSPACE
    </div>

    <h1>{page}</h1>

    <p>
        Integrated molecular data analysis, pathway exploration
        and evidence-guided research prioritization.
    </p>

</div>
""")

render_html("""
<div class="chat-ai">
    <b>Research data notice</b> · This workspace uses synthetic
    demonstration data. It is not a medical device, diagnostic
    tool or clinical decision-support system.
</div>
""")

left, right = st.columns([0.82, 1.18], gap="medium")


# ============================================================
# 6. LEFT PANEL - RESEARCH ASSISTANT
# ============================================================

with left:

    render_html("""
    <div class="kicker">RESEARCH ASSISTANT</div>
    <div class="panel-title" style="font-size:18px;margin-top:6px">
        Omics Research Copilot
    </div>
    <div style="font-size:11px;color:#8fa4bb">
        Explore the cohort and research findings.
    </div>
    """)

    # --------------------------------------------------------
    # MULTI-OMICS OVERVIEW
    # --------------------------------------------------------

    if page == "Multi-Omics Analysis":

        render_html("""
        <div class="chat-user">
            Explore the synthetic ovarian cancer research cohort
            and its molecular data modalities.
        </div>
        """)

        st.markdown("### Workspace Summary")

        st.write(
            f"Loaded **{len(df)} synthetic samples** across four "
            "data modalities: RNA expression, protein abundance, "
            "metabolite profiles and immune-cell composition."
        )

        case_count = int(y.sum())
        control_count = int((1 - y).sum())

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Case-like samples",
                case_count
            )

        with c2:
            st.metric(
                "Control-like samples",
                control_count
            )

        st.markdown("---")
        st.markdown("**Quick Actions**")

        if st.button("Review pathway activity", use_container_width=True):
            st.session_state["page_navigation"] = "Pathway Analysis"
            st.rerun()

        if st.button("Prioritize candidate features", use_container_width=True):
            st.session_state["page_navigation"] = "Target Prioritization"
            st.rerun()

        if st.button("Open model evaluation", use_container_width=True):
            st.session_state["page_navigation"] = "Model Evaluation"
            st.rerun()

    # --------------------------------------------------------
    # PATHWAY ANALYSIS
    # --------------------------------------------------------

    elif page == "Pathway Analysis":

        render_html("""
        <div class="chat-user">
            Explore molecular pathway activity across the
            synthetic research cohort.
        </div>
        """)

        st.markdown("**Pathway Exploration**")

        pscore = pathway_scores(df)

        st.dataframe(
            pscore,
            hide_index=True,
            use_container_width=True
        )

        selected = st.multiselect(
            "Select pathways",
            pscore["Pathway"].tolist(),
            default=pscore["Pathway"].tolist()[:3]
        )

        st.write("Selected pathways:", ", ".join(selected))

    # --------------------------------------------------------
    # TARGET PRIORITIZATION
    # --------------------------------------------------------

    elif page == "Target Prioritization":

        render_html("""
        <div class="chat-user">
            Explore ranked molecular candidates using
            statistical differences and model importance.
        </div>
        """)

        ranking = target_prioritization(
            df,
            importance
        )

        st.markdown("**Candidate ranking**")

        st.dataframe(
            ranking.head(12),
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download candidate ranking",
            ranking.to_csv(index=False),
            "candidate_ranking.csv",
            "text/csv",
            use_container_width=True
        )

    # --------------------------------------------------------
    # COHORT EXPLORER
    # --------------------------------------------------------

    elif page == "Cohort Explorer":

        render_html("""
        <div class="chat-user">
            Inspect individual synthetic samples and
            their molecular measurements.
        </div>
        """)

        sid = st.selectbox(
            "Select sample",
            df.sample_id.tolist()
        )

        sample = df[df.sample_id == sid].iloc[0]

        st.markdown("**Sample information**")

        st.metric(
            "Cohort group",
            str(sample.cohort_group)
        )

        st.metric(
            "Synthetic age",
            f"{sample.age_years:.1f}"
        )

        for layer in OMICS:

            cols = omics_columns(df, layer)

            with st.expander(str(layer)):

                st.dataframe(
                    pd.DataFrame({
                        "Feature": cols,
                        "Value": [sample[c] for c in cols]
                    }),
                    hide_index=True,
                    use_container_width=True
                )

    # --------------------------------------------------------
    # LITERATURE INTELLIGENCE
    # --------------------------------------------------------

    elif page == "Literature Intelligence":

        render_html("""
        <div class="chat-user">
            Search biomedical research literature using PubMed.
        </div>
        """)

        query = st.text_input(
            "PubMed search",
            placeholder="ovarian cancer multi-omics biomarkers"
        )

        limit = st.select_slider(
            "Number of articles",
            options=[3, 5, 8, 10, 15],
            value=5
        )

        if st.button("Search PubMed", use_container_width=True):

            if query.strip():

                try:
                    with st.spinner("Searching research literature..."):
                        st.session_state["pubmed_records"] = (
                            search_pubmed(query, limit)
                        )

                except Exception as e:
                    st.error(f"Search failed: {e}")

            else:
                st.warning("Enter a search query.")

        records = st.session_state.get("pubmed_records", [])

        for rec in records:

            with st.expander(
                rec.get("title", "Research article")[:100]
            ):

                st.write(
                    rec.get("abstract", "No abstract available.")
                )

                if rec.get("url"):
                    st.markdown(
                        f"[Open PubMed ↗]({rec['url']})"
                    )

        st.markdown("---")

        abstract = st.text_area(
            "Paste research abstract",
            height=120
        )

        if st.button("Extract key findings"):

            if len(abstract.strip()) > 30:

                st.session_state["text_summary"] = (
                    summarize(abstract),
                    keywords(abstract)
                )

            else:
                st.warning("Enter a longer abstract.")

        if st.session_state.get("text_summary"):

            st.markdown("**Abstract summary**")

            st.write(
                st.session_state["text_summary"][0]
            )

            st.write("**Keywords**")

            st.write(
                st.session_state["text_summary"][1]
            )

    # --------------------------------------------------------
    # EXPLAINABLE AI
    # --------------------------------------------------------

    elif page == "Explainable AI":

        render_html("""
        <div class="chat-user">
            Explore model feature contributions and generate
            SHAP explanations for the selected classifier.
        </div>

        <div class="chat-ai">
            <b>Explainability</b><br>
            Permutation importance measures the effect of
            feature shuffling on model performance.
            SHAP estimates feature contributions to model output.
        </div>
        """)

        st.markdown("**Top permutation-ranked features**")

        st.dataframe(
            importance.head(12),
            hide_index=True,
            use_container_width=True
        )

        if st.button(
            "Generate SHAP plots",
            use_container_width=True
        ):

            with st.spinner("Generating SHAP explanations..."):

                try:
                    st.session_state["shap_status"] = (
                        generate_shap(MODEL, X)
                    )

                except Exception as e:
                    st.session_state["shap_status"] = str(e)

        if st.session_state.get("shap_status"):
            st.info(st.session_state["shap_status"])

        for filename in ["shap_summary.png", "shap_bar.png"]:

            filepath = ROOT / "reports" / filename

            if filepath.exists():
                st.image(
                    str(filepath),
                    caption=filename.replace("_", " ").replace(".png", ""),
                    use_container_width=True
                )

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    elif page == "Model Evaluation":

        render_html("""
        <div class="chat-user">
            Evaluate the predictive model using held-out
            test data and cross-validation results.
        </div>
        """)

        st.markdown("**Selected model**")

        st.write(
            METRICS["selected_model"].replace("_", " ").title()
        )

        if st.button(
            "View detailed evaluation",
            use_container_width=True
        ):
            st.session_state["evaluation_open"] = True

        if st.session_state.get("evaluation_open"):
            st.json(TEST)

    render_html("""
    <div class="footer-note">
        Research copilot outputs are computational summaries.
        Validate findings independently.
    </div>
    """)


# ============================================================
# 7. RIGHT PANEL - ANALYTICS WORKSPACE
# ============================================================

with right:

    render_html("""
    <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:12px">

        <div class="kicker">
            OMICS ANALYSIS WORKSPACE
        </div>

        <span class="pill">
            ● COHORT LOADED
        </span>

    </div>
    """)

    tabs = st.tabs([
        "1. Multi-Omics",
        "2. Pathways",
        "3. Targets",
        "4. Performance",
        "5. Explainability"
    ])

    # ========================================================
    # TAB 1 - MULTI-OMICS
    # ========================================================

    with tabs[0]:

        st.markdown("**Multi-Omics Analysis · Case vs Control**")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Samples", len(df))
        c2.metric("Features", len(features))
        c3.metric("Case-like", int(y.sum()))
        c4.metric("Data layers", len(OMICS))

        layer = st.selectbox(
            "Molecular layer",
            list(OMICS.keys()),
            key="analytics_layer"
        )

        cols = omics_columns(df, layer)

        chosen = st.selectbox(
            "Select feature",
            cols,
            key="analytics_feature"
        )

        vals = [
            df.loc[df.research_label == 0, chosen].dropna(),
            df.loc[df.research_label == 1, chosen].dropna()
        ]

        fig, ax = plt.subplots(figsize=(7, 3.5))

        ax.boxplot(
            vals,
            patch_artist=True,
            boxprops={
                "facecolor": "#183b50",
                "color": "#39c8b7"
            },
            medianprops={
                "color": "#f0c76b"
            }
        )

        # Matplotlib compatibility fix
        ax.set_xticks([1, 2])
        ax.set_xticklabels([
            "Control cohort",
            "Case cohort"
        ])

        ax.set_title(
            f"{chosen} - Group Distribution",
            color="#dce8f5"
        )

        ax.set_facecolor("#101b2b")
        fig.patch.set_facecolor("#101b2b")

        ax.tick_params(colors="#a9bdd1")

        for spine in ax.spines.values():
            spine.set_color("#344a63")

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("**Molecular layer inventory**")

        st.dataframe(
            group_summary(df),
            hide_index=True,
            use_container_width=True
        )

    # ========================================================
    # TAB 2 - PATHWAY ANALYSIS
    # ========================================================

    with tabs[1]:

        st.markdown("**Pathway Activity Analysis**")

        pscore = pathway_scores(df)

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.barh(
            pscore["Pathway"].iloc[::-1],
            pscore["Mean score"].iloc[::-1],
            color="#278caa"
        )

        ax.set_xlabel(
            "Mean standardized score",
            color="#a9bdd1"
        )

        ax.set_facecolor("#101b2b")
        fig.patch.set_facecolor("#101b2b")

        ax.tick_params(colors="#a9bdd1", labelsize=8)

        for spine in ax.spines.values():
            spine.set_color("#344a63")

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.dataframe(
            pscore,
            hide_index=True,
            use_container_width=True
        )

    # ========================================================
    # TAB 3 - TARGET PRIORITIZATION
    # ========================================================

    with tabs[2]:

        st.markdown("**Molecular Candidate Prioritization**")

        ranking = target_prioritization(
            df,
            importance
        )

        st.dataframe(
            ranking.head(15),
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Export candidate ranking CSV",
            ranking.to_csv(index=False),
            "candidate_ranking.csv",
            "text/csv"
        )

    # ========================================================
    # TAB 4 - MODEL PERFORMANCE
    # ========================================================

    with tabs[3]:

        st.markdown("**Predictive Model Performance**")

        metric_columns = st.columns(4)

        metric_values = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("ROC-AUC", "roc_auc")
        ]

        for col, (label, key) in zip(
            metric_columns,
            metric_values
        ):

            col.metric(
                label,
                f"{TEST[key]:.3f}"
            )

        cv = pd.DataFrame([
            {
                "Model": name.replace("_", " ").title(),
                "CV ROC-AUC": result["roc_auc"]["mean"],
                "CV F1": result["f1"]["mean"]
            }
            for name, result in METRICS["cross_validation"].items()
        ])

        st.markdown("**Cross-validation comparison**")

        st.dataframe(
            cv,
            hide_index=True,
            use_container_width=True
        )

        cm = np.array(TEST["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(4.5, 3.5))

        ax.imshow(cm, cmap="Blues")

        ax.set_xticks([0, 1], ["Control", "Case"])
        ax.set_yticks([0, 1], ["Control", "Case"])

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=13
                )

        ax.set_title(
            "Confusion Matrix",
            color="#dce8f5"
        )

        fig.patch.set_facecolor("#101b2b")
        ax.set_facecolor("#101b2b")

        ax.tick_params(colors="#a9bdd1")

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Complete test metrics**")
        st.json(TEST)

    # ========================================================
    # TAB 5 - EXPLAINABLE AI
    # ========================================================

    with tabs[4]:

        st.markdown("**Explainable AI - Feature Contributions**")

        st.caption(
            "Global feature importance and SHAP model explanations."
        )

        impfile = ROOT / "reports" / "permutation_importance.csv"

        if impfile.exists():

            imp_df = pd.read_csv(impfile)

            st.markdown("**Permutation feature importance**")

            st.dataframe(
                imp_df.head(15),
                hide_index=True,
                use_container_width=True
            )

            top = imp_df.head(12).sort_values(
                "importance_mean"
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.barh(
                top["feature"],
                top["importance_mean"],
                color="#278caa"
            )

            ax.set_xlabel(
                "Decrease in ROC-AUC",
                color="#a9bdd1"
            )

            ax.set_title(
                "Top Feature Importance",
                color="#dce8f5"
            )

            ax.set_facecolor("#101b2b")
            fig.patch.set_facecolor("#101b2b")

            ax.tick_params(colors="#a9bdd1", labelsize=8)

            for spine in ax.spines.values():
                spine.set_color("#344a63")

            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("---")

        if st.button(
            "Generate SHAP Summary and Bar Plots",
            use_container_width=True
        ):

            with st.spinner("Generating SHAP explanations..."):

                try:
                    st.session_state["shap_status"] = (
                        generate_shap(MODEL, X)
                    )
                except Exception as e:
                    st.session_state["shap_status"] = str(e)

        if st.session_state.get("shap_status"):
            st.info(st.session_state["shap_status"])

        for filename in [
            "shap_summary.png",
            "shap_bar.png"
        ]:

            filepath = ROOT / "reports" / filename

            if filepath.exists():

                st.image(
                    str(filepath),
                    caption=filename.replace("_", " ").replace(".png", ""),
                    use_container_width=True
                )


# ============================================================
# 8. FOOTER
# ============================================================

render_html("""
<div class="footer-note">
    OncoOmics Research Intelligence |
    Synthetic Multi-Omics Research Demonstration |
    Not intended for diagnosis or patient-care decisions.
</div>
""")
