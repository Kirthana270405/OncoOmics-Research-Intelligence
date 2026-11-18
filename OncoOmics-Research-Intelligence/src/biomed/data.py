from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "synthetic_ovarian_omics_cohort.csv"

GENES = ["TP53", "BRCA1", "BRCA2", "CCNE1", "MYC", "PIK3CA", "AKT1", "KRAS"]
PROTEINS = ["CA125", "HE4", "MUC16", "ERBB2", "EGFR", "VEGFA"]
METABOLITES = ["lactate", "glucose", "glutamine", "citrate", "choline", "alanine"]
IMMUNE = ["CD8_T", "CD4_T", "B_cells", "NK_cells", "Treg", "macrophages"]

def generate_synthetic_cohort(n=360, seed=42, save=True):
    """Generate reproducible, explicitly synthetic multi-omics research data."""
    rng = np.random.default_rng(seed)
    status = rng.binomial(1, 0.52, n)  # 1 = synthetic case group, 0 = synthetic control group
    age = np.clip(rng.normal(57, 11, n), 25, 85).round(1)
    stage = np.where(status == 1, rng.choice(["I","II","III","IV"], n, p=[.10,.18,.47,.25]),
                     rng.choice(["NA"], n))
    data = {"sample_id":[f"SYN-{i+1:04d}" for i in range(n)],
            "cohort_group":np.where(status==1,"Case cohort","Control cohort"),
            "age_years":age, "research_stage":stage}
    for gene in GENES:
        base = rng.lognormal(mean=2.1, sigma=.55, size=n)
        shift = {"TP53":.48,"BRCA1":-.20,"BRCA2":-.15,"CCNE1":.55,"MYC":.35,
                 "PIK3CA":.28,"AKT1":.20,"KRAS":.18}[gene] * status
        data[f"RNA_{gene}_TPM"] = np.round(base*np.exp(shift+rng.normal(0,.12,n)),3)
    for protein in PROTEINS:
        base = rng.normal(0, .8, n)
        shift = {"CA125":1.0,"HE4":.8,"MUC16":.7,"ERBB2":.35,"EGFR":.25,"VEGFA":.6}[protein]
        data[f"PROT_{protein}_z"] = np.round(base+shift*status,3)
    for metabolite in METABOLITES:
        base = rng.normal(0, 1, n)
        shift = {"lactate":.65,"glucose":.25,"glutamine":-.45,"citrate":-.35,
                 "choline":.55,"alanine":.2}[metabolite]
        data[f"MET_{metabolite}_z"] = np.round(base+shift*status,3)
    for cell in IMMUNE:
        base = rng.beta(2.5, 5, n)
        shift = {"CD8_T":-.07,"CD4_T":-.02,"B_cells":-.01,"NK_cells":-.05,
                 "Treg":.06,"macrophages":.08}[cell]
        data[f"IMM_{cell}_fraction"] = np.round(np.clip(base+shift*status,0,1),4)
    data["research_label"] = status
    df = pd.DataFrame(data)
    if save:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df

def load_cohort(regenerate=False):
    if regenerate or not DATA_PATH.exists():
        return generate_synthetic_cohort()
    return pd.read_csv(DATA_PATH)

def get_feature_columns(df):
    return [c for c in df.columns if c.startswith(("RNA_","PROT_","MET_","IMM_"))]

def get_X_y(df):
    features = get_feature_columns(df)
    return df[features].copy(), df["research_label"].astype(int).copy(), features

def split_data(X, y, test_size=.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
