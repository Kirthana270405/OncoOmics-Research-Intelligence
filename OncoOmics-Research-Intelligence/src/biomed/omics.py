import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

OMICS = {
 "RNA": ("RNA_", "RNA-seq expression"),
 "Protein": ("PROT_", "Protein abundance"),
 "Metabolite": ("MET_", "Metabolite profile"),
 "Immune": ("IMM_", "Immune-cell composition")
}
PATHWAYS = {
 "DNA damage response":["RNA_TP53_TPM","RNA_BRCA1_TPM","RNA_BRCA2_TPM"],
 "PI3K / AKT signaling":["RNA_PIK3CA_TPM","RNA_AKT1_TPM","PROT_EGFR_z"],
 "Cell-cycle regulation":["RNA_CCNE1_TPM","RNA_MYC_TPM","PROT_ERBB2_z"],
 "Tumor metabolism":["MET_lactate_z","MET_glucose_z","MET_glutamine_z","MET_citrate_z","MET_choline_z"],
 "Immune microenvironment":["IMM_CD8_T_fraction","IMM_NK_cells_fraction","IMM_Treg_fraction","IMM_macrophages_fraction"]
}

def omics_columns(df, layer):
    prefix=OMICS[layer][0]
    return [c for c in df.columns if c.startswith(prefix)]

def group_summary(df):
    rows=[]
    for layer in OMICS:
        cols=omics_columns(df,layer)
        rows.append({"Layer":layer,"Features":len(cols),"Description":OMICS[layer][1]})
    return pd.DataFrame(rows)

def pathway_scores(df):
    rows=[]
    for pathway, cols in PATHWAYS.items():
        valid=[c for c in cols if c in df.columns]
        # Average standardized feature values for an exploratory, synthetic pathway score.
        z=(df[valid]-df[valid].mean())/(df[valid].std(ddof=0)+1e-9)
        rows.append({"Pathway":pathway,"Mean score":float(z.mean(axis=1).mean()),
                     "Feature count":len(valid)})
    return pd.DataFrame(rows).sort_values("Mean score",ascending=False)

def differential_features(df):
    rows=[]
    case=df[df.research_label==1]; control=df[df.research_label==0]
    for col in [c for c in df.columns if c.startswith(("RNA_","PROT_","MET_","IMM_"))]:
        a=case[col].dropna(); b=control[col].dropna()
        stat,p=mannwhitneyu(a,b,alternative="two-sided")
        rows.append({"Feature":col,"Case mean":a.mean(),"Control mean":b.mean(),
                     "Difference":a.mean()-b.mean(),"p_value":p})
    out=pd.DataFrame(rows)
    out["-log10(p)"]=-np.log10(out.p_value.clip(lower=1e-300))
    return out.sort_values("p_value")

def target_prioritization(df, importance):
    diff=differential_features(df)
    imp=importance.rename(columns={"feature":"Feature","importance_mean":"Model importance"})
    merged=diff.merge(imp[["Feature","Model importance"]],on="Feature",how="left").fillna({"Model importance":0})
    # Transparent research ranking heuristic; not a drug-target validation score.
    pscore=-np.log10(merged.p_value.clip(lower=1e-300))
    iscore=merged["Model importance"].clip(lower=0)
    merged["Priority score"]=0.55*(pscore/(pscore.max()+1e-9))+0.45*(iscore/(iscore.max()+1e-9))
    return merged.sort_values("Priority score",ascending=False)

def layer_correlation(df):
    features=[c for c in df.columns if c.startswith(("RNA_","PROT_","MET_","IMM_"))]
    corr=df[features].corr(method="spearman")
    groups={k:[c for c in features if c.startswith(v[0])] for k,v in OMICS.items()}
    labels=list(groups)
    matrix=pd.DataFrame(index=labels,columns=labels,dtype=float)
    for a in labels:
        for b in labels:
            cols_a,cols_b=groups[a],groups[b]
            vals=[abs(corr.loc[x,y]) for x in cols_a for y in cols_b if x!=y or a!=b]
            matrix.loc[a,b]=float(np.mean(vals)) if vals else 0
    return matrix
