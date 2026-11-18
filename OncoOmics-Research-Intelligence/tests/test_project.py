import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from biomed.data import generate_synthetic_cohort,get_X_y
from biomed.omics import group_summary,pathway_scores,differential_features,target_prioritization
from biomed.literature import keywords,summarize

def test_synthetic_omics_schema():
    df=generate_synthetic_cohort(n=120,save=False)
    X,y,features=get_X_y(df)
    assert len(df)==120
    assert len(features)>=20
    assert set(y.unique())=={0,1}
    assert any(c.startswith("RNA_") for c in features)
    assert any(c.startswith("PROT_") for c in features)
    assert any(c.startswith("MET_") for c in features)
    assert any(c.startswith("IMM_") for c in features)

def test_analysis_modules():
    df=generate_synthetic_cohort(n=100,save=False)
    assert len(group_summary(df))==4
    assert len(pathway_scores(df))>=4
    assert len(differential_features(df))>10
    imp=differential_features(df).rename(columns={"Feature":"feature"}).assign(importance_mean=.1)
    assert len(target_prioritization(df,imp))>10

def test_literature_helpers():
    text="Ovarian cancer research integrates genomic biomarkers and molecular pathways. Multi-omics analysis supports discovery of disease-associated molecular signatures."
    assert keywords(text)
    assert summarize(text)
