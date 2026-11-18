from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from .modeling import load_model, load_metrics
from .literature import keywords, summarize, search_pubmed
from .data import load_cohort, get_X_y

app=FastAPI(title="OncoOmics Research API",version="1.0.0",
 description="Synthetic multi-omics research demonstration API. Not for clinical use.")

class PredictionInput(BaseModel):
    features: dict[str,float]
class TextInput(BaseModel):
    text: str=Field(min_length=30,max_length=30000)
class SearchInput(BaseModel):
    query: str=Field(min_length=2,max_length=300)
    limit: int=Field(default=5,ge=1,le=20)

@app.get("/")
def root():
    return {"service":"OncoOmics Research Intelligence","status":"ready",
            "data":"synthetic demonstration cohort","clinical_use":False}
@app.get("/health")
def health(): return {"status":"healthy"}
@app.get("/metrics")
def metrics(): return load_metrics()
@app.post("/predict")
def predict(body:PredictionInput):
    model=load_model(); X,_,features=get_X_y(load_cohort())
    missing=[f for f in features if f not in body.features]
    if missing: raise HTTPException(422,detail={"missing_features":missing})
    row=pd.DataFrame([{f:body.features[f] for f in features}])
    p=float(model.predict_proba(row)[0,1])
    return {"research_group_estimate":"case-like" if p>=.5 else "control-like",
            "case_probability":p,"warning":"Synthetic research demonstration only."}
@app.post("/literature/analyze")
def analyze(body:TextInput):
    return {"keywords":keywords(body.text),"extractive_summary":summarize(body.text)}
@app.post("/literature/search")
def literature(body:SearchInput):
    try:return {"query":body.query,"records":search_pubmed(body.query,body.limit)}
    except Exception as e: raise HTTPException(502,detail=str(e))
