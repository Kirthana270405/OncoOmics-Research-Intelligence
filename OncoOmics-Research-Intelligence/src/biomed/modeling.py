from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from .data import load_cohort, get_X_y, split_data

ART = Path("artifacts")
REP = Path("reports")

def candidates(seed=42):
    return {
      "logistic_regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",LogisticRegression(max_iter=2500,class_weight="balanced",random_state=seed))]),
      "random_forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=350,min_samples_leaf=2,class_weight="balanced",random_state=seed,n_jobs=-1))]),
      "hist_gradient_boosting":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",HistGradientBoostingClassifier(max_iter=180,learning_rate=.06,l2_regularization=1.0,random_state=seed))])
    }

def train_and_evaluate(seed=42):
    ART.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)
    df=load_cohort()
    X,y,features=get_X_y(df)
    Xtr,Xte,ytr,yte=split_data(X,y,seed=seed)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    models=candidates(seed); results={}
    for name,model in models.items():
        scores=cross_validate(model,Xtr,ytr,cv=cv,scoring={"roc_auc":"roc_auc","f1":"f1","precision":"precision","recall":"recall"},n_jobs=-1)
        results[name]={m:{"mean":float(np.mean(scores[f"test_{m}"])),"std":float(np.std(scores[f"test_{m}"]))} for m in ["roc_auc","f1","precision","recall"]}
    best_name=max(results,key=lambda k:results[k]["roc_auc"]["mean"])
    best=models[best_name].fit(Xtr,ytr)
    prob=best.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
    metrics={
      "accuracy":float(accuracy_score(yte,pred)),
      "precision":float(precision_score(yte,pred,zero_division=0)),
      "recall":float(recall_score(yte,pred,zero_division=0)),
      "f1":float(f1_score(yte,pred,zero_division=0)),
      "roc_auc":float(roc_auc_score(yte,prob)),
      "pr_auc":float(average_precision_score(yte,prob)),
      "confusion_matrix":confusion_matrix(yte,pred).tolist(),
      "classification_report":classification_report(yte,pred,target_names=["control","case"],output_dict=True),
      "threshold":.5,"test_rows":int(len(yte)),
      "warning":"Synthetic demonstration cohort; metrics are not clinical evidence."
    }
    joblib.dump(best,ART/"multiomics_model.joblib")
    with open(ART/"experiment_metrics.json","w") as f:
        json.dump({"selected_model":best_name,"cross_validation":results,"test":metrics,"features":features,"seed":seed},f,indent=2)
    perm=permutation_importance(best,Xte,yte,n_repeats=12,random_state=seed,scoring="roc_auc",n_jobs=-1)
    imp=pd.DataFrame({"feature":features,"importance_mean":perm.importances_mean,"importance_std":perm.importances_std}).sort_values("importance_mean",ascending=False)
    imp.to_csv(REP/"permutation_importance.csv",index=False)
    return best_name,metrics,results,imp

def load_model():
    p=ART/"multiomics_model.joblib"
    if not p.exists(): train_and_evaluate()
    return joblib.load(p)

def load_metrics():
    p=ART/"experiment_metrics.json"
    if not p.exists(): train_and_evaluate()
    return json.loads(p.read_text())
