import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"src"))
from biomed.data import generate_synthetic_cohort
from biomed.modeling import train_and_evaluate
from biomed.data import load_cohort,get_X_y,split_data
from biomed.explainability import generate_shap

if __name__=="__main__":
    generate_synthetic_cohort()
    name,metrics,cv,importance=train_and_evaluate()
    print("Selected model:",name)
    print("Held-out metrics:")
    for k in ["accuracy","precision","recall","f1","roc_auc","pr_auc"]:
        print(f"{k}: {metrics[k]:.4f}")
    print("Synthetic cohort saved to data/")
    print("Model and metrics saved to artifacts/")
    print("Permutation importance saved to reports/")
