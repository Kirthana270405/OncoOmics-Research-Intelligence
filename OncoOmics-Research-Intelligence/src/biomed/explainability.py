from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_shap(model,X,out_dir="reports"):
    out=Path(out_dir); out.mkdir(exist_ok=True,parents=True)
    try:
        import shap
        sample=X.sample(min(80,len(X)),random_state=42)
        explainer=shap.Explainer(model.predict_proba,sample)
        values=explainer(sample)
        if values.values.ndim==3:
            values=shap.Explanation(values=values.values[:,:,1],base_values=values.base_values[:,1],data=values.data,feature_names=list(X.columns))
        shap.plots.beeswarm(values,max_display=18,show=False)
        plt.tight_layout(); plt.savefig(out/"shap_summary.png",dpi=160,bbox_inches="tight"); plt.close("all")
        shap.plots.bar(values,max_display=18,show=False)
        plt.tight_layout(); plt.savefig(out/"shap_bar.png",dpi=160,bbox_inches="tight"); plt.close("all")
        return "SHAP summary and bar plots generated."
    except ImportError:
        return "SHAP is not installed. Run: pip install shap"
    except Exception as e:
        plt.close("all")
        return f"SHAP generation error: {type(e).__name__}: {e}"
