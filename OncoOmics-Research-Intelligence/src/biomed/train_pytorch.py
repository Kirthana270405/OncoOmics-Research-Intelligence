
"""Optional PyTorch MLP experiment on the synthetic multi-omics cohort."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .data import load_cohort, get_X_y, split_data

class OmicsMLP(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_features,96),nn.ReLU(),nn.Dropout(.25),
            nn.Linear(96,48),nn.ReLU(),nn.Dropout(.15),
            nn.Linear(48,1)
        )
    def forward(self,x): return self.net(x).squeeze(1)

def main(epochs=30,seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    df=load_cohort(); X,y,_=get_X_y(df)
    Xtr,Xte,ytr,yte=split_data(X,y,seed=seed)
    scaler=StandardScaler()
    a=scaler.fit_transform(Xtr).astype("float32")
    b=scaler.transform(Xte).astype("float32")
    ds=TensorDataset(torch.tensor(a),torch.tensor(ytr.to_numpy(),dtype=torch.float32))
    loader=DataLoader(ds,batch_size=32,shuffle=True)
    net=OmicsMLP(a.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    loss_fn=nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        net.train(); losses=[]
        for xb,yb in loader:
            opt.zero_grad(); loss=loss_fn(net(xb),yb); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"Epoch {epoch+1:02d}/{epochs} loss={np.mean(losses):.4f}")
    net.eval()
    with torch.no_grad(): probs=torch.sigmoid(net(torch.tensor(b))).numpy()
    pred=(probs>=.5).astype(int)
    result={"accuracy":float(accuracy_score(yte,pred)),"f1":float(f1_score(yte,pred)),"roc_auc":float(roc_auc_score(yte,probs)),
            "notice":"Synthetic data experiment; not clinical evidence."}
    Path("artifacts").mkdir(exist_ok=True)
    torch.save({"state_dict":net.state_dict(),"input_features":list(X.columns),
                "scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist()},
               "artifacts/pytorch_omics_mlp.pt")
    Path("reports/pytorch_metrics.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
