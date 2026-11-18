# OncoOmics Research Intelligence

A biomedical AI research workspace inspired by modern multi-omics research platforms. It combines a dark, three-panel research dashboard, a synthetic ovarian-cancer multi-omics cohort, predictive modeling, pathway-level exploration, candidate prioritization, PubMed retrieval and an API.

## Project capabilities

- **Multi-omics data pipeline:** reproducible synthetic RNA-expression, protein, metabolite and immune-cell measurements.
- **Cohort exploration:** inspect synthetic samples and compare molecular feature distributions between case-like and control-like groups.
- **Predictive analytics:** compare Logistic Regression, Random Forest and HistGradientBoosting using stratified five-fold cross-validation and held-out evaluation.
- **Pathway exploration:** compute transparent exploratory scores for selected molecular feature groups.
- **Candidate prioritization:** rank features using a documented heuristic combining Mann–Whitney group-difference statistics and model permutation importance.
- **Explainable AI:** permutation feature importance and optional SHAP summary/bar plots.
- **Biomedical NLP:** PubMed search, keyword extraction and extractive summarization.
- **Deployment interface:** Streamlit dashboard and FastAPI endpoints.
- **Optional deep learning:** PyTorch MLP extension can be added as a follow-on experiment.

## Important scientific scope

**All cohort records in this repository are synthetic.** They are generated for software demonstration and method prototyping only. They are not real ovarian cancer patient data, do not represent validated biological effect sizes, and must not be used to infer clinical findings, diagnose disease, recommend treatment or prioritize real therapeutic targets. The pathway scores and target ranking are illustrative computational heuristics, not validated gene-set enrichment or drug-discovery methods. Predictive metrics on synthetic data are not evidence of clinical model performance.

PubMed search retrieves public bibliographic records from NCBI E-utilities when internet access is available. Abstract processing is extractive and local; the application does not generate clinical advice.

## Run in VS Code (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The dashboard opens at `http://localhost:8501`.

## Start the API

Open a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn biomed.api:app --app-dir src --reload
```

API docs: `http://127.0.0.1:8000/docs`

Endpoints:
- `GET /health`
- `GET /metrics`
- `POST /predict`
- `POST /literature/analyze`
- `POST /literature/search`

## Optional PyTorch experiment

```powershell
pip install -r requirements-deep-learning.txt
$env:PYTHONPATH = "src"
python -m biomed.train_pytorch
```

This trains a small MLP on the synthetic multi-omics table and writes its checkpoint and metrics under `artifacts/` and `reports/`.

## Optional SHAP

```powershell
pip install shap
```

Use the Explainable AI option or call `generate_shap()` from the Python package after loading the model and feature matrix. The core app does not require SHAP to run.

## Tests

```powershell
pytest -q
```

## Structure

```text
OncoOmics-Research-Intelligence/
├── app.py
├── train.py
├── requirements.txt
├── requirements-deep-learning.txt
├── .gitignore
├── README.md
├── data/
│   └── synthetic_ovarian_omics_cohort.csv
├── artifacts/
├── reports/
├── src/biomed/
│   ├── data.py
│   ├── modeling.py
│   ├── omics.py
│   ├── literature.py
│   ├── explainability.py
│   └── api.py
└── tests/
```

## Alignment to AI R&D internship responsibilities

| JD area | Project evidence |
|---|---|
| AI model development and optimization | Three-model training and cross-validation pipeline |
| Research and innovation | Multi-omics feature integration and pathway/candidate exploration |
| Data pipelining | Synthetic multi-modal generation, schema, preprocessing and feature extraction |
| Performance evaluation | Held-out metrics, confusion matrix and cross-validation |
| Integration | Streamlit research workspace and FastAPI service |
| Documentation and reporting | Reproducible training script, saved artifacts, tests and this README |
| Python, Pandas, NumPy | Core implementation |
| PyTorch/TensorFlow | Optional deep-learning extension; do not claim it unless implemented and run |
| MLOps | Reproducible artifacts and modular pipeline; production MLOps is not claimed |

## Resume-ready description (edit to reflect only work you have run)

**OncoOmics Research Intelligence | Python, Scikit-learn, Pandas, Streamlit, FastAPI, Biomedical NLP, Explainable AI**

- Built a multi-omics research workspace integrating synthetic RNA, protein, metabolite and immune-cell datasets for cohort-level exploration.
- Developed a reproducible ML pipeline comparing three classifiers with stratified cross-validation and held-out evaluation.
- Implemented exploratory pathway scoring, statistical feature comparison, model-based candidate ranking and explainability reports.
- Integrated PubMed retrieval and text analysis with a Streamlit dashboard and FastAPI endpoints.
