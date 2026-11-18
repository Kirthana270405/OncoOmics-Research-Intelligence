# OncoOmics Research Intelligence – Multi-Omics AI Platform

An AI-powered biomedical research platform for multi-omics data analysis, molecular pathway exploration, target prioritization, and explainable machine learning.

## Overview

OncoOmics Research Intelligence is a research-oriented platform designed to explore ovarian cancer using integrated multi-omics data. It combines data analysis, machine learning, and explainable AI to identify molecular patterns and generate interpretable research insights.

The platform uses **360 synthetic ovarian cancer research samples** across four data modalities:

- RNA Expression
- Protein Abundance
- Metabolite Profiles
- Immune-Cell Composition

## Key Features

### 1. Multi-Omics Analysis
- Explore RNA, protein, metabolite, and immune-cell data.
- Compare case-like and control-like cohorts.
- Identify molecular expression differences.

### 2. Pathway Analysis
- Explore molecular pathway activity.
- Examine relationships between molecular features.
- Identify potentially relevant biological patterns.

### 3. Target Prioritization
- Rank molecular candidates using statistical differences and model-based feature importance.
- Identify high-priority research candidates.

### 4. Cohort Explorer
- Inspect individual synthetic samples.
- Explore molecular measurements across different data modalities.

### 5. Literature Intelligence
- Search biomedical research literature using PubMed.
- Support research exploration through relevant publications.

### 6. Explainable AI
- Generate SHAP-based explanations.
- Explore permutation feature importance.
- Interpret machine learning model predictions.

### 7. Model Evaluation
- Train and compare machine learning models.
- Evaluate predictive performance.
- Visualize ROC curves, precision-recall curves, and confusion matrices.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Frontend | Streamlit |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP, Permutation Importance |
| Visualization | Matplotlib |
| Data Processing | Python Data Science Libraries |

## Project Structure

```text
OncoOmics-Research-Intelligence/
│
├── app.py
├── train.py
├── run_app.bat
├── requirements.txt
├── requirements-deep-learning.txt
├── .env.example
├── .gitignore
│
├── data/
├── src/
│   └── biomed/
├── artifacts/
├── reports/
└── tests/
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Kirthana270405/OncoOmics-Research-Intelligence.git
cd OncoOmics-Research-Intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the platform.

## Research Disclaimer

This project is intended strictly for educational and biomedical research demonstration purposes. It uses synthetic demonstration data and is not a medical device, diagnostic tool, or clinical decision-support system. It must not be used for patient-care decisions.

## Author

**Sai Kirthana Konduru**

- GitHub: https://github.com/Kirthana270405
