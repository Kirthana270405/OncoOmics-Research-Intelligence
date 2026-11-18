@echo off
if not exist .venv (py -3.11 -m venv .venv)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
