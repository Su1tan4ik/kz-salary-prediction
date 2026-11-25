# Salary Prediction — Kazakhstan Job Market (hh.kz)

Streamlit web app for the KBTU Data Mining final project. Enter a vacancy (title, city, role,
experience, skills, …) and get the expected monthly salary in KZT with a calibrated 80% range.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model
Download `0000.parquet` from the Hugging Face dataset
`chagatai-htr/headhunter-kazakhstan-all-regions`, then:
```bash
pip install pyarrow
python train.py path/to/0000.parquet   # writes model/model.joblib
```

The deployed model is the notebook's round-2 gradient boosting without the
sentence-embedding features (so it runs without PyTorch on free hosting).
