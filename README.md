# Landslide Risk Assessment in India Using Bayesian Network Inference

Mini-project for the Department of CSE (AI&ML), RYMEC Ballari, Academic Year 2025-26.

## Problem Statement
<!-- 1-2 short paragraphs: what is the real-world problem, and why it matters to the public. -->

## Objectives
- To build a Bayesian Network model on historical landslide records
- To analyze how trigger, landslide category, season and region influence severity
- To design a Streamlit web app that lets users query landslide risk

## Dataset
- **Source:** NASA Global Landslide Catalog (NASA Open Data Portal)
- **Records:** _to be filled after filtering to India_
- **Key features:** trigger, landslide category, location, date, fatalities, size
- **Target:** severity class (low / medium / high)

## Methodology
1. Data cleaning and missing-value handling
2. Discretization (month -> season, fatalities -> severity classes) and encoding
3. Bayesian Network structure (expert-defined, compared with learned structure)
4. Parameter learning (conditional probability tables) with `pgmpy`
5. Inference with Variable Elimination
6. Evaluation: accuracy, precision, recall, F1, confusion matrix, cross-validation, baseline comparison

## Project Structure
```
├── app.py              # Streamlit web app
├── train_model.py      # preprocessing + training + evaluation
├── notebooks/          # exploration and analysis notebooks
├── data/               # cleaned dataset
├── model/              # trained model (bn_model.pkl)
├── requirements.txt
└── README.md
```

## How to Run Locally
```bash
git clone https://github.com/<your-username>/landslide-bayesian-network.git
cd landslide-bayesian-network
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py           # trains and saves model/bn_model.pkl
streamlit run app.py
```

## Results
<!-- Add accuracy, precision, recall, F1, confusion matrix and screenshots here. -->

## Live Demo
<!-- Paste your Streamlit Community Cloud link here. -->

## Future Scope
- Add rainfall (GPM IMERG / CHIRPS), terrain (SRTM) and land cover as nodes
- Dynamic Bayesian Network for early warning
- Decision network for evacuation / road-closure actions
- District-level risk map

## References
- NASA Global Landslide Catalog
- pgmpy documentation
- Streamlit documentation
