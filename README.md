# Concussion Classifier with Synthetic Clinical Data

## Scope 

This project demonstrates a full-stack machine learning pipeline and interactive dashboard for classifying clinical outcomes using **synthetic resting-state fMRI** data. It includes preprocessing, oversampling, model tuning, evaluation, and an interactive Dash app to visualize results. It simulates a real-world use case in which quantitative neuroimaging metrics might assist in concussion diagnosis. Again, the data used are synthetic, but resemble the original, clinical dataset (pending publication).

## Clinical context

Concussions are mainly diagnosed based on self-reported symptoms. But we can use functional MRI (fMRI) to measure “brain activity” and perform diagnoses more objectively, given that fMRI disturbances are a proxy for the neuropathology of concussion. This will help address the clinical problem of under and/or missed diagnosis of concussion, as well as providing a pathology-based diagnosis that can be used for monitoring of injury progression or recovery.

fMRI data are a timeseries known as the BOLD (Blood Oxygen Level Dependent) signal. This signal can and has been quantified and summarized using several metrics, including the ones below: 

- Mean
- Standard deviation
- Entropy (via Lyapunov exponent and Hurst exponent)
- Amplitude of low-frequency fluctuations (ALFF)
- Fractional ALFF

And each metric is computed for every region of interest (ROI) using the Harvard-Oxford neuroanatomical atlas.

Despite the neuroimaging research into concussion, we do not know about which BOLD metric or ROI is most discriminatory between concussed and healthy brains. 

This analysis helps explore:
- Which metric-ROI pairs best discriminate between concussion and control?
- Which metric is most informative?
- Which ROIs are most relevant?

Results are displayed in a hosted interactive dashboard (it loads slowly on the free version of Render), including ROC and precision-recall curves with plain-English interpretations for each threshold.

## Dashboard

The dashboard includes a series of plots, with the ROC Curve and Precision-Recall curve clickable, such that clicking on a given data point gives a plain English language summary of what it represents.

[![View Dashboard on Render](https://img.shields.io/badge/View%20Live%20App-Render-blue)](https://concussion-diagnosis-model-performance.onrender.com/)

The dashboard includes:
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Learning Curve (F1 Score)
- 3-Fold CV Scores
- Feature Importances
- Glossary of ML terms 

## Data

- `synthetic_data.csv`: synthetic fMRI data (last column = target label)

## Features

- SMOTE for class imbalance
- Gradient Boosting Classifier
- Randomized hyperparameter tuning
- Evaluation: ROC, PR, CV scores, learning curves

## How to Use

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the training script to produce the model and data splits
python rf_classifier.py

# Step 3: Launch the dashboard
python app.py
```
## Notes

Built as part of a data science portfolio (with an accompanying blod post available here: https://bsharma.super.site/projects-database/gradient-boosting), with the publication using the original clinical data pending. Real clinical data was replaced by synthetic equivalents.