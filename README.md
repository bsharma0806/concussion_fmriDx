# RF Classifier with Synthetic Clinical Data

This project demonstrates a machine learning pipeline for classifying clinical outcomes using a synthetic dataset modeled after real-world data.  
It includes preprocessing, oversampling, model tuning, evaluation, and is structured for easy sharing and reproducibility.

Resting state functional magnetic resonance imaging (rs-fMRI) data have the potential to identify the neuropathology associated with concussion. Yet the rs-fMRI signal, a timeseries (known as the BOLD timeseries), can be analyzed and quantified in many ways. Which metric is the most discriminatory between those with concussion and healthy controls is unknown. 

This dataset uses 6 rs-fMRI metrics - namely the BOLD timeseries mean, standard deviation, entropy (measured by the Lyapunov exponent and Hurst exponent), and its amplitiude of low frequency fluctuations (ALFF) and fractional ALFF. For each region of interest (ROI), or anatomical region defined by a common neuroimaging atlas (Harvard-Oxford), all 6 rs-fMRI metrics are computed. 

This analysis allows for an understanding of the following: 
- Which metric-ROI pairs are most discriminatory between concussion subjects and healthy controls? 
- Is one metric better at classifying concusison than the others? 
- Is one ROI better at classifying concussion than the others? 

## Data

- `synthetic_data.csv` is generated using `generate_synthetic_data.py`
- Mirrors statistical structure of actual clinical metrics (not shared due to privacy)

## Features

- SMOTE for class imbalance
- Gradient Boosting classifier
- Hyperparameter tuning with RandomizedSearchCV
- Evaluation with ROC & Precision-Recall curves

## How to Use

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Generate synthetic data
python generate_synthetic_data.py

# Step 3: Open and run the notebook
jupyter notebook rf_classifier_synthetic.ipynb
```

## Reproducible

Use the script with any CSV in the same format to create safe-to-share synthetic data.

## Notes

Built as part of a data science portfolio. Real clinical data was replaced by synthetic equivalents.
