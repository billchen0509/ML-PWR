# Serum Glucose Depletion and Reduced Ketogenesis in Postpartum Weight Retention Revealed by Machine Learning Metabolomics Analysis

**Authors:** Hongjin Chen, Lisa M. Christian, Peyton Greenwood, Elena Papaleo, Matthias S. Klein

## Abstract

Postpartum weight retention (PWR) that persists for more than one year poses long-term health risks, including obesity and maternal metabolic disease. Still, the underlying metabolic mechanisms represent an important knowledge gap, and prediction of PWR risk remains challenging. Metabolomics is a promising avenue for identifying biomarkers for metabolic diseases, and we here present an integrative machine learning (ML) pipeline to predict PWR status across five time points during late pregnancy and postpartum using 1D 1H and 2D 1H-13C HSQC Nuclear Magnetic Resonance (NMR) spectroscopic data of maternal serum. Feature selection and a nested cross-validation workflow were employed to optimize nine ML models. Models combining metadata with selected 2D NMR features outperformed those using 1D NMR or metadata alone. Significant metabolic alterations were identified in individuals in the PWR group, many of which were visible already in the third trimester. This included reduced markers of physical activity, reduced markers of ketogenesis, alterations in branched-chain amino acid (BCAA) metabolism, and, reported for the first time, a consistent depletion of serum glucose levels during development of PWR. Predictive metabolites included, among others, serum glucose, creatinine, acetone, glutathione, ornithine, and xanthurenic acid. Our study demonstrates that integrating 2D NMR with ML increases predictive power for PWR and suggests complex metabolic dynamics. These findings provide novel mechanistic insights and candidate biomarkers for targeted, personalized interventions in postpartum health.

## Project Overview

This repository contains the data-processing, machine-learning, statistical-analysis, deployment, and visualization scripts used to investigate postpartum weight retention (PWR) from late pregnancy through the first postpartum year.

The workflow integrates participant metadata with 1D and 2D NMR serum metabolomic features to:

- compare multiple input configurations for PWR classification;
- identify predictive NMR features;
- evaluate model performance using nested cross-validation;
- examine metadata and metabolite differences between PWR and non-PWR groups;
- refit selected final models for prediction;
- visualize longitudinal metabolite patterns and prediction probabilities.

PWR is coded as `1` and non-PWR as `0` in the analysis scripts.

## Study Visits

| Visit | Time point |
| --- | --- |
| V1 | Third trimester |
| V2 | 4–6 weeks postpartum |
| V3 | 4 months postpartum |
| V4 | 8 months postpartum |
| V5 | 12 months postpartum |

## Input Data

The analysis uses three main types of data:

- participant metadata;
- 1D NMR features;
- 2D NMR features.

Participant-level source data are not included in the public repository because of privacy and ethics restrictions.

## Repository Structure

```text
project/
├── src/
│   ├── 01_process_data_merged.py
│   ├── 02_nested_cv_1d_feature_selection.py
│   ├── 02_nested_cv_2d_feature_selection.py
│   ├── 02_nested_cv_all_2d_nmr_only.py
│   ├── 02_nested_cv_meta_2d_nmr.py
│   ├── 02_nested_cv_meta_only.py
│   ├── 03_summary_results.py
│   ├── 04_visualization_2dvs1d.py
│   ├── 04_visualization_comparison.py
│   ├── 04_visualization_metaselected2dnmr.py
│   ├── 05_meta_association.py
│   ├── 06_deployment.py
│   ├── 07_deployment_summary.py
│   ├── 08_best_performing_1D_2D_models.py
│   ├── 09_Line_plots_1D.py
│   ├── 09_Line_plots_2D.py
│   ├── 09_Line_plots_combined.py
│   ├── 10_Table1.py
│   └── 11_prediction_heatmap.py
├── test/
│   └── result/
│       ├── data/                 # processed data, model summaries, and Table 1 outputs
│       ├── model/                # model-training outputs for selected configurations
│       ├── meta/                 # metadata association results
│       ├── deploy/               # final fitted models and prediction outputs
│       │   └── prediction/
│       └── figure/               # generated figures
└── README.md
```

## Analysis Pipeline

### 1. Data processing

`01_process_data_merged.py` preprocesses metadata and merges it with the 1D and 2D NMR datasets for each visit.

Main steps include:

- removing participants without a PWR outcome;
- cleaning invalid or missing metadata values;
- imputing missing gestational weight gain (`wt_gain`) using the relationship with `weight_change_v1`;
- constructing visit-specific metadata tables;
- extracting participant IDs and visit labels from NMR sample identifiers;
- merging metadata with 1D and 2D NMR data;
- generating visit-specific filtered datasets used for downstream analyses.

### 2. Nested cross-validation model training

Five input configurations are evaluated:

1. **Meta Only**
2. **All 2D NMR Only**
3. **Meta + All 2D NMR**
4. **Meta + Selected 1D NMR**
5. **Meta + Selected 2D NMR**

Nine machine-learning classifiers are evaluated:

- Logistic Regression
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Support Vector Machine
- Random Forest
- K-Nearest Neighbors
- XGBoost
- Decision Tree
- MultiLayer Perceptron

Model development uses stratified nested cross-validation with:

- 10-fold outer cross-validation for performance estimation;
- 10-fold inner cross-validation for hyperparameter tuning;
- ROC AUC as the hyperparameter-selection metric;
- `random_state=42` where applicable.

Reported metrics include accuracy, F1 score, precision, recall, ROC AUC, standard deviations, and standard errors across folds.

### 3. NMR feature selection

For the selected 1D and selected 2D NMR configurations, NMR variables are ranked using Welch's independent-samples t-test within the outer training fold only.

The workflow evaluates top-N feature sets from **5 to 25 NMR features**. Feature selection is repeated separately within each outer fold to reduce information leakage from the held-out test fold.

### 4. Result summarization

`03_summary_results.py` standardizes metric names across model-output formats and generates combined summaries, including:

- the best model for each input configuration and visit;
- the overall best configuration for each visit;
- a combined table of model results across configurations.

### 5. Model-performance visualization

The `04_*.py` scripts generate model-performance figures, including:

- comparisons between selected 1D and selected 2D NMR models;
- comparisons across the five input configurations;
- performance changes across visits;
- ROC curves for selected best-performing models;
- calibration plots and Brier scores for selected models.

Visit labels are displayed using clinically interpretable time-point names rather than only V1–V5.

### 6. Metadata association analysis

`05_meta_association.py` tests associations between participant metadata and PWR status.

The script applies:

- Mann–Whitney U tests for continuous variables;
- Fisher's exact test for eligible 2 × 2 categorical tables;
- chi-square tests for larger categorical tables;
- Benjamini–Hochberg false-discovery-rate correction.

Results are generated both globally and separately by visit.

### 7. Final model refitting and deployment

`06_deployment.py` refits selected final **Meta + Selected 2D NMR** models using the full available training dataset for each visit.

The current deployment plan is:

| Visit | Final model | Selected 2D NMR features |
| --- | --- | ---: |
| V1 | MultiLayer Perceptron | 20 |
| V2 | MultiLayer Perceptron | 10 |
| V3 | Logistic Regression | 5 |
| V4 | Random Forest | 15 |
| V5 | Logistic Regression | 8 |

The script saves:

- final fitted pipelines;
- best hyperparameters;
- selected NMR feature names;
- prediction probabilities and binary predictions.

The current prediction workflow generates held-out prediction files for V1–V4 using a probability threshold of `0.5`.

### 8. Deployment summary

`07_deployment_summary.py` merges visit-specific prediction files into a participant-level wide-format table containing the predicted class and probability for each available visit.

### 9. Best-performing 1D vs 2D models

`08_best_performing_1D_2D_models.py` identifies the highest-performing selected 1D and selected 2D model at each visit and generates a side-by-side comparison of CV ROC AUC values with standard-error bars.

### 10. Longitudinal metabolite analysis

The `09_Line_plots_*.py` scripts visualize longitudinal NMR signal patterns in PWR and non-PWR groups.

For metabolites with multiple annotated NMR signals, the signal with the highest average intensity across visits is selected. At each visit, PWR and non-PWR groups are compared using Welch's t-test.

The scripts also calculate Benjamini–Hochberg FDR-adjusted p-values. The current plotting code displays significance markers using raw `p <= 0.05`.

Available scripts include:

- `09_Line_plots_1D.py`: longitudinal plots for annotated 1D NMR metabolites;
- `09_Line_plots_2D.py`: longitudinal plots for annotated 2D NMR metabolites;
- `09_Line_plots_combined.py`: combined figure for selected 1D and 2D metabolites highlighted in the manuscript.

The combined manuscript-focused plot includes seven selected 2D metabolites and seven selected 1D metabolites.

### 11. Participant characteristics and Table 1

`10_Table1.py` generates the manuscript participant-characteristics tables comparing PWR and non-PWR groups.

Outputs include:

- an overall characteristics table;
- a visit-specific behavioral and weight-change table;
- an Excel workbook containing both tables and analysis notes.

The analysis includes continuous and categorical group comparisons and Benjamini–Hochberg FDR correction across the valid tests in Tables 1a and 1b.

Vigorous physical activity is additionally converted to an estimated numeric frequency per month for the group comparison:

| Original category | Estimated times/month |
| --- | ---: |
| < once/month | 0 |
| once/month | 1 |
| 2–3 times/month | 2.5 |
| once/week | 4 |
| > once/week | 8 |

This numeric physical-activity variable is compared between groups using Student's independent-samples t-test.

### 12. Prediction heatmap

`11_prediction_heatmap.py` generates a participant-by-visit heatmap of predicted PWR probabilities.

- darker cells represent higher predicted probabilities;
- missing predictions are displayed in light gray with `-`;
- probability values are printed within available cells.

## Main Dependencies

The analysis uses the following Python packages:

```text
pandas
numpy
scipy
statsmodels
scikit-learn
xgboost
matplotlib
seaborn
joblib
openpyxl
xlsxwriter
```

## Usage

### Data and directory requirements

Before running the pipeline, prepare:

1. **Private source metadata files** Request from the corresponding author.
2. **External manuscript metabolite tables**  Request from the corresponding author.
3. **Deployment test splits**  Request from the corresponding author.

### Recommended execution order

Core dependency order:

1. `01_process_data_merged.py`
2. all `02_nested_cv_*.py` scripts
3. `03_summary_results.py`
4. `04_visualization_*.py`
5. `05_meta_association.py`
6. `06_deployment.py`
7. `07_deployment_summary.py`
8. `08_best_performing_1D_2D_models.py`
9. `09_Line_plots_*.py`
10. `10_Table1.py`
11. `11_prediction_heatmap.py`

```bash
cd src

python 01_process_data_merged.py

python 02_nested_cv_meta_only.py
python 02_nested_cv_all_2d_nmr_only.py
python 02_nested_cv_meta_2d_nmr.py
python 02_nested_cv_1d_feature_selection.py
python 02_nested_cv_2d_feature_selection.py

python 03_summary_results.py

python 04_visualization_2dvs1d.py
python 04_visualization_comparison.py
python 04_visualization_metaselected2dnmr.py

python 05_meta_association.py

python 06_deployment.py
python 07_deployment_summary.py

python 08_best_performing_1D_2D_models.py

python 09_Line_plots_1D.py
python 09_Line_plots_2D.py
python 09_Line_plots_combined.py

python 10_Table1.py
python 11_prediction_heatmap.py
```

## Key Outputs

Outputs are written under `test/result/` and include:

- visit-specific merged and filtered datasets;
- nested cross-validation model-performance tables;
- selected NMR feature records;
- combined model summaries;
- metadata association results;
- final refitted model pipelines;
- prediction probabilities and predicted labels;
- participant-level prediction summaries;
- ROC and calibration figures;
- model-comparison figures;
- longitudinal 1D and 2D metabolite figures;
- manuscript Table 1 outputs;
- participant prediction heatmaps.

## Data Availability

Due to privacy and ethics restrictions, participant-level data are not publicly uploaded to this repository. Data may be available from the authors upon reasonable request and subject to applicable data-sharing agreements and ethics approvals.

## Citation

If you use this repository, please cite the associated manuscript:

Chen, H., Christian, L. M., Greenwood, P., Papaleo, E., & Klein, M. S. (in preparation). Metabolomic models reveal serum glucose depletion and reduced ketogenesis during development of postpartum weight retention.

The citation will be updated upon publication.
