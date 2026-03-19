# Metabolomic Biomarker Prediction in Postpartum Weight Retention (PWR) using Machine Learning Methods
Author: Hongjin Chen, Peyton Greenwood, Lisa M. Christian, Matthias S. Klein

## Abstract
Postpartum weight retention (PWR) poses risks for long-term maternal metabolic disease, yet the underlying metabolic mechanisms represent an important knowledge gap, and prediction of PWR risk remains challenging. We developed an integrative machine learning (ML) pipeline to predict PWR status across five time points during pregnancy and postpartum using 1D 1H and 2D 1H-13C HSQC Nuclear Magnetic Resonance (NMR) spectroscopic data of maternal serum and clinical metadata. Feature selection and a two-layer nested cross-validation workflow were employed to optimize nine ML models. Models combining metadata with selected 2D NMR features outperformed those using 1D NMR or metadata alone, highlighting the superior informativeness of 2D NMR. Pathway enrichment analysis linked branched-chain amino acid (BCAA) metabolism to insulin resistance in PWR, potentially via the mTORC1/S6K1-IRS1 pathway. Among 52 identified candidate metabolites, 1,5-anhydrosorbitol, glutathione, ornithine, and xanthurenic acid, were prioritized as candidate biomarkers reflecting alterations in glycemic regulation, redox homeostasis, and amino acid and energy metabolism in PWR. Our study demonstrates that integrating 2D NMR with ML effectively predicts PWR and suggests complex metabolic dynamics. These findings provide novel mechanistic insights and candidate biomarkers for targeted, personalized interventions in postpartum health.

### Project Overview
This project investigates whether metadata and NMR-based metabolomic features can improve prediction of postpartum weight retention across visits V1–V5.

### Input Data
- Participant metadata
- 1D NMR features
- 2D NMR features

### Repository Structure
```text
project/
├── src/                         # analysis scripts
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
│   ├── 06_deployment_summary.py
│   └── 07_boxplot_comparison.py
├── test/
│   └── result/
│       ├── data/                # processed datasets (not uploaded publicly)
│       ├── deploy/              # deployment-ready models and prediction outputs
│       └── figure/              # generated figures
└── README.md
```
## Analysis Pipeline

The overall workflow of the repository is:

### 1. Data processing
- Load metadata
- Construct visit-specific merged datasets
- Merge metadata with 1D and 2D NMR features
- Save processed analysis files

### 2. Model training
Run nested cross-validation under multiple input configurations:
- Meta Only
- All 2D NMR Only
- Meta + All 2D NMR
- Meta + Selected 1D NMR
- Meta + Selected 2D NMR

### 3. Feature selection
- Rank NMR variables within training folds only
- Evaluate different top-N selected feature sets for 1D and 2D models

### 4. Result summarization
- Combine model outputs across configurations
- Identify best-performing models by visit

### 5. Visualization
- Compare model performance across visits and feature configurations
- Generate metabolite comparison boxplots
- Produce selected model performance figures

### 6. Metadata association
- Test metadata variables associated with PWR
- Apply multiple-comparison correction

### 7. Deployment
- Refit selected final models
- Save prediction outputs and deployment summaries

## Scripts

### Data processing
- `src/01_process_data_merged.py`: Builds visit-specific merged datasets from metadata and NMR inputs.

### Model training
- `src/02_nested_cv_meta_only.py`: Nested cross-validation using metadata only.
- `src/02_nested_cv_all_2d_nmr_only.py`: Nested cross-validation using all 2D NMR features only.
- `src/02_nested_cv_meta_2d_nmr.py`: Nested cross-validation using metadata + all 2D NMR features.
- `src/02_nested_cv_1d_feature_selection.py`: Nested cross-validation using metadata + selected 1D NMR features.
- `src/02_nested_cv_2d_feature_selection.py`: Nested cross-validation using metadata + selected 2D NMR features.

### Result summarization
- `src/03_summary_results.py`  
  Aggregates results and selects best models by visit.

### Visualization
- `src/04_visualization_2dvs1d.py`  
  Compares selected 1D and 2D models.
- `src/04_visualization_comparison.py`  
  Compares configurations across visits.
- `src/04_visualization_metaselected2dnmr.py`  
  Generates detailed plots for Meta + Selected 2D NMR.

### Metadata association
- `src/05_meta_association.py`  
  Tests metadata variables associated with PWR.

### Deployment
- `src/06_deployment.py`  
  Refits final selected models and exports deployment outputs.
- `src/06_deployment_summary.py`  
  Summarizes deployment predictions across visits.

### Group comparison
- `src/07_boxplot_comparison.py`  
  Generates metabolite boxplots for PWR vs Non-PWR across visits.

## Data Availability

Due to privacy restrictions, participant-level data in `test/result/data/` are not publicly uploaded to this repository. Data may be available from the authors upon reasonable request and subject to applicable data-sharing and ethics restrictions.

## Outputs

The repository produces outputs under `test/result/`, including:
- processed datasets
- deployment results
- generated figures

