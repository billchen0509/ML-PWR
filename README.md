Metabolomic models reveal serum glucose depletion and reduced ketogenesis during development of Postpartum Weight Retention
Author: Hongjin Chen, Lisa M. Christian, Peyton Greenwood, Elena Papaleo, Matthias S. Klein

## Abstract
Postpartum weight retention (PWR) that persists for more than one year poses long-term health risks, including obesity and maternal metabolic disease. Still, the underlying metabolic mechanisms represent an important knowledge gap, and prediction of PWR risk remains challenging. Metabolomics is a promising avenue for identifying biomarkers for metabolic diseases, and we here present an integrative machine learning (ML) pipeline to predict PWR status across five time points during late pregnancy and postpartum using 1D 1H and 2D 1H-13C HSQC Nuclear Magnetic Resonance (NMR) spectroscopic data of maternal serum. Feature selection and a nested cross-validation workflow were employed to optimize nine ML models. Models combining metadata with selected 2D NMR features outperformed those using 1D NMR or metadata alone. Significant metabolic alterations were identified in individuals in the PWR group, many of which were visible already in the third trimester. This included reduced markers of physical activity, reduced markers of ketogenesis, alterations in branched-chain amino acid (BCAA) metabolism, and, reported for the first time, a consistent depletion of serum glucose levels during development of PWR. Predictive metabolites included, among others, serum glucose, creatinine, acetone, glutathione, ornithine, and xanthurenic acid. Our study demonstrates that integrating 2D NMR with ML increases predictive power for PWR and suggests complex metabolic dynamics. These findings provide novel mechanistic insights and candidate biomarkers for targeted, personalized interventions in postpartum health.

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
## Usage

Example workflow:

```bash
python src/01_process_data_merged.py

python src/02_nested_cv_meta_only.py
python src/02_nested_cv_all_2d_nmr_only.py
python src/02_nested_cv_meta_2d_nmr.py
python src/02_nested_cv_1d_feature_selection.py
python src/02_nested_cv_2d_feature_selection.py

python src/03_summary_results.py

python src/04_visualization_2dvs1d.py
python src/04_visualization_comparison.py
python src/04_visualization_metaselected2dnmr.py

python src/05_meta_association.py

python src/06_deployment.py
python src/06_deployment_summary.py

python src/07_boxplot_comparison.py
```
## Data Availability

Due to privacy restrictions, participant-level data in `test/result/data/` are not publicly uploaded to this repository. Data may be available from the authors upon reasonable request and subject to applicable data-sharing and ethics restrictions.

## Citation

If you use this repository, please cite the associated manuscript when available.

## Outputs

The repository produces outputs under `test/result/`, including:
- processed datasets
- deployment results
- generated figures

