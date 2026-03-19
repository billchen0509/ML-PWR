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
 


