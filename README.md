# Metabolomic Biomarker Prediction in Postpartum Weight Retention (PWR) using Machine Learning Methods
Author: Hongjin Chen, Peyton Greenwood, Lisa M. Christian, Matthias S. Klein

## Abstract
Postpartum weight retention (PWR) poses risks for long-term maternal metabolic disease, yet the underlying metabolic mechanisms represent an important knowledge gap, and prediction of PWR risk remains challenging. We developed an integrative machine learning (ML) pipeline to predict PWR status across five time points during pregnancy and postpartum using 1D 1H and 2D 1H-13C HSQC Nuclear Magnetic Resonance (NMR) spectroscopic data of maternal serum and clinical metadata. Feature selection and a two-layer nested cross-validation workflow were employed to optimize nine ML models. Models combining metadata with selected 2D NMR features outperformed those using 1D NMR or metadata alone, highlighting the superior informativeness of 2D NMR. Pathway enrichment analysis linked branched-chain amino acid (BCAA) metabolism to insulin resistance in PWR, potentially via the mTORC1/S6K1-IRS1 pathway. Among 52 identified candidate metabolites, 1,5-anhydrosorbitol, glutathione, ornithine, and xanthurenic acid, were prioritized as candidate biomarkers reflecting alterations in glycemic regulation, redox homeostasis, and amino acid and energy metabolism in PWR. Our study demonstrates that integrating 2D NMR with ML effectively predicts PWR and suggests complex metabolic dynamics. These findings provide novel mechanistic insights and candidate biomarkers for targeted, personalized interventions in postpartum health.

### Objectives
- Evaluate predictive performance of metadata, 1D NMR, 2D NMR, and combined feature sets
- Compare model performance across visits
- Identify selected metabolite features associated with PWR
- Visualize group-level metabolite patterns and model results

### Input Data
- Participant metadata
- 1D NMR features
- 2D NMR features



 


