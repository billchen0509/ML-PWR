# Metabolomic Biomarker Prediction in Postpartum Weight Retention (PWR) using Machine Learning Methods
Author: Hongjin Chen
# Abstract
Postpartum weight retention (PWR) is a significant health concern for longterm maternal metabolic disease. However, predicting PWR remains a challenge, and the underlying
metabolic mechanisms and biomarkers are not fully understood. We applied an integrative machine learning pipeline to predict PWR status across five time points using metabolomic profiles from both 1D and 2D NMR spectroscopy data combined with clinical metadata. Feature selection was performed to identify the most informative NMR
features, and these features were then iteratively combined with metadata into a twolevel
nested crossvalidation
workflow to tune model hyperparameters and evaluate the performance
of nine different machine learning models. Biological interpretation was conducted
through pathway enrichment analysis and comparison of metabolite concentrations between
PWR and NonPWR
groups. The MultiLayer Perceptron (MLP), Random Forest
(RF) and Logistic Regression (LR) models that combined metadata with featureselected
2D NMR data (Meta + Selected 2D NMR) achieved higher crossvalidated
ROC AUC
than models using metadata alone, 1D NMR data or all 2D NMR features across visits.
These results demonstrate the informativeness of 2D NMR compared to 1D NMR data
and the practicability of feature selection using statistical analysis combined with machine
learning models in future clinical implementation. Pathway enrichment analysis revealed
valine, leucine, and isoleucine biosynthesis and degradation as the most enriched pathways,
supporting a link between BranchedChain
Amino Acids (BCAAs) metabolism and
insulin resistance in PWR, potentially via the mTORC1/S6K1IRS1
pathway. We also
identified 42 metabolites potentially associated with PWR; among them, Betaine, 1,5Anhydrosorbitol,
Acetoacetic acid, and 2Hydroxybutyric
acid emerged as the most compelling
biomarker candidates, plausibly reflecting metabolic recovery, glycemic control,
and energy mobilization. This study suggests that machine learning approaches integrating
metabolomic data from 2D NMR with clinical variables can effectively predict PWR.
The results provide novel understandings into the dynamic metabolic changes during the
postpartum period, and those metabolites identified as candidate biomarkers of PWR and
the potential metabolic mechanisms of insulin resistance in PWR via mTORC/S6K1IRS1
require further validation. This work illustrates how predictive machine learning models
can uncover complex patterns in metabolomics studies and potentially provide targeted,
personalized intervention for PWR.
