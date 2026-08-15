SEED-ML: Semen Examination and Evaluation Dataset for Machine Learning

1. General Information

Dataset Name: SEED-ML
Version: 1.0
Authors: Sánchez-Gómez, N.; García-García, J.A.; Navarro-Pando, J.; Escalona-Cuaresma, M.J.
Institution: University of Seville (ES3 Group) / INEBIR, Spain.
Contact: juliangg@us.es

Associated Publication: Sánchez-Gómez et al. (2025). "SEED-ML: A Multi-Parametric Clinical Dataset on Male Infertility for Predictive Modeling and AI Research." Data in Brief.

2. Dataset Overview

SEED-ML is a high-dimensional, multi-parametric clinical dataset designed to support research in computational andrology and artificial intelligence. It contains anonymized records of 10,124 patients extracted from Electronic Health Records (EHR) at a tertiary fertility center. The dataset includes standard semen analysis parameters, detailed morphological classifications, and specialized biochemical markers.

Class Distribution (Ground Truth: diagnostic). The dataset reflects the natural prevalence found in specialized clinical practice. Researchers should account for this inherent class imbalance during model training:

Normozoospermia (NO): 62.68%

OAT Syndrome: 14.22%

Asthenozoospermia: 11.66%

Teratozoospermia: 6.71%

Oligozoospermia: 1.90%

Asthenoteratozoospermia: 1.38%

Oligoasthenozoospermia: 0.96%

Oligoteratozoospermia: 0.34%

Azoospermia (AZOO): 0.16%

3. Data Structure (84 Variables)

To assist researchers from non-medical backgrounds (e.g., Data Scientists), the variables are organized into five functional clusters:

Macroscopic & Physical (Vars 1-9): Physical properties of the seminal plasma (e.g., pH, Volume, Viscosity).

Microscopic & Vitality (Vars 10-21): Core quantitative metrics (e.g., Concentration, Vitality, WHO Morphology).

Detailed Morphology (Pre-treatment) (Vars 22-53): Granular classification of 32 specific structural defects (Prefix: _pre).

Specialized Biomarkers (Vars 54-56): Molecular indicators including Fructose, Citric Acid, and DNA Fragmentation (SCD).

Laboratory Treatment Outcomes (Vars 57-84): Parameters measured after laboratory capacitation (Suffix: _final).

4. Technical Recommendations for Machine Learning

To ensure robust performance and clinical interpretability, the following strategies are recommended:

Feature Prioritization (Tiers):

Tier 1: Use Clusters 1 and 2 to establish clinical baseline performance.

Tier 2: Integrate Clusters 3 and 4 for deep phenotyping and Explainable AI (XAI) research (e.g., SHAP/LIME ranking).

Tier 3: Cluster 5 features should primarily be used as Target Variables for prognostic modeling rather than input features for diagnostic tasks.

Imbalance Management: Due to the "long-tail" distribution, we recommend using SMOTE (Synthetic Minority Over-sampling Technique) or weighted loss functions for minority classes (e.g., AZOO, Oligoterato).

Evaluation Metrics: Avoid simple Accuracy. Use Macro-F1 Score and Balanced Accuracy to properly evaluate performance across all nine diagnostic categories.

5. File Inventory

infertility_man_data.csv: Main dataset file (Delimiter: ;). 

6. Ethics and Privacy
All data have been fully anonymized in accordance with the General Data Protection Regulation (GDPR - EU 2016/679). No personal identifiers, specific birth dates, or location data are included to prevent re-identification.

7. Citation
If you use this dataset in your research, please cite the following:

Sánchez-Gómez, N., García-García, J. A., Navarro-Pando, J., & Escalona-Cuaresma, M. J. (2025). SEED-ML: A Multi-Parametric Clinical Dataset on Male Infertility for Predictive Modeling and AI Research. Mendeley Data, V1, doi: 10.17632/sc8rsz2vd7.1