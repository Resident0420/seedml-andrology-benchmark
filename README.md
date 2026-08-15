# A Reproducible Benchmarking Framework for Machine Learning–Based Multiclass Male Infertility Diagnosis Using the SEED-ML Dataset

## Authors
**Jumin Kim**¹²  
**Kang Byeong Jin**³
¹ Sidney Kimmel Medical College, Thomas Jefferson University, Philadelphia, PA, USA  
² Faculty of Medicine and Surgery, Università Cattolica del Sacro Cuore, Rome, Italy  
³ Department of Urology, Pusan National University Hospital, Busan, Republic of Korea

## Overview

This repository contains the code accompanying the manuscript:

> **A Reproducible Benchmarking Framework for Machine Learning–Based Multiclass Male Infertility Diagnosis Using the SEED-ML Dataset**

The project benchmarks multiple machine learning algorithms for multiclass male infertility diagnosis using routine semen analysis data from the publicly available SEED-ML dataset.

## Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Multilayer Perceptron (MLP)

## Evaluation

Models were evaluated using:

- Repeated Stratified 5-fold Cross-Validation (5 repeats)
- Macro F1 Score
- Balanced Accuracy
- Weighted F1 Score

The highest-performing model was further evaluated using:

- Confusion Matrix
- Per-class Precision, Recall, and F1
- ROC Curves
- Precision–Recall Curves
- Calibration Curves
- Multiclass Brier Score
- SHAP Explainability

## Repository Structure

```
Benchmark/
│
├── data/
├── notebooks/
├── src/
├── results/
├── README.md
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/Resident0420/seedml-andrology-benchmark.git

cd seedml-andrology-benchmark

pip install -r requirements.txt
```

## Dataset

This repository does **not** include the SEED-ML dataset.

Please download Version 3 of the dataset from:

https://data.mendeley.com/datasets/sc8rsz2vd7/3

Sánchez-Gómez N, Garcia-Garcia JA, Navarro-Pando J, Escalona-Cuaresma MJ.
*SEED-ML: A Multi-Parametric Clinical Dataset on Male Infertility for Predictive Modeling and AI Research.*
Mendeley Data. Version 3.
doi:10.17632/sc8rsz2vd7.3

```
data/
    SEED-ML/
        infertility_man_data-v2.csv
        README.txt
```

## Running the Benchmark

Run the benchmark notebook or execute the benchmarking pipeline:

```
Notebook 5
```

or

```python
from src.benchmark import benchmark_models
```

## Reproducibility

All experiments were performed using:

- Fixed random seed (42)
- Standardized preprocessing
- Identical evaluation metrics
- Repeated Stratified Cross-Validation

## Citation

If you use this repository, please cite:

Kim J, Jin KB.

*A Reproducible Benchmarking Framework for Machine Learning–Based Multiclass Male Infertility Diagnosis Using the SEED-ML Dataset.*

(Under review)

## License

MIT License.