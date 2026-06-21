# Supplementary Materials

This repository contains supplementary materials for the manuscript "Multimodal Emotion Recognition Based on Adaptive Fusion Strategy".

The materials include source code, data source documentation, variable definitions, coding schemes, preprocessing scripts, statistical analysis scripts, and tabulated result files used to support the reported findings.

## Repository Structure

```text
.
├── README.md
├── PACKAGE_MANIFEST.md
├── docs/
├── code/
├── scripts/
├── experiments/
├── results/
└── derived_data/
```

## Folder Descriptions

### `docs/`

Contains documentation for readers, editors, and reviewers.

- `DATA_AVAILABILITY_STATEMENT.md`: data availability statement and generative AI use statement.
- `DATA_SOURCES_AND_VARIABLES.md`: dataset sources, variable definitions, coding rules, and transformations.
- `REPRODUCIBILITY_INSTRUCTIONS.md`: instructions for verifying scripts and result files.
- `POINT_BY_POINT_RESPONSE_ANONYMOUS.md`: anonymized point-by-point response to the editorial technical check.
- `SUBMISSION_CHECKLIST.md`: checklist for journal resubmission.

### `code/`

Contains model-related code.

- `model_code/`: model definitions, loaders, multimodal fusion logic, and utility functions.
- `training_scripts/`: training scripts and notebooks for text, audio, and visual models.
- `evaluation_tools/`: model evaluation and test scripts.
- `utility_scripts/`: helper scripts for data processing, diagnostics, and figure preparation.

### `scripts/`

Contains standalone scripts for dataset preparation, model inspection, and data transformation.

### `experiments/`

Contains experimental scripts for weakly aligned multimodal fusion, including ablation studies, robustness analysis, multi-seed stability, statistical significance testing, and figure generation.

### `results/`

Contains result tables used to support the manuscript's tables and figures.

- `prism_data/`: tabulated data for figures and summary tables.
- `weakly_aligned_mre_outputs/`: raw output tables from fusion experiments and statistical analyses.

### `derived_data/`

Contains privacy-safe derived data files and label mapping files when redistribution is permitted.

- `text/`: transformed public text dataset splits.
- `visual/`: visual label mapping files.
- `audio/`: audio label mapping files.

Raw adolescent counseling transcripts are not included because they may contain sensitive mental-health-related information and potentially identifiable information involving minors.

## Public Data Sources

The original public datasets used in this study are available from:

- SMP2020-EWECT: https://smp2020ewect.github.io/
- FER2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- FER2013 reference page: https://datarepository.wolframcloud.com/resources/FER-2013
- CNSCED: https://www.scidb.cn/en/detail?dataSetId=394f27fbc9014cd486951b770fdefa10
- CNSCED project page: https://github.com/wuxlxju/CNSCED

## Data Availability

Publicly shareable derived files, preprocessing scripts, coding schemes, statistical scripts, and summarized result files are included in this package. Raw restricted adolescent counseling transcripts are not included due to privacy and ethical constraints.

