# Reviewer Materials README

## Purpose

This supplementary package supports the manuscript titled "Multimodal Emotion Recognition Based on Adaptive Fusion Strategy". It provides documentation, source code, derived data files, statistical scripts, and result tables required for editors and reviewers to examine the methods and verify the reported findings.

The package is intentionally anonymized. It should not contain author names, affiliations, email addresses, ORCID identifiers, or institution-specific details.

## Package Structure

```text
review_materials/
  README.md
  PACKAGE_MANIFEST.md
  docs/
  code/
  scripts/
  experiments/
  results/
  derived_data/
```

## Folder and File Descriptions

### `README.md`

Top-level guide for readers. It explains what the supplementary package is for and describes each folder.

### `PACKAGE_MANIFEST.md`

Automatically generated list of files included in the review package. It also records the exclusion policy for large, restricted, or identifying files.

### `docs/`

Documentation for editors and reviewers.

- `DATA_AVAILABILITY_STATEMENT.md`: data availability statement and generative AI use statement.
- `DATA_SOURCES_AND_VARIABLES.md`: public data sources, restricted data explanation, variable definitions, and coding scheme.
- `REPRODUCIBILITY_INSTRUCTIONS.md`: instructions for inspecting or rerunning the scripts.
- `POINT_BY_POINT_RESPONSE_ANONYMOUS.md`: anonymized response letter for the editorial technical check.
- `SUBMISSION_CHECKLIST.md`: checklist for uploading revised materials.

### `code/`

Model and evaluation source code.

- `model_code/`: text, audio, visual, and fusion model definitions.
- `training_scripts/`: model training scripts and notebooks.
- `evaluation_tools/`: evaluation utilities and testing scripts.
- `utility_scripts/`: helper scripts for data transformation, figure generation, and diagnostics.

### `scripts/`

Standalone data preparation, model inspection, and dataset transformation scripts.

### `experiments/`

Experimental scripts for weakly aligned multimodal fusion, ablation studies, robustness analysis, significance testing, and figure generation.

### `results/`

Tabulated result files used to support the reported tables and figures.

- `prism_data/`: CSV files used for plotting and table preparation.
- `weakly_aligned_mre_outputs/`: outputs from ablation, robustness, multi-seed, and significance experiments.

### `derived_data/`

Privacy-safe derived or transformed data files, when redistribution is permitted.

- `text/`: transformed text dataset splits derived from public text data.
- `visual/`: visual label mapping files.
- `audio/`: audio label mapping files.

Raw restricted adolescent counseling transcripts are not included.

## Public Data Sources

The original public datasets are available from:

- SMP2020-EWECT: https://smp2020ewect.github.io/
- FER2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- FER2013 reference page: https://datarepository.wolframcloud.com/resources/FER-2013
- CNSCED: https://www.scidb.cn/en/detail?dataSetId=394f27fbc9014cd486951b770fdefa10
- CNSCED project page: https://github.com/wuxlxju/CNSCED

## Restricted Data

Raw adolescent counseling transcripts are not shared because they may contain sensitive mental-health-related information and potentially identifiable information involving minors. To enable verification, this package provides coding schemes, preprocessing scripts, derived aggregate files where appropriate, experiment scripts, statistical code, and result tables.

## Excluded Files

The package excludes:

- raw restricted counseling transcripts,
- raw media files,
- model checkpoints,
- compressed archives,
- Office documents,
- cache files,
- IDE metadata,
- files likely to contain author-identifying metadata.

