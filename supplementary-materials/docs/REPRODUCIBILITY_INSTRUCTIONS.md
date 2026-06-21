# Reproducibility Instructions

This document explains how to inspect or rerun the supplementary code and result-generation scripts.

## Environment

The project is Python-based. GPU acceleration is recommended for full model training, but many result files and statistical checks can be inspected or rerun on CPU.

Recommended core packages:

```bash
pip install torch transformers numpy pandas scikit-learn matplotlib torchvision librosa pillow
```

Optional package for the demo interface:

```bash
pip install streamlit
```

## Code Locations

- Model code: `code/model_code/`
- Training scripts: `code/training_scripts/`
- Evaluation tools: `code/evaluation_tools/`
- Utility scripts: `code/utility_scripts/`
- Standalone preparation scripts: `scripts/`
- Fusion experiments: `experiments/weakly_aligned_mre/`
- Plotting script: `experiments/plot_all_figures.py`

## Data Preparation

Text dataset transformation:

```bash
python scripts/prepare_smp2020_dataset.py
```

Restricted counseling dataset transformation:

```bash
python scripts/prepare_full_dataset_v3.py
```

Visual label mapping:

```bash
python code/model_code/modules/visual_hmtl/prepare_visual_dataset.py
```

Audio label mapping:

```bash
python code/model_code/modules/audio_hmtl/generate_label7_for_audio.py
```

Some commands require original public datasets downloaded from the source providers. Raw restricted counseling transcripts are not distributed in this package.

## Fusion and Robustness Experiments

Run ablation experiments:

```bash
python experiments/weakly_aligned_mre/ablation_study.py
```

Run supplementary robustness experiments:

```bash
python experiments/weakly_aligned_mre/supplementary_experiments.py
```

Run adaptability and stability experiments:

```bash
python experiments/weakly_aligned_mre/adaptability_stability_experiments.py
```

Run statistical significance testing:

```bash
python experiments/weakly_aligned_mre/exp_significance_test.py
```

Run synthetic robustness verification:

```bash
python experiments/weakly_aligned_mre/exp_synth_robustness.py
```

Run normality checks:

```bash
python experiments/weakly_aligned_mre/exp_normality_check.py
```

Generate figures:

```bash
python experiments/plot_all_figures.py
```

## Result Files

Result tables are stored in:

- `results/prism_data/`
- `results/weakly_aligned_mre_outputs/`

The file `PACKAGE_MANIFEST.md` lists all files included in the generated review package.

## Model Checkpoints

Model checkpoint files are not included by default because they are large binary files and are unsuitable for normal repository tracking. The model architectures and loading logic are included in `code/model_code/`.

If model checkpoints are requested by editors or reviewers, they should be provided through a stable large-file storage link with file names, versions, and checksums documented separately.

## Restricted Data

Raw adolescent counseling transcripts are excluded because they may contain sensitive mental-health-related information and potentially identifiable information involving minors. Verification is supported through:

- coding schemes,
- transformation scripts,
- model code,
- experiment scripts,
- statistical analysis scripts,
- aggregate descriptions,
- summarized result files.
