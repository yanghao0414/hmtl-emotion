# Data Sources, Variables, and Coding Scheme

This document describes the datasets, variables, label coding rules, and data transformations used in the study. It is written for journal editors, reviewers, and readers who need to verify the methods and results.

## Public Data Sources

### SMP2020-EWECT Text Dataset

- Source: SMP2020 Weibo Emotion Classification Technology Evaluation.
- Official page: https://smp2020ewect.github.io/
- Data type: Chinese social media text with emotion labels.
- Original label system: neutral, happy, angry, sad, fear, surprise.
- Use in this study: text-modality modeling and conversion to the hierarchical emotion taxonomy.
- Package location: `derived_data/text/`
- Main preprocessing script: `scripts/prepare_smp2020_dataset.py`

### FER2013 Facial Expression Dataset

- Source: Facial Expression Recognition 2013 dataset.
- Original Kaggle source: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- Reference page: https://datarepository.wolframcloud.com/resources/FER-2013
- Data type: facial expression images with expression labels.
- Original label system: anger, disgust, fear, happiness, sadness, surprise, neutral.
- Use in this study: visual-modality modeling after mapping expression categories to the study taxonomy.
- Package location: `derived_data/visual/`
- Main preprocessing scripts: `code/model_code/modules/visual_hmtl/` and `scripts/eval_visual_v4.py`

### CNSCED Audio Dataset

- Source: Chinese Natural Speech Complex Emotion Dataset.
- Science Data Bank page: https://www.scidb.cn/en/detail?dataSetId=394f27fbc9014cd486951b770fdefa10
- Project page: https://github.com/wuxlxju/CNSCED
- Data type: Chinese natural speech samples annotated with emotion information.
- Use in this study: audio-modality modeling after conversion to the study taxonomy.
- Package location: `derived_data/audio/`
- Main preprocessing scripts: `code/model_code/modules/audio_hmtl/`

## Restricted Research Data

The study also uses adolescent counseling text data collected in a psychological counseling context. The raw transcripts are not publicly released because they may contain sensitive mental-health-related content and potentially identifiable information involving minors.

For verification, the supplementary package provides:

- aggregate descriptions of the dataset,
- variable definitions,
- label coding rules,
- preprocessing scripts,
- transformation scripts,
- model training and evaluation code,
- statistical analysis scripts,
- summarized result files.

Raw restricted transcripts are not included in this repository or review package.

## Hierarchical Emotion Coding Scheme

The study uses a hierarchical multi-task target system. Each sample may be represented by five outputs: a fine-grained emotion label, a main emotion label, a polarity label, an arousal value, and a valence value.

### Seven-Class Fine-Grained Emotion Label

Variable name: `label_7` or `label_7_emotion`.

Integer coding:

- `0`: anger
- `1`: anxiety
- `2`: happiness
- `3`: sadness
- `4`: disappointment
- `5`: support
- `6`: calmness

### Four-Class Main Emotion Label

Variable name: `label_4` or `label_4_emotion`.

Integer coding:

- `0`: positive
- `1`: activated negative
- `2`: deactivated negative
- `3`: calmness

Mapping from the seven-class labels:

- positive: happiness, support
- activated negative: anger, anxiety
- deactivated negative: sadness, disappointment
- calmness: calmness

### Three-Class Polarity Label

Variable name: `label_3` or `label_3_polarity`.

Integer coding:

- `0`: positive
- `1`: negative
- `2`: calm or neutral

Mapping from the seven-class labels:

- positive: happiness, support
- negative: anger, anxiety, sadness, disappointment
- calm or neutral: calmness

### Arousal

Variable name: `arousal`.

- Type: continuous value.
- Range: `[0, 1]`.
- Meaning: emotional activation or intensity.
- Model activation: sigmoid.
- Interpretation: higher values indicate stronger emotional activation.

### Valence

Variable name: `valence`.

- Type: continuous value.
- Range: `[-1, 1]`.
- Meaning: affective positivity or negativity.
- Model activation: tanh.
- Interpretation: higher values indicate more positive affect; lower values indicate more negative affect.

## Data Transformations

### Text Modality

- Source labels are mapped to the hierarchical label system.
- Duplicate or low-information records are removed where applicable.
- Data are split into training and evaluation subsets.
- Text is tokenized with `bert-base-chinese`.
- Maximum sequence length is 128 tokens.

### Visual Modality

- Source facial expression categories are mapped to the study taxonomy.
- Images are resized to 224 x 224 pixels.
- Image tensors are normalized with ImageNet-style means and standard deviations.
- Derived labels are stored as CSV files for reproducibility.

### Audio Modality

- Source emotion annotations are mapped to the study taxonomy.
- Audio input is standardized to 16 kHz for Wav2Vec2-based processing.
- Audio features may be extracted or cached where applicable.
- Derived labels are stored as CSV files for reproducibility.

### Fusion Experiments

- Modality predictions are evaluated under weak-alignment, noise, missing-modality, and conflict scenarios.
- Fusion methods include majority voting, weighted voting, confidence-based fusion, and modality reliability estimation.
- Reported metrics include accuracy, macro-F1, robustness under missing modalities, seed stability, and statistical significance.

## Statistical Code and Result Files

Main experiment scripts are located in:

- `experiments/weakly_aligned_mre/`
- `experiments/plot_all_figures.py`

Main result files are located in:

- `results/prism_data/`
- `results/weakly_aligned_mre_outputs/`

The exact file list is recorded in `PACKAGE_MANIFEST.md` after the review package is generated.
