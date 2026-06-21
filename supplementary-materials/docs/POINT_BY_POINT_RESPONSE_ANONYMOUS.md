# Point-by-Point Response to the Editor

Manuscript title: Multimodal Emotion Recognition Based on Adaptive Fusion Strategy

Submission ID: d527bc63-5f10-4fbe-a06b-cae892a04f77

Decision: Minor revision

Important note: This response is intentionally anonymized. Do not add author names, affiliations, email addresses, ORCID IDs, funding acknowledgements, or other identifying information to this file.

## General Response

We thank the editor for the careful technical check and for the opportunity to revise the manuscript. We have addressed the data availability and supporting documentation requirements by preparing a complete review package containing data source information, variable definitions, coding schemes, data transformation descriptions, preprocessing scripts, experimental scripts, statistical code, and result files. We have also added a Data Availability statement to the revised manuscript and will enter the same statement in the submission system under Declarations.

## Comment 1

Request for data and supporting documentation. To support peer review, all datasets and/or materials underpinning the research should be made available to the referees and editors, including full raw datasets and refined or cleaned versions, variable definitions, coding schemes, transformations, formulae, statistical code, data processing scripts, and any additional documentation required to verify the methods and findings. If files are not uploaded directly, stable links to public repositories should be provided. The manuscript must include a Data Availability statement explaining what data have been shared, where they can be accessed, and why any data cannot be shared.

## Response to Comment 1

We have prepared a complete data and reproducibility package for editorial and peer-review verification. The package includes:

- Data source documentation for the public datasets used in the study: SMP2020-EWECT, FER2013, and CNSCED.
- Stable source links for the public datasets.
- A description of the restricted adolescent counseling text data and the reason why raw transcripts cannot be publicly released.
- Variable definitions and coding schemes for the 7-class emotion label, 4-class main emotion label, 3-class polarity label, arousal score, and valence score.
- Data transformation and preprocessing documentation for text, visual, audio, and multimodal fusion experiments.
- Source code for model definitions, data preprocessing, label mapping, model evaluation, fusion experiments, statistical significance testing, robustness experiments, and figure generation.
- Tabulated result files used to support the reported tables and figures.
- Reproducibility instructions explaining how reviewers can inspect or rerun the key analyses.

The raw adolescent counseling transcripts are not publicly released because they may contain sensitive mental-health-related content and potentially identifiable information involving minors. To enable verification without compromising privacy, we provide anonymized aggregate statistics, label definitions, coding schemes, preprocessing and transformation scripts, experimental configurations, training/evaluation code, statistical analysis scripts, and summarized result files. Access to restricted materials may be considered upon reasonable request, subject to ethical approval, data-use agreements, and privacy protection requirements.

The following Data Availability statement has been added to the end of the revised manuscript:

> The data, preprocessing scripts, variable coding schemes, experimental code, and main result files used in this study are available as supplementary materials or through a public repository prepared for peer review. The shared materials include source-code files, preprocessing and transformation scripts, label mapping rules, experimental configurations, statistical analysis scripts, and tabulated result files used to generate the reported tables and figures.
>
> The study uses publicly available datasets and restricted research data. The public datasets include the SMP2020-EWECT Weibo Emotion Classification dataset, the FER2013 facial expression dataset, and the Chinese Natural Speech Complex Emotion Dataset (CNSCED). These datasets can be accessed from their original sources: SMP2020-EWECT at https://smp2020ewect.github.io/, FER2013 at https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data, and CNSCED at https://www.scidb.cn/en/detail?dataSetId=394f27fbc9014cd486951b770fdefa10.
>
> Cleaned or transformed files derived from the public datasets are shared where redistribution is permitted and where the files do not contain restricted personal information. This study also uses adolescent counseling text data collected in a psychological counseling context. Because these data may contain sensitive mental-health-related content and potentially identifiable information involving minors, the raw counseling transcripts cannot be made publicly available. To support peer review and verification, the authors provide anonymized aggregate statistics, label definitions, coding schemes, preprocessing and transformation scripts, experimental configurations, model training and evaluation code, and summarized result files. Access to restricted materials may be considered upon reasonable request, subject to ethical approval, data-use agreements, and privacy protection requirements.

## Comment 2

Please provide the Data Availability statement on the submission system under Declarations as well. The journal does not accept "NO" in this section. The same Data Availability statement should be included in both the manuscript and the submission system.

## Response to Comment 2

We have prepared the Data Availability statement for inclusion in both the revised manuscript and the Declarations section of the submission system. We will not enter "NO" in the Declarations field. The statement entered in the submission system will match the Data Availability statement included at the end of the revised manuscript.

For the Declarations field, we will use the following version:

> The data, preprocessing scripts, variable coding schemes, experimental code, and main result files used in this study are available as supplementary materials or through a public repository prepared for peer review. The public datasets used in this study are SMP2020-EWECT, FER2013, and CNSCED, available from https://smp2020ewect.github.io/, https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data, and https://www.scidb.cn/en/detail?dataSetId=394f27fbc9014cd486951b770fdefa10, respectively. Cleaned or transformed files derived from these public datasets are shared where redistribution is permitted. The adolescent counseling text data used in this study cannot be publicly released because they may contain sensitive mental-health-related information and potentially identifiable information involving minors. To enable verification, anonymized aggregate statistics, label definitions, coding schemes, preprocessing scripts, experimental configurations, training/evaluation code, statistical analysis scripts, and summarized result files are provided. Access to restricted materials may be considered upon reasonable request, subject to ethical approval, data-use agreements, and privacy protection requirements.

## Final Check

We have checked that this response letter does not contain author names, affiliations, email addresses, ORCID IDs, or other author-identifying details.
