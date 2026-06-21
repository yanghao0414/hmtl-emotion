# Minor Revision Submission Checklist

Deadline: 23 Jun 2026

Submission ID: d527bc63-5f10-4fbe-a06b-cae892a04f77

Manuscript title: Multimodal Emotion Recognition Based on Adaptive Fusion Strategy

## Required Uploads

### Revised Manuscript

Before uploading the revised manuscript:

- Add a `Data Availability` section at the end of the manuscript.
- Use the statement in `docs/DATA_AVAILABILITY_STATEMENT.md`.
- Make sure the same statement is also entered in the submission system under `Declarations`.
- Strengthen limitations where needed, especially regarding restricted counseling data, label mapping, weakly aligned fusion settings, and generalization to real-world settings.

### Point-by-Point Response

Upload the anonymized response file:

- `docs/POINT_BY_POINT_RESPONSE_ANONYMOUS.md`

Before upload:

- Do not add author names.
- Do not add affiliations.
- Do not add email addresses.
- Do not add ORCID identifiers.
- Do not add acknowledgements or funding details.
- Do not include local file paths that reveal personal information.

### Supplementary Materials

Upload the generated supplementary package or provide a stable public repository link.

The package should include:

- `README.md`
- `PACKAGE_MANIFEST.md`
- `docs/`
- `code/`
- `scripts/`
- `experiments/`
- `results/`
- `derived_data/`

## Declarations Field

Do not enter:

```text
NO
```

Use the short Data Availability statement in `docs/DATA_AVAILABILITY_STATEMENT.md`.

## Files to Exclude

Exclude by default:

- raw restricted counseling transcripts,
- raw image/audio/video files,
- model checkpoint files,
- compressed archives,
- Office documents,
- cache files,
- IDE metadata,
- files with author-identifying metadata.

Typical excluded extensions:

- `.docx`
- `.pptx`
- `.pdf`
- `.pt`
- `.pth`
- `.bin`
- `.onnx`
- `.pkl`
- `.zip`
- `.tar`
- `.gz`
- `.wav`
- `.mp3`
- `.mp4`

## Public Repository Checks

If a public repository is used:

- The repository must be accessible without login.
- The repository link should be stable.
- The top-level README must explain every folder.
- The data availability statement must be included.
- The generative AI use statement must be included if required by the journal.
- Raw restricted counseling data must not be uploaded.
- Author-identifying information must not appear in anonymous review files.

## Final Check

- The manuscript includes the Data Availability statement.
- The submission system includes the same Data Availability statement.
- The point-by-point response is anonymous.
- The supplementary package has a top-level `README.md`.
- All Markdown files are in English.
- Restricted data are explained but not publicly released.
- Public dataset source links are provided.
- Scripts and result files are included for verification.
