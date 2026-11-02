# `data` — dataset and splits

These are the only inputs the model saw. Everything else in the repository is
produced by the notebooks.

## Files

| File | Rows | What it is |
|---|---:|---|
| `final_labeled_reports.csv` | 3,955 | the labelled dataset: report text plus 14 CheXpert labels |
| `train_set.csv` | 2,768 | 70% split, used for 5-shot exemplars and LoRA training |
| `val_set.csv` | 395 | 10% split, used for few-shot evaluation, LoRA selection, and the sanity check |
| `test_set.csv` | 792 | 20% split, scored once with the selected adapters |
| `prompt_templates.json` | — | the exact direct and structure prefixes/suffixes written by `05_sft_data.ipynb` |

Each CSV has 18 columns: `id`, `findings`, `impression`, `full_report`, and the
14 finding columns.

## Provenance

1. `preprocessing.ipynb` downloads the public NLMCXR reports from OpenI, parses
   the `FINDINGS` and `IMPRESSION` sections of each XML file, and joins them into
   `full_report`.
2. The same notebook runs the Stanford CheXpert labeler over `full_report` to
   produce silver labels for the 14 observations.
3. It writes `final_labeled_reports.csv` to the MyDrive root.
4. `01_data_preparation.ipynb` splits that file into the three CSVs above.

`processed_reports.csv` (the report text before labelling) is an intermediate
inside `preprocessing.ipynb` and is not archived here.

## How the split was made

```python
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df  = train_test_split(temp_df, test_size=0.6667, random_state=42)
```

The split is at report level. It is unstratified, not grouped by patient, and
not de-duplicated by report text. Exact-text overlap between splits is measured
as a sensitivity check in `results/tables/table_duplicate_text_sensitivity.csv`.

The three files partition `final_labeled_reports.csv` exactly: the union of their
`id` values is the 3,955 dataset IDs, no ID appears in two splits, and every
report's text and labels are identical to the corresponding row of the dataset.
The split is clean at the report-ID level but not at the text level: 192 of 792
test reports share exact text with a training or validation report. The overlap
is quantified in `results/tables/table_duplicate_text_sensitivity.csv`.

## Label semantics

| Value | Meaning |
|---:|---|
| `1.0` | present |
| `0.0` | absent |
| `-1.0` | uncertain |
| empty | not mentioned |

The labels are silver labels from an automated labeler, not independent
radiologist annotations. The evaluation excludes uncertain truth per label
(U-Ignore) and treats blank truth as absent; see the scoring rule in the root
README.

Positive (`1.0`) counts per split:

| Finding | Train | Val | Test |
|---|---:|---:|---:|
| No Finding | 997 | 154 | 318 |
| Enlarged Cardiomediastinum | 141 | 17 | 33 |
| Cardiomegaly | 489 | 53 | 106 |
| Lung Lesion | 129 | 16 | 26 |
| Lung Opacity | 480 | 56 | 131 |
| Edema | 42 | 5 | 8 |
| Consolidation | 44 | 1 | 15 |
| Pneumonia | 61 | 10 | 14 |
| Atelectasis | 173 | 15 | 52 |
| Pneumothorax | 28 | 3 | 9 |
| Pleural Effusion | 148 | 15 | 30 |
| Pleural Other | 29 | 4 | 10 |
| Fracture | 100 | 20 | 21 |
| Support Devices | 200 | 27 | 53 |

## Licensing

The reports come from the public OpenI / NLMCXR collection and remain subject to
its terms. The labels and derived splits are included for reproducibility; the
code is released under the MIT licence.
