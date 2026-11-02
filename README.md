# Evaluating a Small Open-Weight Large Language Model for Structured Information Extraction from Chest X-Ray Reports: A Comparison of Direct and Structure-Based Pipelines

Code, data, and saved results for an Honours research report at the University of
the Witwatersrand. The study asks whether a small instruction-tuned language model
can extract the 14 CheXpert observations from a free-text chest X-ray report,
and whether a structure-based pipeline that returns an anatomical JSON object is
better than a direct pipeline that returns one label per line.

The notebooks are the implementation. There is no second copy of the pipeline
in library form; every number in the report comes from the files under
`results/`.

## How to read this repository

| Directory | What it holds | Start here if you want to... |
|---|---|---|
| `data/` | the labelled dataset and the three splits | see what the model was given |
| `notebooks/` | the executed Colab notebooks, numbered in run order | see how each stage was run |
| `results/` | every prediction and metric file the notebooks wrote | check a reported number |
| `analysis/` | one script that recomputes the post-hoc tables | verify the report's tables without a GPU |
| `requirements/` | the recorded package versions per environment | rebuild a runtime |

`data/README.md` explains the dataset and splits. `results/README.md` explains
every result file and records what is missing from the archive.

## Experiment at a glance

| Item | Setting |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct`, 4-bit NF4, float16 compute |
| Data | OpenI / NLMCXR reports, silver labels from the Stanford CheXpert labeler |
| Split | 2,768 train / 395 validation / 792 test, report-level 70/10/20 |
| Direct pipeline | 14 lines `Finding=value`; `1` present, `0` absent, `-1` uncertain, `NaN` not mentioned |
| Structure-based pipeline | one JSON object in six anatomical sections; `present`, `absent`, `uncertain`, `not_mentioned` |
| Conditions | zero-shot, 5-shot (seeds 42/123/456/789/1011), LoRA fine-tuned |
| Primary metric | macro F1 over the 13 clinical labels; No Finding reported separately |
| Significance | paired bootstrap over reports, 2,000 resamples, seed 42 |

## Headline results

Test set, 792 reports, generated once per pipeline with the selected seed-42
adapters in the frozen inference stack:

| Pipeline | macro13 | macro14 | No Finding | 14-field coverage |
|---|---:|---:|---:|---:|
| Direct pipeline | 0.8461 | 0.8529 | 0.9412 | 99.87% |
| Structure-based pipeline | 0.8825 | 0.8864 | 0.9371 | 100% |

Structure minus direct: **+0.0364**, 95% CI **[-0.0129, +0.0873]**, bootstrap
P(delta > 0) = **0.938**. The interval includes zero, so the test set does not
establish a reliable advantage for the structured format.

Validation scores for context:

| Condition | Direct pipeline | Structure-based pipeline |
|---|---:|---:|
| Zero-shot | 0.4229 | 0.4557 |
| 5-shot, mean of seeds 42/123/456 | 0.4552 (sd 0.012) | 0.4328 (sd 0.022) |
| LoRA, selected on validation (optimistic) | 0.8449 | 0.8755 |

The selected LoRA validation score is optimistic because the configuration was
chosen on that split. Training stability, measured by retraining with seeds 123
and 456, is reported in `results/tables/table_stability.csv`.

## Pipeline and run order

Each row is one notebook. "Reads" and "Writes" use repository paths; the mapping
to the Colab Drive paths is in the next section.

| Step | Notebook | Reads | Writes | Purpose |
|---:|---|---|---|---|
| 0 | `preprocessing.ipynb` | NLMCXR XML (downloaded) | `data/final_labeled_reports.csv` | extract FINDINGS/IMPRESSION, run the CheXpert labeler |
| 1 | `01_data_preparation.ipynb` | `data/final_labeled_reports.csv` | `data/train_set.csv`, `val_set.csv`, `test_set.csv` | make the 2,768 / 395 / 792 split |
| 2 | `02_direct_pipeline.ipynb` | `data/train_set.csv`, `val_set.csv` | `results/prompted/direct_*_results.csv` | zero-shot and 5-shot, direct pipeline |
| 3 | `03_structure_pipeline.ipynb` | `data/train_set.csv`, `val_set.csv` | `results/prompted/structure_*_results.csv` | zero-shot and 5-shot, structure-based pipeline |
| 4 | `04_evaluate.ipynb` | the twelve `results/prompted/*_results.csv` | `results/prompted/unified_metrics.csv` | score the prompted runs, paired bootstrap |
| 5 | `05_sft_data.ipynb` | `data/train_set.csv` | `data/prompt_templates.json`, `sft_*.jsonl` | build completion-only SFT data and prompt templates |
| 6 | `06_lora_search.ipynb` | `data/train_set.csv`, `val_set.csv`, `data/prompt_templates.json`, `sft_*.jsonl` | `results/lora_search/` | 8 LoRA configs per pipeline, select on validation macro13 |
| 7 | `07a_stability.ipynb` | the same inputs plus `results/lora_search/selected_configs.json` | `results/stability/` | retrain the selected configs with seeds 123 and 456 |
| 8 | `07b_final_eval.ipynb` | `data/val_set.csv`, `test_set.csv`, `data/prompt_templates.json`, `results/lora_search/selected_configs.json`, the selected adapters | `results/final_eval/` | validation sanity check, then one test evaluation |

Dependencies: steps 2 and 3 need step 1; step 4 needs steps 2 and 3; step 5
needs step 1; step 6 needs step 5; step 7 needs step 6; step 8 needs the step-6
adapters and the data splits. Step 4 is independent of steps 5–8, and step 7 is
a stability check that does not feed step 8.

Each notebook installs its own stack and must run in a fresh Colab runtime. Do
not install the fine-tuning and frozen-evaluation stacks together.
`requirements/prompted.txt` covers steps 2–3, `requirements/finetune.txt` covers
steps 6–7, and `requirements/frozen-eval.txt` covers step 8; steps 0, 1, 4, and
5 are CPU-only, and `requirements/analysis.txt` covers the reproduction script.

The 5-shot exemplars are drawn independently per seed with
`train_df.sample(n=10, replace=False, random_state=seed)`. Both pipelines use
the same ten exemplars at a given seed.

Seeds 42/123/456 for both pipelines ran on a Colab T4 and form the primary
few-shot comparison. Seeds 789/1011 were a sensitivity check; the structure
runs for those two seeds were completed on a Kaggle T4, so the five-seed mean
is platform-confounded and is not the primary result.

## Rerunning on Colab: where the files go

The notebooks read and write absolute Drive paths. To rerun, place the
repository files as follows.

| Repository path | Colab / Drive path |
|---|---|
| `data/final_labeled_reports.csv` | `/content/drive/MyDrive/final_labeled_reports.csv` |
| `data/train_set.csv`, `val_set.csv`, `test_set.csv` | `/content/drive/MyDrive/cxr_eval_data/` (the same names) |
| `data/prompt_templates.json` | `/content/drive/MyDrive/cxr_eval_data/prompt_templates.json` |
| `results/prompted/*.csv` | `/content/drive/MyDrive/cxr_eval_data/` (the same names) |
| `results/lora_search/` | `/content/drive/MyDrive/cxr_eval_data/lora_search/` |
| `results/stability/` | `/content/drive/MyDrive/cxr_eval_data/stability/` |
| `results/final_eval/` | `/content/drive/MyDrive/cxr_eval_data/final_eval/` |

Two details from the recorded runs:

- `preprocessing.ipynb` writes `processed_reports.csv` and then
  `final_labeled_reports.csv` to the MyDrive root. Only the final labelled file
  is archived here; `processed_reports.csv` is an intermediate that the labeler
  step consumes.
- `02_direct_pipeline.ipynb` reads the splits through `files.upload()`, not the
  Drive mount. `03_structure_pipeline.ipynb` also reads them through upload, and
  in the recorded run it wrote its CSVs to the MyDrive root. All twelve prompted
  CSVs are archived together under `results/prompted/`.

## Reproducing the reported tables

No GPU is needed. From the repository root:

```bash
pip install -r requirements/analysis.txt
python analysis/reproduce_tables.py
```

The script reads `results/final_eval/*/test_predictions.csv`, the search and
stability manifests, and the three split files under `data/`. It recomputes:

- final test metrics and per-label F1;
- the paired bootstrap for structure minus direct;
- the exact-text duplicate sensitivity analysis;
- the matched-configuration comparison and the seed-stability summary.

It also recomputes macro13 for each per-configuration validation prediction
file on disk and checks it against `results/lora_search/search_results.csv`.

It writes the tables to `results/tables/` and asserts the values frozen in the
report. A successful run ends with:

```
PASS: test metrics, bootstrap, and duplicate-text table match the frozen results.
```

The prompted zero-shot and few-shot table (`results/prompted/unified_metrics.csv`)
is produced by `04_evaluate.ipynb`, which is included with its output.

## Scoring rule

- Uncertain silver truth (`-1`) is excluded per label (U-Ignore).
- Blank silver truth is treated as absent.
- Only a predicted `present` (`1`) is positive; absent, uncertain,
  not-mentioned, invalid, and failed output all count as absent.
- Every report is scored. A failed generation is never dropped.
- Macro F1 is computed over the 13 clinical labels; No Finding is reported
  separately. Macro14 is shown only for continuity with the earlier notebooks.

## What is not included

- **Adapter weights.** The 19 adapter directories are not included. Individual
  `adapter_model.safetensors` files are 47–194 MB, and the selected structure
  adapter alone is 194 MB, over GitHub's 100 MB limit. The selected
  configurations, their validation predictions, and the final test predictions
  are included. The weights are available from the author on request.
- **SFT JSONL files.** `05_sft_data.ipynb` regenerates both files
  deterministically from `data/train_set.csv`, with a full label round-trip
  check.
- **One LoRA validation prediction file** that was absent from the Drive
  download (`structure_r16_a16_lr0.0002_ep1_s42`). Its score is still in
  `results/lora_search/search_results.csv`.
- **`processed_reports.csv`.** The report-only intermediate from preprocessing;
  it is reproduced by the first cells of `preprocessing.ipynb`.
- **The two legacy fine-tuning notebooks.** They use a different prompt, the
  old 593-report validation split, and only 200 validation reports. They are
  not part of the reported protocol and were deliberately left out.

## Data and ethics

The reports are from the public OpenI / NLMCXR chest X-ray collection. They are
de-identified and openly available. The labels are silver labels produced by the
Stanford CheXpert labeler, not independent radiologist annotations. No private
or identifiable patient data are included. The split is unstratified, not
grouped by patient, and not deduplicated by report text; exact-text overlap is
quantified as a robustness check rather than removed.

The three split files partition `final_labeled_reports.csv` exactly: 3,955
unique report IDs, no overlap between splits, and identical report text and
labels for every ID.

No credentials are stored in this repository. The notebooks read the Hugging
Face token from Colab userdata (`HF_TOKEN`) or Kaggle secrets.

## Citation

```bibtex
@software{ntimana2026cxr,
  author  = {Ntimana, Risuna},
  title   = {Evaluating a Small Open-Weight Large Language Model for Structured Information Extraction from Chest X-Ray Reports: A Comparison of Direct and Structure-Based Pipelines},
  year    = {2026},
  note    = {Honours research report, University of the Witwatersrand}
}
```

Code is released under the MIT licence. The OpenI reports remain subject to
their original terms.
