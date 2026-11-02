"""Recompute the post-hoc tables reported in the report from saved predictions.

This script does not run a model and does not reimplement either pipeline. It
reads the per-report predictions produced by the notebooks and applies the
frozen scoring rule:

  * uncertain silver truth (-1) is excluded per label (U-Ignore);
  * blank silver truth is treated as absent;
  * only a predicted present ("1") is positive;
  * every other or missing prediction is absent;
  * every report is scored, and a failed output is never dropped.

It writes the test, duplicate-text, matched-configuration, and stability tables
to results/tables/ and asserts the values frozen in the report. It also checks
that every per-configuration validation prediction file on disk reproduces the
macro13 recorded for that configuration in the search manifest.

Usage (no GPU, no Drive):
    python analysis/reproduce_tables.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

NO_FINDING = "No Finding"
LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CLINICAL = [label for label in LABELS if label != NO_FINDING]


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = {"id", "n_found"} | {f"truth_{p}" for p in LABELS} | {f"pred_{p}" for p in LABELS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return df


def truth_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for label in LABELS:
        values = pd.to_numeric(df[f"truth_{label}"], errors="coerce").to_numpy(dtype=float)
        out[label] = np.where(values == -1, np.nan, np.where(np.isnan(values), 0.0, values))
    return out


def prediction_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array(
        [[1.0 if df[f"pred_{p}"].iloc[i] == "1" else 0.0 for p in LABELS] for i in range(len(df))]
    )


def f1_per_label(truth: dict[str, np.ndarray], pred: np.ndarray) -> dict[str, float]:
    scores = {}
    for column, label in enumerate(LABELS):
        target = truth[label]
        predicted = pred[:, column]
        mask = ~np.isnan(target)
        tp = int(((target == 1) & (predicted == 1) & mask).sum())
        fp = int(((target == 0) & (predicted == 1) & mask).sum())
        fn = int(((target == 1) & (predicted == 0) & mask).sum())
        denominator = 2 * tp + fp + fn
        scores[label] = 2 * tp / denominator if denominator else 0.0
    return scores


def evaluate(truth: dict[str, np.ndarray], pred: np.ndarray) -> dict:
    per_label = f1_per_label(truth, pred)
    return {
        "macro13": float(np.mean([per_label[p] for p in CLINICAL])),
        "macro14": float(np.mean(list(per_label.values()))),
        NO_FINDING: per_label[NO_FINDING],
        "per_label": per_label,
    }


def macro13_at(truth: dict[str, np.ndarray], pred: np.ndarray, index: np.ndarray) -> float:
    values = []
    for column, label in enumerate(LABELS):
        if label == NO_FINDING:
            continue
        target = truth[label][index]
        predicted = pred[index, column]
        mask = ~np.isnan(target)
        tp = int(((target == 1) & (predicted == 1) & mask).sum())
        fp = int(((target == 0) & (predicted == 1) & mask).sum())
        fn = int(((target == 1) & (predicted == 0) & mask).sum())
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def paired_bootstrap(
    truth: dict[str, np.ndarray],
    pred_structure: np.ndarray,
    pred_direct: np.ndarray,
    B: int = 2000,
    seed: int = 42,
) -> dict:
    n = len(next(iter(truth.values())))
    observed = macro13_at(truth, pred_structure, np.arange(n)) - macro13_at(
        truth, pred_direct, np.arange(n)
    )
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for b in range(B):
        index = rng.choice(n, n, replace=True)
        diffs[b] = macro13_at(truth, pred_structure, index) - macro13_at(
            truth, pred_direct, index
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "observed_delta": float(observed),
        "boot_mean_delta": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_delta_gt_0": float(np.mean(diffs > 0)),
        "boot_sd": float(diffs.std(ddof=1)),
        "B": B,
        "seed": seed,
        "n_reports": n,
    }


def subset(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    by_id = {str(value): position for position, value in enumerate(df["id"])}
    return df.iloc[[by_id[i] for i in ids]].reset_index(drop=True)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    direct = load_predictions(RESULTS / "final_eval" / "direct" / "test_predictions.csv")
    structure = load_predictions(RESULTS / "final_eval" / "structure" / "test_predictions.csv")
    if direct["id"].tolist() != structure["id"].tolist():
        raise ValueError("direct and structure test predictions are not in the same report order")
    if len(direct) != 792:
        raise ValueError(f"expected 792 test reports, found {len(direct)}")

    truth = truth_arrays(direct)
    truth_structure = truth_arrays(structure)
    for label in LABELS:
        if not np.allclose(truth[label], truth_structure[label], equal_nan=True):
            raise ValueError(f"test truth differs between pipelines for {label}")

    direct_pred = prediction_matrix(direct)
    structure_pred = prediction_matrix(structure)

    metrics = {
        "direct": evaluate(truth, direct_pred),
        "structure": evaluate(truth, structure_pred),
    }

    test_rows = []
    for pipeline, df, pred in (
        ("direct", direct, direct_pred),
        ("structure", structure, structure_pred),
    ):
        m = metrics[pipeline]
        found = pd.to_numeric(df["n_found"]).to_numpy()
        test_rows.append(
            {
                "pipeline": pipeline,
                "n": len(df),
                "macro13": m["macro13"],
                "macro14": m["macro14"],
                "NoFinding": m[NO_FINDING],
                "valid14": float(np.mean(found == 14)),
                "usable_any": float(np.mean(found > 0)),
            }
        )
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(TABLES / "table_test_metrics.csv", index=False)

    per_label = pd.DataFrame(
        {
            "label": LABELS,
            "direct_f1": [metrics["direct"]["per_label"][p] for p in LABELS],
            "structure_f1": [metrics["structure"]["per_label"][p] for p in LABELS],
        }
    )
    per_label["diff_S_minus_D"] = per_label["structure_f1"] - per_label["direct_f1"]
    per_label.to_csv(TABLES / "table_test_per_label.csv", index=False)

    test_boot = paired_bootstrap(truth, structure_pred, direct_pred)
    pd.DataFrame([{"comparison": "structure - direct", **test_boot}]).to_csv(
        TABLES / "table_test_bootstrap.csv", index=False
    )

    # Exact-text overlap is measured, not removed: a test report is a duplicate
    # when its full report text occurs verbatim in the train or validation split.
    train = pd.read_csv(DATA / "train_set.csv", dtype=str, keep_default_na=False)
    val = pd.read_csv(DATA / "val_set.csv", dtype=str, keep_default_na=False)
    test_text = pd.read_csv(DATA / "test_set.csv", dtype=str, keep_default_na=False)
    train_text = set(train["full_report"])
    val_text = set(val["full_report"])
    flags = pd.DataFrame(
        {
            "id": test_text["id"],
            "duplicate_in_train": test_text["full_report"].isin(train_text),
            "duplicate_in_val": test_text["full_report"].isin(val_text),
        }
    )
    flags["duplicate_in_either"] = flags["duplicate_in_train"] | flags["duplicate_in_val"]
    flags.to_csv(TABLES / "duplicate_text_flags.csv", index=False)

    clean_ids = flags.loc[~flags["duplicate_in_either"], "id"].astype(str).tolist()
    sensitivity_rows = []
    for name, ids in (("full test", flags["id"].astype(str).tolist()), ("no exact-text overlap", clean_ids)):
        d_sub = subset(direct, ids)
        s_sub = subset(structure, ids)
        t_sub = truth_arrays(d_sub)
        d_metrics = evaluate(t_sub, prediction_matrix(d_sub))
        s_metrics = evaluate(t_sub, prediction_matrix(s_sub))
        boot = paired_bootstrap(
            t_sub, prediction_matrix(s_sub), prediction_matrix(d_sub)
        )
        sensitivity_rows.append(
            {
                "subset": name,
                "n": len(ids),
                "direct_macro13": d_metrics["macro13"],
                "direct_macro14": d_metrics["macro14"],
                "direct_NoFinding": d_metrics[NO_FINDING],
                "structure_macro13": s_metrics["macro13"],
                "structure_macro14": s_metrics["macro14"],
                "structure_NoFinding": s_metrics[NO_FINDING],
                "delta_macro13": boot["observed_delta"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_delta_gt_0": boot["p_delta_gt_0"],
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES / "table_duplicate_text_sensitivity.csv", index=False)

    search = pd.read_csv(RESULTS / "lora_search" / "search_results.csv")
    key = ["r", "alpha", "lr", "epochs"]
    direct_search = search.loc[search["pipeline"] == "direct", key + ["macro13"]].rename(
        columns={"macro13": "direct_macro13"}
    )
    structure_search = search.loc[search["pipeline"] == "structure", key + ["macro13"]].rename(
        columns={"macro13": "structure_macro13"}
    )
    matched = direct_search.merge(structure_search, on=key, how="inner")
    matched["diff_S_minus_D"] = matched["structure_macro13"] - matched["direct_macro13"]
    matched.sort_values(key).to_csv(TABLES / "table_matched_configurations.csv", index=False)

    checked = 0
    for _, row in search.iterrows():
        path = RESULTS / "lora_search" / row["pipeline"] / row["config"] / "val_predictions.csv"
        if not path.exists():
            continue
        frame = load_predictions(path)
        recomputed = evaluate(truth_arrays(frame), prediction_matrix(frame))["macro13"]
        if abs(recomputed - float(row["macro13"])) > 1e-9:
            raise AssertionError(
                f"{row['config']}: predictions give macro13 {recomputed}, "
                f"manifest records {row['macro13']}"
            )
        checked += 1
    print(f"Per-config validation files reproduce the search manifest: {checked}/16 configs on disk")

    stability = pd.read_csv(RESULTS / "stability" / "stability_summary.csv")
    stability_rows = []
    for pipeline in ("direct", "structure"):
        sub = stability[stability["pipeline"] == pipeline].set_index("seed")
        all_three = sub.loc[[42, 123, 456], "macro13"].to_numpy(dtype=float)
        stability_rows.append(
            {
                "pipeline": pipeline,
                "seed42_macro13": float(sub.loc[42, "macro13"]),
                "seeds123_456_mean": float(sub.loc[[123, 456], "macro13"].mean()),
                "seeds123_456_sd": float(sub.loc[[123, 456], "macro13"].std(ddof=1)),
                "all3_mean": float(all_three.mean()),
                "all3_sd": float(all_three.std(ddof=1)),
                "seed42_minus_all3_mean": float(sub.loc[42, "macro13"] - all_three.mean()),
            }
        )
    pd.DataFrame(stability_rows).to_csv(TABLES / "table_stability.csv", index=False)

    print("\nFinal test (792 reports)")
    print(test_table.to_string(index=False))
    print("\nPaired bootstrap, structure - direct (macro13)")
    print(pd.DataFrame([test_boot]).to_string(index=False))
    print("\nDuplicate-text sensitivity")
    print(
        sensitivity[
            ["subset", "n", "direct_macro13", "structure_macro13", "delta_macro13", "ci_lo", "ci_hi", "p_delta_gt_0"]
        ].to_string(index=False)
    )
    print(f"\nWrote tables to {TABLES}")

    expected = {
        "direct macro13": (metrics["direct"]["macro13"], 0.8460906236367575),
        "structure macro13": (metrics["structure"]["macro13"], 0.8825349639373775),
        "observed delta": (test_boot["observed_delta"], 0.03644434030062005),
        "ci low": (test_boot["ci_lo"], -0.012947525672708685),
        "ci high": (test_boot["ci_hi"], 0.08734048563208698),
        "p(delta>0)": (test_boot["p_delta_gt_0"], 0.938),
        "clean n": (float(len(clean_ids)), 600.0),
        "clean direct": (sensitivity.loc[1, "direct_macro13"], 0.8403642577678992),
        "clean structure": (sensitivity.loc[1, "structure_macro13"], 0.8762052941744481),
        "clean delta": (sensitivity.loc[1, "delta_macro13"], 0.0358410364065489),
        "clean ci low": (sensitivity.loc[1, "ci_lo"], -0.009915071531278616),
        "clean ci high": (sensitivity.loc[1, "ci_hi"], 0.0837717117896347),
        "clean p(delta>0)": (sensitivity.loc[1, "p_delta_gt_0"], 0.9365),
    }
    for name, (got, want) in expected.items():
        if abs(got - want) > 1e-9:
            raise AssertionError(f"{name}: got {got!r}, expected {want!r}")
    print("\nPASS: test metrics, bootstrap, and duplicate-text table match the frozen results.")


if __name__ == "__main__":
    main()
