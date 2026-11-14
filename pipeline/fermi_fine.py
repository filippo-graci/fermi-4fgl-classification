# %% [markdown]
# # fermi_fine.py

# %%

"""
Fine-grain model training & evaluation — Module 2.

Runs after `fermi_common.py` in the same namespace: `SEED`, `VERSION_TAG`,
`SELECTION_METRIC`, `MODEL_SPECS`, `PCA_VARIANTS`, `SCALER_VARIANTS`,
`WEIGHTING_MODES`, `DTreeCVPrunedClassifier`, `io_for`, `get_section_logger`,
`save_result_multi`, `publish_split_summary`, `load_all_results`, `pbar`,
`SPLIT_ID_DEFAULT`, etc. are in scope, not imported.

Scope: flat fine-grain classification only — Setup A (`fine_all`, `bcu`
retained) and Setup B (`fine_nobcu`, `bcu` excluded). The hierarchical
workflows live in `fermi_merged.py` / `fermi_3levels.py`.

How it works:
- Reads Module 1's per-split output under
  `fermi_module1_processing_{VERSION_TAG}/split_XX/`. `discover_splits()`
  reads Module 1's `multi_split_manifest.json`, and the whole Setup A/B
  training-and-selection procedure is wrapped in
  `run_fine_module_for_split(split_id)`, run once per split (§0 below).
- HistGBC: `fermi_common.py`'s `select_and_refit_val_only` swaps in the
  leakage-safe `StratifiedEarlyStoppingHGBC` for HistGBC's OOF pass and
  train-refit, never touching a test set. This module passes its
  `HISTGBC_MAX_ITER_CEILING` (2000) through and reports the
  `histgbc_deployable_used` column in its leaderboards.
- `build_fine_context` asserts that every feature-set tag's `source_name`
  sequence is identical to the first tag's, so labels can never be paired
  with the wrong feature rows (shape-only check, with a warning, for an
  export without `source_name`).
- Leaderboards include the CV `_std` columns, and
  `plot_leaderboard_with_ci` draws them as error bars.
- Logging: every print goes through a `label_set`-scoped logger
  (`get_section_logger`) — milestones at `NOTICE` (console + that branch's
  log file), per-candidate detail at `INFO` (file only), failures at
  `WARNING`/`ERROR`. Module-wide banners use the `"_module"` pseudo label set.
- I/O: `io_for(OUTPUT_ROOT, split_id)` gives every branch its own
  `results/{label_set}.json` and `logs/{label_set}.log`, alongside the
  full-ledger `results/results_log.json`. `publish_split_summary` copies
  each split's leaderboard / winner_info / comparison artifacts into
  `OUTPUT_ROOT/results_aggregated/{split_id}/`, and
  `build_all_splits_summary` concatenates every split's leaderboard into
  `results_aggregated/all_splits_summary.csv`.
- Plotting / safe-run helpers are per-split, per-`label_set` closures
  (`make_safe_run`, explicit `plots_dir=`); no global name is reassigned.

Methodological notes (for the paper's methods section):
- fsrq/ssrq/css are merged before any split; four class-weighting modes;
  PCA-variant and scaler ablation axes; six MLP topologies.
- The Setup A vs. B comparison is descriptive, not a paired test: the two
  setups have different class cardinality.
- `DTreeCVPrunedClassifier` uses a local, K-Fold-only override (§1b below),
  a deliberate, compute-driven divergence from `fermi_common.py`'s
  adaptive-LOO default used by the hierarchical modules.
"""
import json
import warnings
from pathlib import Path
from typing import Optional, Sequence
from dataclasses import dataclass, field
warnings.filterwarnings("ignore")

# %%
import numpy as np
import pandas as pd
import joblib

# %%
from sklearn.preprocessing import LabelEncoder

# %%
"""
Captured BEFORE this module could redefine safe_run/plot_confusion_matrix/
plot_comparison_table/MODEL_SPECS below. This module never reassigns
the bare `plot_confusion_matrix`/`plot_comparison_table`/`safe_run`
global names (see the header docstring) — this capture is kept purely as cheap
insurance against an out-of-order notebook re-execution, not because a
later reassignment in THIS file needs guarding against.
"""
COMMON_MODEL_SPECS = MODEL_SPECS
_safe_run_common = globals().get("_BASE_SAFE_RUN", safe_run)
_plot_confusion_matrix_common = globals().get("_BASE_PLOT_CONFUSION_MATRIX", plot_confusion_matrix)
globals().setdefault("_BASE_PLOT_CONFUSION_MATRIX", _plot_confusion_matrix_common)
_plot_comparison_table_common = globals().get("_BASE_PLOT_COMPARISON_TABLE", plot_comparison_table)
globals().setdefault("_BASE_PLOT_COMPARISON_TABLE", _plot_comparison_table_common)

# %% [markdown]
# ## Section 0 — Multi-split discovery

# %%
"""


`fermi_processing.py` writes one folder per validation split under
`fermi_module1_processing_{VERSION_TAG}/`, and records which splits
exist in a root-level `multi_split_manifest.json`. `discover_splits`
below is deliberately defensive about the manifest's exact schema: it accepts
the most likely key names for a split-id list, handles either bare
strings or `{"id": ...}` dicts, and falls back — loudly, with a printed
warning, never silently — to assuming only the primary split exists if
the manifest is missing or doesn't match. If the manifest schema ever
changes, this is the one function to adjust; nothing downstream needs to change.

`ENABLE_MULTI_SPLIT_VALIDATION` mirrors Module 1's own default-off
switch of the same name: off by default for fast iteration (trains only
the primary split), on to run this module's entire Setup A/B procedure
once per split Module 1 produced — which is what actually lets the
paper report how much the winning configuration moves across draws of
the data.
"""

# %%
PROCESSING_ROOT = _cfg("PROCESSING_ROOT", Path(f"./fermi_module1_processing_{VERSION_TAG}/"))
OUTPUT_ROOT = _cfg("OUTPUT_ROOT", Path(f"./outputs_{VERSION_TAG}_{SELECTION_METRIC}/"))

# %%
ENABLE_MULTI_SPLIT_VALIDATION = _cfg("MODULE_MULTI_SPLIT", False)

FORCE_RECOMPUTE = _cfg("FINE_FORCE_RECOMPUTE", True)  # True: purge a split's fine-module outputs and rerun from scratch,
                          # bypassing the "already complete" read-back shortcut in §9.


# %%
def discover_splits(processing_root: Path) -> list:
    """Reads fermi_processing.py's multi_split_manifest.json for the
    list of split ids it produced. Defensive against schema uncertainty
    — see the section markdown above."""
    manifest_path = processing_root / "multi_split_manifest.json"
    if not manifest_path.exists():
        print(f"  ⚠ {manifest_path} not found — assuming only '{SPLIT_ID_DEFAULT}' exists "
              "(a single-split Module 1 run, or an export predating the multi-split manifest).")
        return [SPLIT_ID_DEFAULT]
    with open(manifest_path) as f:
        manifest = json.load(f)
    raw = manifest.get("splits") or manifest.get("split_ids") or manifest.get("available_splits")
    if not raw:
        print(f"  ⚠ {manifest_path} exists but no recognized split-list key "
              "('splits' / 'split_ids' / 'available_splits') was found — assuming only "
              f"'{SPLIT_ID_DEFAULT}' exists. Inspect the manifest and adjust discover_splits() "
              "if fermi_processing.py uses a different schema.")
        return [SPLIT_ID_DEFAULT]
    split_ids = []
    for s in raw:
        if isinstance(s, dict):
            if "split_dir" in s:
                split_ids.append(Path(s["split_dir"]).name)
            elif "id" in s:
                split_ids.append(s["id"])
            else:
                raise KeyError(
                    f"discover_splits: manifest split entry has neither 'split_dir' nor "
                    f"'id' — keys found: {sorted(s.keys())}. fermi_processing.py's "
                    "actual schema (see process_split's return, line ~1169) uses "
                    "'split_dir'; adjust here if that changes again."
                )
        else:
            split_ids.append(str(s))
    return split_ids or [SPLIT_ID_DEFAULT]


# %%
_DISCOVERED_SPLITS = discover_splits(PROCESSING_ROOT)
SPLITS_TO_RUN = _DISCOVERED_SPLITS if ENABLE_MULTI_SPLIT_VALIDATION else _DISCOVERED_SPLITS[:1]

# %%
print(f"Config ready — SEED={SEED}  VERSION_TAG={VERSION_TAG}  USE_GPU={USE_GPU}  "
      f"SELECTION_METRIC={SELECTION_METRIC}")
print(f"  Module 1 output root -> {PROCESSING_ROOT.resolve()}")
print(f"  Discovered splits    -> {_DISCOVERED_SPLITS}")
print(f"  ENABLE_MULTI_SPLIT_VALIDATION={ENABLE_MULTI_SPLIT_VALIDATION}  -> running: {SPLITS_TO_RUN}")
print(f"  Outputs (metric-suffix) -> {OUTPUT_ROOT.resolve()}")

# %% [markdown]
# ## Section 1 — Physical merge / model-family config
#
# `ssrq`/`css` are merged into `fsrq` before any split or selection
# (astrophysical and empirical justification in fermi_processing.py),
# applied identically here since Module 1 already applies it before
# feature selection too.

# %%
APPLY_PHYSICAL_MERGE = True
MERGE_FINE_CLASSES = {"ssrq": "fsrq", "css": "fsrq"}

# %%
FINE_SETUP_KEYS = _cfg("FINE_SETUP_KEYS", ["A", "B"])
FINE_SETUP_LABELS = {
    "A": "fine_all_merged" if APPLY_PHYSICAL_MERGE else "fine_all",
    "B": "fine_nobcu_merged" if APPLY_PHYSICAL_MERGE else "fine_nobcu",
}

# %%
"""
HistGBC max_iter ceiling for this module's StratifiedEarlyStoppingHGBC
calls (2000). Passed explicitly into `select_and_refit_val_only`'s
`histgbc_max_iter_ceiling=` parameter — this is the one piece of
module-specific configuration `fermi_common.py`'s own HistGBC fix
still needs from a caller.
"""
HISTGBC_MAX_ITER_CEILING = 2000


# %% [markdown]
# ## Section 1b — Local K-Fold-only DecisionTree override
#
# `fermi_common.DTreeCVPrunedClassifier` supports adaptive Leave-
# One-Out CV for very small classes/branches. For **fine-grain
# classification specifically**, this module overrides that and forces
# plain Stratified-K-Fold CV always — LOO would make the whole sweep
# (roughly 15 models × ~6 tag/PCA variants × 4 weighting modes, per
# setup, for BOTH Setup A and Setup B, per split) dramatically slower, and
# Setup A in particular has several fine classes with only 1-2 training
# samples that would otherwise trigger it constantly. Implemented via
# thresholds of -1 that can never fire, registered in a **local**
# `MODEL_SPECS` copy — `fermi_common.MODEL_SPECS` itself is left
# untouched, so the (not yet migrated) hierarchical modules keep their own
# adaptive-LOO DecisionTree.
#
# Note for the paper (not fixable in code): this
# is a real, disclosed divergence from the shared default: a reader
# comparing this module's DecisionTree numbers against the hierarchical
# modules' DecisionTree numbers should be told the two used different CV
# policies for that one model. That sentence belongs in the paper's
# methods text; it cannot be fixed by code.

# %%
def _build_dtree_kfold_only(pca_variance=None, scaler_cls=None, stopping_metric=None):
    """Fine-grain-module-local override of common's `build_dtree`.
    Passing thresholds of -1 for both `loo_min_class_count` and
    `loo_min_branch_train` means the adaptive-LOO condition can never
    fire, so `DTreeCVPrunedClassifier.fit` always falls through to its
    `StratifiedKFold` branch. `scaler_cls` is accepted-and-ignored
    (trees never take a scaler). `stopping_metric` is threaded through
    for API completeness with `DTreeCVPrunedClassifier` — left
    `None` here (as everywhere else in this module) so it resolves
    dynamically from `SELECTION_METRIC` at fit time, same as every other
    model family."""
    return DTreeCVPrunedClassifier(
        random_state=SEED,
        loo_min_class_count=-1,
        loo_min_branch_train=-1,
        stopping_metric=stopping_metric,
    )


# %%
FINE_MODEL_SPECS = {k: dict(v) for k, v in COMMON_MODEL_SPECS.items()}
FINE_MODEL_SPECS["DecisionTree"] = dict(
    build=_build_dtree_kfold_only, build_gpu=_build_dtree_kfold_only,
    weight_support="sample_weight", supports_pca=False, scale_sensitive=False,
)
ALL_MODEL_NAMES = list(FINE_MODEL_SPECS.keys())


def _fine_get_builder(model_name: str):
    """Fine's own builder lookup. A bare `_get_builder` would resolve to whatever
    merged/3levels left in the shared kernel (3levels' THREELEVEL_MODEL_SPECS), so this
    one is uniquely named and a later file's `_get_builder` can't shadow it."""
    return get_builder(model_name, FINE_MODEL_SPECS)


# %% [markdown]
# ## Section 2 — Load Persisted Setup Data (A & B only), per split
#
# Reads Module 1's exported per-tag CSVs for one split, for each setup,
# applies the physical fsrq/ssrq/css merge at load time, and refits a
# local `LabelEncoder` on the merged training labels since Module 1's
# persisted encoder was fit on the unmerged taxonomy.
#
# `_load_tag_split` now also returns each
# tag's `source_name` column (Module 1's traceability tag). This is what
# lets `build_fine_context` verify — rather than silently assume — that
# every feature-set tag's exported rows for one setup/split share
# identical order, before applying one shared boolean mask (the
# train_core/validation split, later `_select_winners`' positional
# `.iloc[mask]` indexing) across every tag's frame. A mismatch here would
# previously have paired the wrong labels with the wrong feature rows with
# no error raised at all.

# %%
@dataclass
class DatasetContext:
    """Bundles one fine-grain setup (A or B), for one split: every
    Module-1 feature-set tag available for it, plus the shared
    train/test label arrays."""
    key: str
    label_set: str
    split_id: str
    x_train_by_tag: dict
    x_test_by_tag: dict
    y_train_enc: np.ndarray
    y_test_enc: np.ndarray
    label_encoder: LabelEncoder
    class_names: np.ndarray
    tags: list = field(default_factory=list)


# %%
def _load_tag_split(source_dir: Path, setup_key: str, tag: str, split: str):
    path = source_dir / "datasets" / f"{setup_key}_{split}_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected Module-1 export missing: {path}")
    df = pd.read_csv(path)
    y_str = df["label"].astype(str).to_numpy()
    if APPLY_PHYSICAL_MERGE:
        y_str = np.array([MERGE_FINE_CLASSES.get(c, c) for c in y_str])
    source_name = df["source_name"].to_numpy() if "source_name" in df.columns else None
    x = df.drop(columns=["label", "label_encoded", "source_name"], errors="ignore")
    return x, y_str, source_name


# %%
def build_fine_context(source_dir: Path, feature_metadata: dict, setup_key: str,
                        split_id: str) -> DatasetContext:
    label_set = FINE_SETUP_LABELS[setup_key]
    setup_meta = feature_metadata["setups"][setup_key]
    tags = [t for t in setup_meta["feature_sets"] if t != "ALL_FEATURES"]
    if PCA_BASE_TAG not in setup_meta["feature_sets"]:
        raise ValueError(
            f"Setup {setup_key}: PCA_BASE_TAG='{PCA_BASE_TAG}' not among "
            f"Module 1's exported feature sets {list(setup_meta['feature_sets'])} — "
            "adjust PCA_BASE_TAG or re-export Module 1."
        )
    # PCA_BASE_TAG may point at "ALL_FEATURES", which `tags` deliberately
    # excludes from the non-PCA 99/90/80(+DECORR) sweep. Load it as an
    # extra entry in x_train_by_tag/x_test_by_tag only — NOT into `tags`
    # itself, so ctx.tags (handed to _select_winners as the non-PCA
    # candidate list) is unchanged.
    load_tags = tags if PCA_BASE_TAG in tags else tags + [PCA_BASE_TAG]

    x_train_by_tag, x_test_by_tag = {}, {}
    y_train_str = y_test_str = None
    train_source_name_ref = test_source_name_ref = None
    for tag in load_tags:
        x_tr, y_tr_str, sn_tr = _load_tag_split(source_dir, setup_key, tag, "train")
        x_te, y_te_str, sn_te = _load_tag_split(source_dir, setup_key, tag, "test")

        # Row-alignment guard — see section markdown above.
        if sn_tr is not None:
            if train_source_name_ref is None:
                train_source_name_ref = sn_tr
            elif not np.array_equal(sn_tr, train_source_name_ref):
                raise ValueError(
                    f"Setup {setup_key} [{split_id}]: tag '{tag}' TRAIN rows are not "
                    f"row-aligned with tag '{tags[0]}' (source_name order/content "
                    "differs). Every tag's export for one setup/split must share "
                    "identical row order — check fermi_processing.py's per-tag "
                    "export routine before trusting anything downstream of this."
                )
        elif x_train_by_tag:
            ref_n = len(next(iter(x_train_by_tag.values())))
            print(f"  ⚠ Setup {setup_key} [{split_id}] tag '{tag}': no 'source_name' column in "
                  "this export — cannot verify row alignment against other tags. Falling back "
                  f"to a shape-only check (n_rows={len(x_tr)} vs {ref_n}).")
            if len(x_tr) != ref_n:
                raise ValueError(f"Setup {setup_key} [{split_id}] tag '{tag}': TRAIN row count "
                                  f"mismatch ({len(x_tr)} vs {ref_n}) — cannot proceed safely "
                                  "without source_name to verify true alignment.")

        if sn_te is not None:
            if test_source_name_ref is None:
                test_source_name_ref = sn_te
            elif not np.array_equal(sn_te, test_source_name_ref):
                raise ValueError(
                    f"Setup {setup_key} [{split_id}]: tag '{tag}' TEST rows are not "
                    f"row-aligned with tag '{tags[0]}'."
                )

        x_train_by_tag[tag] = x_tr
        x_test_by_tag[tag] = x_te
        if y_train_str is None:
            y_train_str, y_test_str = y_tr_str, y_te_str

    if APPLY_PHYSICAL_MERGE:
        le = LabelEncoder().fit(y_train_str)
        known = set(le.classes_)
        test_known = np.isin(y_test_str, list(known))
        n_dropped = int((~test_known).sum())
        if n_dropped:
            print(f"  [{setup_key}, {split_id}, merged] dropping {n_dropped} test rows with "
                  f"labels unseen in train: {sorted(set(y_test_str[~test_known]) - known)}")
        y_train_enc = le.transform(y_train_str)
        y_test_enc = le.transform(y_test_str[test_known])
        x_test_by_tag = {t: x.iloc[test_known].reset_index(drop=True)
                          for t, x in x_test_by_tag.items()}
    else:
        encoder_filename = {
            "A": "label_encoder_fine_A.joblib",
            "B": "label_encoder_fine.joblib",
        }[setup_key]
        le = joblib.load(source_dir / "encoders" / encoder_filename)
        y_train_enc = le.transform(y_train_str)
        y_test_enc = le.transform(y_test_str)

    return DatasetContext(
        key=setup_key, label_set=label_set, split_id=split_id,
        x_train_by_tag=x_train_by_tag, x_test_by_tag=x_test_by_tag,
        y_train_enc=y_train_enc, y_test_enc=y_test_enc,
        label_encoder=le, class_names=le.classes_, tags=tags,
    )


# %% [markdown]
# ## Section 3 — Per-branch `safe_run`
#
# Results/logs are per-(split, label_set) rather than per-module, so
# `safe_run` is too: `make_safe_run` builds one closure per branch,
# writing failures to BOTH that split's full aggregator AND this
# label_set's own per-branch file (via `save_result_multi`, through
# common's `safe_run`), with the failure line routed to that branch's
# logger at `ERROR` instead of a bare `print`.

# %%
def make_safe_run(io_bundle: dict, label_set: str):
    branch_results_path = io_bundle["results_dir"] / f"{_safe_filename(label_set)}.json"
    logger = get_section_logger(label_set, io_bundle["logs_dir"])

    def safe_run(run_fn, *args, context="", **kwargs):
        return _safe_run_common(
            run_fn, *args,
            results_path=[io_bundle["split_results_path"], branch_results_path],
            context=context, logger=logger, **kwargs,
        )
    return safe_run


# %% [markdown]
# ## Section 4 — Uncertainty-aware leaderboard figure
#
# `cv_score_estimator` already computes a bootstrap standard deviation for
# every metric on every leaderboard row; `select_and_refit_val_only`
# already carries it through as `val_{metric}_std`. The printed summary
# table shows it, and `plot_leaderboard_with_ci` is a
# small companion to the shared `plot_comparison_table` (kept as-is, for
# consistency with every other module's leaderboard figure) that adds
# error bars using the uncertainty that was already computed (no new
# statistics).

# %%
def plot_leaderboard_with_ci(best_per_model_df: pd.DataFrame, metric_col: str, std_col: str,
                              plots_dir: Path, title: str, filename: str, logger=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = best_per_model_df.sort_values(metric_col, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.42)))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df[metric_col], xerr=df[std_col].fillna(0.0), capsize=3,
            color="steelblue", edgecolor="k", linewidth=0.4, ecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model_name"])
    ax.set_xlabel(f"{metric_col.replace('_', ' ').title()} (± bootstrap std)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = plots_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    msg = f"  Saved -> {out_path}"
    (logger.notice(msg) if logger is not None else print(msg))
    return out_path


# %% [markdown]
# ## Section 4b — Train-vs-validation accuracy-per-step curve [NEW]

# %%
# %% [markdown]
# ## Section 4b — Train-vs-validation accuracy-per-step curve [NEW]
#
# The five staged estimators in fermi_common.py's Section 1.4
# (RandomForest/GBC/HistGBC/every MLP_* architecture) already grow in
# chunks (trees / boosting rounds / epochs) against an internal
# leakage-safe validation split. `select_and_refit_val_only`'s one final
# refit per surviving model now asks that refit to also record a
# (step, train_{SELECTION_METRIC}, val_{SELECTION_METRIC}) trace via
# `dispatch_fit(..., record_history=True)` — no hardcoded accuracy, this
# tracks whatever SELECTION_METRIC currently is. LDA/LogReg/KNN/SVM/
# DecisionTree/CatBoost have no natural per-step axis in this codebase
# and are silently skipped (`history_` absent/empty on those estimators).

# %%
def plot_train_val_curve(model, model_name: str, label_set: str, plots_dir: Path, logger=None):
    clf = model.named_steps[model.steps[-1][0]] if hasattr(model, "steps") else model
    history = getattr(clf, "history_", None)
    if not history:
        return None
    metric_name = getattr(clf, "history_metric_", SELECTION_METRIC)
    df = pd.DataFrame(history)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["step"], df["train_score"], marker="o", label="train", color="indianred")
    ax.plot(df["step"], df["val_score"], marker="o", label="validation", color="steelblue")
    ax.set_xlabel("Training step (trees / boosting rounds / epochs)")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{model_name} [{label_set}] — train vs. validation {metric_name} vs. step")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = plots_dir / f"train_val_curve_{_safe_filename(f'{model_name}_{label_set}')}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    msg = f"  Saved -> {out_path}"
    (logger.notice(msg) if logger is not None else print(msg))
    return out_path


# %% [markdown]
# ## Section 5 — Run one setup's sweep (`run_setup_pipeline`)
#
# Calls `select_and_refit_val_only` — the Stage 0/1/2 coarse-to-fine
# funnel plus a genuinely held-out OOF cross-validation pass per surviving
# model, never touching the test set. The resulting leaderboard is a
# *selection* artifact rather than a directly reportable number — that
# framing belongs in the paper's methods text. Passes `logs_dir=`/
# `extra_results_paths=`/`histgbc_max_iter_ceiling=` (branch-scoped IO
# bundle), and exports the `_std` uncertainty columns and the
# `histgbc_deployable_used` flag into both the printed table and the
# companion CI figure.

# %% [markdown]
# ### Section 5a — Top-15 model summary (visualization + CSV export)

# %%
# %% [markdown]
# 
#
# `best_per_model` (built just below, Section 5) is already ONE ROW PER
# SURVIVING MODEL FAMILY — `len(ALL_MODEL_NAMES)` rows, which for this
# module's registry (9 base families + 6 MLP_ARCHITECTURES topologies)
# is exactly 15 — sorted descending by `val_{SELECTION_METRIC}` by
# `select_and_refit_val_only` itself. This cell does no new selection;
# it just formats/labels/persists a compact, plotting-ready summary
# separate from the full leaderboard file (which also carries the
# heavier `fitted_summary`/`hyperparameters`/`normalization` columns).

# %%
TOP_K_SUMMARY = 15

def summarize_top_models_fine(best_per_model: pd.DataFrame, metric_key: str, top_k: int,
                               key: str, label_set: str, split_dir: Path,
                               logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    df = best_per_model[best_per_model[f"val_{metric_key}"].notna()].copy()
    df = df.sort_values(f"val_{metric_key}", ascending=False).head(top_k).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["full_model_name"] = (df["model_name"].astype(str) + " [" + df["feature_set"].astype(str)
                              + "/" + df["weighting"].astype(str) + "]")
    cols = ["rank", "full_model_name", "model_name", "feature_set", "tag", "pca_variance",
            "scaler_name", "weighting",
            "val_accuracy", "val_bal_accuracy", "val_macro_f1", "val_kappa",
            "train_accuracy", "train_bal_accuracy", "train_macro_f1", "train_kappa"]
    summary = df[[c for c in cols if c in df.columns]]

    label = f"Setup {key} [{label_set}]"
    msg = f"\nTOP-{len(summary)} MODELS — {label} (sorted by val_{metric_key})"
    (logger.notice(msg) if logger is not None else print(msg))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        printed = summary.to_string(index=False)
    (logger.notice(printed) if logger is not None else print(printed))

    out_csv = split_dir / f"top{top_k}_summary_{_safe_filename(f'{key}_{label_set}')}.csv"
    summary.to_csv(out_csv, index=False)
    (logger.notice(f"  Saved -> {out_csv}") if logger is not None else print(f"  Saved -> {out_csv}"))
    return summary


# %%
def run_setup_pipeline(ctx: DatasetContext, io_bundle: dict):
    logger = get_section_logger(ctx.label_set, io_bundle["logs_dir"])
    safe_run_for_ctx = make_safe_run(io_bundle, ctx.label_set)
    branch_results_path = io_bundle["results_dir"] / f"{_safe_filename(ctx.label_set)}.json"

    def _final_hook(model, model_name, oof_true, oof_pred, class_names, label_set, win):
        _plot_confusion_matrix_common(f"{model_name}_final_val_oof", oof_true, oof_pred,
                                       class_names, label_set=label_set,
                                       plots_dir=io_bundle["plots_dir"], logger=logger)
        plot_train_val_curve(model, model_name, label_set, io_bundle["plots_dir"], logger=logger)  # [NEW]
        
        
    winner_info, best_per_model = select_and_refit_val_only(
        ctx.x_train_by_tag, ctx.y_train_enc, ctx.class_names, ctx.label_set, ctx.tags,
        ALL_MODEL_NAMES, safe_run_for_ctx, io_bundle["split_results_path"],
        model_specs=FINE_MODEL_SPECS, get_builder_fn=_fine_get_builder,
        context_prefix=f"{ctx.split_id}|{ctx.key}", final_hook=_final_hook,
        coarse_survival_fraction=0.10,
        val_split_report_path=io_bundle["split_dir"] / f"val_split_report_{ctx.label_set}.json",
        logs_dir=io_bundle["logs_dir"],
        extra_results_paths=[branch_results_path],
        run_histgbc_deployable=True, # explicit — this is the fix, not an incidental default
        histgbc_max_iter_ceiling=HISTGBC_MAX_ITER_CEILING,
    )
    if not winner_info:
        raise RuntimeError(
            f"SETUP {ctx.key} [{ctx.split_id}]: every model failed selection — check the "
            f"⚠ SKIPPED lines in {io_bundle['logs_dir']}/{ctx.label_set}.log and that branch's "
            "results file entries with status='failed'."
        )

    # Uniform key across all three pipelines, so a cross-pipeline ranking
    # script can read "generalization_bal_accuracy" for every source
    # (fine/merged/3-level). NOT an unbiased generalization estimate: it is
    # the pooled-OOF CV score of the configuration that this same CV chose,
    # on features selected on the same rows, so it is optimistic (Cawley &
    # Talbot 2010; Ambroise & McLachlan 2002). Only Stage 5's test score is.
    for _c in ("accuracy", "bal_accuracy", "macro_f1", "kappa"):
        best_per_model[f"generalization_{_c}"] = best_per_model[f"val_{_c}"]

    winner_msg = (f"\n[{ctx.key} | {ctx.split_id}] Winning feature configuration per model "
                  f"(selection_metric={SELECTION_METRIC}, top-10% coarse survival):")
    logger.notice(winner_msg)
    for m, w in winner_info.items():
        line = (f"  {m:14s}: tag={w['tag']:12s}  scaler={str(w.get('scaler_name')):8s}  "
                f"weighting={w['weighting']:10s}  cv_score={w.get('cv_score', float('nan')):.4f}"
                f"±{w.get('cv_score_std', float('nan')):.4f}")
        logger.notice(line)

    winner_info_path = io_bundle["split_dir"] / f"winner_info_{ctx.key}_{ctx.label_set}.json"
    with open(winner_info_path, "w") as f:
        json.dump(winner_info, f, indent=2, default=str)
    logger.notice(f"  Saved -> {winner_info_path}")

    logger.notice(f"\nSETUP {ctx.key} [{ctx.label_set}, {ctx.split_id}] — FINAL LEADERBOARD "
                  f"(sorted by val_{SELECTION_METRIC}, NO TEST SET USED)")
    # _std columns + histgbc_deployable_used surfaced —
    # previously computed but dropped before reaching this table.
    print_cols = ["model_name", "feature_set", "variant",
                  "val_accuracy", "val_accuracy_std",
                  "val_bal_accuracy", "val_bal_accuracy_std",
                  "val_macro_f1", "val_macro_f1_std",
                  "val_kappa", "val_kappa_std",
                  "train_accuracy", "train_accuracy_std", "train_bal_accuracy", "train_bal_accuracy_std",
                  "train_macro_f1", "train_macro_f1_std", "train_kappa", "train_kappa_std",
                  "cv_mode", "n_folds", "histgbc_deployable_used"]
    print_cols = [c for c in print_cols if c in best_per_model.columns]
    logger.info(best_per_model[print_cols].to_string(index=False))

    _plot_comparison_table_common(
        results_list=best_per_model.to_dict("records"), plots_dir=io_bundle["plots_dir"],
        metric=f"val_{SELECTION_METRIC}",
        title=f"Setup {ctx.key} [{ctx.label_set}, {ctx.split_id}] — Best Config per Model "
              f"(val_{SELECTION_METRIC})",
        filename=f"leaderboard_{ctx.key}_{ctx.label_set}.png",
        logger=logger,
    )
    # Same leaderboard, with the bootstrap CI this
    # module already computes but the point-estimate chart above discards.
    plot_leaderboard_with_ci(
        best_per_model, metric_col=f"val_{SELECTION_METRIC}", std_col=f"val_{SELECTION_METRIC}_std",
        plots_dir=io_bundle["plots_dir"],
        title=f"Setup {ctx.key} [{ctx.label_set}, {ctx.split_id}] — val_{SELECTION_METRIC} "
              f"± bootstrap std",
        filename=f"leaderboard_{ctx.key}_{ctx.label_set}_with_ci.png",
        logger=logger,
    )

    leaderboard_csv = io_bundle["split_dir"] / f"leaderboard_{ctx.key}_{ctx.label_set}.csv"
    leaderboard_json = io_bundle["split_dir"] / f"leaderboard_{ctx.key}_{ctx.label_set}.json"
    best_per_model.to_csv(leaderboard_csv, index=False)
    with open(leaderboard_json, "w") as f:
        json.dump(best_per_model.to_dict("records"), f, indent=2, default=str)
    logger.notice(f"  Saved -> {leaderboard_csv.name}, {leaderboard_json.name}")
    
    summarize_top_models_fine(best_per_model, SELECTION_METRIC, TOP_K_SUMMARY,
                               ctx.key, ctx.label_set, io_bundle["split_dir"], logger)

    return winner_info, best_per_model


# %% [markdown]
# ## Section 6 — Setup A vs Setup B: Effect of Retaining `bcu`
#
# A and B differ in class cardinality (`bcu` retained vs. excluded), so
# this isolates the marginal per-model effect of including `bcu` — it is
# **not** a claim that both setups solve an identical task. Both the
# `SELECTION_METRIC`-selected validation metrics and their train-side
# counterparts are compared side by side, so a reader can spot whether
# Setup A's/B's advantage for a given model is genuine or mostly a
# train/val-gap artifact. **Kept descriptive, not a formal paired
# significance test** — deliberately: A and B compare
# different-cardinality tasks, so a formal test's null hypothesis is
# already a harder thing to state cleanly here than in, say, Module 4's
# Point 3 same-task comparison; a descriptive delta with both setups'
# train/val quartets shown side by side is the more honest presentation
# for this specific comparison.

# %%
def compare_setup_a_vs_b(best_a: pd.DataFrame, best_b: pd.DataFrame, io_bundle: dict, logger) -> pd.DataFrame:
    metric_cols = ["val_accuracy", "val_bal_accuracy", "val_macro_f1", "val_kappa",
                   "train_accuracy", "train_bal_accuracy", "train_macro_f1", "train_kappa"]
    a = best_a[["model_name"] + metric_cols].rename(columns={c: f"{c}_A" for c in metric_cols})
    b = best_b[["model_name"] + metric_cols].rename(columns={c: f"{c}_B" for c in metric_cols})
    merged = a.merge(b, on="model_name", how="inner")
    merged["delta_bal_acc"] = merged["val_bal_accuracy_A"] - merged["val_bal_accuracy_B"]
    merged["delta_macro_f1"] = merged["val_macro_f1_A"] - merged["val_macro_f1_B"]
    merged = merged.sort_values("delta_bal_acc")

    header = "\n" + "=" * 70 + "\nSETUP A (bcu retained) vs SETUP B (bcu excluded) — VALIDATION metrics, no test\n" + "=" * 70
    logger.notice(header)
    logger.info(merged.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, max(5, len(merged) * 0.5)))
    y_pos = np.arange(len(merged))
    ax.barh(y_pos - 0.2, merged["val_bal_accuracy_A"], height=0.4, label="Setup A (+bcu)", color="indianred")
    ax.barh(y_pos + 0.2, merged["val_bal_accuracy_B"], height=0.4, label="Setup B (−bcu)", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["model_name"])
    ax.set_xlabel("Balanced Accuracy (best config per model, VALIDATION)")
    ax.set_title("Effect of Retaining bcu on Fine-Grain Classification (val only)")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = io_bundle["plots_dir"] / "setup_A_vs_B_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.notice(f"  Saved -> {out_path}")

    merged.to_csv(io_bundle["split_dir"] / "setup_A_vs_B_comparison.csv", index=False)
    return merged


# %% [markdown]
# ## Section 7 — MLP Architecture Comparison
#
# Restricted to rows whose `model_name` is one of the six
# `MLP_ARCHITECTURES` topologies, per setup's leaderboard — this answers
# "does going wider/deeper actually help this fine-grain task, or does it
# just add training cost." Train-side metrics are included in the printed
# table (and were already present in the underlying CSV, since `best_df`
# is ledger-derived).

# %%
def compare_mlp_architectures(best_per_model_by_setup: dict, io_bundle: dict, logger,
                               split_id: str) -> pd.DataFrame:
    rows = []
    for setup_key, best_df in best_per_model_by_setup.items():
        mlp_rows = best_df[best_df["model_name"].isin(MLP_ARCHITECTURES.keys())].copy()
        mlp_rows["setup"] = setup_key
        mlp_rows["n_hidden_layers"] = mlp_rows["model_name"].map(
            lambda m: len(MLP_ARCHITECTURES[m]))
        mlp_rows["n_total_neurons"] = mlp_rows["model_name"].map(
            lambda m: int(sum(MLP_ARCHITECTURES[m])))
        rows.append(mlp_rows)
    comparison = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if comparison.empty:
        logger.warning("  ⚠ No MLP_* rows found in any setup's leaderboard — skipping comparison.")
        return comparison

    sort_col = f"val_{SELECTION_METRIC}"
    comparison = comparison.sort_values(["setup", sort_col], ascending=[True, False])
    header = ("\n" + "=" * 70 +
              f"\nMLP ARCHITECTURE COMPARISON (best config per topology, per setup; "
              f"sorted by {sort_col}, VALIDATION only, {split_id})\n" + "=" * 70)
    logger.notice(header)
    print_cols = ["model_name", "feature_set", "variant",
                  "val_accuracy", "val_bal_accuracy", "val_macro_f1", "val_kappa",
                  "train_accuracy", "train_bal_accuracy", "train_macro_f1", "train_kappa"]
    logger.info(comparison[print_cols]
                .to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for setup_key in best_per_model_by_setup:
        sub = comparison[comparison["setup"] == setup_key]
        if sub.empty:
            continue
        best_row = sub.iloc[0]
        logger.notice(f"\n→ Setup {setup_key} best MLP topology: {best_row['model_name']} "
                      f"({MLP_ARCHITECTURES[best_row['model_name']]}) {sort_col}={best_row[sort_col]:.4f}")

        fig, ax = plt.subplots(figsize=(9, max(4, len(sub) * 0.5)))
        sub_sorted = sub.sort_values(sort_col)
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(sub_sorted)))
        bars = ax.barh(sub_sorted["model_name"], sub_sorted[sort_col],
                        color=colors, edgecolor="k", linewidth=0.4)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
        ax.set_xlabel(f"{sort_col} (best feature-set/weighting config, validation)")
        ax.set_title(f"MLP Topology Comparison — Setup {setup_key} [{split_id}]")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        out_path = io_bundle["plots_dir"] / f"mlp_architecture_comparison_{setup_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.notice(f"  Saved -> {out_path}")

    out_csv = io_bundle["split_dir"] / "mlp_architecture_comparison.csv"
    comparison.to_csv(out_csv, index=False)
    logger.notice(f"  Saved -> {out_csv.name}")
    return comparison


# %% [markdown]
# ## Section 8 — Run one split end to end (`run_fine_module_for_split`)
#
# Orchestrates one split: resolves that split's Module-1 datasets and
# feature metadata, sets up the per-split IO bundle (`io_for`), builds both setups' `DatasetContext`s, runs `run_setup_pipeline`
# for each, runs the A-vs-B and MLP-topology comparisons, publishes this
# split's final artifacts into `results_aggregated/{split_id}/`
# (`publish_split_summary` — a plain file copy, no new computation), and
# returns everything a later cross-split summary needs.

# %%
import shutil

def _purge_split_outputs(split_id: str, output_root: Path) -> None:
    split_dir = output_root / split_id
    agg_dir = output_root / "results_aggregated" / split_id
    removed = []
    for d in (split_dir, agg_dir):
        if d.exists():
            shutil.rmtree(d)
            removed.append(d)
    if removed:
        print(f"  🗑 [{split_id}] FORCE_RECOMPUTE — deleted: {[str(d) for d in removed]}")
    else:
        print(f"  🗑 [{split_id}] FORCE_RECOMPUTE — nothing to delete.")


# %%
def run_fine_module_for_split(split_id: str) -> dict:
    source_dir = PROCESSING_ROOT / split_id
    feature_metadata_path = source_dir / "diagnostics" / "feature_metadata.json"
    if not feature_metadata_path.exists():
        raise FileNotFoundError(
            f"{feature_metadata_path} not found — run fermi_processing.py "
            f"(Module 1) first, or check that split '{split_id}' actually exists "
            "under PROCESSING_ROOT."
        )
    with open(feature_metadata_path) as f:
        feature_metadata = json.load(f)

    io_bundle = io_for(OUTPUT_ROOT, split_id)
    module_logger = get_section_logger("_module", io_bundle["logs_dir"])
    module_logger.notice(f"\n{'=' * 78}\nMODULE 2 (Fine-Grain) — split '{split_id}'\n{'=' * 78}")
    module_logger.notice(f"  datasets   -> {(source_dir / 'datasets').resolve()}")
    module_logger.notice(f"  results    -> {io_bundle['results_dir'].resolve()}")
    module_logger.notice(f"  logs       -> {io_bundle['logs_dir'].resolve()}")
    module_logger.notice(f"  plots      -> {io_bundle['plots_dir'].resolve()}")

    setup_keys_here = [k for k in FINE_SETUP_KEYS if k in feature_metadata["setups"]]

    contexts = {key: build_fine_context(source_dir, feature_metadata, key, split_id) for key in setup_keys_here}
    for _k, _ctx in contexts.items():
        module_logger.notice(
            f"Context {_k} [{_ctx.label_set:11s}]: {len(_ctx.class_names):>2} classes, "
            f"tags={_ctx.tags}, train={len(_ctx.y_train_enc):>5}, "
            f"test={len(_ctx.y_test_enc):>5}, 'bcu' present={'bcu' in _ctx.class_names}"
        )

    module_logger.notice(f"\nRunning fine-grain sweeps for setups {setup_keys_here} — "
                          f"selection_metric={SELECTION_METRIC}  split={split_id}")
    winner_info_by_setup, best_per_model_by_setup = {}, {}
    for _setup_key in pbar(setup_keys_here, desc=f"[{split_id}] Setup {'/'.join(setup_keys_here)}"):
        winner_info_by_setup[_setup_key], best_per_model_by_setup[_setup_key] = run_setup_pipeline(contexts[_setup_key], io_bundle)

    if "A" in setup_keys_here:
        ab_comparison = compare_setup_a_vs_b(best_per_model_by_setup["A"], best_per_model_by_setup["B"], io_bundle, module_logger)
    else:
        ab_comparison = None
        module_logger.notice(f"  Setup A skipped for split '{split_id}' (split_idx != 0) — A-vs-B comparison not applicable.")

    mlp_comparison = compare_mlp_architectures(best_per_model_by_setup, io_bundle, module_logger, split_id)

    module_logger.notice("=" * 70)
    module_logger.notice(f"Module 2 (Fine-Grain Classification) complete — split '{split_id}'.")
    module_logger.notice(f"  Selection metric   : {SELECTION_METRIC}  (kappa/macro-F1 diagnostic-only, "
                          "never a sort key)")
    for _key in setup_keys_here:
        module_logger.notice(f"  Setup {_key} [{FINE_SETUP_LABELS[_key]}] leaderboard rows: "
                              f"{len(best_per_model_by_setup[_key])}")
    if "A" in setup_keys_here:
        module_logger.notice(f"  A-vs-B comparison  : {io_bundle['split_dir'] / 'setup_A_vs_B_comparison.csv'}")
    if not mlp_comparison.empty:
        module_logger.notice(f"  MLP topology sweep : "
                              f"{io_bundle['split_dir'] / 'mlp_architecture_comparison.csv'} "
                              f"({len(MLP_ARCHITECTURES)} topologies x {len(setup_keys_here)} setups)")
    module_logger.notice("  DecisionTree CV    : Stratified-K-Fold ONLY (adaptive LOO disabled for "
                          "this module — see _build_dtree_kfold_only; state this explicitly in the "
                          "paper's methods text)")
    module_logger.notice("  HistGBC            : deployable StratifiedEarlyStoppingHGBC used for its "
                          "OOF pass and train-refit — see the "
                          "'histgbc_deployable_used' leaderboard column")
    module_logger.notice(f"  Split results ledger : {io_bundle['split_results_path']}")
    module_logger.notice(f"  Metadata manifest    : {io_bundle['metadata_manifest_path']}")
    module_logger.notice(f"  Plots                : {io_bundle['plots_dir']}")
    module_logger.notice("=" * 70)

    summary_files = [io_bundle["metadata_manifest_path"], io_bundle["split_dir"] / "mlp_architecture_comparison.csv"]
    if "A" in setup_keys_here:
        summary_files.append(io_bundle["split_dir"] / "setup_A_vs_B_comparison.csv")
    for _setup_key in setup_keys_here:
        _label_set = FINE_SETUP_LABELS[_setup_key]
        summary_files += [io_bundle["split_dir"] / f"leaderboard_{_setup_key}_{_label_set}.csv",
                           io_bundle["split_dir"] / f"leaderboard_{_setup_key}_{_label_set}.json",
                           io_bundle["split_dir"] / f"winner_info_{_setup_key}_{_label_set}.json",
                           io_bundle["split_dir"] / f"val_split_report_{_label_set}.json"]
    publish_split_summary(split_id, OUTPUT_ROOT, summary_files)

    return {
        "split_id": split_id,
        "winner_info_by_setup": winner_info_by_setup,
        "best_per_model_by_setup": best_per_model_by_setup,
        "ab_comparison": ab_comparison,
        "mlp_comparison": mlp_comparison,
    }


# %% [markdown]
# ## Section 9 — Run every requested split, then build the cross-split summary
#
# `build_all_splits_summary`
# concatenates every split's Setup A/B leaderboard into one long table —
# trivially one split's worth of rows today (`ENABLE_MULTI_SPLIT_VALIDATION`
# is off by default), growing automatically the moment it's turned on.
# This is the file that answers "how much does the winning configuration
# move across draws of the data," which is the entire point of Module 1's
# multi-split design.

# %%
def _reconstruct_result_from_leaderboards(split_id: str, output_root: Path) -> dict:
    """Read back an already-completed split's per-setup leaderboards instead
    of recomputing them. `best_per_model_by_setup[key]` is exactly what
    run_setup_pipeline() saved to leaderboard_{key}_{label_set}.csv (L621-626),
    so this is a byte-for-byte reload — the only thing build_all_splits_summary()
    actually reads out of the result dict."""

    agg_dir = output_root / "results_aggregated" / split_id
    best_per_model_by_setup = {}
    for setup_key in FINE_SETUP_KEYS:
        label_set = FINE_SETUP_LABELS[setup_key]
        csv_path = agg_dir / f"leaderboard_{setup_key}_{label_set}.csv"
        if not csv_path.exists():
            continue  # Setup A skipped for this split
        best_per_model_by_setup[setup_key] = pd.read_csv(csv_path)
    return {"split_id": split_id, "winner_info_by_setup": None,
            "best_per_model_by_setup": best_per_model_by_setup,
            "ab_comparison": None, "mlp_comparison": None}


# %%
def build_all_splits_summary_fine(output_root: Path, all_splits_results: dict):
    rows = []
    for split_id, result in all_splits_results.items():
        for setup_key, df in result["best_per_model_by_setup"].items():
            df = df.copy()
            df["split_id"] = split_id
            df["setup_key"] = setup_key
            rows.append(df)
    if not rows:
        print("  ⚠ build_all_splits_summary_fine: nothing to summarize.")
        return None
    combined = pd.concat(rows, ignore_index=True)
    out_dir = output_root / "results_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_splits_summary.csv"
    combined.to_csv(out_path, index=False)
    print(f"  ✓ all_splits_summary -> {out_path}  "
          f"({len(combined)} rows across {len(all_splits_results)} split(s))")
    return out_path


# %%
# %%
import traceback

ALL_SPLITS_RESULTS = {}
FAILED_SPLITS = []
for _split_id in pbar(SPLITS_TO_RUN, desc="Fine module — splits"):
    _done_path = OUTPUT_ROOT / "results_aggregated" / _split_id / f"leaderboard_B_{FINE_SETUP_LABELS['B']}.csv"
    if FORCE_RECOMPUTE:
        _purge_split_outputs(_split_id, OUTPUT_ROOT)
    elif _done_path.exists():
        print(f"  ⏭ [{_split_id}] already complete — reading leaderboards back, no recompute.")
        ALL_SPLITS_RESULTS[_split_id] = _reconstruct_result_from_leaderboards(_split_id, OUTPUT_ROOT)
        continue
    try:
        ALL_SPLITS_RESULTS[_split_id] = run_fine_module_for_split(_split_id)
        # A traceback left by an earlier failed attempt is stale once the split succeeds.
        (OUTPUT_ROOT / "results_aggregated" / _split_id / "CRASH_traceback.txt").unlink(missing_ok=True)
    except Exception:
        _err_dir = OUTPUT_ROOT / "results_aggregated" / _split_id
        _err_dir.mkdir(parents=True, exist_ok=True)
        _err_path = _err_dir / "CRASH_traceback.txt"
        _err_path.write_text(traceback.format_exc())
        FAILED_SPLITS.append(_split_id)
        print(f"  ✗ [{_split_id}] CRASHED — traceback -> {_err_path}. Continuing to next split.")

if FAILED_SPLITS:
    print(f"⚠ {len(FAILED_SPLITS)}/{len(SPLITS_TO_RUN)} failed: {FAILED_SPLITS}. "
          "Fix and just re-run this cell — completed splits are skipped automatically.")

# %%
# %%
ALL_SPLITS_SUMMARY_PATH = build_all_splits_summary_fine(OUTPUT_ROOT, ALL_SPLITS_RESULTS)

print("=" * 78)
print("Module 2 (Fine-Grain) — ALL REQUESTED SPLITS COMPLETE.")
print(f"  Splits run          : {SPLITS_TO_RUN}")
print(f"  Cross-split summary  : {ALL_SPLITS_SUMMARY_PATH}")

# %%
print("=" * 78)
print("Module 2 (Fine-Grain) — ALL REQUESTED SPLITS COMPLETE.")
print(f"  Splits run          : {SPLITS_TO_RUN}")
print(f"  Cross-split summary  : {ALL_SPLITS_SUMMARY_PATH}")
print(f"  Per-split artifacts  : {OUTPUT_ROOT / 'results_aggregated'}/{{split_id}}/")
print("  Reminder — this module's own leaderboards are a model-SELECTION artifact, "
      "not a directly reportable result: the paper's headline fine-grain number should "
      "come from fermi_validation.py's nested-CV-corrected score, not the naive "
      "argmax reported here.")
print("=" * 78)

# %%
for split_id, result in ALL_SPLITS_RESULTS.items():
    split_dir = OUTPUT_ROOT / split_id
    for setup_key, best_df in result["best_per_model_by_setup"].items():
        summarize_top_models_fine(
            best_df, SELECTION_METRIC, TOP_K_SUMMARY,
            setup_key, FINE_SETUP_LABELS[setup_key], split_dir, logger=None
        )
