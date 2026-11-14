# %% [markdown]
# # fermi_validation.py

# %% [markdown]
# ## Cross-Pipeline Validation — Module 4
#
# Runs against the shared kernel
# (`fermi_common.py` — `SEED`, `VERSION_TAG`, `SELECTION_METRIC`,
# `USE_GPU`, `MODEL_SPECS`, `ALL_MODEL_NAMES`, `MLP_ARCHITECTURES`,
# `PCA_BASE_TAG`, `PCA_VARIANTS`, `SCALER_VARIANTS`, `WEIGHTING_MODES`,
# `DEFAULT_SCALER_NAME`, `COARSE_VAL_FRACTION`, `COARSE_SURVIVAL_FRACTION`,
# `COARSE_MIN_SURVIVORS`, `TAXONOMY`, `SUPERCLASS_ORDER`, `_select_winners`,
# `fit_component`, `get_builder`, `weight_support_for`, `dispatch_fit`,
# `cv_score_estimator`, `_score_by_name`, `evaluate_model`, `load_results`,
# `results_to_dataframe`, `_safe_filename`, `get_section_logger`, `pbar`,
# `SPLIT_ID_DEFAULT`, `io_for`, `save_result_multi`, `publish_split_summary`,
# `load_all_results` — bare names, in-scope, not imported cross-file), and
# against the already-executed `fermi_processing.py` (Module 1),
# `fermi_fine.py` (Module 2), `fermi_merged.py` (Module 3, 2-level),
# and `fermi_3levels.py` (Module 3-alt, 3-level) — this file reads their
# persisted per-split artifacts and, for context rebuilding, uses those
# three modules' own `build_fine_context`/`build_merged_context`/
# `build_3level_context` functions (bare names, still in kernel scope).
#
# **Fixed choice, not re-derived per family**: every stage below runs
# against fine **Setup B** (`fine_nobcu_merged`, `bcu` excluded) — matching
# `FINE_SETUP_KEY = "B"` hardcoded in both `fermi_merged.py` and
# `fermi_3levels.py`. Setup A is never a fair fourth competitor here
# (different class cardinality), so restricting to Setup B keeps every
# family's `SELECTION_METRIC` scores comparable on the identical
# labeled-row pool and identical fine-class taxonomy.
#
# ## Design notes
#
# 1. **Inputs.** Every sibling-module leaderboard/JSON this file reads
#    (`level1_leaderboard`, `level1_gate_leaderboard`,
#    `level2_blazar_subgate_leaderboard`, `level2_nonblazar_router_leaderboard`,
#    `level3_branches`, `merged_classification_metrics.json`,
#    `3level_classification_metrics.json`, per-branch `results/branch_{sc}.json`
#    / `results/branch_{sc}_L3.json`) is read from `PROCESSING_ROOT` /
#    `OUTPUT_ROOT`. The `_m2_*`-prefixed merged-specific chain-CV helpers
#    (Section 1) are required: `fermi_merged.py` and `fermi_3levels.py` both
#    define bare top-level `build_branch_datasets`/`_fit_and_route_chain_fold`/
#    `_predict_proba_safe`/`_resolve_full_length_tag_frame`, and 3-level runs
#    last in the shared kernel, so it wins the bare names.
# 2. **Per-family "every model validated" CSVs + per-node leaderboard CSVs +
#    end-to-end CSV** (Section 12) — every candidate Stage 3 scored, not just
#    the Stage 4 winner, lands in a CSV with train/val metrics AND their
#    cross-split error (std); for `merged`/`3level` an additional CSV per
#    hierarchy node (router/branch, or gate/subgate/nonblazar_router/branch)
#    is exported from the sibling module's own per-node leaderboard, plus a
#    contracted end-to-end architecture name and a `node_models` dict
#    recording which model won at every node (Section 2's
#    `build_end_to_end_name`).
# 3. **Full cross-pipeline report with CI on every metric** (Section 14) —
#    every `REPORT_METRICS` column (including `precision_macro`/
#    `recall_macro`) gets a mean, a std ("error"), AND a 95% CI, computed
#    differently depending on what the estimate is (see Section 2's
#    `_metric_ci_across_splits` docstring for the CI-vs-error rationale).
# 4. **Plain accuracy is one of four Borda ranking criteria**:
#    `QUALITY_METRICS = ("accuracy","bal_accuracy","macro_f1","kappa")`, a
#    deliberate, disclosed choice. Every Borda ranking in this file (Stage
#    1/3/4's candidate-level selection, Stage 8's family-level selection)
#    uses this constant.
# 5. **Unassociated-source inference** lives in `fermi_inference.py`, which
#    does not import this file (that would re-run the whole validation
#    driver, since this file has no `if __name__` guard): it re-derives
#    `SPLIT0`/`VALIDATION_MODELS_DIR`/etc. with the same formulas used here
#    and loads Stage 7's deployed `.joblib` models plus Stage 8's
#    `cross_pipeline_validation_report.json` from disk. Run this file first
#    (through Stage 8, at least), then `fermi_inference.py`.
# 6. **Per-class metrics export** (Section 16) — precision/recall/F1/support
#    per fine class for each family's Stage-5 winner, since the
#    macro-averaged `QUALITY_METRICS` alone hide which specific rare classes
#    (`hmb`/`nov`/`sey`-scale) are driving a low macro score.
# 7. **Calibration / Brier-score check** (Section 16b, opt-in via
#    `RUN_CALIBRATION_CHECK_NOW`) — is a source predicted at ~0.8 confidence
#    actually correct ~80% of the time? A per-split multiclass Brier score
#    plus a 10-bin reliability table answer that directly, reusing
#    `_fit_predict_merged_core`/`_fit_predict_3level_core`'s
#    `return_proba=True` path. It is opt-in because it is the one section
#    past Stage 5 that refits from scratch (once per family and split);
#    nothing downstream reads `CALIBRATION_CHECK`. Checkpointed under
#    `"stage9c_calibration"`.
# 8. **Rare-class hierarchical partial-pooling shrinkage** (Section 16c) —
#    reporting-ONLY by design: a closed-form empirical-Bayes (Beta-Binomial,
#    method-of-moments / James-Stein-type) reanalysis of the per-fine-class
#    confusion counts Stage 5 already produced, pooled across
#    `VALIDATION_SPLITS` — no new CV folds, no refits. Grouping comes from
#    the taxonomy (`TAXONOMY`/`SUPERCLASS_ORDER`/`BLAZAR_SUPERCLASSES`):
#    `fine` gets 1 level (every fine class pooled toward one grand mean),
#    `merged` 2 levels (fine nested in its superclass), `3level` 3 levels
#    (fine nested in superclass nested in the gate's BLAZAR/OTHER split).
#    **It never feeds back into `QUALITY_METRICS`/`SELECTION_METRIC`, Borda
#    ranking, or any architecture selection** — using shrinkage-adjusted
#    metrics as a selection criterion would need its own validation first.
#    Checkpointed as `"stage9d_shrinkage"`, depending only on `"stage5"`.
# 9. **Computational cost control, without touching combo-grid width,
#    repeat count, or fold count.** Two correctness-preserving levers: (a) a
#    node-fit dedup cache (`node_cache`, Section 1a/1b) shared across every
#    candidate scored in the same Stage 1 (per (context, seed)) or Stage 3
#    (per split) sweep — `StratifiedKFold`'s fold boundaries never depend on
#    candidate config, so two candidates sharing an identical router/branch/
#    gate/subgate/nonblazar_router config get that fit only once; (b)
#    `STAGE3_SHORTLIST_TOP_K` (gated behind `REDUCE_STAGE3_CANDIDATES`) trims
#    Stage 3's more expensive rescoring to the top K of Stage 1's
#    Borda-ranked shortlist per family. Disclosed caveat: Stage 1's
#    single-split ranking is itself noisy, so a candidate that would have
#    overtaken the top K on confirmation can be dropped.
# 10. **NaN-safe config reconstruction.** A branch model with no PCA
#    round-trips through `json.dump` as a literal `NaN`, not `null`;
#    `_nan_to_none` is applied in both `_leaderboard_row_to_cfg` and
#    `_branch_cfgs_from_metrics`, so `PCA(n_components=nan)` can never be
#    built.
# 11. **Optional / reference-only code.** Stage 2 (winner's-curse nested-CV
#    diagnostic), Stage 6 (family-vs-family significance testing, with
#    Cohen's d and Holm-Bonferroni correction) and the SHAP explainability
#    section are fully written but sit in the DEAD CODE section at the end
#    of this file, not wired into `STAGE_DEPENDENCIES`, the Section 13
#    driver, or Stage 8's report. Active stages: 1, 3, 4, 5, 7, 8, 9. See
#    the DEAD CODE section's docstring for how to re-enable any of them.

# %%
import json
import warnings
from pathlib import Path
from typing import Optional, Sequence
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import t as _t_dist
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, classification_report, confusion_matrix
import joblib

# %% [markdown]
# ## Section 0 — Config, constants, output layout

# %%
PROCESSING_ROOT = _cfg("PROCESSING_ROOT", Path(f"./fermi_module1_processing_{VERSION_TAG}/"))
OUTPUT_ROOT = _cfg("OUTPUT_ROOT", Path(f"./outputs_{VERSION_TAG}_{SELECTION_METRIC}/"))
VALIDATION_DIR = OUTPUT_ROOT / "validation"
VALIDATION_LOGS_DIR = VALIDATION_DIR / "logs"
VALIDATION_RESULTS_DIR = VALIDATION_DIR / "results"
VALIDATION_PLOTS_DIR = VALIDATION_DIR / "plots"
VALIDATION_MODELS_DIR = VALIDATION_DIR / "deployed_models"
VALIDATION_CHECKPOINTS_DIR = VALIDATION_DIR / "checkpoints"
# Dedicated CSV-report tree — every requirement-#1/#2 CSV in this
# file lands here, kept separate from VALIDATION_RESULTS_DIR's existing
# .npz/.json artifacts so "give me every CSV" is one directory listing.
VALIDATION_CSV_DIR = VALIDATION_DIR / "csv_reports"
VALIDATION_CSV_NODES_DIR = VALIDATION_CSV_DIR / "per_node_leaderboards"
# Every confusion matrix this file now saves (every
# Stage 1/3 candidate — requirement #3 — plus each family's Stage-5 winner on both validation and
# test — requirement #4) lands here: a CSV (always) under VALIDATION_CONFUSION_DIR, mirrored as a
# PNG heatmap (best-effort — see `_save_confusion_matrix`) under VALIDATION_CONFUSION_PLOTS_DIR.
VALIDATION_CONFUSION_DIR = VALIDATION_CSV_DIR / "confusion_matrices"
VALIDATION_CONFUSION_PLOTS_DIR = VALIDATION_PLOTS_DIR / "confusion_matrices"
for _d in (VALIDATION_LOGS_DIR, VALIDATION_RESULTS_DIR, VALIDATION_PLOTS_DIR, VALIDATION_MODELS_DIR,
           VALIDATION_CHECKPOINTS_DIR, VALIDATION_CSV_DIR, VALIDATION_CSV_NODES_DIR,
           VALIDATION_CONFUSION_DIR / "stage1_shortlist", VALIDATION_CONFUSION_DIR / "stage3_confirmation",
           VALIDATION_CONFUSION_DIR / "stage5_winners",
           VALIDATION_CONFUSION_PLOTS_DIR / "stage1_shortlist", VALIDATION_CONFUSION_PLOTS_DIR / "stage3_confirmation",
           VALIDATION_CONFUSION_PLOTS_DIR / "stage5_winners"):
    _d.mkdir(parents=True, exist_ok=True)

# Per-stage checkpointing. Checkpoints live under
# VALIDATION_CHECKPOINTS_DIR, nested under OUTPUT_ROOT (VERSION_TAG + SELECTION_METRIC-scoped),
# so a checkpoint from a different VERSION_TAG/SELECTION_METRIC run is never silently picked up.
# List stage names here to force them (and, transitively, anything downstream — see
# STAGE_DEPENDENCIES) to ignore any cached checkpoint and recompute.
# Valid names: "split_contexts", "stage1", "stage3".."stage5", "stage7", "stage8", "stage8b_csv",
# "stage9b_perclass", "stage9c_calibration" (only runs if RUN_CALIBRATION_CHECK_NOW=True — see
# Section 16b), "stage9d_shrinkage". "stage2" and "stage6" live in
# the DEAD CODE section at the end of this file and are not part of the automatic dependency
# cascade; unassociated-source inference lives in fermi_inference.py, outside this file's
# checkpoint namespace.
FORCE_RECOMPUTE_STAGES = set(_cfg("FORCE_RECOMPUTE_STAGES", set()))  # e.g. {"stage1"} after a bug fix that changes Stage 1's scoring

FINE_SETUP_KEY_VALIDATION = "B"
# Which architecture families to validate. ["fine"] skips the hierarchical ones everywhere
# (their Module-3 searches don't need to run).
FAMILIES = list(_cfg("FAMILIES", ["fine", "merged", "3level"]))
FAMILY_LABELS = {"fine": "Fine-Grain (flat)", "merged": "Hierarchical Merged (2-level)",
                  "3level": "Hierarchical 3-Level"}

# --- Constants table -------------------------------------------------------
N_VALIDATION_SPLITS = _cfg("N_VALIDATION_SPLITS", 5)
TOP_K_SHORTLIST = _cfg("TOP_K_SHORTLIST", 15)
N_REPEATS_SHORTLIST_CV = _cfg("N_REPEATS_SHORTLIST_CV", 5)
N_CONFIRMATION_FOLDS = 5
STAGE3_MIN_SPLITS_ELIGIBLE = min(4, N_VALIDATION_SPLITS)  # of N_VALIDATION_SPLITS (1 = single-split mode)
STAGE3_MIN_SPLITS_SELECTED = N_VALIDATION_SPLITS  # the chosen winner needs every split
# Per-split selection ("option A"): Stages 1-4 run inside EACH split on that split's own
# training data (Modules 2/3 must have searched every split), and Stage 5 scores each split's
# own winner on that split's test set. No selection step then reads data that overlaps a test
# set, so the Stage-5 mean estimates the performance of the whole selection procedure
# (Varma & Simon 2006; Krstajic et al. 2014). Split 0's selection is the one deployed.
PER_SPLIT_SELECTION = _cfg("PER_SPLIT_SELECTION", False)
if PER_SPLIT_SELECTION:
    STAGE3_MIN_SPLITS_ELIGIBLE = STAGE3_MIN_SPLITS_SELECTED = 1  # each split only sees itself
STAGE4_WINNERS_BY_SPLIT = None  # {split_id: Stage-4 winners}, filled by the driver under PER_SPLIT_SELECTION
# {family: model} — deploy this Stage-3 candidate instead of the Borda winner (matched against
# full_model_name, the config's model_name, or the hierarchical combo_name). Forces Stage 4 and
# everything after it to recompute.
WINNER_OVERRIDE = dict(_cfg("WINNER_OVERRIDE", {}))
if WINNER_OVERRIDE:
    FORCE_RECOMPUTE_STAGES |= {"stage4"}


def _winner_for(stage4_winners: dict, fam: str, split_id: str) -> dict:
    """The winner to refit/score on `split_id`: that split's own selection under
    PER_SPLIT_SELECTION, otherwise the single shared Stage-4 winner."""
    return ((STAGE4_WINNERS_BY_SPLIT or {}).get(split_id) or stage4_winners)[fam]

# Stage 3 re-scores every Stage-1 shortlisted candidate on
# EVERY validation split (merged/3level candidates are the expensive ones —
# each split re-runs a full chain-CV). Since Stage 1's output is already
# Borda-sorted best-first (see `run_stage1`), carrying only the top K into
# Stage 3 drops the long, noisy tail of the shortlist without touching any
# combo-grid width, repeat count, or fold count (those were explicitly kept
# at full resolution). Gated behind a flag, defaulting True; set
# REDUCE_STAGE3_CANDIDATES=False to carry the full TOP_K_SHORTLIST-sized (or
# TOP_N_* combinatorial) shortlist into Stage 3.
REDUCE_STAGE3_CANDIDATES = True
STAGE3_SHORTLIST_TOP_K = _cfg("STAGE3_SHORTLIST_TOP_K", 10)

# --- Section 16b (calibration/Brier check) speed knob ----------
# Unlike Section 16 (per-class metrics) and Section 16c (rare-class
# shrinkage), which only re-read Stage 5's already-cached
# `predictions_by_split`, Section 16b REFITS Stage 4's winning architecture
# from scratch — once per (family, split), i.e. 3 * len(VALIDATION_SPLITS)
# refits, with merged/3level each being a full chain-CV routing fit, not a
# single model. It was the one reporting-only stage past Stage 5 still
# wired into the automatic driver with no opt-in, unlike Stage 2/6/SHAP,
# all DEAD CODE now (see the section at the end of this file). Mirrors
# RUN_WINNERS_CURSE_DIAGNOSTIC_NOW/RUN_SHAP_EXPLAINABILITY_NOW's pattern;
# default False. Nothing downstream in this file reads CALIBRATION_CHECK
# (verified: not in STAGE_DEPENDENCIES for "stage8", not referenced by the
# DEAD CODE section), so leaving this off costs nothing but the
# Brier-score/reliability-table CSVs themselves. Still checkpointed under
# "stage9c_calibration" when it does run.
RUN_CALIBRATION_CHECK_NOW = _cfg("RUN_CALIBRATION_CHECK_NOW", True)

# --- merged/3-level combo-grid widths, mirrored from their own modules ----
TOP_N_ROUTERS, TOP_N_BRANCH_COMBOS = 3, 5                       # merged
TOP_N_GATES, TOP_N_BLAZAR_SUBGATES, TOP_N_NONBLAZAR_ROUTERS = 3, 3, 3  # 3-level

_module_logger = get_section_logger("_module", VALIDATION_LOGS_DIR)
_module_logger.notice(f"\n{'=' * 78}\nMODULE 4 (Cross-Pipeline Validation) — fermi_validation.py\n{'=' * 78}")
_module_logger.notice(f"  Selection metric        : {SELECTION_METRIC}")
_module_logger.notice(f"  Fine setup used          : {FINE_SETUP_KEY_VALIDATION}")
_module_logger.notice(f"  Processing root          : {PROCESSING_ROOT.resolve()}")
_module_logger.notice(f"  Output root              : {OUTPUT_ROOT.resolve()}")
_module_logger.notice(f"  Validation dir           : {VALIDATION_DIR.resolve()}")
_module_logger.notice(f"  CSV reports dir          : {VALIDATION_CSV_DIR.resolve()}")

CHECKPOINT_SCHEMA_VERSION = 3  # bump whenever a stage's checkpoint gains/loses keys (Stage 1/3/4/5/7/8
# rows carry per-candidate/per-winner confusion-matrix dicts+paths and, Stage 5, winner train/
# validation/test metrics-with-error) so older checkpoints are treated as stale and recomputed.

STAGE_DEPENDENCIES = {
    "split_contexts": [],
    "stage1": ["split_contexts"],
    "stage3": ["split_contexts", "stage1"],
    "stage4": ["stage3"],
    "stage5": ["split_contexts", "stage4"],
    "stage7": ["split_contexts", "stage4"],
    # "stage6" lives in the DEAD CODE section and is not part of this cascade.
    "stage8": ["stage1", "stage3", "stage4", "stage5", "stage7"],
    # CSV exports read stage1/stage3/stage4/stage5's already-computed rows plus each
    # sibling module's persisted per-node leaderboards straight off disk — no new computation,
    # so this is cheap to force independently of stage8's own JSON report.
    "stage8b_csv": ["stage1", "stage3", "stage4", "stage5"],
    # Unassociated-source inference lives in fermi_inference.py, outside this namespace.
    # Per-class metrics (Section 16) only reads Stage 5's already-cached split
    # predictions — cheap, and independent of anything else in this file.
    "stage9b_perclass": ["stage5"],
    # Calibration/Brier-score check (Section 16b) refits Stage 4's winners per split
    # to get soft-voted probabilities (never captured by Stage 5's own hard-prediction refit).
    "stage9c_calibration": ["stage4", "split_contexts"],
    # Rare-class shrinkage (Section 16c) reads Stage 5's already-cached
    # predictions_by_split (identical source to stage9b_perclass) — no new computation, so it is
    # cheap to force independently, and deliberately depends on nothing past "stage5": it must
    # never cascade-force, or be forced by, any selection stage (stage1/3/4) or stage8's report.
    "stage9d_shrinkage": ["stage5"],
    # Shift-aware test metrics (Section 16a): Stage 5's cached predictions plus a cheap
    # domain classifier per split; no classifier from the pipeline is refit.
    "stage9e_shift": ["stage5"],
    # "stage2" and "stage6" deliberately have NO entry here — see the DEAD CODE section at the end
    # of this file. Both are checkpointed under their own name if you run them manually, but
    # nothing in the automatic dependency cascade above forces or is forced by either.
}


def _stage_effectively_forced(name: str) -> bool:
    seen, stack = set(), [name]
    while stack:
        cur = stack.pop().split("@")[0]  # "stage1@split_01" (per-split selection) forces like "stage1"
        if cur in seen:
            continue
        seen.add(cur)
        if cur in FORCE_RECOMPUTE_STAGES:
            return True
        stack.extend(STAGE_DEPENDENCIES.get(cur, []))
    return False


def _checkpoint_path(name: str) -> Path:
    return VALIDATION_CHECKPOINTS_DIR / f"{name}.joblib"


def _load_checkpoint(name: str):
    if _stage_effectively_forced(name):
        _module_logger.notice(f"  ↻ '{name}' (or an upstream stage it depends on) is in "
                               "FORCE_RECOMPUTE_STAGES — ignoring any cached checkpoint.")
        return None
    path = _checkpoint_path(name)
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            _module_logger.warning(f"  ⚠ checkpoint '{name}' has a stale schema_version — recomputing.")
            return None
        _module_logger.notice(f"  -> '{name}' loaded from checkpoint ({path}) — skipping recomputation.")
        return payload["data"]
    except Exception as e:
        _module_logger.warning(f"  ⚠ failed to load checkpoint '{name}' ({type(e).__name__}: {e}) — recomputing.")
        return None


def _save_checkpoint(name: str, data) -> None:
    path = _checkpoint_path(name)
    try:
        joblib.dump({"schema_version": CHECKPOINT_SCHEMA_VERSION, "data": data}, path)
        _module_logger.notice(f"  -> '{name}' checkpoint saved -> {path}")
    except Exception as e:
        _module_logger.warning(f"  ⚠ failed to save checkpoint '{name}' ({type(e).__name__}: {e}) — "
                                "continuing without it; this stage will simply re-run next time.")


def _run_cached(name: str, compute_fn, summarize_fn=None):
    """`summarize_fn`, when given, is called
    with the stage's result whenever that result came from a checkpoint hit
    — i.e. exactly the case where `compute_fn` (and therefore every
    `logger.notice(...)` summary line normally printed while computing that
    stage) never runs. This makes a checkpoint-loaded run print the same
    per-step summary a freshly-computed run does, instead of only the
    terse "-> 'name' loaded from checkpoint" line. On a cache MISS,
    `summarize_fn` is deliberately NOT called here — `compute_fn` (the
    stage's own `run_stageN`) already prints its summary inline while it
    computes, so calling `summarize_fn` again afterwards would just
    duplicate that output.

    If `summarize_fn` RETURNS a non-`None` value,
    that value replaces `cached` as this call's result (and as what gets
    returned to the driver). Every existing `summarize_fn` implicitly
    returns `None` (no `return` statement) so this is a no-op for them —
    it exists for report-authoring stages (`stage8`, `stage8b_csv`) whose
    entire job is cheap pure-aggregation of OTHER stages' already-computed
    results: without this, a checkpoint hit for e.g. `stage8` would print a
    summary of whatever `cross_pipeline_validation_report.json` happened to
    contain from the run that ORIGINALLY computed it, even if an upstream
    stage's checkpoint was since backfilled/regenerated in place (see
    `_backfill_stage3_metric_std`/`_regenerate_confusion_matrix_artifacts`)
    — the JSON file on disk would then stay frozen at its stale content
    forever, since nothing else ever rewrites it. Their `summarize_fn`s
    rebuild fresh from the live upstream results and return the result,
    so it propagates both to this function's return value and back onto
    disk."""
    cached = _load_checkpoint(name)
    if cached is not None:
        if summarize_fn is not None:
            try:
                replacement = summarize_fn(cached)
                if replacement is not None:
                    cached = replacement
            except Exception as e:
                _module_logger.warning(f"  ⚠ summarize_fn for checkpointed stage '{name}' failed "
                                        f"({type(e).__name__}: {e}) — data is still valid, only the "
                                        "printed summary is affected.")
        return cached
    result = compute_fn()
    _save_checkpoint(name, result)
    return result


def _val_discover_splits(processing_root: Path) -> list:
    """Every split Module 1 actually produced — independent of what
    fine/merged/3-level individually chose to loop over. Reads the
    same `multi_split_manifest.json` shape `fermi_processing.py`
    writes (`{"splits": [{"split_idx","split_seed","split_dir",...}, ...]}`)."""
    manifest_path = processing_root / "multi_split_manifest.json"
    if not manifest_path.exists():
        _module_logger.warning(f"  ⚠ {manifest_path} not found — assuming only "
                                f"'{SPLIT_ID_DEFAULT}' exists.")
        return [SPLIT_ID_DEFAULT]
    with open(manifest_path) as f:
        manifest = json.load(f)
    raw = manifest.get("splits") or []
    if not raw:
        return [SPLIT_ID_DEFAULT]
    return [Path(s["split_dir"]).name if isinstance(s, dict) and "split_dir" in s else str(s) for s in raw]


ALL_DISCOVERED_SPLITS = _val_discover_splits(PROCESSING_ROOT)
VALIDATION_SPLITS = ALL_DISCOVERED_SPLITS[:N_VALIDATION_SPLITS]
if len(VALIDATION_SPLITS) < N_VALIDATION_SPLITS:
    _module_logger.warning(
        f"  ⚠ Only {len(VALIDATION_SPLITS)}/{N_VALIDATION_SPLITS} split(s) discovered under "
        f"{PROCESSING_ROOT} ({VALIDATION_SPLITS}). Stage 3/5/6 need genuine independent draws — "
        "re-run fermi_processing.py with ENABLE_MULTI_SPLIT_VALIDATION=True and "
        f"N_VALIDATION_SPLITS={N_VALIDATION_SPLITS} for the full design. Proceeding with what "
        "exists; every downstream stage's own n (and its confidence intervals) will reflect this.")
SPLIT0 = VALIDATION_SPLITS[0]
SPLIT0_DIR = OUTPUT_ROOT / SPLIT0
_module_logger.notice(f"  Validation splits         : {VALIDATION_SPLITS}")

with open(PROCESSING_ROOT / "multi_split_manifest.json") as _f:
    _MANIFEST = json.load(_f)
SPLIT_ROW_COUNTS = {
    (Path(s["split_dir"]).name if isinstance(s, dict) and "split_dir" in s else str(s)):
        {"n_train": s.get("n_train_B"), "n_test": s.get("n_test_B")}
    for s in _MANIFEST.get("splits", [])
}


def _seed_for_split(split_idx: int) -> int:
    """Matches fermi_processing.py's own split_seed = SEED + i convention exactly."""
    return SEED + split_idx


def _split_idx_of(split_id: str) -> int:
    return VALIDATION_SPLITS.index(split_id)


# %% [markdown]
# ## Section 1 — Re-ported merged-specific chain-CV helpers (`_m2_*`)
#
# Exact ports of `fermi_merged.py`'s own `build_branch_datasets`,
# `_fit_and_route_chain_fold`, `_predict_proba_safe`,
# `_resolve_full_length_tag_frame` — needed because both
# `fermi_merged.py` and `fermi_3levels.py` define these bare top-level
# names in the shared kernel. 3-level runs last in the shared kernel
# and its versions win the bare names — this file uses the live bare
# names directly for the 3-level family and these `_m2_`-prefixed local
# copies for merged.

# %% [markdown]
# ### Section 1a — Node-fit cache
#
# `fermi_merged.py`'s own top-15 grid is `TOP_N_ROUTERS(3) x
# TOP_N_BRANCH_COMBOS(5)` candidates built from only 3 distinct router
# configs and 5 distinct branch-combo configs; `fermi_3levels.py`'s is
# `TOP_N_GATES(3) x TOP_N_BLAZAR_SUBGATES(3) x TOP_N_NONBLAZAR_ROUTERS(3) =
# 27` candidates from only 3+3+3=9 distinct node configs, sharing ONE fixed
# branch-config set across all 27. Scoring every candidate independently
# would refit every shared node from scratch once per candidate
# that happens to reuse it — for a fixed `(ctx, seed)`, the CV fold
# boundaries are identical across every candidate (they depend only on
# `y_super_str`/`n_splits`/`seed`, never on a candidate's own cfg), so a
# node's per-fold fit is byte-for-byte reusable across every candidate that
# specifies the identical `(model_name, tag, pca_variance, scaler_name,
# weighting)` at that node. `node_cache` (a plain dict, scoped to one
# `(ctx, seed)` "sweep" and shared across every candidate scored within
# it) plus `_cfg_key`/`_cached_fit` below implement that reuse — same
# fits, same numbers, no approximation, just no redundant computation.
# Stage 1/3's driver loops (Sections 4/6) create one `node_cache` per
# `(ctx, seed)` and pass it into every candidate call for that sweep;
# Stage 5/7/9 (single winner, nothing to share) simply don't pass one, so
# each call gets its own private, empty cache — behaviorally identical to
# uncached scoring for every single-candidate caller.

# %%
def _cfg_key(cfg: Optional[dict]) -> tuple:
    """Hashable identity for a {"model_name","tag","pca_variance",
    "scaler_name","weighting"} cfg dict — two textually-identical cfgs
    (regardless of dict object identity) collide on this key, which is
    exactly the "same architecture+features+weighting, so the same fit"
    equivalence node-fit caching needs."""
    if cfg is None:
        return None
    return (cfg.get("model_name"), cfg.get("tag"), cfg.get("pca_variance"),
            cfg.get("scaler_name"), cfg.get("weighting"))


def _cached_fit(node_cache: Optional[dict], key, fit_fn):
    """Returns `node_cache[key]` if present, else calls `fit_fn()`, stores,
    and returns it. `node_cache=None` degrades to "always fit" (no cache),
    which is the correct behavior for any single-candidate caller that
    doesn't opt in."""
    if node_cache is None:
        return fit_fn()
    if key in node_cache:
        return node_cache[key]
    val = fit_fn()
    node_cache[key] = val
    return val


# %%
def _m2_predict_proba_safe(model, X, n_classes):
    """Coerces a pandas- or cuDF-flavored `predict_proba` return value to a
    plain ndarray before any numpy-style column indexing — mirrors the fix
    `fermi_3levels.py`'s own (different, unreachable-from-here) copy of
    this function applies for the same reason (GPU/cuML builders can return
    a DataFrame-like object)."""
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if hasattr(proba, "to_numpy"):
                proba = proba.to_numpy()
            elif hasattr(proba, "values") and not isinstance(proba, np.ndarray):
                proba = proba.values
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] == n_classes:
                return proba
            # e.g. a fit that never saw a rare class returns fewer columns: indexing them
            # by global class position would silently misalign -> hard fallback instead.
            print(f"  ⚠ predict_proba returned shape {proba.shape}, expected {n_classes} columns "
                  f"({type(model).__name__}) — using one-hot from predict() instead.")
        except Exception:
            pass
    pred_idx = np.asarray(model.predict(X))
    onehot = np.zeros((len(pred_idx), n_classes))
    onehot[np.arange(len(pred_idx)), pred_idx] = 1.0
    return onehot


def _m2_resolve_full_length_tag_frame(ctx, tag: str, local_meta: Optional[dict]):
    if tag in ctx.x_train_by_tag:
        return ctx.x_train_by_tag[tag]
    if local_meta and tag in local_meta.get("feature_sets", {}):
        return ctx.x_train_by_tag["ALL_FEATURES"][local_meta["feature_sets"][tag]]
    raise KeyError(f"[merged] Tag '{tag}' is neither a global/router-local tag nor a known branch-local tag.")


def _m2_build_branch_datasets(ctx, route_str_train: np.ndarray = None) -> dict:
    route_str_train = ctx.y_train_super_str if route_str_train is None else route_str_train
    branches = {}
    for sc in SUPERCLASS_ORDER:
        tr_mask = route_str_train == sc
        fine_tr = ctx.y_train_fine_str[tr_mask]
        if len(set(fine_tr)) < 2:
            branches[sc] = None
            continue
        from sklearn.preprocessing import LabelEncoder
        le_branch = LabelEncoder().fit(fine_tr)
        y_tr_enc = le_branch.transform(fine_tr)

        x_train_by_tag = {tag: ctx.x_train_by_tag[tag][tr_mask].reset_index(drop=True) for tag in ctx.tags}
        branch_tags = list(ctx.tags)
        local_meta = ctx.branch_meta.get(sc)
        x_tr_all = None
        if local_meta is not None:
            x_tr_all = ctx.x_train_by_tag["ALL_FEATURES"][tr_mask].reset_index(drop=True)
            for local_tag, feats in local_meta["feature_sets"].items():
                x_train_by_tag[local_tag] = x_tr_all[feats].reset_index(drop=True)
                branch_tags.append(local_tag)
        # `ctx.tags` excludes "ALL_FEATURES" by construction (build_merged_context's
        # own ctx_tags filter), so a branch with no local feature selection (local_meta is None)
        # never gets an "ALL_FEATURES"/PCA_BASE_TAG entry in its own x_train_by_tag at all — this
        # is the ONE thing this re-ported local copy was missing relative to the authoritative
        # fermi_merged.py::build_branch_datasets, and it's what let a PCA-over-ALL_FEATURES
        # branch candidate crash with a bare `KeyError: 'ALL_FEATURES'` inside `_fit_predict_
        # merged_core`'s `branch["x_train_by_tag"][cfg["tag"]]` lookup (Section 8) whenever that
        # candidate's branch model used tag=PCA_BASE_TAG="ALL_FEATURES".
        if PCA_BASE_TAG not in x_train_by_tag:
            if x_tr_all is None:
                x_tr_all = ctx.x_train_by_tag["ALL_FEATURES"][tr_mask].reset_index(drop=True)
            x_train_by_tag[PCA_BASE_TAG] = x_tr_all

        branches[sc] = {
            "x_train_by_tag": x_train_by_tag, "tags": branch_tags,
            "y_train_enc": y_tr_enc, "class_names": le_branch.classes_,
            "label_encoder": le_branch, "n_train": int(tr_mask.sum()), "local_meta": local_meta,
        }
    return branches


def _m2_fit_and_route_chain_fold(ctx, tr_idx, eval_idx, router_cfg, branch_cfgs: dict,
                                  router_meta, branch_datasets: dict, seed=SEED,
                                  node_cache: Optional[dict] = None, fold_idx: int = 0):
    """`node_cache`/`fold_idx` added — see Section 1a. Both default
    to "no sharing" (`node_cache=None`), so a call site that doesn't pass
    them gets plain uncached behavior."""
    from sklearn.preprocessing import LabelEncoder
    y_super_full, y_fine_full = ctx.y_train_super_str, ctx.y_train_fine_str
    y_super_tr = y_super_full[tr_idx]

    x_router_full = _m2_resolve_full_length_tag_frame(ctx, router_cfg["tag"], router_meta)
    router_key = ("router", _cfg_key(router_cfg), fold_idx)
    router_fold = _cached_fit(node_cache, router_key, lambda: fit_component(
        router_cfg, MODEL_SPECS, get_builder, x_router_full.iloc[tr_idx],
        SUPER_LE.transform(y_super_tr), seed))

    branch_folds = {}
    for sc in SUPERCLASS_ORDER:
        cfg = branch_cfgs.get(sc)
        if cfg is None:
            branch_folds[sc] = None
            continue
        sc_mask_tr = y_super_tr == sc
        fine_tr_sc = y_fine_full[tr_idx][sc_mask_tr]
        if len(set(fine_tr_sc)) < 2:
            branch_folds[sc] = None
            continue
        branch_key = ("branch", sc, _cfg_key(cfg), fold_idx)

        def _fit_branch(cfg=cfg, sc=sc, fine_tr_sc=fine_tr_sc, sc_mask_tr=sc_mask_tr):
            le_branch = LabelEncoder().fit(fine_tr_sc)
            branch_local_meta = branch_datasets.get(sc, {}).get("local_meta") if branch_datasets.get(sc) else None
            x_branch_full = _m2_resolve_full_length_tag_frame(ctx, cfg["tag"], branch_local_meta)
            model = fit_component(cfg, MODEL_SPECS, get_builder, x_branch_full.iloc[tr_idx][sc_mask_tr],
                                   le_branch.transform(fine_tr_sc), seed)
            return (model, le_branch)

        try:
            branch_folds[sc] = _cached_fit(node_cache, branch_key, _fit_branch)
        except KeyError:
            branch_folds[sc] = None

    x_router_eval = x_router_full.iloc[eval_idx]
    all_superclasses = list(SUPER_LE.classes_)
    super_idx = {s: i for i, s in enumerate(all_superclasses)}
    p_super = _m2_predict_proba_safe(router_fold, x_router_eval, len(all_superclasses))
    pred_super_eval = np.array(all_superclasses)[p_super.argmax(axis=1)]

    all_fine_classes = list(ctx.fine_class_names)
    fine_idx = {c: i for i, c in enumerate(all_fine_classes)}
    p_fine = np.zeros((len(eval_idx), len(all_fine_classes)))

    for sc in SUPERCLASS_ORDER:
        p_sc = p_super[:, super_idx[sc]]
        bf = branch_folds.get(sc)
        if bf is None:
            fine_tr_sc = y_fine_full[tr_idx][y_super_tr == sc]
            if len(fine_tr_sc):
                fallback = pd.Series(fine_tr_sc).mode().iloc[0]
                p_fine[:, fine_idx[fallback]] += p_sc
            continue
        model, le_branch = bf
        branch_local_meta = branch_datasets.get(sc, {}).get("local_meta") if branch_datasets.get(sc) else None
        x_branch_full = _m2_resolve_full_length_tag_frame(ctx, branch_cfgs[sc]["tag"], branch_local_meta)
        x_branch_eval = x_branch_full.iloc[eval_idx]
        p_branch = _m2_predict_proba_safe(model, x_branch_eval, len(le_branch.classes_))
        for k, fc in enumerate(le_branch.classes_):
            p_fine[:, fine_idx[fc]] += p_sc * p_branch[:, k]

    pred_fine_eval = np.array(all_fine_classes)[p_fine.argmax(axis=1)]
    return pred_fine_eval, pred_super_eval


# %% [markdown]
# ### Section 1b — Local cached 3-level chain-fold fit (`_l3_*`)
#
# There is no analogous collision problem for 3-level (its bare
# `_fit_and_route_chain_fold` is exactly the version that wins in the
# shared kernel), but that bare function has no `node_cache` parameter and
# validation.py cannot add one to a sibling module's own top-level
# function — so, exactly like the `_m2_*` merged re-port above, this file
# carries a LOCAL cached copy of 3-level's per-fold chain-fit logic
# (verified against `fermi_3levels.py`'s real `_fit_and_route_chain_
# fold`/`build_gate_labels`, minus that module's own optional
# `hard_routing_log` parameter, which this
# local copy does not need since `_m2_predict_proba_safe` already coerces
# every backend's `predict_proba` output uniformly).

# %%
def _l3_fit_and_route_chain_fold_cached(ctx, tr_idx, eval_idx, fold_idx, gate_cfg, subgate_cfg, nonblazar_cfg,
                                         branch_cfgs: dict, gate_meta, subgate_meta, nonblazar_meta,
                                         branch_datasets: dict, node_cache: Optional[dict], seed=SEED):
    from sklearn.preprocessing import LabelEncoder
    y_super_full, y_fine_full = ctx.y_train_super_str, ctx.y_train_fine_str
    y_super_tr = y_super_full[tr_idx]
    tr_mask_blazar = np.isin(y_super_tr, list(BLAZAR_SUPERCLASSES))
    tr_mask_nonblazar = ~tr_mask_blazar

    x_gate_full = _m2_resolve_full_length_tag_frame(ctx, gate_cfg["tag"], gate_meta)
    gate_key = ("gate", _cfg_key(gate_cfg), fold_idx)
    gate_model = _cached_fit(node_cache, gate_key, lambda: fit_component(
        gate_cfg, MODEL_SPECS, get_builder, x_gate_full.iloc[tr_idx],
        GATE_LE.transform(build_gate_labels(y_super_tr)), seed))

    x_sub_full = _m2_resolve_full_length_tag_frame(ctx, subgate_cfg["tag"], subgate_meta)
    subgate_key = ("blazar_subgate", _cfg_key(subgate_cfg), fold_idx)
    subgate_model = _cached_fit(node_cache, subgate_key, lambda: fit_component(
        subgate_cfg, MODEL_SPECS, get_builder,
        x_sub_full.iloc[tr_idx][tr_mask_blazar].reset_index(drop=True),
        BLAZAR_SUBGATE_LE.transform(y_super_tr[tr_mask_blazar]), seed))

    x_nb_full = _m2_resolve_full_length_tag_frame(ctx, nonblazar_cfg["tag"], nonblazar_meta)
    nb_key = ("nonblazar_router", _cfg_key(nonblazar_cfg), fold_idx)
    nonblazar_model = _cached_fit(node_cache, nb_key, lambda: fit_component(
        nonblazar_cfg, MODEL_SPECS, get_builder,
        x_nb_full.iloc[tr_idx][tr_mask_nonblazar].reset_index(drop=True),
        NONBLAZAR_ROUTER_LE.transform(y_super_tr[tr_mask_nonblazar]), seed))

    branch_folds = {}
    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        cfg = branch_cfgs.get(sc)
        if cfg is None:
            branch_folds[sc] = None
            continue
        sc_mask_tr = y_super_tr == sc
        fine_tr_sc = y_fine_full[tr_idx][sc_mask_tr]
        if len(set(fine_tr_sc)) < 2:
            branch_folds[sc] = None
            continue
        branch_key = ("branch", sc, _cfg_key(cfg), fold_idx)

        def _fit_branch(cfg=cfg, sc=sc, fine_tr_sc=fine_tr_sc, sc_mask_tr=sc_mask_tr):
            le_branch = LabelEncoder().fit(fine_tr_sc)
            branch_local_meta = branch_datasets.get(sc, {}).get("local_meta") if branch_datasets.get(sc) else None
            x_branch_full = _m2_resolve_full_length_tag_frame(ctx, cfg["tag"], branch_local_meta)
            model = fit_component(cfg, MODEL_SPECS, get_builder, x_branch_full.iloc[tr_idx][sc_mask_tr],
                                   le_branch.transform(fine_tr_sc), seed)
            return (model, le_branch)

        try:
            branch_folds[sc] = _cached_fit(node_cache, branch_key, _fit_branch)
        except KeyError:
            branch_folds[sc] = None

    x_gate_eval = x_gate_full.iloc[eval_idx]
    p_gate = _m2_predict_proba_safe(gate_model, x_gate_eval, len(GATE_LE.classes_))
    p_blazar = p_gate[:, list(GATE_LE.classes_).index("BLAZAR")]
    p_other = p_gate[:, list(GATE_LE.classes_).index("OTHER")]

    x_sub_eval = x_sub_full.iloc[eval_idx]
    p_subgate = _m2_predict_proba_safe(subgate_model, x_sub_eval, len(BLAZAR_SUBGATE_LE.classes_))
    x_nb_eval = x_nb_full.iloc[eval_idx]
    p_router = _m2_predict_proba_safe(nonblazar_model, x_nb_eval, len(NONBLAZAR_ROUTER_LE.classes_))

    all_superclasses = list(SUPER_LE.classes_)
    super_idx = {s: i for i, s in enumerate(all_superclasses)}
    n_eval = len(eval_idx)
    p_super = np.zeros((n_eval, len(all_superclasses)))
    for j, sc in enumerate(BLAZAR_SUBGATE_LE.classes_):
        p_super[:, super_idx[sc]] = p_blazar * p_subgate[:, j]
    for j, sc in enumerate(NONBLAZAR_ROUTER_LE.classes_):
        p_super[:, super_idx[sc]] = p_other * p_router[:, j]

    all_fine_classes = list(ctx.fine_class_names)
    fine_idx = {c: i for i, c in enumerate(all_fine_classes)}
    p_fine = np.zeros((n_eval, len(all_fine_classes)))
    for sc in BLAZAR_SUPERCLASSES:
        fines = sorted(c for c, s in TAXONOMY.items() if s == sc and c in ctx.fine_class_names)
        p_fine[:, fine_idx[fines[0]]] += p_super[:, super_idx[sc]]
    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        p_sc = p_super[:, super_idx[sc]]
        bf = branch_folds.get(sc)
        if bf is None:
            fine_tr_sc = y_fine_full[tr_idx][y_super_tr == sc]
            if len(fine_tr_sc):
                fallback = pd.Series(fine_tr_sc).mode().iloc[0]
                p_fine[:, fine_idx[fallback]] += p_sc
            continue
        model, le_branch = bf
        branch_local_meta = branch_datasets.get(sc, {}).get("local_meta") if branch_datasets.get(sc) else None
        try:
            x_branch_full = _m2_resolve_full_length_tag_frame(ctx, branch_cfgs[sc]["tag"], branch_local_meta)
        except KeyError:
            fine_tr_sc = y_fine_full[tr_idx][y_super_tr == sc]
            fallback = pd.Series(fine_tr_sc).mode().iloc[0] if len(fine_tr_sc) else all_fine_classes[0]
            p_fine[:, fine_idx[fallback]] += p_sc
            continue
        x_branch_eval = x_branch_full.iloc[eval_idx]
        p_branch = _m2_predict_proba_safe(model, x_branch_eval, len(le_branch.classes_))
        for k, fc in enumerate(le_branch.classes_):
            p_fine[:, fine_idx[fc]] += p_sc * p_branch[:, k]

    pred_fine_eval = np.array(all_fine_classes)[p_fine.argmax(axis=1)]
    pred_super_eval = np.array(all_superclasses)[p_super.argmax(axis=1)]
    return pred_fine_eval, pred_super_eval


# %% [markdown]
# ## Section 2 — Metric primitives, Borda ranking, end-to-end naming
#
# Stage 1/3 call the `_val_score_*` functions many times per candidate (15
# repeats x up to 27 candidates x up to 5 splits) — these wrappers compute
# only the pooled-OOF `SELECTION_METRIC` score plus the full metric quartet,
# nothing else (no plotting/logging).
#
# `QUALITY_METRICS` includes plain `"accuracy"`
# as a 4th, equally-weighted Borda ranking criterion, alongside
# `bal_accuracy`/`macro_f1`/`kappa` — a deliberate, disclosed choice
# (accuracy is often left diagnostic-only under class imbalance). This constant feeds `_borda_rank`
# everywhere ranking happens in this file (Stage 1/3/4's candidate-level
# selection, Stage 8's family-level selection).
#
# `REPORT_METRICS` is a strict superset of
# `QUALITY_METRICS` — it adds `precision_macro`/`recall_macro` — used only
# by the CSV-export stage (Section 12) and the extended train/val/test
# report (Section 14). Nothing that ranks anything reads `REPORT_METRICS`;
# it exists purely so the requested "acc, bal_acc, macro_f1, kappa,
# precision, recall etc." CSV columns don't require re-deriving predictions
# a second time.

# %%
QUALITY_METRICS = ("accuracy", "bal_accuracy", "macro_f1", "kappa") # accuracy added — see above
REPORT_METRICS = QUALITY_METRICS + ("precision_macro", "recall_macro")


def _metric_value(metric_name: str, y_true, y_pred) -> float:
    """Single dispatch point for every per-metric computation in this file
    — `_score_by_name` (common) covers accuracy/bal_accuracy/macro_f1;
    kappa/precision_macro/recall_macro aren't in its supported set, so
    they're computed directly here via sklearn."""
    if metric_name == "kappa":
        return float(cohen_kappa_score(y_true, y_pred))
    if metric_name == "precision_macro":
        return float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    if metric_name == "recall_macro":
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return float(_score_by_name(metric_name, y_true, y_pred))


def _full_metric_dict(y_true, y_pred, metrics: Sequence[str] = QUALITY_METRICS) -> dict:
    """One set of hard predictions -> `{metric: value}` for every metric in
    `metrics` (default: the 4 Borda `QUALITY_METRICS`; pass `REPORT_METRICS`
    for the 6-column CSV/report variant — see `_full_report_metric_dict`)."""
    return {m: _metric_value(m, y_true, y_pred) for m in metrics}


def _full_report_metric_dict(y_true, y_pred) -> dict:
    """The 6 `REPORT_METRICS` from one set of hard predictions —
    what Section 12/14's CSVs actually want, without touching any Borda
    ranking code (which still reads only `QUALITY_METRICS`)."""
    return _full_metric_dict(y_true, y_pred, metrics=REPORT_METRICS)


def _mean_metric_dicts(dicts: list, metrics: Sequence[str] = REPORT_METRICS) -> dict:
    """Component-wise mean of a list of per-metric dicts (e.g. one per
    validation split), skipping non-finite entries per-metric.
    Defaults to `REPORT_METRICS` (superset of `QUALITY_METRICS`) rather
    than `QUALITY_METRICS` alone — every call site in this file now
    aggregates dicts built via `_full_report_metric_dict`, so this default
    carries precision_macro/recall_macro through every aggregation for
    free; any dict missing those 2 keys (e.g. fine's val-side, which
    genuinely cannot compute them — see `_val_score_fine_candidate`) simply
    contributes no values for them and the result is nan, never a crash.
    Every Borda/overfit-gap reader below still looks up only the specific
    `QUALITY_METRICS` keys it needs, so this widening is purely additive."""
    out = {}
    for m in metrics:
        vals = [d[m] for d in dicts if d is not None and m in d and np.isfinite(d.get(m, float("nan")))]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    return out


def _std_metric_dicts(dicts: list, metrics: Sequence[str] = REPORT_METRICS, ddof: int = 1) -> dict:
    """Component-wise sample std
    (ddof=1) of a list of per-metric dicts. Returns nan for a metric with
    fewer than 2 finite observations (a std computed from 1 point is not
    an error bar, it's a fabrication) rather than silently returning 0.
    Defaults to `REPORT_METRICS` — see `_mean_metric_dicts` above."""
    out = {}
    for m in metrics:
        vals = [d[m] for d in dicts if d is not None and m in d and np.isfinite(d.get(m, float("nan")))]
        out[m] = float(np.std(vals, ddof=ddof)) if len(vals) >= 2 else float("nan")
    return out


def _metric_ci(y_true, y_pred) -> dict:
    """Uncertainty of every REPORT_METRICS column for ONE evaluated set of predictions (a split's
    test rows, a split's pooled OOF validation rows, or a training refit): stratified bootstrap,
    with the Brodersen et al. (2010) posterior for balanced accuracy — see `bootstrap_metric_ci`."""
    return bootstrap_metric_ci(y_true, y_pred, lambda t, p, w: _full_report_metric_dict(t, p))


# %% [markdown]
# ### Section 2a — Confusion-matrix helpers
#
# **Deliberately built on STRING labels, never encoded ints, once more than
# one split/repeat is being pooled.** Each validation split builds its OWN
# `LabelEncoder` (`_val_build_contexts` -> each sibling module's own context
# builder), and `sklearn.preprocessing.LabelEncoder` assigns integer codes
# by sorting whatever class SET was actually present in the data it was
# fit on — if one split's OOF/train pool happens to be missing a rare fine
# class another split has, that split's codes shift relative to every other
# split's. Concatenating raw encoded arrays from different splits and
# decoding the pooled result with a single (e.g. split-0) encoder would
# silently mislabel rows wherever the codes disagree. Every OOF array this
# file collects for a confusion matrix is therefore a STRING class-name
# array (each candidate-scoring function's own `ctx.label_encoder`/
# `ctx.fine_label_encoder` — the same one used to build that array in the
# first place — is what does the decoding, right where the array is
# produced, never re-applied later to a different split's codes); pooling
# is then plain string concatenation, immune to any cross-split code
# mismatch by construction.
#
# `_compute_confusion_matrix` turns a pair of STRING label arrays plus a
# fixed `labels` ordering (normally `list(some_context.label_encoder.
# classes_)` — any one split's encoder works equally well here since it
# only fixes the matrix's row/column ORDER, not how any array gets
# decoded) into a plain, JSON/CSV-serializable dict (`{"labels","matrix",
# "n"}`), always shaped over the FULL `labels` set so a class with zero
# rows in this particular pooled sample still gets an all-zero row/column
# instead of silently shrinking the matrix. `_save_confusion_matrix`
# persists that dict as a CSV (always — the actual "saved" deliverable
# requirements #3/#4 ask for) and, best-effort, a PNG heatmap (matplotlib
# is imported lazily here, not at module load time, so a plotting failure
# or a missing matplotlib never takes down a stage that otherwise
# succeeded — only the PNG is skipped, with a warning).

# %%
def _compute_confusion_matrix(y_true_str: np.ndarray, y_pred_str: np.ndarray, labels: Sequence[str]) -> Optional[dict]:
    y_true_str, y_pred_str = np.asarray(y_true_str), np.asarray(y_pred_str)
    if len(y_true_str) == 0 or len(y_pred_str) == 0:
        return None
    labels = list(labels)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)
    return {"labels": [str(c) for c in labels], "matrix": cm.tolist(), "n": int(len(y_true_str))}


def _pooled_confusion_matrix(y_true_list: list, y_pred_list: list, labels: Sequence[str]) -> Optional[dict]:
    """Concatenates a list of per-repeat/per-split STRING (y_true, y_pred)
    label arrays (e.g. one pair per Stage-1 repeat or per Stage-3
    validation split, each already decoded by ITS OWN split's encoder
    before being handed here) into one pooled confusion matrix — the
    natural aggregate for "the" confusion matrix of a candidate scored
    more than once, since raw counts sum validly across independent
    repeats/splits even though the underlying fold boundaries (and each
    split's own integer encoding, irrelevant here) differ."""
    y_true_list = [np.asarray(a) for a in (y_true_list or []) if a is not None and len(a)]
    y_pred_list = [np.asarray(a) for a in (y_pred_list or []) if a is not None and len(a)]
    if not y_true_list or not y_pred_list:
        return None
    return _compute_confusion_matrix(np.concatenate(y_true_list), np.concatenate(y_pred_list), labels)


def _save_confusion_matrix(cm_dict: Optional[dict], csv_dir: Path, plot_dir: Path, stem: str,
                            title: str, logger=None) -> Optional[str]:
    """Writes `{csv_dir}/{stem}.csv` (raw counts, rows=true, cols=predicted,
    over the full label set), `{csv_dir}/{stem}_row_normalized.csv` (each row
    divided by its true-class total, i.e. per-class recall — undefined/NaN
    rows for a true class with zero support in this pooled sample), and,
    best-effort, `{plot_dir}/{stem}.png`.

    The PNG is colored by the ROW-NORMALIZED matrix (0-1, per-true-class
    recall), not raw counts: under this catalog's severe class imbalance
    (bll/fsrq outnumber rare classes like nov/sfr/hmb by 2-3 orders of
    magnitude), a linear raw-count color scale is dominated entirely by the
    bll/fsrq cells — every rare-class row reads as blank/white regardless of
    how well that class is actually classified, and the bll/fsrq block's own
    off-diagonal cells can look deceptively as dark as the diagonal even when
    the actual confusion rate between them is small, simply because both are
    large numbers on the same saturated scale. This mirrors the project's own
    validation-protocol principle of preferring balance-aware metrics over
    raw-count/accuracy readings. Raw counts are still shown as the cell text
    annotations and remain fully available in the CSV. Returns the raw-count
    CSV path (the CSV is the one deliverable this function guarantees — every
    caller treats a `None` return as "confusion matrix unavailable for this
    candidate/split combination", e.g. zero valid OOF rows) or `None` if
    `cm_dict` itself is `None`."""
    if cm_dict is None:
        return None
    csv_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(cm_dict["matrix"], index=cm_dict["labels"], columns=cm_dict["labels"])
    df.index.name = "true\\pred"
    csv_path = csv_dir / f"{stem}.csv"
    df.to_csv(csv_path)

    row_sums = df.values.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_arr = np.where(row_sums > 0, df.values / np.where(row_sums == 0, 1.0, row_sums), np.nan)
    norm_df = pd.DataFrame(norm_arr, index=cm_dict["labels"], columns=cm_dict["labels"])
    norm_df.index.name = "true\\pred (row-normalized = per-class recall)"
    norm_df.to_csv(csv_dir / f"{stem}_row_normalized.csv")

    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_lab = len(cm_dict["labels"])
        fig_side = max(6.0, 0.45 * n_lab + 3.0)
        fig, ax = plt.subplots(figsize=(fig_side, fig_side))
        plot_arr = np.nan_to_num(norm_arr, nan=0.0)
        im = ax.imshow(plot_arr, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(n_lab)); ax.set_xticklabels(cm_dict["labels"], rotation=90, fontsize=7)
        ax.set_yticks(range(n_lab)); ax.set_yticklabels(cm_dict["labels"], fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{title}\n(n={cm_dict['n']}, color = row-normalized recall, "
                      "text = raw count)", fontsize=9)
        # Raw-count text annotation per cell, readable against both light and
        # dark cells (row-normalized value, not the raw count, sets the text
        # color threshold).
        raw_arr = df.values
        for i in range(n_lab):
            for j in range(n_lab):
                v = raw_arr[i, j]
                if v == 0:
                    continue
                txt_color = "white" if plot_arr[i, j] > 0.5 else "black"
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6, color=txt_color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row-normalized (recall)", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{stem}.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        if logger is not None:
            logger.warning(f"  ⚠ confusion-matrix PNG for '{stem}' failed ({type(e).__name__}: {e}) "
                            "— the CSV was still saved.")
    return str(csv_path)


def _regenerate_confusion_matrix_artifacts(cm_dict: Optional[dict], csv_path_str: Optional[str],
                                            title: str, logger=None) -> None:
    """Rewrites one confusion matrix's CSV/row-normalized-CSV/
    PNG purely from an already-computed `cm_dict` plus the CSV path this file
    itself saved it under on an earlier run — no CV, no refit, no GPU work.
    Called from `_summarize_stage1`/`_summarize_stage3`/`_summarize_stage5`
    (the checkpoint-HIT path — see `_run_cached`'s docstring) so that a
    plotting-only change to `_save_confusion_matrix` (e.g. the
    row-normalized coloring) takes effect on every existing candidate's
    and winner's PNG the very next run, without needing to bump
    `CHECKPOINT_SCHEMA_VERSION` or force any stage back into recomputing its
    actual scores. `csv_dir`/`plot_dir` are reconstructed from the stored CSV
    path rather than re-deriving the candidate's index (Borda re-sorts the
    candidate list in place, so a freshly-enumerated index would not
    necessarily match the index baked into the original filename)."""
    if cm_dict is None or not csv_path_str:
        return
    csv_path = Path(csv_path_str)
    csv_dir = csv_path.parent
    try:
        rel = csv_dir.relative_to(VALIDATION_CONFUSION_DIR)
    except ValueError:
        rel = Path(".")
    plot_dir = VALIDATION_CONFUSION_PLOTS_DIR / rel
    _save_confusion_matrix(cm_dict, csv_dir, plot_dir, csv_path.stem, title, logger=logger)


def _candidate_cm_stem(fam: str, idx: int, cand: dict) -> str:
    """A filesystem-safe, stable-within-a-run filename stem for one Stage-1/
    Stage-3 candidate's confusion-matrix files — the running index (assigned
    in each family's pre-Borda-rank discovery order) guarantees uniqueness
    even if two candidates would otherwise contract to the same name."""
    if fam == "fine":
        c = cand["cfg"]
        ident = f"{c.get('model_name')}_{c.get('tag')}_pca{c.get('pca_variance')}_{c.get('scaler_name')}_{c.get('weighting')}"
    else:
        ident = cand.get("combo_name") or cand.get("full_model_name") or "candidate"
    return f"{idx:03d}_{_safe_filename(ident)}"


def _overfit_gap(row: dict) -> float:
    """Mean, over `QUALITY_METRICS`, of max(0, train_metric - val_metric)
    for one candidate/family `row` (a dict carrying `train_metrics`/
    `val_metrics` sub-dicts). Clipped at 0 so a model that generalizes
    BETTER than its own training score is never rewarded for it. Smaller is
    better; this is the 4th (of 5) Borda criterion."""
    tm, vm = row.get("train_metrics") or {}, row.get("val_metrics") or {}
    gaps = [max(0.0, tm[m] - vm[m]) for m in QUALITY_METRICS
            if np.isfinite(tm.get(m, float("nan"))) and np.isfinite(vm.get(m, float("nan")))]
    return float(np.mean(gaps)) if gaps else float("nan")


def _borda_criteria_from_metrics() -> list:
    """The 5 equally-weighted ranking criteria shared by every Borda
    ranking in this file: each of `QUALITY_METRICS` (now 4, incl.
    accuracy — see the section markdown) on the validation side (higher
    better), plus the train-val overfitting gap (lower better)."""
    criteria = [(lambda row, m=m: (row.get("val_metrics") or {}).get(m, float("nan")), False)
                for m in QUALITY_METRICS]
    criteria.append((_overfit_gap, True))
    return criteria


def _borda_rank(rows: list, criteria: list) -> list:
    """Ranks `rows` by the average of their fractional ranks (ties share the
    mean rank) across `criteria` — a Borda count (Dwork et al. 2001, WWW,
    "Rank aggregation methods for the Web"): `(key_fn,
    ascending)` pairs. A non-finite key value always ranks worst for its
    criterion. Mutates each row in place (`borda_score`, lower better) and
    returns `rows` re-sorted best-first."""
    n = len(rows)
    if n == 0:
        return rows
    rank_sums = np.zeros(n)
    for key_fn, ascending in criteria:
        vals = np.array([key_fn(r) for r in rows], dtype=float)
        signed = vals if ascending else -vals
        signed = np.where(np.isfinite(vals), signed, np.inf)
        order = np.argsort(signed, kind="mergesort")
        ranks = np.empty(n)
        sorted_signed = signed[order]
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_signed[j + 1] == sorted_signed[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        rank_sums += ranks
    for row, total in zip(rows, rank_sums):
        row["borda_score"] = float(total / len(criteria))
    return sorted(rows, key=lambda r: r["borda_score"])


# %% [markdown]
# ### Section 2b — End-to-end architecture naming
#
# `build_end_to_end_name` gives every candidate/winner a single, human-
# readable, CONTRACTED name (short model-family abbreviations joined by the
# architecture's own routing structure) plus a `node_models` dict recording
# the FULL, uncontracted model name actually used at every node of that
# architecture — the contraction is for display/CSV-column-width, the dict
# is the source of truth for "which model did node X use". Both are attached
# to every candidate row starting at Stage 1 and carried through Stage 3/4/
# 5/7/8/9 unchanged (never recomputed downstream — always read off the row
# that was ranked/selected).

# %%
MODEL_NAME_ABBREV = {
    "LDA": "LDA", "LogReg": "LR", "KNN": "KNN", "SVM": "SVM",
    "DecisionTree": "DT", "RandomForest": "RF", "GBC": "GBC",
    "HistGBC": "HGBC", "CatBoost": "CB",
}
for _mlp_name in MLP_ARCHITECTURES:
    # e.g. "MLP_1x64" -> "MLP1x64", "MLP_4x512_256_128_64" -> "MLP4x512-256-128-64"
    MODEL_NAME_ABBREV[_mlp_name] = _mlp_name.replace("MLP_", "MLP").replace("_", "-")

SUPERCLASS_ABBREV = {
    "BLAZAR_HE": "BHE", "BLAZAR_LE": "BLE", "AGN_OTHER": "AGN",
    "PSR_ISOLATED": "PSR", "BINARY_SYSTEMS": "BIN", "SNR_PWN": "SNR", "GALACTIC_EXT": "GAL",
}


def contract_model_name(model_name: Optional[str]) -> str:
    if model_name is None:
        return "passthrough"
    return MODEL_NAME_ABBREV.get(model_name, model_name)


def _branches_contracted_str(branch_cfgs: dict, superclass_order: Sequence[str]) -> str:
    parts = []
    for sc in superclass_order:
        cfg = branch_cfgs.get(sc)
        abbrev_sc = SUPERCLASS_ABBREV.get(sc, sc)
        parts.append(f"{abbrev_sc}:{contract_model_name(cfg['model_name'] if cfg else None)}")
    return ",".join(parts)


def build_end_to_end_name(family: str, **cfgs) -> tuple:
    """Returns `(full_model_name, node_models)`.

    `full_model_name` — a compact, contracted, human-readable identity for
    the whole end-to-end architecture:
      - fine:    "<ModelAbbrev>"                                    e.g. "HGBC"
      - merged:  "<RouterAbbrev>->{<sc>:<BranchAbbrev>,...}"         e.g. "RF->{BHE:SVM,BLE:RF,...}"
      - 3level:  "<GateAbbrev>/<SubgateAbbrev>/<RouterAbbrev>->{...}" e.g. "RF/SVM/LR->{PSR:KNN,...}"

    `node_models` — {node_label: FULL (uncontracted) model_name actually
    used at that node}, e.g. {"fine_model": "HistGBC"} or {"router": "RandomForest",
    "branch_BLAZAR_HE": "SVM", ...} — this is the per-node bookkeeping
    requirement (#1a): every model name that fed into the end-to-end
    architecture, in one place, never abbreviated.

    Expected kwargs per family:
      fine:    cfg=<{"model_name",...}>
      merged:  router_cfg=<cfg>, branch_cfgs=<{sc: cfg or None}>
      3level:  gate_cfg=<cfg>, subgate_cfg=<cfg>, nonblazar_cfg=<cfg>, branch_cfgs=<{sc: cfg or None}>
    """
    if family == "fine":
        cfg = cfgs["cfg"]
        return contract_model_name(cfg["model_name"]), {"fine_model": cfg["model_name"]}

    if family == "merged":
        router_cfg, branch_cfgs = cfgs["router_cfg"], cfgs["branch_cfgs"]
        full_name = f"{contract_model_name(router_cfg['model_name'])}->{{{_branches_contracted_str(branch_cfgs, SUPERCLASS_ORDER)}}}"
        node_models = {"router": router_cfg["model_name"]}
        for sc in SUPERCLASS_ORDER:
            bc = branch_cfgs.get(sc)
            node_models[f"branch_{sc}"] = bc["model_name"] if bc else "passthrough"
        return full_name, node_models

    if family == "3level":
        gate_cfg, subgate_cfg = cfgs["gate_cfg"], cfgs["subgate_cfg"]
        nonblazar_cfg, branch_cfgs = cfgs["nonblazar_cfg"], cfgs["branch_cfgs"]
        nb_order = [sc for sc in SUPERCLASS_ORDER if sc not in BLAZAR_SUPERCLASSES]
        full_name = (f"{contract_model_name(gate_cfg['model_name'])}/"
                     f"{contract_model_name(subgate_cfg['model_name'])}/"
                     f"{contract_model_name(nonblazar_cfg['model_name'])}"
                     f"->{{{_branches_contracted_str(branch_cfgs, nb_order)}}}")
        node_models = {"gate": gate_cfg["model_name"], "blazar_subgate": subgate_cfg["model_name"],
                       "nonblazar_router": nonblazar_cfg["model_name"]}
        for sc in nb_order:
            bc = branch_cfgs.get(sc)
            node_models[f"branch_{sc}"] = bc["model_name"] if bc else "passthrough"
        return full_name, node_models

    raise ValueError(f"Unknown family '{family}'")


def _attach_end_to_end_name(family: str, row: dict) -> dict:
    """Mutates+returns `row` in place with `full_model_name`/`node_models`
    attached, dispatching on which cfg keys `row` carries."""
    if family == "fine":
        full_name, node_models = build_end_to_end_name("fine", cfg=row["cfg"])
    elif family == "merged":
        full_name, node_models = build_end_to_end_name("merged", router_cfg=row["router_cfg"],
                                                         branch_cfgs=row["branch_cfgs"])
    else:
        full_name, node_models = build_end_to_end_name("3level", gate_cfg=row["gate_cfg"],
                                                         subgate_cfg=row["subgate_cfg"],
                                                         nonblazar_cfg=row["nonblazar_cfg"],
                                                         branch_cfgs=row["branch_cfgs"])
    row["full_model_name"], row["node_models"] = full_name, node_models
    return row


# %% [markdown]
# ### Section 2c — Per-candidate CV scoring wrappers
#
# `_val_score_fine_candidate`/`_val_score_merged_chain_candidate`/
# `_val_score_3level_chain_candidate` each return
# `{"selection_mean", "n", "val_metrics", "train_metrics"}` —
# `val_metrics`/`train_metrics` are `QUALITY_METRICS` dicts (now including
# accuracy, via the Section 2 constant).

# %%
def _val_score_fine_candidate(ctx, cfg: dict, seed: int, n_splits: int = N_CONFIRMATION_FOLDS,
                               return_oof: bool = False, with_ci: bool = False) -> dict:
    """Every validation metric comes from the SAME pooled out-of-fold predictions that give the
    selection score (`cv_score_estimator(collect_oof=True)`), so one row never mixes two OOF
    definitions. `with_ci` adds
    `val_ci`/`train_ci` (`_metric_ci`) — Stage 3 asks for them, Stage 1 has its repeat std."""
    spec = MODEL_SPECS[cfg["model_name"]]
    support = weight_support_for(cfg["model_name"])
    builder = get_builder(cfg["model_name"])
    build_kwargs = {"pca_variance": cfg.get("pca_variance")}
    if spec.get("scale_sensitive", False):
        scaler_name = cfg.get("scaler_name")
        build_kwargs["scaler_cls"] = SCALER_VARIANTS[scaler_name] if scaler_name else None
    x = ctx.x_train_by_tag[cfg["tag"]]
    y = np.asarray(ctx.y_train_enc)
    result = cv_score_estimator(builder, build_kwargs, x, y, cfg["weighting"], support,
                                 SELECTION_METRIC, n_splits=n_splits, seed=seed, show_progress=False,
                                 collect_oof=True)
    oof_true = np.asarray(result["oof_true"]).reshape(-1)
    oof_pred = np.asarray(result["oof_pred"]).reshape(-1)  # some GPU builders predict shape (n, 1)
    val_metrics = _full_report_metric_dict(oof_true, oof_pred)
    model = fit_component(cfg, MODEL_SPECS, get_builder, x, y, seed)
    y_pred_train = np.asarray(model.predict(x)).reshape(-1)
    train_metrics = _full_report_metric_dict(y, y_pred_train)
    out = {"selection_mean": result["mean"], "n": result["oof_n"],
           "val_metrics": val_metrics, "train_metrics": train_metrics}
    if with_ci:
        out["val_ci"], out["train_ci"] = _metric_ci(oof_true, oof_pred), _metric_ci(y, y_pred_train)
    if return_oof:
        # STRING class names via THIS split's own encoder (Section 2a: pool on strings, never codes).
        out["oof_y_true"] = ctx.label_encoder.inverse_transform(oof_true)
        out["oof_y_pred"] = ctx.label_encoder.inverse_transform(oof_pred)
    return out


def _val_score_merged_chain_candidate(ctx, router_cfg: dict, branch_cfgs: dict, router_meta,
                                       branch_datasets: dict, seed: int, n_splits: int = N_CONFIRMATION_FOLDS,
                                       node_cache: Optional[dict] = None, return_oof: bool = False,
                                       with_ci: bool = False) -> dict:
    """`node_cache`, when passed by the caller (Stage 1/3's driver
    loops — see Section 1a), is shared across every candidate scored at
    this `(ctx, seed)` so router/branch fits common to multiple candidates
    are computed once. `node_cache=None` (the default — Stage 5/7/9's
    single-winner call sites) gives plain uncached scoring."""
    y_super_str, y_fine_str = ctx.y_train_super_str, ctx.y_train_fine_str
    n = len(y_super_str)
    classes, counts = np.unique(y_super_str, return_counts=True)
    rare_mask = np.isin(y_super_str, classes[counts < n_splits])
    strat_idx, rare_idx = np.where(~rare_mask)[0], np.where(rare_mask)[0]
    y_strat = y_super_str[strat_idx]
    strat_min = int(np.unique(y_strat, return_counts=True)[1].min()) if len(strat_idx) else 0
    n_eff = max(2, min(n_splits, strat_min)) if strat_min >= 2 else 2
    skf = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)

    oof_fine_pred = np.empty(n, dtype=object)
    oof_touched = np.zeros(n, dtype=bool)
    for fold_idx, (tr_local, va_local) in enumerate(skf.split(strat_idx, y_strat)):
        tr_idx = np.concatenate([strat_idx[tr_local], rare_idx])
        va_idx = strat_idx[va_local]
        pred_fine_va, _ = _m2_fit_and_route_chain_fold(ctx, tr_idx, va_idx, router_cfg, branch_cfgs,
                                                         router_meta, branch_datasets, seed=seed,
                                                         node_cache=node_cache, fold_idx=fold_idx)
        oof_fine_pred[va_idx], oof_touched[va_idx] = pred_fine_va, True

    y_true = ctx.fine_label_encoder.transform(y_fine_str[oof_touched])
    y_pred = ctx.fine_label_encoder.transform(oof_fine_pred[oof_touched])
    selection_mean = _score_by_name(SELECTION_METRIC, y_true, y_pred)
    val_metrics = _full_report_metric_dict(y_true, y_pred)  # free — same OOF preds as selection_mean

    y_pred_train = _fit_predict_merged_on_train(ctx, router_cfg, branch_cfgs, router_meta,
                                                 branch_datasets, seed=seed, node_cache=node_cache)
    y_true_train = ctx.fine_label_encoder.transform(y_fine_str)
    train_metrics = _full_report_metric_dict(y_true_train, y_pred_train)
    out = {"selection_mean": selection_mean, "n": int(oof_touched.sum()),
           "val_metrics": val_metrics, "train_metrics": train_metrics}
    if with_ci:
        out["val_ci"], out["train_ci"] = _metric_ci(y_true, y_pred), _metric_ci(y_true_train, y_pred_train)
    if return_oof:
        # STRING class names (see Section 2a) — the plain
        # string arrays that were already sitting right here before `.transform()` produced the
        # encoded `y_true`/`y_pred` above, never re-derived.
        out["oof_y_true"], out["oof_y_pred"] = y_fine_str[oof_touched], oof_fine_pred[oof_touched]
    return out


def _val_score_3level_chain_candidate(ctx, gate_cfg: dict, subgate_cfg: dict, nonblazar_cfg: dict,
                                       branch_cfgs: dict, gate_meta, subgate_meta, nonblazar_meta,
                                       branch_datasets: dict, seed: int, n_splits: int = N_CONFIRMATION_FOLDS,
                                       node_cache: Optional[dict] = None, return_oof: bool = False,
                                       with_ci: bool = False) -> dict:
    """`node_cache` — same contract as the merged candidate scorer
    above; routes through the LOCAL cached `_l3_fit_and_route_chain_fold_cached`
    (Section 1b) instead of the sibling module's bare, uncached
    `_fit_and_route_chain_fold`."""
    y_super_str, y_fine_str = ctx.y_train_super_str, ctx.y_train_fine_str
    n = len(y_super_str)
    classes, counts = np.unique(y_super_str, return_counts=True)
    rare_mask = np.isin(y_super_str, classes[counts < n_splits])
    strat_idx, rare_idx = np.where(~rare_mask)[0], np.where(rare_mask)[0]
    y_strat = y_super_str[strat_idx]
    strat_min = int(np.unique(y_strat, return_counts=True)[1].min()) if len(strat_idx) else 0
    n_eff = max(2, min(n_splits, strat_min)) if strat_min >= 2 else 2
    skf = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)

    oof_fine_pred = np.empty(n, dtype=object)
    oof_touched = np.zeros(n, dtype=bool)
    for fold_idx, (tr_local, va_local) in enumerate(skf.split(strat_idx, y_strat)):
        tr_idx = np.concatenate([strat_idx[tr_local], rare_idx])
        va_idx = strat_idx[va_local]
        pred_fine_va, _ = _l3_fit_and_route_chain_fold_cached(
            ctx, tr_idx, va_idx, fold_idx, gate_cfg, subgate_cfg, nonblazar_cfg,
            branch_cfgs, gate_meta, subgate_meta, nonblazar_meta, branch_datasets, node_cache, seed=seed)
        oof_fine_pred[va_idx], oof_touched[va_idx] = pred_fine_va, True

    y_true = ctx.fine_label_encoder.transform(y_fine_str[oof_touched])
    y_pred = ctx.fine_label_encoder.transform(oof_fine_pred[oof_touched])
    selection_mean = _score_by_name(SELECTION_METRIC, y_true, y_pred)
    val_metrics = _full_report_metric_dict(y_true, y_pred)

    y_pred_train = _fit_predict_3level_on_train(ctx, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs,
                                                 gate_meta, subgate_meta, nonblazar_meta,
                                                 branch_datasets, seed=seed, node_cache=node_cache)
    y_true_train = ctx.fine_label_encoder.transform(y_fine_str)
    train_metrics = _full_report_metric_dict(y_true_train, y_pred_train)
    out = {"selection_mean": selection_mean, "n": int(oof_touched.sum()),
           "val_metrics": val_metrics, "train_metrics": train_metrics}
    if with_ci:
        out["val_ci"], out["train_ci"] = _metric_ci(y_true, y_pred), _metric_ci(y_true_train, y_pred_train)
    if return_oof:
        # STRING class names — see the merged candidate
        # scorer above for the identical rationale.
        out["oof_y_true"], out["oof_y_pred"] = y_fine_str[oof_touched], oof_fine_pred[oof_touched]
    return out


def _val_repeated_score(score_fn, n_repeats: int, seed_offset: int = 0) -> dict:
    """Runs `score_fn(seed)` n_repeats times with seeds SEED+seed_offset+r,
    catching per-repeat failures (nan-tolerant mean). Returns the
    `_val_score_*` shape, repeat-averaged, plus `selection_std` and
    `n_successful_repeats`.

    ALSO now returns `val_metrics_std`/
    `train_metrics_std` — the per-metric sample std ACROSS REPEATS, so a
    Stage-1 shortlist row carries a genuine error bar on every metric, not
    just on the scalar `selection_mean`."""
    sel_scores = []
    val_dicts, train_dicts = [], []
    oof_true_list, oof_pred_list = [], []  # collected only when
    # `score_fn` was built with `return_oof=True` — see `run_stage1`'s call sites.
    for r in range(n_repeats):
        try:
            res = score_fn(SEED + seed_offset + r)
            sel = res.get("selection_mean") if res is not None else None
            if sel is None or not np.isfinite(sel):
                continue
            sel_scores.append(sel)
            val_dicts.append(res["val_metrics"])
            train_dicts.append(res["train_metrics"])
            if "oof_y_true" in res and "oof_y_pred" in res:
                oof_true_list.append(res["oof_y_true"])
                oof_pred_list.append(res["oof_y_pred"])
        except Exception as e:
            _module_logger.warning(f"  ⚠ repeated-score attempt {r} failed: {type(e).__name__}: {e}")
    if not sel_scores:
        nan_metrics = {m: float("nan") for m in REPORT_METRICS}
        return {"selection_mean": float("nan"), "selection_std": float("nan"),
                "val_metrics": dict(nan_metrics), "train_metrics": dict(nan_metrics),
                "val_metrics_std": dict(nan_metrics), "train_metrics_std": dict(nan_metrics),
                "n_successful_repeats": 0, "oof_y_true": [], "oof_y_pred": []}
    return {
        "selection_mean": float(np.mean(sel_scores)), "selection_std": float(np.std(sel_scores)),
        "val_metrics": _mean_metric_dicts(val_dicts), "train_metrics": _mean_metric_dicts(train_dicts),
        "val_metrics_std": _std_metric_dicts(val_dicts), "train_metrics_std": _std_metric_dicts(train_dicts),
        "n_successful_repeats": len(sel_scores), "oof_y_true": oof_true_list, "oof_y_pred": oof_pred_list,
    }


def _val_repeated_score_shared_cache(candidates: list, score_fn, n_repeats: int,
                                      seed_offset: int = 0, desc: Optional[str] = None) -> list:
    """Same contract/output shape as
    `_val_repeated_score`, one aggregated dict per candidate in input order,
    but with the repeat-seed loop made the OUTER loop and candidates the
    INNER loop, so every candidate scored at a given seed `r` shares ONE
    `node_cache` dict. This is a correctness-preserving speedup, not an
    approximation: `StratifiedKFold(n_splits=..., shuffle=True,
    random_state=seed)`'s fold boundaries depend only on
    (labels, n_splits, seed) — never on a candidate's model/tag/pca/scaler/
    weighting config — so whenever two candidates in the same Stage 1/3
    sweep happen to share an identical router/branch/gate/subgate/
    nonblazar_router config, the (ctx, seed, fold_idx) fit is byte-for-byte
    the same model either way; caching it and reusing it for the second
    candidate changes nothing about what gets fit or scored, it just avoids
    re-fitting something already fit. Use this in place of calling
    `_val_repeated_score` once per candidate for merged/3level sweeps
    (`score_fn(cand, seed, node_cache) -> dict`, matching the
    `_val_score_merged_chain_candidate`/`_val_score_3level_chain_candidate`
    return shape); the fine family has no shared sub-components to dedup
    and should keep using plain `_val_repeated_score`."""
    per_cand_sel: list = [[] for _ in candidates]
    per_cand_val: list = [[] for _ in candidates]
    per_cand_train: list = [[] for _ in candidates]
    per_cand_oof_true: list = [[] for _ in candidates]
    per_cand_oof_pred: list = [[] for _ in candidates]
    repeat_iter = pbar(range(n_repeats), desc=desc) if desc else range(n_repeats)
    for r in repeat_iter:
        seed = SEED + seed_offset + r
        node_cache: dict = {}
        for i, cand in enumerate(candidates):
            try:
                res = score_fn(cand, seed, node_cache)
                sel = res.get("selection_mean") if res is not None else None
                if sel is None or not np.isfinite(sel):
                    continue
                per_cand_sel[i].append(sel)
                per_cand_val[i].append(res["val_metrics"])
                per_cand_train[i].append(res["train_metrics"])
                if "oof_y_true" in res and "oof_y_pred" in res:
                    per_cand_oof_true[i].append(res["oof_y_true"])
                    per_cand_oof_pred[i].append(res["oof_y_pred"])
            except Exception as e:
                _module_logger.warning(f"  ⚠ shared-cache repeated-score: seed-repeat {r}, "
                                        f"candidate {i} failed: {type(e).__name__}: {e}")
    out = []
    for sel_scores, val_dicts, train_dicts, oof_true, oof_pred in zip(
            per_cand_sel, per_cand_val, per_cand_train, per_cand_oof_true, per_cand_oof_pred):
        if not sel_scores:
            nan_metrics = {m: float("nan") for m in REPORT_METRICS}
            out.append({"selection_mean": float("nan"), "selection_std": float("nan"),
                        "val_metrics": dict(nan_metrics), "train_metrics": dict(nan_metrics),
                        "val_metrics_std": dict(nan_metrics), "train_metrics_std": dict(nan_metrics),
                        "n_successful_repeats": 0, "oof_y_true": [], "oof_y_pred": []})
        else:
            out.append({
                "selection_mean": float(np.mean(sel_scores)), "selection_std": float(np.std(sel_scores)),
                "val_metrics": _mean_metric_dicts(val_dicts), "train_metrics": _mean_metric_dicts(train_dicts),
                "val_metrics_std": _std_metric_dicts(val_dicts), "train_metrics_std": _std_metric_dicts(train_dicts),
                "n_successful_repeats": len(sel_scores), "oof_y_true": oof_true, "oof_y_pred": oof_pred,
            })
    return out


# %% [markdown]
# ## Section 3 — Loading persisted per-split contexts and leaderboards
#
# Every path/key read below matches what each sibling module writes: fine's
# `FINE_SETUP_LABELS`/`leaderboard_{key}_{label_set}.csv` naming, merged's
# `merged_classification_metrics.json`'s `level1_leaderboard` key and its
# per-branch `results/branch_{sc}.json` files, and 3-level's
# `3level_classification_metrics.json`'s `level1_gate_leaderboard`/
# `level2_blazar_subgate_leaderboard`/`level2_nonblazar_router_leaderboard`/
# `level3_branches` keys.

# %%
def _val_load_feature_metadata(split_id: str) -> dict:
    with open(PROCESSING_ROOT / split_id / "diagnostics" / "feature_metadata.json") as f:
        return json.load(f)


def _val_build_contexts(split_id: str):
    """Returns (fine_ctx, merged_ctx, threelevel_ctx) for one split, built
    fresh from Module 1's exports via each sibling module's own bare
    context builder."""
    source_dir = PROCESSING_ROOT / split_id
    feature_metadata = _val_load_feature_metadata(split_id)
    fine_ctx = build_fine_context(source_dir, feature_metadata, FINE_SETUP_KEY_VALIDATION, split_id)
    merged_ctx = build_merged_context(source_dir, feature_metadata, split_id) if "merged" in FAMILIES else None
    threelevel_ctx = (build_3level_context(source_dir, feature_metadata, split_id)
                      if "3level" in FAMILIES else None)
    return fine_ctx, merged_ctx, threelevel_ctx


def _nan_to_none(v):
    """A blank cell in a CSV-loaded leaderboard column comes
    back as `float('nan')` from pandas, not `None` — `pca_variance is not
    None` (fermi_common.py's `_pipeline_steps`) and `if scaler_name
    else None` both pass NaN straight through: `PCA(n_components=nan)`
    raises InvalidParameterError, and `SCALER_VARIANTS[nan]` would
    KeyError. Every {pca_variance, scaler_name} pulled off a leaderboard
    row, OR out of a persisted metrics JSON block, goes through this one
    normalization point — moved above `_branch_cfgs_from_metrics` (was
    defined after it, only reachable from `_leaderboard_row_to_cfg`) so
    both ingestion points can use it."""
    return None if isinstance(v, float) and np.isnan(v) else v


def _branch_cfgs_from_metrics(level_branches: dict, superclass_order: list) -> dict:
    """Reconstructs a fixed {sc: cfg} dict from a persisted metrics JSON's
    level2_branches/level3_branches block. `level3_branches`
    (in `3level_classification_metrics.json`) is written by
    `fermi_3levels.py` through the same DataFrame->records->json.dump
    path as its leaderboards — a branch model with no PCA round-trips as a
    literal JSON `NaN` there exactly like a blank leaderboard CSV cell
    does, not as `null`/`None`. This was the one {pca_variance,
    scaler_name} construction site in this file NOT already routed through
    `_nan_to_none` (only `_leaderboard_row_to_cfg`'s CSV-sourced path was
    guarded) — the same `PCA(n_components=nan)` crash Stage 1/3 already
    hit for merged candidates was equally reachable here for EVERY
    3-level Stage 1 candidate, since `branch_cfgs_fixed` (built from this
    function) is shared across all `TOP_N_GATES * TOP_N_BLAZAR_SUBGATES *
    TOP_N_NONBLAZAR_ROUTERS` of them. `_build_branch_cfg_variants`
    (merged's competing-branch path) never calls this function — it goes
    through `_leaderboard_row_to_cfg` instead — so merged was never
    exposed to this specific gap."""
    cfgs = {}
    for sc in superclass_order:
        info = level_branches.get(sc, {})
        if not info or "note" in info or "model_name" not in info:
            cfgs[sc] = None
            continue
        cfgs[sc] = {"model_name": info["model_name"], "tag": info["tag"],
                     "pca_variance": _nan_to_none(info.get("pca_variance")),
                     "scaler_name": _nan_to_none(info.get("scaler_name")),
                     "weighting": info.get("weighting", "none")}
    return cfgs


def _leaderboard_row_to_cfg(row: dict) -> dict:
    # str(): read back from a leaderboard CSV, a tag like "90" becomes the int 90.
    return {"model_name": row["model_name"], "tag": str(row["tag"]),
            "pca_variance": _nan_to_none(row.get("pca_variance")),
            "scaler_name": _nan_to_none(row.get("scaler_name")), "weighting": row.get("weighting", "none")}


def _build_branch_cfg_variants(branch_leaderboards: dict, top_n_combos: int) -> list:
    """Builds `top_n_combos` independent {sc: cfg} branch-combo variants by
    taking the i-th-ranked candidate for every superclass in variant i
    (falling back to the last available rank if a branch has fewer
    candidates than `top_n_combos`, or None if it has none at all)."""
    variants = []
    for i in range(top_n_combos):
        variant = {}
        for sc, lb in branch_leaderboards.items():
            if lb is None or lb.empty:
                variant[sc] = None
                continue
            idx = min(i, len(lb) - 1)
            row = lb.sort_values(f"val_{SELECTION_METRIC}", ascending=False).iloc[idx].to_dict()
            variant[sc] = _leaderboard_row_to_cfg(row)
        variants.append(variant)
    return variants


# %% [markdown]
# ## Section 4 — Stage 1: coarse shortlist per family [selection, split 0 only]
#
# Every shortlisted candidate is re-scored with `N_REPEATS_SHORTLIST_CV=5`
# independently-reshuffled `N_CONFIRMATION_FOLDS`-fold passes on split 0's
# training data only. Every candidate row now also gets
# `full_model_name`/`node_models` attached at construction time
# (`_attach_end_to_end_name`) — requirement #1a.

# %%
def _stage1_fine_candidates(split0_dir: Path) -> list:
    lb_path = split0_dir / f"leaderboard_{FINE_SETUP_KEY_VALIDATION}_{FINE_SETUP_LABELS[FINE_SETUP_KEY_VALIDATION]}.csv"
    lb = pd.read_csv(lb_path)
    lb = lb[lb[f"val_{SELECTION_METRIC}"].notna()].sort_values(f"val_{SELECTION_METRIC}", ascending=False)
    cands = [{"cfg": _leaderboard_row_to_cfg(row.to_dict()), "naive_val_score": row[f"val_{SELECTION_METRIC}"]}
             for _, row in lb.head(TOP_K_SHORTLIST).iterrows()]
    return [_attach_end_to_end_name("fine", c) for c in cands]


def _stage1_merged_candidates(split0_dir: Path) -> list:
    with open(split0_dir / "merged_classification_metrics.json") as f:
        mm = json.load(f)
    router_lb = pd.DataFrame(mm["level1_leaderboard"])
    router_lb = router_lb[router_lb[f"val_{SELECTION_METRIC}"].notna()]
    top_router_rows = router_lb.sort_values(f"val_{SELECTION_METRIC}", ascending=False).head(TOP_N_ROUTERS).to_dict("records")

    branch_leaderboards = {}
    for sc in SUPERCLASS_ORDER:
        p = split0_dir / "results" / f"{_safe_filename(f'branch_{sc}')}.json"
        if p.exists():
            df = results_to_dataframe(p)
            branch_leaderboards[sc] = (df[df[f"val_{SELECTION_METRIC}"].notna()]
                                        if (not df.empty and f"val_{SELECTION_METRIC}" in df.columns) else pd.DataFrame())
        else:
            branch_leaderboards[sc] = pd.DataFrame()
    branch_cfg_variants = _build_branch_cfg_variants(branch_leaderboards, TOP_N_BRANCH_COMBOS)

    candidates = []
    for rrow in top_router_rows:
        router_cfg = _leaderboard_row_to_cfg(rrow)
        for bi, branch_cfgs_variant in enumerate(branch_cfg_variants):
            cand = {
                "combo_name": f"{router_cfg['model_name']}__branch{bi}",
                "router_cfg": router_cfg, "branch_cfgs": branch_cfgs_variant,
                "naive_val_score": rrow[f"val_{SELECTION_METRIC}"],
            }
            candidates.append(_attach_end_to_end_name("merged", cand))
    return candidates


def _stage1_3level_candidates(split0_dir: Path) -> list:
    with open(split0_dir / "3level_classification_metrics.json") as f:
        mm = json.load(f)
    gate_lb = pd.DataFrame(mm["level1_gate_leaderboard"])
    subgate_lb = pd.DataFrame(mm["level2_blazar_subgate_leaderboard"])
    router_lb = pd.DataFrame(mm["level2_nonblazar_router_leaderboard"])
    gate_top = gate_lb[gate_lb[f"val_{SELECTION_METRIC}"].notna()].sort_values(
        f"val_{SELECTION_METRIC}", ascending=False).head(TOP_N_GATES).to_dict("records")
    subgate_top = subgate_lb[subgate_lb[f"val_{SELECTION_METRIC}"].notna()].sort_values(
        f"val_{SELECTION_METRIC}", ascending=False).head(TOP_N_BLAZAR_SUBGATES).to_dict("records")
    router_top = router_lb[router_lb[f"val_{SELECTION_METRIC}"].notna()].sort_values(
        f"val_{SELECTION_METRIC}", ascending=False).head(TOP_N_NONBLAZAR_ROUTERS).to_dict("records")
    branch_cfgs_fixed = _branch_cfgs_from_metrics(mm["level3_branches"], NONBLAZAR_SUPERCLASS_ORDER)

    candidates = []
    for grow in gate_top:
        for srow in subgate_top:
            for nrow in router_top:
                gate_cfg, subgate_cfg, nonblazar_cfg = (_leaderboard_row_to_cfg(grow),
                                                          _leaderboard_row_to_cfg(srow),
                                                          _leaderboard_row_to_cfg(nrow))
                combo_name = f"{gate_cfg['model_name']}+{subgate_cfg['model_name']}+{nonblazar_cfg['model_name']}"
                cand = {
                    "combo_name": combo_name, "gate_cfg": gate_cfg, "subgate_cfg": subgate_cfg,
                    "nonblazar_cfg": nonblazar_cfg, "branch_cfgs": branch_cfgs_fixed,
                    "naive_val_score": (grow[f"val_{SELECTION_METRIC}"] + srow[f"val_{SELECTION_METRIC}"]
                                        + nrow[f"val_{SELECTION_METRIC}"]) / 3.0,
                }
                candidates.append(_attach_end_to_end_name("3level", cand))
    return candidates


def _attach_stage1_confusion_matrix(fam: str, idx: int, cand: dict, rs: dict, labels: Sequence[str], logger) -> None:
    """Pools the OOF (validation-side)
    predictions collected across every successful Stage-1 repeat into one
    confusion matrix, saves it under
    VALIDATION_CONFUSION_DIR/stage1_shortlist/{fam}/, and attaches both the
    in-memory dict and the on-disk path to `cand` — so it survives into the
    "stage1" checkpoint and is available to every downstream reader without
    recomputation. Never touches test data (Stage 1 is selection-only)."""
    cm = _pooled_confusion_matrix(rs.get("oof_y_true"), rs.get("oof_y_pred"), labels)
    stem = _candidate_cm_stem(fam, idx, cand)
    csv_path = _save_confusion_matrix(
        cm, VALIDATION_CONFUSION_DIR / "stage1_shortlist" / fam, VALIDATION_CONFUSION_PLOTS_DIR / "stage1_shortlist" / fam,
        stem, f"Stage 1 [{fam}] {cand.get('full_model_name', stem)} — pooled OOF (validation)", logger=logger)
    cand["confusion_matrix"], cand["confusion_matrix_path"] = cm, csv_path


def _summarize_stage1(out: dict) -> None:
    logger = get_section_logger("stage1_shortlist", VALIDATION_LOGS_DIR)
    for fam in FAMILIES:
        for cand in out[fam]:
            _regenerate_confusion_matrix_artifacts(
                cand.get("confusion_matrix"), cand.get("confusion_matrix_path"),
                f"Stage 1 [{fam}] {cand.get('full_model_name', 'candidate')} — pooled OOF (validation)",
                logger=logger)
        top = out[fam][0] if out[fam] else None
        if top:
            logger.notice(f"  -> [{fam}] shortlist top: {top['full_model_name']}  "
                           f"stage1_mean={top['stage1_mean']:.4f}±{top['stage1_std']:.4f}  "
                           f"borda_score={top['borda_score']:.3f}  "
                           f"confusion_matrix -> {top.get('confusion_matrix_path')}")
        logger.notice(f"  [{fam}] {len(out[fam])} shortlisted candidate(s), each with a saved "
                       "pooled-OOF confusion matrix (requirement #3) — plot artifacts refreshed from "
                       "cache to pick up any plotting-only fix.")


def _require_some_success(out: dict, score_key: str, stage: str) -> None:
    """Raise (so `_run_cached` saves no checkpoint) when every candidate of a family failed —
    a cached all-NaN stage would otherwise be silently reused by every later --resume."""
    for fam, rows in out.items():
        if rows and not any(np.isfinite(r.get(score_key, float("nan"))) for r in rows):
            raise RuntimeError(f"[{stage}|{fam}] every candidate failed — see the warnings above; "
                               "nothing checkpointed.")


def run_stage1(fine_ctx, merged_ctx, threelevel_ctx, split_dir: Optional[Path] = None) -> dict:
    """`split_dir`: whose Module-2/3 leaderboards to shortlist from (default split 0);
    the contexts must be that same split's."""
    split_dir = split_dir or SPLIT0_DIR
    logger = get_section_logger("stage1_shortlist", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSTAGE 1 — coarse shortlist per family, {split_dir.name} only\n{'=' * 70}")
    out = {}

    fine_cands = _stage1_fine_candidates(split_dir)
    logger.notice(f"[fine] {len(fine_cands)} shortlisted candidate(s) — repeated-CV rescoring "
                   f"({N_REPEATS_SHORTLIST_CV} seeds x {N_CONFIRMATION_FOLDS}-fold)...")
    for idx, cand in enumerate(pbar(fine_cands, desc="[stage1|fine]")):
        rs = _val_repeated_score(
            lambda seed, cand=cand: _val_score_fine_candidate(fine_ctx, cand["cfg"], seed, return_oof=True),
            N_REPEATS_SHORTLIST_CV)
        cand.update({"stage1_mean": rs["selection_mean"], "stage1_std": rs["selection_std"],
                     "n_successful_repeats": rs["n_successful_repeats"],
                     "val_metrics": rs["val_metrics"], "train_metrics": rs["train_metrics"],
                     "val_metrics_std": rs["val_metrics_std"], "train_metrics_std": rs["train_metrics_std"]})
        _attach_stage1_confusion_matrix("fine", idx, cand, rs, list(fine_ctx.label_encoder.classes_), logger)
    out["fine"] = _borda_rank(fine_cands, _borda_criteria_from_metrics())

    if "merged" in FAMILIES:
        merged_cands = _stage1_merged_candidates(split_dir)
        merged_branch_datasets = _m2_build_branch_datasets(merged_ctx)
        logger.notice(f"[merged] {len(merged_cands)} shortlisted whole-system candidate(s) — repeated chain-CV "
                       f"rescoring ({N_REPEATS_SHORTLIST_CV} seeds x {N_CONFIRMATION_FOLDS}-fold), "
                       "node-fit cache shared across candidates per seed...")
        merged_rs_list = _val_repeated_score_shared_cache(
            merged_cands,
            lambda cand, seed, node_cache: _val_score_merged_chain_candidate(
                merged_ctx, cand["router_cfg"], cand["branch_cfgs"], merged_ctx.router_meta,
                merged_branch_datasets, seed, node_cache=node_cache, return_oof=True),
            N_REPEATS_SHORTLIST_CV, desc="[stage1|merged] seeds")
        for idx, (cand, rs) in enumerate(zip(merged_cands, merged_rs_list)):
            cand.update({"stage1_mean": rs["selection_mean"], "stage1_std": rs["selection_std"],
                         "n_successful_repeats": rs["n_successful_repeats"],
                         "val_metrics": rs["val_metrics"], "train_metrics": rs["train_metrics"],
                         "val_metrics_std": rs["val_metrics_std"], "train_metrics_std": rs["train_metrics_std"]})
            _attach_stage1_confusion_matrix("merged", idx, cand, rs, list(merged_ctx.fine_label_encoder.classes_), logger)
        out["merged"] = _borda_rank(merged_cands, _borda_criteria_from_metrics())

    if "3level" in FAMILIES:
        threelevel_cands = _stage1_3level_candidates(split_dir)
        threelevel_branch_datasets = build_branch_datasets(threelevel_ctx)
        logger.notice(f"[3level] {len(threelevel_cands)} shortlisted whole-system candidate(s) — repeated chain-CV "
                       f"rescoring ({N_REPEATS_SHORTLIST_CV} seeds x {N_CONFIRMATION_FOLDS}-fold), "
                       "node-fit cache shared across candidates per seed...")
        threelevel_rs_list = _val_repeated_score_shared_cache(
            threelevel_cands,
            lambda cand, seed, node_cache: _val_score_3level_chain_candidate(
                threelevel_ctx, cand["gate_cfg"], cand["subgate_cfg"], cand["nonblazar_cfg"], cand["branch_cfgs"],
                threelevel_ctx.gate_meta, threelevel_ctx.subgate_meta, threelevel_ctx.nonblazar_meta,
                threelevel_branch_datasets, seed, node_cache=node_cache, return_oof=True),
            N_REPEATS_SHORTLIST_CV, desc="[stage1|3level] seeds")
        for idx, (cand, rs) in enumerate(zip(threelevel_cands, threelevel_rs_list)):
            cand.update({"stage1_mean": rs["selection_mean"], "stage1_std": rs["selection_std"],
                         "n_successful_repeats": rs["n_successful_repeats"],
                         "val_metrics": rs["val_metrics"], "train_metrics": rs["train_metrics"],
                         "val_metrics_std": rs["val_metrics_std"], "train_metrics_std": rs["train_metrics_std"]})
            _attach_stage1_confusion_matrix("3level", idx, cand, rs, list(threelevel_ctx.fine_label_encoder.classes_), logger)
        out["3level"] = _borda_rank(threelevel_cands, _borda_criteria_from_metrics())

    _require_some_success(out, "stage1_mean", "stage1")
    _summarize_stage1(out)
    return out


# %% [markdown]
#

# %% [markdown]
# ## Section 5 — Stage 2: winner's-curse diagnostic — RETIRED (DEAD CODE)
#
# `run_stage2` and its helpers live in the DEAD CODE section at the
# end of this file. The code is fully runnable but is not wired into
# `STAGE_DEPENDENCIES`, the Section 13 driver, or Stage 8's persisted
# report. See the DEAD CODE section's own docstring for how to resurrect
# it.

# %% [markdown]
# ## Section 6 — Stage 3: cross-split confirmation of the shortlist [selection, splits 0-4, train only]
#
# Each shortlisted candidate is scored on EVERY discovered split's training
# data with a plain, single-seed `StratifiedKFold(n_splits=
# N_CONFIRMATION_FOLDS)`. Rows now ALSO retain the raw per-split
# metric dicts (`per_split_val_metrics`/`per_split_train_metrics`, keyed by
# split_id) alongside the existing cross-split MEAN dicts — needed by
# Section 12's CSV export (requirement #1's "with error": the error is the
# std of these per-split values) and by Section 14's CI report. The
# aggregated `val_metrics`/`train_metrics`/`cross_split_mean` fields Stage
# 4's Borda ranking already reads are unchanged in meaning.

# %%
def _stage3_shortlist(stage1_shortlists: dict, fam: str, logger) -> list:
    """Trims Stage 1's already-Borda-sorted (best-first)
    shortlist to the top `STAGE3_SHORTLIST_TOP_K` before Stage 3's much more
    expensive cross-split rescoring, when `REDUCE_STAGE3_CANDIDATES` is True
    (the default). This is a breadth trim on already-ranked candidates, not
    a change to any combo-grid width, repeat count, or fold count — those
    stay at full resolution. Caveat (disclosed and accepted): Stage 1's
    single-split ranking is itself noisy, so a candidate that would have
    overtaken the top K on cross-split confirmation can be dropped here;
    set REDUCE_STAGE3_CANDIDATES=False to disable this trim entirely."""
    full = stage1_shortlists[fam]
    if not REDUCE_STAGE3_CANDIDATES or len(full) <= STAGE3_SHORTLIST_TOP_K:
        return full
    logger.notice(f"  [{fam}] REDUCE_STAGE3_CANDIDATES=True — carrying top "
                   f"{STAGE3_SHORTLIST_TOP_K}/{len(full)} Stage-1-ranked candidate(s) into Stage 3.")
    return full[:STAGE3_SHORTLIST_TOP_K]


def _attach_stage3_confusion_matrix(fam: str, idx: int, row: dict, oof_true_list: list, oof_pred_list: list,
                                     labels: Sequence[str], logger) -> None:
    """Same pooling contract as Stage 1's
    `_attach_stage1_confusion_matrix`, but pooling across VALIDATION SPLITS
    (one pair of already-STRING-decoded OOF arrays per split that produced
    a valid score — see Section 2a — rather than across repeats) — Stage
    3's own unit of replication."""
    cm = _pooled_confusion_matrix(oof_true_list, oof_pred_list, labels)
    stem = _candidate_cm_stem(fam, idx, row)
    csv_path = _save_confusion_matrix(
        cm, VALIDATION_CONFUSION_DIR / "stage3_confirmation" / fam,
        VALIDATION_CONFUSION_PLOTS_DIR / "stage3_confirmation" / fam,
        stem, f"Stage 3 [{fam}] {row.get('full_model_name', stem)} — pooled cross-split OOF (validation)",
        logger=logger)
    row["confusion_matrix"], row["confusion_matrix_path"] = cm, csv_path


def _backfill_stage3_metric_std(row: dict) -> None:
    """Older cached "stage3" checkpoints may store only the raw `per_split_val_metrics`/
    `per_split_train_metrics` dicts on every row without
    `val_metrics_std`/`train_metrics_std` — so those fields were
    silently missing from the JSON report's `stage3_cross_split_scores`
    section (the CSV export alone recomputed them on the fly, which is why
    the CSV showed error bars but the JSON didn't). Backfills them here,
    in place, from data ALREADY sitting in the checkpoint — no re-scoring,
    no cross-split CV rerun — so this becomes correct on checkpoint-hit runs
    too, without forcing Stage 3 back into recomputing."""
    if row.get("val_metrics_std") is not None and row.get("train_metrics_std") is not None:
        return
    per_split_val = row.get("per_split_val_metrics") or {}
    per_split_train = row.get("per_split_train_metrics") or {}
    valid_splits = [sid for sid, v in (row.get("per_split_scores") or {}).items() if np.isfinite(v)]
    if row.get("val_metrics_std") is None:
        row["val_metrics_std"] = _std_metric_dicts([per_split_val[s] for s in valid_splits if s in per_split_val])
    if row.get("train_metrics_std") is None:
        row["train_metrics_std"] = _std_metric_dicts([per_split_train[s] for s in valid_splits if s in per_split_train])


def _summarize_stage3(out: dict) -> None:
    logger = get_section_logger("stage3_confirmation", VALIDATION_LOGS_DIR)
    for fam in FAMILIES:
        for row in out[fam]:
            _backfill_stage3_metric_std(row)
            _regenerate_confusion_matrix_artifacts(
                row.get("confusion_matrix"), row.get("confusion_matrix_path"),
                f"Stage 3 [{fam}] {row.get('full_model_name', 'candidate')} — pooled cross-split OOF (validation)",
                logger=logger)
        n_eligible = sum(1 for r in out[fam] if r["n_valid_splits"] >= STAGE3_MIN_SPLITS_ELIGIBLE)
        logger.notice(f"  [{fam}] {n_eligible}/{len(out[fam])} candidate(s) eligible "
                       f"(>={STAGE3_MIN_SPLITS_ELIGIBLE}/{len(VALIDATION_SPLITS)} valid splits)")
        n_with_std = sum(1 for r in out[fam] if r.get("val_metrics_std") and r.get("train_metrics_std"))
        logger.notice(f"  [{fam}] {n_with_std}/{len(out[fam])} candidate(s) have cross-split "
                       "train+val metric STD ('with error', requirement #2) — backfilled from cached "
                       "per-split data if this checkpoint predates that field.")
        n_with_cm = sum(1 for r in out[fam] if r.get("confusion_matrix") is not None)
        logger.notice(f"  [{fam}] {n_with_cm}/{len(out[fam])} candidate(s) have a saved pooled cross-split "
                       "confusion matrix (requirement #3).")
        top = out[fam][0] if out[fam] else None
        if top:
            logger.notice(f"  -> [{fam}] Stage-3 top (pre-Stage-4-eligibility): {top['full_model_name']}  "
                           f"cross_split_mean={top['cross_split_mean']:.4f}  "
                           f"confusion_matrix -> {top.get('confusion_matrix_path')}")


def run_stage3(stage1_shortlists: dict, split_contexts: dict, splits: Optional[Sequence[str]] = None) -> dict:
    """`splits`: which splits' training data to score on (default: all VALIDATION_SPLITS;
    one split under PER_SPLIT_SELECTION)."""
    splits = list(splits or VALIDATION_SPLITS)
    logger = get_section_logger("stage3_confirmation", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSTAGE 3 — cross-split confirmation, splits {splits}, train only\n{'=' * 70}")
    out = {}

    fine_shortlist = _stage3_shortlist(stage1_shortlists, "fine", logger)
    merged_shortlist = _stage3_shortlist(stage1_shortlists, "merged", logger) if "merged" in FAMILIES else []
    threelevel_shortlist = (_stage3_shortlist(stage1_shortlists, "3level", logger)
                            if "3level" in FAMILIES else [])

    # --- fine ---------------------------------------------------------------
    # No shared sub-components across fine candidates (each is a single flat
    # estimator config) — nothing for a node_cache to dedup, so this stays a
    # plain candidate-outer loop.
    fine_rows = []
    for idx, cand in enumerate(pbar(fine_shortlist, desc="[stage3|fine] candidates")):
        per_split_scores, per_split_val, per_split_train = {}, {}, {}
        per_split_val_ci, per_split_train_ci = {}, {}
        oof_true_list, oof_pred_list = [], []
        for split_id in splits:
            fine_ctx = split_contexts[split_id][0]
            try:
                res = _val_score_fine_candidate(fine_ctx, cand["cfg"], seed=_seed_for_split(_split_idx_of(split_id)),
                                                 return_oof=True, with_ci=True)
                per_split_scores[split_id] = res["selection_mean"]
                per_split_val[split_id], per_split_train[split_id] = res["val_metrics"], res["train_metrics"]
                per_split_val_ci[split_id], per_split_train_ci[split_id] = res["val_ci"], res["train_ci"]
                oof_true_list.append(res["oof_y_true"]); oof_pred_list.append(res["oof_y_pred"])
            except Exception as e:
                logger.warning(f"  ⚠ [fine|{cand['cfg']['model_name']}] split {split_id} failed: {type(e).__name__}: {e}")
                per_split_scores[split_id] = float("nan")
        valid = [v for v in per_split_scores.values() if np.isfinite(v)]
        valid_splits = [sid for sid, v in per_split_scores.items() if np.isfinite(v)]
        row = {"cfg": cand["cfg"], "full_model_name": cand["full_model_name"],
               "node_models": cand["node_models"], "per_split_scores": per_split_scores,
               "cross_split_mean": float(np.mean(valid)) if valid else float("nan"),
               "n_valid_splits": len(valid),
               "val_metrics": _mean_metric_dicts([per_split_val[s] for s in valid_splits]),
               "train_metrics": _mean_metric_dicts([per_split_train[s] for s in valid_splits]),
               # Cross-split STD ("with error") materialized as a
               # first-class field here — previously only the raw `per_split_*_metrics` dicts were
               # stored on the row, so std existed for the CSV export (which computed it on the fly)
               # but never reached the in-memory row itself, and therefore never reached the JSON
               # report's `stage3_cross_split_scores` section either.
               "val_metrics_std": _std_metric_dicts([per_split_val[s] for s in valid_splits]),
               "train_metrics_std": _std_metric_dicts([per_split_train[s] for s in valid_splits]),
               "per_split_val_metrics": per_split_val, "per_split_train_metrics": per_split_train,
               "per_split_val_ci": per_split_val_ci, "per_split_train_ci": per_split_train_ci}
        _attach_stage3_confusion_matrix("fine", idx, row, oof_true_list, oof_pred_list,
                                         list(FINE_CTX_SPLIT0.label_encoder.classes_), logger)
        fine_rows.append(row)
    out["fine"] = fine_rows

    if "merged" in FAMILIES:
        # --- merged ---------------------------------------------------------------
        # split-outer / candidate-inner, one shared node_cache
        # per split: StratifiedKFold's fold boundaries at a given split's seed
        # don't depend on candidate config, so candidates in this shortlist that
        # share an identical router/branch config get that fit only once per split.
        merged_branch_ds_by_split = {sid: _m2_build_branch_datasets(split_contexts[sid][1]) for sid in splits}
        n_merged = len(merged_shortlist)
        merged_scores = [dict() for _ in range(n_merged)]
        merged_val = [dict() for _ in range(n_merged)]
        merged_train = [dict() for _ in range(n_merged)]
        merged_val_ci = [dict() for _ in range(n_merged)]
        merged_train_ci = [dict() for _ in range(n_merged)]
        merged_oof_true = [[] for _ in range(n_merged)]
        merged_oof_pred = [[] for _ in range(n_merged)]
        for split_id in pbar(splits, desc="[stage3|merged] splits (shared node-fit cache)"):
            merged_ctx = split_contexts[split_id][1]
            seed = _seed_for_split(_split_idx_of(split_id))
            node_cache: dict = {}
            for i, cand in enumerate(merged_shortlist):
                try:
                    res = _val_score_merged_chain_candidate(
                        merged_ctx, cand["router_cfg"], cand["branch_cfgs"], merged_ctx.router_meta,
                        merged_branch_ds_by_split[split_id], seed=seed, node_cache=node_cache, return_oof=True,
                        with_ci=True)
                    merged_scores[i][split_id] = res["selection_mean"]
                    merged_val[i][split_id], merged_train[i][split_id] = res["val_metrics"], res["train_metrics"]
                    merged_val_ci[i][split_id], merged_train_ci[i][split_id] = res["val_ci"], res["train_ci"]
                    merged_oof_true[i].append(res["oof_y_true"]); merged_oof_pred[i].append(res["oof_y_pred"])
                except Exception as e:
                    logger.warning(f"  ⚠ [merged|{cand['combo_name']}] split {split_id} failed: {type(e).__name__}: {e}")
                    merged_scores[i][split_id] = float("nan")
        merged_rows = []
        for i, cand in enumerate(merged_shortlist):
            per_split_scores, per_split_val, per_split_train = merged_scores[i], merged_val[i], merged_train[i]
            valid = [v for v in per_split_scores.values() if np.isfinite(v)]
            valid_splits = [sid for sid, v in per_split_scores.items() if np.isfinite(v)]
            row = {"router_cfg": cand["router_cfg"], "branch_cfgs": cand["branch_cfgs"],
                   "combo_name": cand["combo_name"], "full_model_name": cand["full_model_name"],
                   "node_models": cand["node_models"], "per_split_scores": per_split_scores,
                   "cross_split_mean": float(np.mean(valid)) if valid else float("nan"),
                   "n_valid_splits": len(valid),
                   "val_metrics": _mean_metric_dicts([per_split_val[s] for s in valid_splits]),
                   "train_metrics": _mean_metric_dicts([per_split_train[s] for s in valid_splits]),
                   "val_metrics_std": _std_metric_dicts([per_split_val[s] for s in valid_splits]),
                   "train_metrics_std": _std_metric_dicts([per_split_train[s] for s in valid_splits]),
                   "per_split_val_metrics": per_split_val, "per_split_train_metrics": per_split_train,
                   "per_split_val_ci": merged_val_ci[i], "per_split_train_ci": merged_train_ci[i]}
            _attach_stage3_confusion_matrix("merged", i, row, merged_oof_true[i], merged_oof_pred[i],
                                             list(MERGED_CTX_SPLIT0.fine_label_encoder.classes_), logger)
            merged_rows.append(row)
        out["merged"] = merged_rows

    if "3level" in FAMILIES:
        # --- 3level ---------------------------------------------------------------
        # Same split-outer / candidate-inner shared-cache restructuring as merged above.
        threelevel_branch_ds_by_split = {sid: build_branch_datasets(split_contexts[sid][2]) for sid in splits}
        n_3l = len(threelevel_shortlist)
        tl_scores = [dict() for _ in range(n_3l)]
        tl_val = [dict() for _ in range(n_3l)]
        tl_train = [dict() for _ in range(n_3l)]
        tl_val_ci = [dict() for _ in range(n_3l)]
        tl_train_ci = [dict() for _ in range(n_3l)]
        tl_oof_true = [[] for _ in range(n_3l)]
        tl_oof_pred = [[] for _ in range(n_3l)]
        for split_id in pbar(splits, desc="[stage3|3level] splits (shared node-fit cache)"):
            tl_ctx = split_contexts[split_id][2]
            seed = _seed_for_split(_split_idx_of(split_id))
            node_cache: dict = {}
            for i, cand in enumerate(threelevel_shortlist):
                try:
                    res = _val_score_3level_chain_candidate(
                        tl_ctx, cand["gate_cfg"], cand["subgate_cfg"], cand["nonblazar_cfg"], cand["branch_cfgs"],
                        tl_ctx.gate_meta, tl_ctx.subgate_meta, tl_ctx.nonblazar_meta,
                        threelevel_branch_ds_by_split[split_id], seed=seed, node_cache=node_cache, return_oof=True,
                        with_ci=True)
                    tl_scores[i][split_id] = res["selection_mean"]
                    tl_val[i][split_id], tl_train[i][split_id] = res["val_metrics"], res["train_metrics"]
                    tl_val_ci[i][split_id], tl_train_ci[i][split_id] = res["val_ci"], res["train_ci"]
                    tl_oof_true[i].append(res["oof_y_true"]); tl_oof_pred[i].append(res["oof_y_pred"])
                except Exception as e:
                    logger.warning(f"  ⚠ [3level|{cand['combo_name']}] split {split_id} failed: {type(e).__name__}: {e}")
                    tl_scores[i][split_id] = float("nan")
        threelevel_rows = []
        for i, cand in enumerate(threelevel_shortlist):
            per_split_scores, per_split_val, per_split_train = tl_scores[i], tl_val[i], tl_train[i]
            valid = [v for v in per_split_scores.values() if np.isfinite(v)]
            valid_splits = [sid for sid, v in per_split_scores.items() if np.isfinite(v)]
            row = {"gate_cfg": cand["gate_cfg"], "subgate_cfg": cand["subgate_cfg"],
                   "nonblazar_cfg": cand["nonblazar_cfg"], "branch_cfgs": cand["branch_cfgs"],
                   "combo_name": cand["combo_name"], "full_model_name": cand["full_model_name"],
                   "node_models": cand["node_models"], "per_split_scores": per_split_scores,
                   "cross_split_mean": float(np.mean(valid)) if valid else float("nan"),
                   "n_valid_splits": len(valid),
                   "val_metrics": _mean_metric_dicts([per_split_val[s] for s in valid_splits]),
                   "train_metrics": _mean_metric_dicts([per_split_train[s] for s in valid_splits]),
                   "val_metrics_std": _std_metric_dicts([per_split_val[s] for s in valid_splits]),
                   "train_metrics_std": _std_metric_dicts([per_split_train[s] for s in valid_splits]),
                   "per_split_val_metrics": per_split_val, "per_split_train_metrics": per_split_train,
                   "per_split_val_ci": tl_val_ci[i], "per_split_train_ci": tl_train_ci[i]}
            _attach_stage3_confusion_matrix("3level", i, row, tl_oof_true[i], tl_oof_pred[i],
                                             list(THREELEVEL_CTX_SPLIT0.fine_label_encoder.classes_), logger)
            threelevel_rows.append(row)
        out["3level"] = threelevel_rows

    # Attach `borda_score` to every row HERE (ranked over all of that family's Stage-3
    # candidates), not only when `run_stage4` happens to execute and mutate these same row objects
    # in place. `run_stage4` still computes its OWN, authoritative ranking restricted to eligible
    # rows only (>=STAGE3_MIN_SPLITS_ELIGIBLE valid splits) when it actually runs, and that
    # overwrites this value for eligible rows — this pre-ranking only guarantees `borda_score` is
    # never silently missing/NaN in the persisted "stage3" checkpoint (and therefore in Section 12's
    # CSVs / Stage 8's report) on a run where "stage3" is a checkpoint hit but "stage4" is forced
    # independently, or vice versa, since the two checkpoints are saved/loaded independently and a
    # loaded checkpoint is a fresh set of Python objects, not the ones a PRIOR run's `run_stage4`
    # mutated in memory.
    for fam in FAMILIES:
        out[fam] = _borda_rank(out[fam], _borda_criteria_from_metrics())

    _require_some_success(out, "cross_split_mean", "stage3")
    _summarize_stage3(out)
    return out


# %% [markdown]
# ## Section 7 — Stage 4: final architecture selection [selection]
#
# Borda (average-rank) combination of `QUALITY_METRICS` (now 4, incl.
# accuracy) plus the train-val overfitting gap — 5 equally-weighted
# criteria. If the Borda winner doesn't have `STAGE3_MIN_SPLITS_SELECTED`
# valid splits, the next-best-ranked eligible candidate with full split
# coverage is chosen instead.

# %%
def _summarize_stage4(winners: dict) -> None:
    logger = get_section_logger("stage4_selection", VALIDATION_LOGS_DIR)
    for fam in FAMILIES:
        chosen = winners[fam]
        logger.notice(f"  -> [{fam}] SELECTED: {chosen['full_model_name']}  borda_score={chosen['borda_score']:.3f}  "
                       f"cross_split_mean={chosen['cross_split_mean']:.4f}  n_valid_splits={chosen['n_valid_splits']}  "
                       f"confusion_matrix (inherited from Stage 3) -> {chosen.get('confusion_matrix_path')}")


def run_stage4(stage3_results: dict) -> dict:
    logger = get_section_logger("stage4_selection", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSTAGE 4 — final architecture selection "
                   f"[Borda rank of Stage 3: accuracy/bal_accuracy/macro_f1/kappa + overfit gap]\n{'=' * 70}")
    winners = {}
    for fam in FAMILIES:
        eligible = [r for r in stage3_results[fam] if r["n_valid_splits"] >= STAGE3_MIN_SPLITS_ELIGIBLE
                    and np.isfinite(r["cross_split_mean"])]
        if not eligible:
            raise RuntimeError(f"[stage4|{fam}] no candidate reached {STAGE3_MIN_SPLITS_ELIGIBLE}/"
                                f"{len(VALIDATION_SPLITS)} valid splits — cannot select a final architecture.")
        ranked = _borda_rank(eligible, _borda_criteria_from_metrics())
        if fam in WINNER_OVERRIDE:
            want = WINNER_OVERRIDE[fam]
            chosen = next((r for r in ranked if want in (r.get("full_model_name"), r.get("combo_name"),
                                                        (r.get("cfg") or {}).get("model_name"))), None)
            if chosen is None:
                raise RuntimeError(f"[stage4|{fam}] WINNER_OVERRIDE '{want}' is not among the Stage-3 candidates: "
                                   f"{[r.get('full_model_name') for r in ranked]}")
            logger.warning(f"  ⚠ [{fam}] WINNER_OVERRIDE: deploying '{chosen.get('full_model_name')}' "
                           f"(Borda rank {ranked.index(chosen) + 1}) instead of '{ranked[0].get('full_model_name')}'.")
            winners[fam] = chosen
            continue
        chosen = next((r for r in ranked if r["n_valid_splits"] >= STAGE3_MIN_SPLITS_SELECTED), None)
        if chosen is None:
            logger.warning(f"  ⚠ [{fam}] top-ranked candidate(s) lack {STAGE3_MIN_SPLITS_SELECTED}/"
                            f"{STAGE3_MIN_SPLITS_SELECTED} valid "
                            "splits — falling back to the best-ranked candidate with full split coverage.")
            chosen = ranked[0]
        winners[fam] = chosen
    _summarize_stage4(winners)
    return winners


# %% [markdown]
# ## Section 8 — Stage 5: held-out test evaluation [estimation]
#
# For each split, refit the family's Stage-4 winner on that split's FULL
# training data and evaluate ONCE on that split's held-out test partition.
# Under the `single_split` and `per_split` designs no selection step has
# read rows of that test partition; under the legacy `shared` design the
# test rows overlap other splits' training data, so the scores are
# optimistic. Every split's evaluation carries its own row-level 95% CI
# (`_metric_ci`).

# %%
def _resolve_test_frame_merged(ctx, tag: str, local_meta):
    if tag in ctx.x_test_by_tag:
        return ctx.x_test_by_tag[tag]
    if local_meta and tag in local_meta.get("feature_sets", {}):
        return ctx.x_test_by_tag["ALL_FEATURES"][local_meta["feature_sets"][tag]]
    raise KeyError(f"[merged] Tag '{tag}' has no resolvable TEST-side frame.")


def _fit_predict_merged_core(ctx, router_cfg, branch_cfgs, router_meta, branch_datasets,
                              resolve_frame, n_rows: int, seed=SEED, return_proba: bool = False,
                              node_cache: Optional[dict] = None):
    """Shared fit-and-route logic for the merged (2-level) chain, used by
    both the TEST-side and TRAIN-side evaluators. `return_proba`
    optionally also returns the full `p_fine` soft-voted probability
    matrix (needed by fermi_inference.py's confidence export and by
    SHAP, now DEAD CODE at the end of this file)
    without duplicating this function a third time. `node_cache`
    dedups full-training-set refits across candidates that share an
    identical router/branch config (e.g. Stage 5 re-evaluating the same
    winner across the train-side calibration/SHAP paths); cache keys use
    a "*_full" prefix so they never collide with the fold-scoped keys
    used by `_m2_fit_and_route_chain_fold`."""
    router_key = ("router_full", _cfg_key(router_cfg))
    router_model = _cached_fit(node_cache, router_key, lambda: fit_component(
        router_cfg, MODEL_SPECS, get_builder, ctx.x_train_by_tag[router_cfg["tag"]],
        ctx.y_train_super_enc, seed))
    branch_models = {}
    for sc in SUPERCLASS_ORDER:
        cfg = branch_cfgs.get(sc)
        branch = branch_datasets.get(sc)
        if cfg is None or branch is None:
            branch_models[sc] = None
            continue
        branch_key = ("branch_full", sc, _cfg_key(cfg))
        model = _cached_fit(node_cache, branch_key, lambda cfg=cfg, branch=branch: fit_component(
            cfg, MODEL_SPECS, get_builder, branch["x_train_by_tag"][cfg["tag"]],
            branch["y_train_enc"], seed))
        branch_models[sc] = (model, branch["label_encoder"])

    x_router = resolve_frame(ctx, router_cfg["tag"], router_meta)
    all_superclasses = list(SUPER_LE.classes_)
    super_idx = {s: i for i, s in enumerate(all_superclasses)}
    p_super = _m2_predict_proba_safe(router_model, x_router, len(all_superclasses))

    all_fine_classes = list(ctx.fine_class_names)
    fine_idx = {c: i for i, c in enumerate(all_fine_classes)}
    p_fine = np.zeros((n_rows, len(all_fine_classes)))

    for sc in SUPERCLASS_ORDER:
        p_sc = p_super[:, super_idx[sc]]
        bm = branch_models.get(sc)
        branch = branch_datasets.get(sc)
        if bm is None:
            fine_tr_sc = ctx.y_train_fine_str[ctx.y_train_super_str == sc]
            if len(fine_tr_sc):
                fallback = pd.Series(fine_tr_sc).mode().iloc[0]
                p_fine[:, fine_idx[fallback]] += p_sc
            continue
        model, le_branch = bm
        cfg = branch_cfgs[sc]
        local_meta = branch["local_meta"]
        try:
            x_branch = resolve_frame(ctx, cfg["tag"], local_meta)
        except KeyError:
            fine_tr_sc = ctx.y_train_fine_str[ctx.y_train_super_str == sc]
            fallback = pd.Series(fine_tr_sc).mode().iloc[0] if len(fine_tr_sc) else all_fine_classes[0]
            p_fine[:, fine_idx[fallback]] += p_sc
            continue
        p_branch = _m2_predict_proba_safe(model, x_branch, len(le_branch.classes_))
        for k, fc in enumerate(le_branch.classes_):
            p_fine[:, fine_idx[fc]] += p_sc * p_branch[:, k]

    pred_fine_str = np.array(all_fine_classes)[p_fine.argmax(axis=1)]
    pred_fine_enc = ctx.fine_label_encoder.transform(pred_fine_str)
    return (pred_fine_enc, p_fine) if return_proba else pred_fine_enc


def _fit_predict_merged_on_test(ctx, router_cfg, branch_cfgs, router_meta, branch_datasets, seed=SEED,
                                 return_proba: bool = False, node_cache: Optional[dict] = None):
    return _fit_predict_merged_core(ctx, router_cfg, branch_cfgs, router_meta, branch_datasets,
                                     _resolve_test_frame_merged, len(ctx.y_test_fine_enc), seed,
                                     return_proba=return_proba, node_cache=node_cache)


def _fit_predict_merged_on_train(ctx, router_cfg, branch_cfgs, router_meta, branch_datasets, seed=SEED,
                                  return_proba: bool = False, node_cache: Optional[dict] = None):
    return _fit_predict_merged_core(ctx, router_cfg, branch_cfgs, router_meta, branch_datasets,
                                     _m2_resolve_full_length_tag_frame, len(ctx.y_train_super_str), seed,
                                     return_proba=return_proba, node_cache=node_cache)


def _resolve_test_frame_3level(ctx, tag: str, local_meta):
    if tag in ctx.x_test_by_tag:
        return ctx.x_test_by_tag[tag]
    if local_meta and tag in local_meta.get("feature_sets", {}):
        return ctx.x_test_by_tag["ALL_FEATURES"][local_meta["feature_sets"][tag]]
    raise KeyError(f"[3level] Tag '{tag}' has no resolvable TEST-side frame.")


def _fit_predict_3level_core(ctx, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs,
                              gate_meta, subgate_meta, nonblazar_meta, branch_datasets,
                              resolve_frame, n_rows: int, seed=SEED, node_cache: Optional[dict] = None,
                              return_proba: bool = False):
    """Shared fit-and-route logic for the 3-level chain. `node_cache`
    dedups full-training-set refits (see `_fit_predict_merged_core`);
    node key prefixes ("gate_full", "blazar_subgate_full",
    "nonblazar_router_full", "branch_full") are namespaced separately
    from the fold-scoped keys used by `_l3_fit_and_route_chain_fold_cached`.
    `return_proba` mirrors `_fit_predict_merged_core`'s — also
    returns the full soft-voted `p_fine` matrix, needed by Section 16b's
    calibration/Brier-score check."""
    tr_mask_blazar = np.isin(ctx.y_train_super_str, list(BLAZAR_SUPERCLASSES))
    tr_mask_nonblazar = ~tr_mask_blazar

    x_gate_tr = _m2_resolve_full_length_tag_frame(ctx, gate_cfg["tag"], gate_meta) \
        if gate_cfg["tag"] in ctx.x_train_by_tag or (gate_meta and gate_cfg["tag"] in gate_meta.get("feature_sets", {})) \
        else ctx.x_train_by_tag[gate_cfg["tag"]]
    gate_key = ("gate_full", _cfg_key(gate_cfg))
    gate_model = _cached_fit(node_cache, gate_key, lambda: fit_component(
        gate_cfg, MODEL_SPECS, get_builder, x_gate_tr,
        GATE_LE.transform(build_gate_labels(ctx.y_train_super_str)), seed))

    x_sub_tr_full = (ctx.x_train_by_tag[subgate_cfg["tag"]] if subgate_cfg["tag"] in ctx.x_train_by_tag
                      else ctx.x_train_by_tag["ALL_FEATURES"][subgate_meta["feature_sets"][subgate_cfg["tag"]]])
    subgate_key = ("blazar_subgate_full", _cfg_key(subgate_cfg))
    subgate_model = _cached_fit(node_cache, subgate_key, lambda: fit_component(
        subgate_cfg, MODEL_SPECS, get_builder, x_sub_tr_full[tr_mask_blazar].reset_index(drop=True),
        BLAZAR_SUBGATE_LE.transform(ctx.y_train_super_str[tr_mask_blazar]), seed))

    x_nb_tr_full = (ctx.x_train_by_tag[nonblazar_cfg["tag"]] if nonblazar_cfg["tag"] in ctx.x_train_by_tag
                     else ctx.x_train_by_tag["ALL_FEATURES"][nonblazar_meta["feature_sets"][nonblazar_cfg["tag"]]])
    nonblazar_key = ("nonblazar_router_full", _cfg_key(nonblazar_cfg))
    nonblazar_model = _cached_fit(node_cache, nonblazar_key, lambda: fit_component(
        nonblazar_cfg, MODEL_SPECS, get_builder,
        x_nb_tr_full[tr_mask_nonblazar].reset_index(drop=True),
        NONBLAZAR_ROUTER_LE.transform(ctx.y_train_super_str[tr_mask_nonblazar]), seed))

    branch_models = {}
    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        cfg = branch_cfgs.get(sc)
        branch = branch_datasets.get(sc)
        if cfg is None or branch is None:
            branch_models[sc] = None
            continue
        branch_key = ("branch_full", sc, _cfg_key(cfg))
        model = _cached_fit(node_cache, branch_key, lambda cfg=cfg, branch=branch: fit_component(
            cfg, MODEL_SPECS, get_builder, branch["x_train_by_tag"][cfg["tag"]],
            branch["y_train_enc"], seed))
        branch_models[sc] = (model, branch["label_encoder"])

    x_gate_eval = resolve_frame(ctx, gate_cfg["tag"], gate_meta)
    p_gate = _m2_predict_proba_safe(gate_model, x_gate_eval, len(GATE_LE.classes_))
    p_blazar = p_gate[:, list(GATE_LE.classes_).index("BLAZAR")]
    p_other = p_gate[:, list(GATE_LE.classes_).index("OTHER")]

    x_sub_eval = resolve_frame(ctx, subgate_cfg["tag"], subgate_meta)
    p_subgate = _m2_predict_proba_safe(subgate_model, x_sub_eval, len(BLAZAR_SUBGATE_LE.classes_))
    x_nb_eval = resolve_frame(ctx, nonblazar_cfg["tag"], nonblazar_meta)
    p_router = _m2_predict_proba_safe(nonblazar_model, x_nb_eval, len(NONBLAZAR_ROUTER_LE.classes_))

    all_superclasses = list(SUPER_LE.classes_)
    super_idx = {s: i for i, s in enumerate(all_superclasses)}
    p_super = np.zeros((n_rows, len(all_superclasses)))
    for j, sc in enumerate(BLAZAR_SUBGATE_LE.classes_):
        p_super[:, super_idx[sc]] = p_blazar * p_subgate[:, j]
    for j, sc in enumerate(NONBLAZAR_ROUTER_LE.classes_):
        p_super[:, super_idx[sc]] = p_other * p_router[:, j]

    all_fine_classes = list(ctx.fine_class_names)
    fine_idx = {c: i for i, c in enumerate(all_fine_classes)}
    p_fine = np.zeros((n_rows, len(all_fine_classes)))
    for sc in BLAZAR_SUPERCLASSES:
        fines = sorted(c for c, s in TAXONOMY.items() if s == sc and c in ctx.fine_class_names)
        p_fine[:, fine_idx[fines[0]]] += p_super[:, super_idx[sc]]
    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        p_sc = p_super[:, super_idx[sc]]
        bm = branch_models.get(sc)
        branch = branch_datasets.get(sc)
        if bm is None:
            fine_tr_sc = ctx.y_train_fine_str[ctx.y_train_super_str == sc]
            if len(fine_tr_sc):
                fallback = pd.Series(fine_tr_sc).mode().iloc[0]
                p_fine[:, fine_idx[fallback]] += p_sc
            continue
        model, le_branch = bm
        cfg = branch_cfgs[sc]
        local_meta = branch["local_meta"]
        try:
            x_branch_eval = resolve_frame(ctx, cfg["tag"], local_meta)
        except KeyError:
            fine_tr_sc = ctx.y_train_fine_str[ctx.y_train_super_str == sc]
            fallback = pd.Series(fine_tr_sc).mode().iloc[0] if len(fine_tr_sc) else all_fine_classes[0]
            p_fine[:, fine_idx[fallback]] += p_sc
            continue
        p_branch = _m2_predict_proba_safe(model, x_branch_eval, len(le_branch.classes_))
        for k, fc in enumerate(le_branch.classes_):
            p_fine[:, fine_idx[fc]] += p_sc * p_branch[:, k]

    pred_fine_str = np.array(all_fine_classes)[p_fine.argmax(axis=1)]
    pred_fine_enc = ctx.fine_label_encoder.transform(pred_fine_str)
    return (pred_fine_enc, p_fine) if return_proba else pred_fine_enc


def _fit_predict_3level_on_test(ctx, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs,
                                 gate_meta, subgate_meta, nonblazar_meta, branch_datasets, seed=SEED,
                                 node_cache: Optional[dict] = None, return_proba: bool = False):
    return _fit_predict_3level_core(ctx, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs,
                                     gate_meta, subgate_meta, nonblazar_meta, branch_datasets,
                                     _resolve_test_frame_3level, len(ctx.y_test_fine_enc), seed,
                                     node_cache=node_cache, return_proba=return_proba)


def _fit_predict_3level_on_train(ctx, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs,
                                  gate_meta, subgate_meta, nonblazar_meta, branch_datasets, seed=SEED,
                                  node_cache: Optional[dict] = None, return_proba: bool = False):
    return _fit_predict_3level_core(ctx, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs,
                                     gate_meta, subgate_meta, nonblazar_meta, branch_datasets,
                                     _m2_resolve_full_length_tag_frame, len(ctx.y_train_super_str), seed,
                                     node_cache=node_cache, return_proba=return_proba)


def _label_encoder_for(fam: str, ctx_tuple):
    fine_ctx, merged_ctx, tl_ctx = ctx_tuple
    return fine_ctx.label_encoder if fam == "fine" else merged_ctx.fine_label_encoder if fam == "merged" \
        else tl_ctx.fine_label_encoder


def _winner_label_encoder(fam: str, split_contexts: dict):
    return _label_encoder_for(fam, split_contexts[SPLIT0])


def _attach_stage5_winner_confusion_matrices(fam: str, winner: dict, out_fam: dict, split_contexts: dict, logger) -> None:
    """Every deployed-winner confusion
    matrix this file saves, in one place:
      - VALIDATION — the winner's OWN Stage-3 pooled cross-split confusion
        matrix (`winner["confusion_matrix"]`, attached back in Stage 3 —
        requirement #3's machinery already computed this; re-saving it
        here under a "stage5_winners" name just makes it discoverable
        alongside the winner's test-side numbers without re-deriving it a
        second time from raw predictions).
      - TEST — pooled across every valid split's genuine held-out test
        predictions (`out_fam["predictions_by_split"]`), the first
        confusion matrix in this file ever built from test data, and only
        for the one architecture per family that reached deployment.
        `predictions_by_split` stores each split's (y_true, y_pred) as
        THAT split's own encoded ints — decoded to strings with that SAME split's own
        encoder before pooling (see Section 2a's docstring for why this
        matters: a single fixed encoder cannot safely decode every split's
        codes)."""
    labels = list(_winner_label_encoder(fam, split_contexts).classes_)

    val_cm = winner.get("confusion_matrix")
    val_path = _save_confusion_matrix(
        val_cm, VALIDATION_CONFUSION_DIR / "stage5_winners", VALIDATION_CONFUSION_PLOTS_DIR / "stage5_winners",
        f"{fam}_validation", f"Stage 5 [{fam}] winner {winner.get('full_model_name')} — VALIDATION "
        "(Stage-3 pooled cross-split OOF)", logger=logger)

    pred_cache = out_fam.get("predictions_by_split") or {}
    test_true_str, test_pred_str = [], []
    for split_id, pair in pred_cache.items():
        if pair is None:
            continue
        y_true_enc, y_pred_enc = pair
        if y_true_enc is None or y_pred_enc is None or len(y_true_enc) == 0:
            continue
        split_encoder = _label_encoder_for(fam, split_contexts[split_id])
        test_true_str.append(split_encoder.inverse_transform(np.asarray(y_true_enc)))
        test_pred_str.append(split_encoder.inverse_transform(np.asarray(y_pred_enc)))
    test_cm = _pooled_confusion_matrix(test_true_str, test_pred_str, labels)
    test_path = _save_confusion_matrix(
        test_cm, VALIDATION_CONFUSION_DIR / "stage5_winners", VALIDATION_CONFUSION_PLOTS_DIR / "stage5_winners",
        f"{fam}_test", f"Stage 5 [{fam}] winner {winner.get('full_model_name')} — TEST "
        f"({sum(len(a) for a in test_true_str)} pooled held-out split(s))", logger=logger)

    out_fam["confusion_matrix_validation"], out_fam["confusion_matrix_validation_path"] = val_cm, val_path
    out_fam["confusion_matrix_test"], out_fam["confusion_matrix_test_path"] = test_cm, test_path


def _attach_stage5_winner_metrics_with_error(fam: str, winner: dict, out_fam: dict) -> None:
    """The deployed winner's TRAIN (Stage-5 refit, scored on its own training rows — a memorization
    diagnostic), VALIDATION (its Stage-3 pooled OOF score, what Stage 4 selected on) and TEST
    (Stage 5) metrics, each with mean/std/95% CI over the splits (Nadeau-Bengio t), or — with a
    single split — that split's own row-level CI (`_metric_ci`)."""
    out_fam["winner_per_split_validation_ci"] = winner.get("per_split_val_ci") or {}
    for stage, dicts, cis in (
            ("train", out_fam.get("per_split_train_metrics"), out_fam.get("per_split_train_ci")),
            ("validation", winner.get("per_split_val_metrics"), winner.get("per_split_val_ci")),
            ("test", out_fam.get("per_split_val_metrics"), out_fam.get("per_split_test_ci"))):
        full = _metric_ci_across_splits(dicts or {}, per_split_ci=cis)
        out_fam[f"winner_{stage}_metrics_ci"] = full
        out_fam[f"winner_{stage}_metrics_mean"] = {m: v["mean"] for m, v in full.items()}
        out_fam[f"winner_{stage}_metrics_std"] = {m: v["std"] for m, v in full.items()}


def export_winner_full_report_csv(stage5_results: dict) -> str:
    """One row per family: the deployed
    winner's TRAIN/VALIDATION/TEST metrics (mean±std, every REPORT_METRICS
    column) plus the paths of its two saved confusion matrices — the
    single, explicit "tested winner models ... logged and saved" deliverable
    requirement #4 asks for, in addition to (not instead of) Section 14's
    fuller CI-based report."""
    logger = get_section_logger("stage5_test_eval", VALIDATION_LOGS_DIR)
    rows, per_split_rows = [], []
    for fam in FAMILIES:
        r = stage5_results[fam]
        row = {"family": fam, "full_model_name": r.get("full_model_name"),
               "confusion_matrix_validation_path": r.get("confusion_matrix_validation_path"),
               "confusion_matrix_test_path": r.get("confusion_matrix_test_path")}
        for stage in ("train", "validation", "test"):
            full = r.get(f"winner_{stage}_metrics_ci") or {}
            for m in REPORT_METRICS:
                for k in ("mean", "std", "ci_low", "ci_high", "method"):
                    row[f"{stage}_{m}_{k}"] = (full.get(m) or {}).get(k, float("nan"))
        rows.append(row)
        # Long format: every split, every stage, every metric, with its own row-level CI.
        for stage, cis in (("train", r.get("per_split_train_ci")),
                           ("validation", r.get("winner_per_split_validation_ci")),
                           ("test", r.get("per_split_test_ci"))):
            for split_id, ci_split in (cis or {}).items():
                for m, v in ci_split.items():
                    per_split_rows.append({"family": fam, "split_id": split_id, "stage": stage, "metric": m, **v})
    path = VALIDATION_CSV_DIR / "winner_full_report.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    per_split_path = VALIDATION_CSV_DIR / "winner_per_split_metrics_ci.csv"
    pd.DataFrame(per_split_rows).to_csv(per_split_path, index=False)
    logger.notice(f"  Winner metrics per split/stage with row-level 95% CI -> {per_split_path}")
    logger.notice(f"  Winner full train/validation/test report (with errors) + confusion-matrix "
                   f"paths (requirement #4) -> {path}")
    return str(path)


def _summarize_stage5(out: dict) -> None:
    logger = get_section_logger("stage5_test_eval", VALIDATION_LOGS_DIR)
    for fam in FAMILIES:
        out_fam = out[fam]
        winner_name = out_fam.get("full_model_name", "winner")
        val_cm = out_fam.get("confusion_matrix_validation")
        _regenerate_confusion_matrix_artifacts(
            val_cm, out_fam.get("confusion_matrix_validation_path"),
            f"Stage 5 [{fam}] winner {winner_name} — VALIDATION (Stage-3 pooled cross-split OOF)",
            logger=logger)
        test_cm = out_fam.get("confusion_matrix_test")
        _regenerate_confusion_matrix_artifacts(
            test_cm, out_fam.get("confusion_matrix_test_path"),
            f"Stage 5 [{fam}] winner {winner_name} — TEST "
            f"({test_cm.get('n') if test_cm else 0} pooled held-out split(s))",
            logger=logger)
        for split_id, tci in (out_fam.get("per_split_test_ci") or {}).items():
            split_metric_str = ", ".join(f"{m}={tci[m]['value']:.4f} [{tci[m]['ci_low']:.4f},{tci[m]['ci_high']:.4f}]"
                                         for m in REPORT_METRICS)
            logger.notice(f"  [{fam}|{split_id}] test (95% CI): {split_metric_str}")
        ci = out_fam["ci"]
        logger.notice(f"  -> [{fam}] mean test {SELECTION_METRIC} = {ci['mean']:.4f}  "
                       f"95% CI [{ci.get('ci_low', float('nan')):.4f}, {ci.get('ci_high', float('nan')):.4f}]  "
                       f"(n={ci['n']}, {ci.get('method', 'nadeau_bengio_t')})  overfit_gap={_overfit_gap(out_fam):.4f}")
        test_full_ci = out_fam.get("test_full_ci") or {}
        if test_full_ci:
            full_metric_str = "  ".join(
                f"{m}={test_full_ci[m]['mean']:.4f}±{test_full_ci[m]['std']:.4f} "
                f"CI[{test_full_ci[m]['ci_low']:.4f},{test_full_ci[m]['ci_high']:.4f}]"
                for m in REPORT_METRICS)
            logger.notice(f"     [{fam}] full test metrics (mean±std, 95% CI): {full_metric_str}")
        logger.notice(f"     [{fam}] winner train/validation/test metrics with error + confusion "
                       f"matrices (requirement #4) — validation -> {out_fam.get('confusion_matrix_validation_path')}"
                       f"  |  test -> {out_fam.get('confusion_matrix_test_path')}")
    export_winner_full_report_csv(out)


def run_stage5(stage4_winners: dict, split_contexts: dict) -> dict:
    logger = get_section_logger("stage5_test_eval", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSTAGE 5 — held-out test evaluation, {len(VALIDATION_SPLITS)} split(s) [estimation]\n{'=' * 70}")
    out = {}
    for fam in FAMILIES:
        winner = stage4_winners[fam]
        per_split = {}
        pred_cache = {}
        per_split_val_metrics, per_split_train_metrics = {}, {}
        per_split_test_ci, per_split_train_ci = {}, {}
        for split_id in pbar(VALIDATION_SPLITS, desc=f"[stage5|{fam}]"):
            fine_ctx, merged_ctx, tl_ctx = split_contexts[split_id]
            winner = _winner_for(stage4_winners, fam, split_id)
            try:
                if fam == "fine":
                    cfg = winner["cfg"]
                    model = fit_component(cfg, MODEL_SPECS, get_builder, fine_ctx.x_train_by_tag[cfg["tag"]],
                                           fine_ctx.y_train_enc)
                    y_pred = model.predict(fine_ctx.x_test_by_tag[cfg["tag"]])
                    y_true = fine_ctx.y_test_enc
                    y_pred_train = model.predict(fine_ctx.x_train_by_tag[cfg["tag"]])
                    y_true_train = fine_ctx.y_train_enc
                elif fam == "merged":
                    branch_ds = _m2_build_branch_datasets(merged_ctx)
                    y_pred = _fit_predict_merged_on_test(merged_ctx, winner["router_cfg"], winner["branch_cfgs"],
                                                          merged_ctx.router_meta, branch_ds)
                    y_true = merged_ctx.y_test_fine_enc
                    y_pred_train = _fit_predict_merged_on_train(merged_ctx, winner["router_cfg"],
                                                                 winner["branch_cfgs"], merged_ctx.router_meta,
                                                                 branch_ds)
                    y_true_train = merged_ctx.fine_label_encoder.transform(merged_ctx.y_train_fine_str)
                else:
                    branch_ds = build_branch_datasets(tl_ctx)
                    y_pred = _fit_predict_3level_on_test(
                        tl_ctx, winner["gate_cfg"], winner["subgate_cfg"], winner["nonblazar_cfg"],
                        winner["branch_cfgs"], tl_ctx.gate_meta, tl_ctx.subgate_meta, tl_ctx.nonblazar_meta, branch_ds)
                    y_true = tl_ctx.y_test_fine_enc
                    y_pred_train = _fit_predict_3level_on_train(
                        tl_ctx, winner["gate_cfg"], winner["subgate_cfg"], winner["nonblazar_cfg"],
                        winner["branch_cfgs"], tl_ctx.gate_meta, tl_ctx.subgate_meta, tl_ctx.nonblazar_meta, branch_ds)
                    y_true_train = tl_ctx.fine_label_encoder.transform(tl_ctx.y_train_fine_str)
                score = _score_by_name(SELECTION_METRIC, y_true, y_pred)
                per_split[split_id] = score
                pred_cache[split_id] = (y_true, y_pred)
                per_split_val_metrics[split_id] = _full_report_metric_dict(y_true, y_pred)
                per_split_train_metrics[split_id] = _full_report_metric_dict(y_true_train, y_pred_train)
                per_split_test_ci[split_id] = _metric_ci(y_true, y_pred)
                per_split_train_ci[split_id] = _metric_ci(y_true_train, y_pred_train)
                tci = per_split_test_ci[split_id]
                split_metric_str = ", ".join(f"{m}={tci[m]['value']:.4f} [{tci[m]['ci_low']:.4f},{tci[m]['ci_high']:.4f}]"
                                             for m in REPORT_METRICS)
                logger.notice(f"  [{fam}|{split_id}] test (95% CI): {split_metric_str}")
            except Exception as e:
                logger.error(f"  ⚠ [{fam}|{split_id}] test evaluation FAILED: {type(e).__name__}: {e}")
                per_split[split_id] = float("nan")

        winner = stage4_winners[fam]  # split 0's (or the shared) winner, for naming/attachments
        valid_scores = [v for v in per_split.values() if np.isfinite(v)]
        valid_splits = [sid for sid, v in per_split.items() if np.isfinite(v)]
        test_full_ci = _metric_ci_across_splits({s: per_split_val_metrics[s] for s in valid_splits},
                                                per_split_ci=per_split_test_ci)
        ci = _repeated_cv_ci(valid_scores)
        ci["method"] = "nadeau_bengio_t"
        if ci["n"] == 1:  # one split: its own row-level CI is the only uncertainty there is
            t = test_full_ci[SELECTION_METRIC]
            ci.update(std=t["std"], ci_low=t["ci_low"], ci_high=t["ci_high"], method=t["method"])
        val_metrics = _mean_metric_dicts([per_split_val_metrics[s] for s in valid_splits])
        train_metrics = _mean_metric_dicts([per_split_train_metrics[s] for s in valid_splits])
        # [full-metrics logging] mean/std("error")/95% CI for every REPORT_METRICS column, not just
        # SELECTION_METRIC — reuses the same `_std_metric_dicts`/`_metric_ci_across_splits` helpers
        # Section 14's report builder already relies on, so the console log and the exported CSV
        # can never disagree on these numbers.
        val_metrics_std = _std_metric_dicts([per_split_val_metrics[s] for s in valid_splits])
        train_metrics_std = _std_metric_dicts([per_split_train_metrics[s] for s in valid_splits])
        out[fam] = {"per_split_scores": per_split, "ci": ci, "predictions_by_split": pred_cache,
                     "per_split_test_ci": per_split_test_ci, "per_split_train_ci": per_split_train_ci,
                     "val_metrics": val_metrics, "train_metrics": train_metrics,
                     "val_metrics_std": val_metrics_std, "train_metrics_std": train_metrics_std,
                     "test_full_ci": test_full_ci,
                     "per_split_val_metrics": per_split_val_metrics, "per_split_train_metrics": per_split_train_metrics,
                     "full_model_name": winner.get("full_model_name"), "node_models": winner.get("node_models"),
                     "winner_by_split": {sid: _winner_for(stage4_winners, fam, sid).get("full_model_name")
                                         for sid in VALIDATION_SPLITS}}
        # "tested winner models ... logged and saved with all train,
        # validation and test metrics with errors + confusion matrix on validation + confusion matrix
        # on test" \u2014 the winner's own Stage-3 validation confusion matrix (requirement #3's
        # machinery) plus a freshly-pooled TEST confusion matrix from this stage's own predictions,
        # and the train/validation/test metric triple with error bars on every REPORT_METRICS column.
        _attach_stage5_winner_confusion_matrices(fam, winner, out[fam], split_contexts, logger)
        _attach_stage5_winner_metrics_with_error(fam, winner, out[fam])
        logger.notice(f"  -> [{fam}] mean test {SELECTION_METRIC} = {ci['mean']:.4f}  "
                       f"95% CI [{ci.get('ci_low', float('nan')):.4f}, {ci.get('ci_high', float('nan')):.4f}]  "
                       f"(n={ci['n']})  overfit_gap={_overfit_gap(out[fam]):.4f}")
        full_metric_str = "  ".join(
            f"{m}={test_full_ci[m]['mean']:.4f}\u00b1{test_full_ci[m]['std']:.4f} "
            f"CI[{test_full_ci[m]['ci_low']:.4f},{test_full_ci[m]['ci_high']:.4f}]"
            for m in REPORT_METRICS
        )
        logger.notice(f"     [{fam}] full test metrics (mean\u00b1std, 95% CI): {full_metric_str}")
        logger.notice(f"     [{fam}] winner confusion matrices (requirement #4) \u2014 "
                       f"validation -> {out[fam].get('confusion_matrix_validation_path')}  |  "
                       f"test -> {out[fam].get('confusion_matrix_test_path')}")
    export_winner_full_report_csv(out)
    return out


# %%
# CI helpers used by Stage 5 (`run_stage5` above calls `_repeated_cv_ci` to build each family's
# `ci` dict; Python only needs the def to exist by the time the call runs).
def _nb_sem(std: float, n: int) -> float:
    """Nadeau & Bengio (2003) corrected standard error of a mean over n random
    train/test re-splits of ONE pool: std * sqrt(1/n + n_test/n_train). The
    splits share rows, so the naive std/sqrt(n) (independent splits) is too
    small — by ~1.8x here (n=5, n_test/n_train=0.43)."""
    counts = SPLIT_ROW_COUNTS.get(SPLIT0, {})
    rho = counts["n_test"] / counts["n_train"] if counts.get("n_train") and counts.get("n_test") else 0.0
    return float(std * np.sqrt(1.0 / n + rho))


def _repeated_cv_ci(scores: Sequence[float], confidence: float = 0.95) -> dict:
    arr = np.asarray([s for s in scores if s is not None and np.isfinite(s)], dtype=float)
    n = len(arr)
    if n < 2:
        return {"mean": float(arr.mean()) if n else float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": n, "confidence": confidence}
    mean, sem = float(arr.mean()), _nb_sem(arr.std(ddof=1), n)
    tmult = float(_t_dist.ppf(0.5 + confidence / 2, df=n - 1))
    return {"mean": mean, "std": float(arr.std(ddof=1)), "ci_low": mean - tmult * sem,
            "ci_high": mean + tmult * sem, "n": n, "confidence": confidence}


def _metric_ci_across_splits(per_split_dicts: dict, metrics: Sequence[str] = REPORT_METRICS,
                              confidence: float = 0.95, per_split_ci: Optional[dict] = None) -> dict:
    """`per_split_dicts`: {split_id: {metric: value}} (e.g. Stage 3/5's
    `per_split_val_metrics`/`per_split_train_metrics`). Returns
    {metric: {"mean","std","ci_low","ci_high","n"}} — same small-sample
    Student-t construction as `_repeated_cv_ci`, applied independently to
    every metric column (`method="nadeau_bengio_t"`). With a single split
    there is nothing to average, so the split's own row-level CI from
    `per_split_ci` ({split_id: `_metric_ci` output}) is reported instead
    (`method` says which); without it, std/CI are nan."""
    out = {}
    for m in metrics:
        vals = np.asarray([d[m] for d in per_split_dicts.values()
                            if d is not None and m in d and np.isfinite(d.get(m, float("nan")))], dtype=float)
        n = len(vals)
        if n < 2:
            sid = next((s for s, d in per_split_dicts.items()
                        if d is not None and np.isfinite(d.get(m, float("nan")))), None)
            b = ((per_split_ci or {}).get(sid) or {}).get(m)
            out[m] = ({"mean": b["value"], "std": b["std"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                       "n": n, "method": b["method"], "posterior_mean": b.get("posterior_mean", float("nan"))} if b else
                      {"mean": float(vals.mean()) if n else float("nan"), "std": float("nan"),
                       "ci_low": float("nan"), "ci_high": float("nan"), "n": n, "method": "none"})
            continue
        mean, sem = float(vals.mean()), _nb_sem(vals.std(ddof=1), n)
        tmult = float(_t_dist.ppf(0.5 + confidence / 2, df=n - 1))
        out[m] = {"mean": mean, "std": float(vals.std(ddof=1)), "ci_low": mean - tmult * sem,
                  "ci_high": mean + tmult * sem, "n": n, "method": "nadeau_bengio_t"}
    return out


# %% [markdown]
# ## Section 9 — Stage 6: family-vs-family significance testing — RETIRED (DEAD CODE)
#
# `run_stage6` and its support functions (Nadeau-Bengio corrected t-test,
# McNemar, stratified paired bootstrap, Cohen's d, Holm-Bonferroni) live
# in the DEAD CODE section at the end of this file. They are runnable but
# not wired into the Section 13 driver or Stage 8's report (`run_stage8`
# takes no `stage6` argument, and `cross_pipeline_validation_report.json`
# has no `stage6_pairwise_significance` key). `_repeated_cv_ci`, which
# Stage 5 depends on, sits at the end of Section 8. See the DEAD CODE
# section's own docstring for how to resurrect Stage 6.

# %% [markdown]
# ## Section 10 — Stage 7: deployment refit [deployment]
#
# Refit each family's Stage-4 architecture on split 0's train+test pooled
# — the simpler, design-doc-sanctioned alternative to pooling across all
# 5 splits (which would require reconciling different splits'
# independently-re-selected feature-name sets). This model is never
# itself re-evaluated; its expected performance is implied by Stage 5.
# Every deployed `.joblib` payload now also carries
# `full_model_name`/`node_models` (Section 2b) straight from the Stage-4
# winner — so loading a deployed model for inference (fermi_inference.py)
# or SHAP (now DEAD CODE at the end of this file) never has to re-derive its
# own name from raw configs.

# %%
def _summarize_stage7(out: dict) -> None:
    logger = get_section_logger("stage7_deployment", VALIDATION_LOGS_DIR)
    for fam in FAMILIES:
        logger.notice(f"  [{fam}] deployed -> {out[fam]['model_path']}  (n={out[fam]['n_pooled_rows']})  "
                       f"{out[fam].get('full_model_name')}")


def run_stage7(stage4_winners: dict, split_contexts: dict) -> dict:
    logger = get_section_logger("stage7_deployment", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSTAGE 7 — deployment refit on split 0 train+test pooled\n{'=' * 70}")
    fine_ctx, merged_ctx, tl_ctx = split_contexts[SPLIT0]
    out = {}

    # --- fine -------------------------------------------------------------
    winner_fine = stage4_winners["fine"]
    cfg = winner_fine["cfg"]
    x_pool = pd.concat([fine_ctx.x_train_by_tag[cfg["tag"]], fine_ctx.x_test_by_tag[cfg["tag"]]], ignore_index=True)
    y_pool = np.concatenate([fine_ctx.y_train_enc, fine_ctx.y_test_enc])
    model = fit_component(cfg, MODEL_SPECS, get_builder, x_pool, y_pool)
    # Calibrator fit on out-of-fold predictions of the same pooled rows (Section 1.3j of common);
    # training priors are the pooled class frequencies, for the EM prior re-estimation at inference.
    calibrator = fit_calibrator(cfg, MODEL_SPECS, get_builder, x_pool, y_pool)
    train_priors = np.bincount(y_pool, minlength=len(fine_ctx.label_encoder.classes_)) / len(y_pool)
    logger.notice(f"  [fine] calibrator: temperature={calibrator.temperature_:.3f} "
                  f"(weighting '{cfg.get('weighting')}' undone first)")
    path = VALIDATION_MODELS_DIR / "fine_deployed.joblib"
    joblib.dump({"model": model, "cfg": cfg, "label_encoder": fine_ctx.label_encoder,
                 "calibrator": calibrator, "train_priors": train_priors,
                 "features": list(x_pool.columns),
                 "full_model_name": winner_fine.get("full_model_name"),
                 "node_models": winner_fine.get("node_models")}, path)
    out["fine"] = {"model_path": str(path), "n_pooled_rows": int(len(y_pool)),
                   "full_model_name": winner_fine.get("full_model_name")}
    logger.notice(f"  [fine] deployed -> {path}  (n={len(y_pool)})  {winner_fine.get('full_model_name')}")

    if "merged" in FAMILIES:
        # --- merged -------------------------------------------------------------
        winner = stage4_winners["merged"]
        router_cfg, branch_cfgs = winner["router_cfg"], winner["branch_cfgs"]
        y_super_pool_str = np.concatenate([merged_ctx.y_train_super_str, merged_ctx.y_test_super_str])
        y_fine_pool_str = np.concatenate([merged_ctx.y_train_fine_str, merged_ctx.y_test_fine_str])

        def _pool_tag(ctx, tag, local_meta):
            tr = _m2_resolve_full_length_tag_frame(ctx, tag, local_meta)
            te = _resolve_test_frame_merged(ctx, tag, local_meta)
            return pd.concat([tr, te], ignore_index=True)

        x_router_pool = _pool_tag(merged_ctx, router_cfg["tag"], merged_ctx.router_meta)
        router_model = fit_component(router_cfg, MODEL_SPECS, get_builder, x_router_pool,
                                      SUPER_LE.transform(y_super_pool_str))
        branch_models_deploy = {}
        for sc in SUPERCLASS_ORDER:
            bcfg = branch_cfgs.get(sc)
            if bcfg is None:
                branch_models_deploy[sc] = None
                continue
            sc_mask = y_super_pool_str == sc
            fine_sc = y_fine_pool_str[sc_mask]
            if len(set(fine_sc)) < 2:
                branch_models_deploy[sc] = None
                continue
            from sklearn.preprocessing import LabelEncoder
            le_branch = LabelEncoder().fit(fine_sc)
            local_meta = merged_ctx.branch_meta.get(sc)
            try:
                x_branch_pool = _pool_tag(merged_ctx, bcfg["tag"], local_meta)[sc_mask].reset_index(drop=True)
            except KeyError:
                branch_models_deploy[sc] = None
                continue
            model_b = fit_component(bcfg, MODEL_SPECS, get_builder, x_branch_pool, le_branch.transform(fine_sc))
            branch_models_deploy[sc] = {"model": model_b, "label_encoder": le_branch}
        path = VALIDATION_MODELS_DIR / "merged_deployed.joblib"
        joblib.dump({"router_model": router_model, "router_cfg": router_cfg, "branch_models": branch_models_deploy,
                     "branch_cfgs": branch_cfgs, "fine_label_encoder": merged_ctx.fine_label_encoder,
                     "router_meta": merged_ctx.router_meta, "branch_meta": merged_ctx.branch_meta,
                     "full_model_name": winner.get("full_model_name"),
                     "node_models": winner.get("node_models")}, path)
        out["merged"] = {"model_path": str(path), "n_pooled_rows": int(len(y_super_pool_str)),
                          "full_model_name": winner.get("full_model_name")}
        logger.notice(f"  [merged] deployed -> {path}  (n={len(y_super_pool_str)})  {winner.get('full_model_name')}")

    if "3level" in FAMILIES:
        # --- 3level -------------------------------------------------------------
        winner3 = stage4_winners["3level"]
        gate_cfg, subgate_cfg, nonblazar_cfg = winner3["gate_cfg"], winner3["subgate_cfg"], winner3["nonblazar_cfg"]
        branch_cfgs3 = winner3["branch_cfgs"]
        y_super_pool_str3 = np.concatenate([tl_ctx.y_train_super_str, tl_ctx.y_test_super_str])
        y_fine_pool_str3 = np.concatenate([tl_ctx.y_train_fine_str, tl_ctx.y_test_fine_str])
        blazar_mask_pool = np.isin(y_super_pool_str3, list(BLAZAR_SUPERCLASSES))

        def _pool_tag3(ctx, tag, local_meta):
            tr = (ctx.x_train_by_tag[tag] if tag in ctx.x_train_by_tag
                  else ctx.x_train_by_tag["ALL_FEATURES"][local_meta["feature_sets"][tag]])
            te = _resolve_test_frame_3level(ctx, tag, local_meta)
            return pd.concat([tr, te], ignore_index=True)

        x_gate_pool = _pool_tag3(tl_ctx, gate_cfg["tag"], tl_ctx.gate_meta)
        gate_model = fit_component(gate_cfg, MODEL_SPECS, get_builder, x_gate_pool,
                                    GATE_LE.transform(build_gate_labels(y_super_pool_str3)))
        x_sub_pool = _pool_tag3(tl_ctx, subgate_cfg["tag"], tl_ctx.subgate_meta)
        subgate_model = fit_component(subgate_cfg, MODEL_SPECS, get_builder, x_sub_pool[blazar_mask_pool].reset_index(drop=True),
                                       BLAZAR_SUBGATE_LE.transform(y_super_pool_str3[blazar_mask_pool]))
        x_nb_pool = _pool_tag3(tl_ctx, nonblazar_cfg["tag"], tl_ctx.nonblazar_meta)
        nonblazar_model = fit_component(nonblazar_cfg, MODEL_SPECS, get_builder,
                                         x_nb_pool[~blazar_mask_pool].reset_index(drop=True),
                                         NONBLAZAR_ROUTER_LE.transform(y_super_pool_str3[~blazar_mask_pool]))
        branch_models_deploy3 = {}
        for sc in NONBLAZAR_SUPERCLASS_ORDER:
            bcfg = branch_cfgs3.get(sc)
            if bcfg is None:
                branch_models_deploy3[sc] = None
                continue
            sc_mask = y_super_pool_str3 == sc
            fine_sc = y_fine_pool_str3[sc_mask]
            if len(set(fine_sc)) < 2:
                branch_models_deploy3[sc] = None
                continue
            from sklearn.preprocessing import LabelEncoder
            le_branch = LabelEncoder().fit(fine_sc)
            local_meta = tl_ctx.branch_meta.get(sc)
            try:
                x_branch_pool = _pool_tag3(tl_ctx, bcfg["tag"], local_meta)[sc_mask].reset_index(drop=True)
            except KeyError:
                branch_models_deploy3[sc] = None
                continue
            model_b = fit_component(bcfg, MODEL_SPECS, get_builder, x_branch_pool, le_branch.transform(fine_sc))
            branch_models_deploy3[sc] = {"model": model_b, "label_encoder": le_branch}
        path = VALIDATION_MODELS_DIR / "3level_deployed.joblib"
        joblib.dump({"gate_model": gate_model, "gate_cfg": gate_cfg, "subgate_model": subgate_model,
                     "subgate_cfg": subgate_cfg, "nonblazar_model": nonblazar_model, "nonblazar_cfg": nonblazar_cfg,
                     "branch_models": branch_models_deploy3, "branch_cfgs": branch_cfgs3,
                     "fine_label_encoder": tl_ctx.fine_label_encoder, "gate_meta": tl_ctx.gate_meta,
                     "subgate_meta": tl_ctx.subgate_meta, "nonblazar_meta": tl_ctx.nonblazar_meta,
                     "branch_meta": tl_ctx.branch_meta,
                     "full_model_name": winner3.get("full_model_name"),
                     "node_models": winner3.get("node_models")}, path)
        out["3level"] = {"model_path": str(path), "n_pooled_rows": int(len(y_super_pool_str3)),
                          "full_model_name": winner3.get("full_model_name")}
        logger.notice(f"  [3level] deployed -> {path}  (n={len(y_super_pool_str3)})  {winner3.get('full_model_name')}")

    return out


# %% [markdown]
# ## Section 11 — Stage 8: persistence / report schema
#
# `stage2` is Optional — Section 13's driver no longer calls Stage 2
# automatically (it is now DEAD CODE — see the section at the end of this
# file), so this function must tolerate `stage2=None` (never yet run this
# session) gracefully rather than assuming the dict shape Stage 2 returns.
# Stage 6 (family-vs-family significance testing) is in the DEAD CODE
# section and does not feed the report this function builds.

# %%
def _jsonable_candidate(cand: dict) -> dict:
    # "confusion_matrix" (the full matrix dict, requirement #3) is deliberately
    # dropped here — it's already saved to its own CSV/PNG (`confusion_matrix_path`, kept) under
    # VALIDATION_CONFUSION_DIR, and embedding every candidate's full matrix a second time, inline,
    # in cross_pipeline_validation_report.json would bloat that file for no reader that doesn't
    # already have the CSV path to go read it from.
    return {k: v for k, v in cand.items() if k not in ("model", "label_encoder", "confusion_matrix")}


def _build_stage8_report(stage1, stage2, stage3, stage4, stage5, stage7) -> dict:
    """Pure dict-aggregation half of `run_stage8` — reads
    ONLY the already-computed stage1-7 dicts, no model fitting, no I/O.
    Split out so it can be called again cheaply on a `stage8` checkpoint
    HIT (see `_refresh_stage8_report`) to pick up any upstream backfill."""

    # Family-level Borda ranking — same criteria as Stage 4, across the 3 families, on the Stage-4
    # winners' TRAINING-side (Stage 3) scores, never on Stage 5's test scores: choosing the best of
    # three on the test set would bias the reported test score of the chosen family upward.
    def _pick_family(winners: dict) -> list:
        rows = [{"family": fam, "val_metrics": winners[fam]["val_metrics"],
                 "train_metrics": winners[fam]["train_metrics"]} for fam in FAMILIES]
        return _borda_rank(rows, _borda_criteria_from_metrics())

    family_ranked = _pick_family(stage4)
    corrected_winner_family = family_ranked[0]["family"]
    per_split_selection = None
    if STAGE4_WINNERS_BY_SPLIT:
        # The whole procedure, as run inside each split: pick the family on that split's training
        # side, score it on that split's test set. The mean is the procedure's performance estimate.
        fam_by_split = {sid: _pick_family(w)[0]["family"] for sid, w in STAGE4_WINNERS_BY_SPLIT.items()}
        scores = {sid: stage5[fam]["per_split_scores"].get(sid) for sid, fam in fam_by_split.items()}
        per_split_selection = {"family_by_split": fam_by_split, "test_score_by_split": scores,
                               "procedure_test_ci": _repeated_cv_ci(list(scores.values())),
                               "selection_metric": SELECTION_METRIC}
    family_borda_ranking = [{"family": r["family"], "borda_score": r["borda_score"],
                              "val_metrics": r["val_metrics"], "train_metrics": r["train_metrics"],
                              "overfit_gap": _overfit_gap(r)} for r in family_ranked]

    stage2_section = stage2 if stage2 else {
        "status": "not_run",
        "note": ("Stage 2 (winner's-curse nested-CV diagnostic) is in the DEAD CODE section — see "
                 "the DEAD CODE section at the end of this file. Flip RUN_WINNERS_CURSE_DIAGNOSTIC_NOW "
                 "there to True and run that cell to populate this section, then re-run Stage 8 "
                 "(FORCE_RECOMPUTE_STAGES={'stage8'}) to fold it into this report."),
    }

    report = {
        "selection_metric": SELECTION_METRIC, "version_tag": VERSION_TAG, "seed": SEED,
        "validation_splits": VALIDATION_SPLITS, "n_validation_splits_requested": N_VALIDATION_SPLITS,
        "fine_setup_used": FINE_SETUP_KEY_VALIDATION,
        "stage1_shortlists": {
            fam: [{**_jsonable_candidate(c)} for c in stage1[fam]] for fam in FAMILIES
        },
        "stage2_optimism_diagnostic": stage2_section,
        "stage3_cross_split_scores": {
            # `val_metrics_std`/`train_metrics_std` ("with error")
            # were being computed for the CSV export only (`_stage3_row_to_csv_row`) but silently
            # dropped from this JSON report — every Stage-3 candidate's row now carries them as a
            # first-class field (see the row-construction loops above), so they're included here too,
            # not just for the Stage-4/5 winner.
            fam: [{"candidate": (r.get("combo_name") or r["cfg"]["model_name"]),
                   "full_model_name": r.get("full_model_name"), "node_models": r.get("node_models"),
                   "per_split_scores": r["per_split_scores"], "cross_split_mean": r["cross_split_mean"],
                   "n_valid_splits": r["n_valid_splits"], "val_metrics": r["val_metrics"],
                   "train_metrics": r["train_metrics"],
                   "val_metrics_std": r.get("val_metrics_std"), "train_metrics_std": r.get("train_metrics_std"),
                   "borda_score": r.get("borda_score"),
                   "confusion_matrix_path": r.get("confusion_matrix_path")} for r in stage3[fam]]
            for fam in FAMILIES
        },
        "stage3_shortlist_reduction": {
            "reduce_stage3_candidates": REDUCE_STAGE3_CANDIDATES,
            "stage3_shortlist_top_k": STAGE3_SHORTLIST_TOP_K if REDUCE_STAGE3_CANDIDATES else None,
        },
        "stage4_selected_architecture": {
            fam: _jsonable_candidate(stage4[fam]) for fam in FAMILIES
        },
        "stage5_test_scores": {
            fam: {"per_split_scores": stage5[fam]["per_split_scores"], "ci": stage5[fam]["ci"],
                  "test_full_ci": stage5[fam].get("test_full_ci"),
                  "per_split_test_ci": stage5[fam].get("per_split_test_ci"),
                  "winner_train_metrics_ci": stage5[fam].get("winner_train_metrics_ci"),
                  "winner_validation_metrics_ci": stage5[fam].get("winner_validation_metrics_ci"),
                  "winner_test_metrics_ci": stage5[fam].get("winner_test_metrics_ci"),
                  "val_metrics": stage5[fam]["val_metrics"], "train_metrics": stage5[fam]["train_metrics"],
                  "full_model_name": stage5[fam].get("full_model_name"),
                  "node_models": stage5[fam].get("node_models"),
                  "overfit_gap": _overfit_gap(stage5[fam]),
                  # Winner train/validation/test metrics with error
                  # (mean+std, every REPORT_METRICS column) plus the two saved confusion-matrix paths.
                  "winner_train_metrics_mean": stage5[fam].get("winner_train_metrics_mean"),
                  "winner_train_metrics_std": stage5[fam].get("winner_train_metrics_std"),
                  "winner_validation_metrics_mean": stage5[fam].get("winner_validation_metrics_mean"),
                  "winner_validation_metrics_std": stage5[fam].get("winner_validation_metrics_std"),
                  "winner_test_metrics_mean": stage5[fam].get("winner_test_metrics_mean"),
                  "winner_test_metrics_std": stage5[fam].get("winner_test_metrics_std"),
                  "confusion_matrix_validation_path": stage5[fam].get("confusion_matrix_validation_path"),
                  "confusion_matrix_test_path": stage5[fam].get("confusion_matrix_test_path")}
            for fam in FAMILIES
        },
        "stage7_deployment": stage7,
        "corrected_winner_family": corrected_winner_family,
        "per_split_selection": per_split_selection,
        "evaluation_design": (
            f"Per-split selection over {len(VALIDATION_SPLITS)} splits: model search and Stages 1-4 ran inside "
            "each split on its own training data, and each split's winners were scored on that split's test "
            "set, so no selection step read rows overlapping a test set. Stage-5 means (per family) and "
            "per_split_selection.procedure_test_ci (family also chosen inside each split) estimate the "
            "performance of the selection procedure; CIs use the Nadeau-Bengio correction. The deployed "
            "models are split 0's selection, refit on split 0's train+test."
            if STAGE4_WINNERS_BY_SPLIT else
            "Single split: search and selection used only its training data and the family was chosen on "
            "Stage-3 training-side scores, so the Stage-5 test score is untouched by selection (no CI: n=1)."
            if len(VALIDATION_SPLITS) == 1 else
            f"Shared selection over {len(VALIDATION_SPLITS)} splits (legacy): search ran on split 0 and Stage 3 "
            "scored every split's training data; each split's test set overlaps that data, so Stage-5 test "
            "scores are optimistic. The family was chosen on Stage-3 training-side scores."),
        "family_borda_ranking": family_borda_ranking,
        "family_identity": FAMILY_LABELS,
        "three_way_role_disclosure": (
            "Stage 1-4 are model-SELECTION only (never read test data). Stage 5 is ESTIMATION on the "
            "held-out test partition(s) (see evaluation_design for which rows selection could have seen). "
            "Stage 7 is DEPLOYMENT: a final refit on more data than any evaluation step used, never itself "
            "re-evaluated. corrected_winner_family is chosen on the Stage-3 TRAINING-side scores, never on "
            "Stage 5's test scores. Stage 6 (family-vs-family significance testing) is in the DEAD CODE "
            "section and does not contribute to this report."
        ),
        "ranking_methodology_note": (
            "Both the Stage 4 per-family architecture selection and this corrected_winner_family "
            "are chosen by Borda (average-rank) combination of 4 equally-weighted criteria — accuracy, "
            "bal_accuracy, macro_f1, kappa (each: higher rank is better) and the train-val gap on those "
            "same 4 metrics, clipped at 0 (lower rank is better, i.e. less overfitting) — rather than a "
            "raw weighted average or a single-metric argmax. Ranks (not raw metric values) are averaged "
            "because these metrics are not on a common scale (kappa is chance-corrected; accuracy is not "
            "balanced-class-aware) and are mutually correlated by construction, so naive raw-value "
            "averaging would silently overweight whichever metric has the widest spread — a scale-free "
            "rank average (a Borda count; Dwork et al. 2001) avoids this. The train-val gap uses in-sample "
            "training predictions; it is a deliberate penalty on overfitting, a stated preference rather "
            "than a published selection rule. Plain "
            "accuracy is included as a 4th ranking criterion (a deliberate choice; it is often left "
            "diagnostic-only under class imbalance); the raw metrics are kept verbatim in "
            "stage3_cross_split_scores/stage5_test_scores, so this choice is auditable against the "
            "un-ranked raw metric regardless."
        ),
        "cost_control_note": (
            "Stage 1/3's chain-CV candidate scoring shares one node-fit cache per (context, seed) "
            "[Stage 1] or per split [Stage 3] across every candidate scored in that sweep, since "
            "StratifiedKFold's fold boundaries depend only on (labels, n_splits, seed) — never on a "
            "candidate's config — so two candidates that happen to share an identical router/branch/gate/"
            "subgate/nonblazar_router config get that fit only once; this changes nothing about what is "
            "fit or scored. Separately, when REDUCE_STAGE3_CANDIDATES=True (see "
            "stage3_shortlist_reduction above), Stage 3 only re-scores the top STAGE3_SHORTLIST_TOP_K of "
            "Stage 1's already-Borda-ranked shortlist per family, rather than the full shortlist — a "
            "breadth trim on top of already-ranked candidates, not a change to any combo-grid width, "
            "repeat count, or fold count."
        ),
    }
    return report


def _persist_stage8_report(report: dict, stage5: dict, logger) -> None:
    """I/O half of `run_stage8` — writes
    `cross_pipeline_validation_report.json` and the per-family split-0 OOF
    `.npz` files. Split out so `_refresh_stage8_report` can call it again on
    a checkpoint hit without duplicating this logic."""
    report_path = VALIDATION_DIR / "cross_pipeline_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.notice(f"  Saved -> {report_path}")

    for fam in FAMILIES:
        np.savez(VALIDATION_RESULTS_DIR / f"{fam}_split0_test_oof.npz",
                  y_true=stage5[fam]["predictions_by_split"].get(SPLIT0, (np.array([]), np.array([])))[0],
                  y_pred=stage5[fam]["predictions_by_split"].get(SPLIT0, (np.array([]), np.array([])))[1])
    logger.notice(f"  Per-family split-0 OOF .npz files -> {VALIDATION_RESULTS_DIR}")


def run_stage8(stage1, stage2, stage3, stage4, stage5, stage7) -> dict:
    logger = get_section_logger("stage8_persistence", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSTAGE 8 — persistence / report schema\n{'=' * 70}")
    report = _build_stage8_report(stage1, stage2, stage3, stage4, stage5, stage7)
    _persist_stage8_report(report, stage5, logger)
    _summarize_stage8(report)
    return report


def _summarize_stage8(report: dict) -> None:
    logger = get_section_logger("stage8_persistence", VALIDATION_LOGS_DIR)
    logger.notice(f"  Corrected winner family (Stage 5-sourced, Borda-ranked): {report['corrected_winner_family']}")
    for r in report["family_borda_ranking"]:
        logger.notice(f"    [{r['family']:7s}] borda_score={r['borda_score']:.3f}  "
                       f"overfit_gap={r['overfit_gap']:.4f}  val_metrics={r['val_metrics']}")
    logger.notice(f"  Report -> {VALIDATION_DIR / 'cross_pipeline_validation_report.json'}")


def _refresh_stage8_report(cached_report: dict) -> dict:
    """Used as `stage8`'s `summarize_fn` (checkpoint-
    HIT path only). `run_stage8` is pure aggregation of stage1-7's
    already-computed results — cheap to redo in full — so on a `stage8`
    cache hit this rebuilds `cross_pipeline_validation_report.json` from the
    LIVE `STAGE1_RESULTS`/`STAGE3_RESULTS`/.../`STAGE7_DEPLOYMENT` globals
    (which may have just been backfilled in place by an upstream stage's own
    `summarize_fn`, e.g. `_backfill_stage3_metric_std`) and rewrites it to
    disk, rather than leaving the file frozen at whatever content the
    ORIGINAL `run_stage8` call happened to write. Returns the fresh report
    so `_run_cached` uses it (see that function's docstring) instead of the
    stale cached one for the rest of this run too."""
    logger = get_section_logger("stage8_persistence", VALIDATION_LOGS_DIR)
    report = _build_stage8_report(STAGE1_RESULTS, STAGE2_RESULTS, STAGE3_RESULTS, STAGE4_WINNERS,
                                   STAGE5_RESULTS, STAGE7_DEPLOYMENT)
    _persist_stage8_report(report, STAGE5_RESULTS, logger)
    _summarize_stage8(report)
    return report


# %% [markdown]
# ## Section 12 — CSV report export [reporting] — requirement #1
#
# Three CSV deliverables per family, sourced entirely from Stage 1/3/4's
# already-computed rows plus each sibling module's own persisted split-0
# leaderboard/branch-results JSON — no new model fitting:
#
# 1. **`all_models_validated_{fam}.csv`** (requirement #1) — every Stage-3
#    cross-split-scored candidate (not just the Stage-4 winner), one row
#    each, with every `REPORT_METRICS` column's cross-split MEAN and STD
#    ("with error") on both the train and val side, plus its contracted
#    `full_model_name`/`node_models` (requirement #1a) and Borda score.
# 2. **`per_node_leaderboards/*.csv`** (requirement #1b, merged/3level
#    only) — one CSV per hierarchy node (router, each branch; or gate,
#    blazar_subgate, nonblazar_router, each branch), dumped straight from
#    that sibling module's own split-0 leaderboard/branch-results JSON —
#    the node-level candidates it screened before Stage 1 ever built a
#    whole-system combo out of them.
# 3. **`end_to_end_performance_{fam}.csv`** (requirement #1c) — a slimmer,
#    presentation-ready ranking of the same Stage-3 candidates as (1),
#    keyed by `full_model_name`, flagging which row Stage 4 actually
#    selected (`is_selected_winner`).
#
# Checkpointed as `"stage8b_csv"` (Section 0's `STAGE_DEPENDENCIES`) —
# cheap to force independently of Stage 8's own JSON report.

# %%
def _report_metrics_row(prefix: str, mean_dict: dict, std_dict: Optional[dict] = None) -> dict:
    row = {}
    for m in REPORT_METRICS:
        row[f"{prefix}_{m}_mean"] = (mean_dict or {}).get(m, float("nan"))
        row[f"{prefix}_{m}_std"] = (std_dict or {}).get(m, float("nan"))
    return row


def _stage3_row_to_csv_row(fam: str, r: dict) -> dict:
    # `val_metrics_std`/`train_metrics_std` are now materialized directly on the row
    # by `run_stage3` itself (see its row-construction loops) — falls back to recomputing from the
    # raw `per_split_*_metrics` dicts only for an older checkpoint without them.
    val_std = r.get("val_metrics_std")
    if val_std is None:
        per_split_val = r.get("per_split_val_metrics") or {}
        val_std = _std_metric_dicts(list(per_split_val.values()), metrics=REPORT_METRICS) if per_split_val else {}
    train_std = r.get("train_metrics_std")
    if train_std is None:
        per_split_train = r.get("per_split_train_metrics") or {}
        train_std = _std_metric_dicts(list(per_split_train.values()), metrics=REPORT_METRICS) if per_split_train else {}
    row = {
        "family": fam, "candidate": r.get("combo_name") or r["cfg"]["model_name"],
        "full_model_name": r.get("full_model_name"),
        "node_models": json.dumps(r.get("node_models") or {}),
        "cross_split_mean": r.get("cross_split_mean"), "n_valid_splits": r.get("n_valid_splits"),
        "borda_score": r.get("borda_score"), "confusion_matrix_path": r.get("confusion_matrix_path"),
    }
    row.update(_report_metrics_row("val", r.get("val_metrics") or {}, val_std))
    row.update(_report_metrics_row("train", r.get("train_metrics") or {}, train_std))
    return row


def _stage1_row_to_csv_row(fam: str, r: dict) -> dict:
    row = {
        "family": fam, "candidate": r.get("combo_name") or r["cfg"]["model_name"],
        "full_model_name": r.get("full_model_name"),
        "node_models": json.dumps(r.get("node_models") or {}),
        "stage1_mean": r.get("stage1_mean"), "stage1_std": r.get("stage1_std"),
        "n_successful_repeats": r.get("n_successful_repeats"), "borda_score": r.get("borda_score"),
        "confusion_matrix_path": r.get("confusion_matrix_path"),
    }
    row.update(_report_metrics_row("val", r.get("val_metrics") or {}, r.get("val_metrics_std") or {}))
    row.update(_report_metrics_row("train", r.get("train_metrics") or {}, r.get("train_metrics_std") or {}))
    return row


def export_all_models_shortlisted_csv(stage1_results: dict) -> dict:
    """"stage 1/3 models must be saved with
    all the validation and train metrics with the errors, not only the
    winner" — the Stage-1 half of that: every shortlisted candidate (not
    just whichever ones survive Stage 3's breadth trim, and never just the
    eventual Stage-4 winner), with its full train/validation
    REPORT_METRICS, mean±std across the N_REPEATS_SHORTLIST_CV repeats,
    plus its saved confusion-matrix path (requirement #3). Mirrors
    `export_all_models_validated_csv` below, which does the equivalent for
    Stage 3's cross-split-confirmed candidates."""
    logger = get_section_logger("stage8b_csv", VALIDATION_LOGS_DIR)
    out_paths = {}
    for fam in FAMILIES:
        rows = [_stage1_row_to_csv_row(fam, r) for r in stage1_results[fam]]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("stage1_mean", ascending=False)
        path = VALIDATION_CSV_DIR / f"all_models_shortlisted_{fam}.csv"
        df.to_csv(path, index=False)
        out_paths[fam] = str(path)
        logger.notice(f"  [{fam}] {len(df)} Stage-1 shortlisted candidate(s), every metric with "
                       f"repeat-to-repeat error -> {path}")
    return out_paths


def export_all_models_validated_csv(stage3_results: dict) -> dict:
    logger = get_section_logger("stage8b_csv", VALIDATION_LOGS_DIR)
    out_paths = {}
    for fam in FAMILIES:
        rows = [_stage3_row_to_csv_row(fam, r) for r in stage3_results[fam]]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("cross_split_mean", ascending=False)
        path = VALIDATION_CSV_DIR / f"all_models_validated_{fam}.csv"
        df.to_csv(path, index=False)
        out_paths[fam] = str(path)
        logger.notice(f"  [{fam}] {len(df)} validated candidate(s), every metric with cross-split "
                       f"error -> {path}")
    return out_paths


def _candidate_identity(fam: str, r: dict) -> tuple:
    """A stable, config-based identity for a Stage-3 row — used only to
    flag which row Stage 4 selected in the end-to-end CSV below, since a
    row reloaded from a checkpoint is a different Python object than the
    one `run_stage4` returns even when it represents the same candidate."""
    if fam == "fine":
        return ("fine", _cfg_key(r["cfg"]))
    if fam == "merged":
        branch_key = tuple(sorted((sc, _cfg_key(c)) for sc, c in r["branch_cfgs"].items()))
        return ("merged", _cfg_key(r["router_cfg"]), branch_key)
    branch_key = tuple(sorted((sc, _cfg_key(c)) for sc, c in r["branch_cfgs"].items()))
    return ("3level", _cfg_key(r["gate_cfg"]), _cfg_key(r["subgate_cfg"]), _cfg_key(r["nonblazar_cfg"]), branch_key)


def export_end_to_end_performance_csv(stage3_results: dict, stage4_winners: dict) -> dict:
    logger = get_section_logger("stage8b_csv", VALIDATION_LOGS_DIR)
    out_paths = {}
    for fam in FAMILIES:
        winner_identity = _candidate_identity(fam, stage4_winners[fam])
        rows = []
        for r in stage3_results[fam]:
            row = {
                "family": fam, "full_model_name": r.get("full_model_name"),
                "is_selected_winner": _candidate_identity(fam, r) == winner_identity,
                "cross_split_mean": r.get("cross_split_mean"), "n_valid_splits": r.get("n_valid_splits"),
                "borda_score": r.get("borda_score"), "confusion_matrix_path": r.get("confusion_matrix_path"),
            }
            for m in QUALITY_METRICS:
                row[f"val_{m}_mean"] = (r.get("val_metrics") or {}).get(m, float("nan"))
                row[f"train_{m}_mean"] = (r.get("train_metrics") or {}).get(m, float("nan"))
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("cross_split_mean", ascending=False)
        path = VALIDATION_CSV_DIR / f"end_to_end_performance_{fam}.csv"
        df.to_csv(path, index=False)
        out_paths[fam] = str(path)
        n_winner = int(df["is_selected_winner"].sum()) if not df.empty else 0
        logger.notice(f"  [{fam}] end-to-end performance CSV ({len(df)} row(s), "
                       f"{n_winner} flagged as selected winner) -> {path}")
    return out_paths


def export_per_node_leaderboards_csv() -> dict:
    """Dumps each hierarchy node's OWN split-0
    leaderboard/branch-results — exactly what `fermi_merged.py`/
    `fermi_3levels.py` already persisted, before Stage 1 ever combined
    node-level candidates into a whole-system combo — straight to CSV. No
    re-scoring; this reads the identical files `_stage1_merged_candidates`/
    `_stage1_3level_candidates` already parse.

    The merged branch loop below now iterates
    `NONBLAZAR_SUPERCLASS_ORDER`, not the full `SUPERCLASS_ORDER` — BLAZAR_HE/
    BLAZAR_LE collapse to exactly one fine class after the physical fsrq/ssrq/css
    merge (fermi_processing.py), so `_m2_build_branch_datasets`'s own
    `len(set(fine_tr)) < 2` guard means `fermi_merged.py` never fits or
    persists a branch model for them — structurally identical to why the
    3-level loop right below only ever iterated `NONBLAZAR_SUPERCLASS_ORDER`.
    The old full-order loop hit this expected absence every run and logged it
    as a WARNING; it's now skipped without noise."""
    logger = get_section_logger("stage8b_csv", VALIDATION_LOGS_DIR)
    out_paths = {}

    with open(SPLIT0_DIR / "merged_classification_metrics.json") as f:
        mm = json.load(f)
    router_lb = pd.DataFrame(mm["level1_leaderboard"])
    p = VALIDATION_CSV_NODES_DIR / "merged_router_leaderboard.csv"
    router_lb.to_csv(p, index=False)
    out_paths["merged_router"] = str(p)
    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        bp = SPLIT0_DIR / "results" / f"{_safe_filename(f'branch_{sc}')}.json"
        if not bp.exists():
            logger.warning(f"  ⚠ [merged|branch_{sc}] {bp} not found — skipping its leaderboard CSV.")
            continue
        df = results_to_dataframe(bp)
        p = VALIDATION_CSV_NODES_DIR / f"merged_branch_{sc}_leaderboard.csv"
        df.to_csv(p, index=False)
        out_paths[f"merged_branch_{sc}"] = str(p)

    with open(SPLIT0_DIR / "3level_classification_metrics.json") as f:
        mm3 = json.load(f)
    for node_key, lb_key in (("gate", "level1_gate_leaderboard"),
                              ("blazar_subgate", "level2_blazar_subgate_leaderboard"),
                              ("nonblazar_router", "level2_nonblazar_router_leaderboard")):
        df = pd.DataFrame(mm3[lb_key])
        p = VALIDATION_CSV_NODES_DIR / f"3level_{node_key}_leaderboard.csv"
        df.to_csv(p, index=False)
        out_paths[f"3level_{node_key}"] = str(p)
    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        bp = SPLIT0_DIR / "results" / f"{_safe_filename(f'branch_{sc}_L3')}.json"
        if not bp.exists():
            logger.warning(f"  ⚠ [3level|branch_{sc}_L3] {bp} not found — skipping its leaderboard CSV.")
            continue
        df = results_to_dataframe(bp)
        p = VALIDATION_CSV_NODES_DIR / f"3level_branch_{sc}_leaderboard.csv"
        df.to_csv(p, index=False)
        out_paths[f"3level_branch_{sc}"] = str(p)

    logger.notice(f"  Per-node leaderboard CSVs ({len(out_paths)} file(s)) -> {VALIDATION_CSV_NODES_DIR}")
    return out_paths


def _summarize_stage8b_csv(out: dict) -> None:
    logger = get_section_logger("stage8b_csv", VALIDATION_LOGS_DIR)
    for key, paths in out.items():
        if isinstance(paths, dict):
            for sub_key, p in paths.items():
                logger.notice(f"  [{key}|{sub_key}] -> {p}")
        else:
            logger.notice(f"  [{key}] -> {paths}")
    logger.notice(f"  All Stage 8b CSVs written under {VALIDATION_CSV_DIR}")


def run_stage8b_csv(stage1_results: dict, stage3_results: dict, stage4_winners: dict) -> dict:
    logger = get_section_logger("stage8b_csv", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSTAGE 8b — CSV report export [requirements #1/#2]\n{'=' * 70}")
    out = {
        "all_models_shortlisted": export_all_models_shortlisted_csv(stage1_results),
        "all_models_validated": export_all_models_validated_csv(stage3_results),
        "end_to_end_performance": export_end_to_end_performance_csv(stage3_results, stage4_winners),
        "per_node_leaderboards": export_per_node_leaderboards_csv() if FAMILIES != ["fine"] else {},
    }
    logger.notice(f"  All Stage 8b CSVs written under {VALIDATION_CSV_DIR}")
    return out


def _refresh_stage8b_csv(cached_out: dict) -> dict:
    """Used as `stage8b_csv`'s `summarize_fn`
    (checkpoint-HIT path only) — same rationale as `_refresh_stage8_report`.
    `run_stage8b_csv`'s exporters (`export_all_models_shortlisted_csv` etc.)
    only ever read already-computed Stage-1/3/4 rows straight off the live
    `STAGE1_RESULTS`/`STAGE3_RESULTS`/`STAGE4_WINNERS` globals and write
    CSVs — no new model fitting — so re-running it on a `stage8b_csv` cache
    hit is cheap and picks up any upstream backfill (e.g. the
    Stage-3 std backfill) instead of leaving `all_models_validated_{fam}.csv`
    etc. frozen at their previously-exported content."""
    out = run_stage8b_csv(STAGE1_RESULTS, STAGE3_RESULTS, STAGE4_WINNERS)
    _summarize_stage8b_csv(out)
    return out


# %% [markdown]
# ## Section 13 — Driver
#
# Every stage runs through `_run_cached` (Section 0) — each one loads
# straight from `VALIDATION_CHECKPOINTS_DIR` instead of recomputing if a
# valid checkpoint from an earlier run of this script already exists. A
# crash, a Ctrl-C, or a deliberate stop partway through therefore never
# loses whatever finished. To force specific stages to recompute anyway
# (e.g. after a bug fix), list them in `FORCE_RECOMPUTE_STAGES` near the
# top of Section 0 — that also cascades to every stage downstream of the
# ones you list (see `STAGE_DEPENDENCIES`).
#
# **Stage 2 and Stage 6 are deliberately NOT part of this driver** — both
# are DEAD CODE (see the section at the end of this file); this driver
# proceeds straight from Stage 1 to Stage 3, and from Stage 5 to Stage 7.
# Stage 9 (unassociated-source inference) isn't part of this driver either
# any more — it now lives in the standalone `fermi_inference.py`,
# meant to be run after this file.

# %%
_module_logger.notice("\nBuilding per-split contexts (fine/merged/3level) for every validation split...")


def _build_split_contexts():
    contexts = {}
    for _split_id in pbar(VALIDATION_SPLITS, desc="Building contexts"):
        contexts[_split_id] = _val_build_contexts(_split_id)
    return contexts


def _summarize_split_contexts(contexts: dict) -> None:
    """Split-context building has no
    `run_stageN` of its own to print an inline summary, so this is the
    checkpoint-hit summary for it: which splits were loaded and each
    split's known train/test row counts (from the multi-split manifest —
    Section 0)."""
    for split_id in VALIDATION_SPLITS:
        counts = SPLIT_ROW_COUNTS.get(split_id, {})
        _module_logger.notice(f"  [{split_id}] contexts loaded (fine/merged/3level)  "
                               f"n_train={counts.get('n_train')}  n_test={counts.get('n_test')}")


SPLIT_CONTEXTS = _run_cached("split_contexts", _build_split_contexts, summarize_fn=_summarize_split_contexts)
FINE_CTX_SPLIT0, MERGED_CTX_SPLIT0, THREELEVEL_CTX_SPLIT0 = SPLIT_CONTEXTS[SPLIT0]

# Stage 2 intentionally NOT run here — see the DEAD CODE section. STAGE2_RESULTS starts as None
# every fresh run of this driver; run_stage8 already tolerates stage2=None gracefully.
STAGE2_RESULTS = None
if PER_SPLIT_SELECTION:
    # Stages 1-4 inside every split, on that split's own leaderboards and training data only.
    # Split 0's results also stand in as STAGE1/3/4 for the CSV exports, deployment and report.
    # Other splits' confusion-matrix files go to a per-split subfolder so they don't overwrite split 0's.
    STAGE4_WINNERS_BY_SPLIT, _s1_by_split, _s3_by_split = {}, {}, {}
    _cm_dirs = (VALIDATION_CONFUSION_DIR, VALIDATION_CONFUSION_PLOTS_DIR)
    for _sid in VALIDATION_SPLITS:
        _module_logger.notice(f"\n### Per-split selection — {_sid} ###")
        VALIDATION_CONFUSION_DIR, VALIDATION_CONFUSION_PLOTS_DIR = (
            _cm_dirs if _sid == SPLIT0 else tuple(d / _sid for d in _cm_dirs))
        _s1_by_split[_sid] = _run_cached(f"stage1@{_sid}", lambda: run_stage1(
            *SPLIT_CONTEXTS[_sid], split_dir=OUTPUT_ROOT / _sid), summarize_fn=_summarize_stage1)
        _s3_by_split[_sid] = _run_cached(f"stage3@{_sid}", lambda: run_stage3(
            _s1_by_split[_sid], SPLIT_CONTEXTS, splits=[_sid]), summarize_fn=_summarize_stage3)
        STAGE4_WINNERS_BY_SPLIT[_sid] = _run_cached(f"stage4@{_sid}", lambda: run_stage4(_s3_by_split[_sid]),
                                                    summarize_fn=_summarize_stage4)
    VALIDATION_CONFUSION_DIR, VALIDATION_CONFUSION_PLOTS_DIR = _cm_dirs
    STAGE1_RESULTS, STAGE3_RESULTS = _s1_by_split[SPLIT0], _s3_by_split[SPLIT0]
    STAGE4_WINNERS = STAGE4_WINNERS_BY_SPLIT[SPLIT0]
else:
    STAGE1_RESULTS = _run_cached("stage1", lambda: run_stage1(FINE_CTX_SPLIT0, MERGED_CTX_SPLIT0,
                                                              THREELEVEL_CTX_SPLIT0),
                                  summarize_fn=_summarize_stage1)
    STAGE3_RESULTS = _run_cached("stage3", lambda: run_stage3(STAGE1_RESULTS, SPLIT_CONTEXTS),
                                  summarize_fn=_summarize_stage3)
    STAGE4_WINNERS = _run_cached("stage4", lambda: run_stage4(STAGE3_RESULTS), summarize_fn=_summarize_stage4)
STAGE5_RESULTS = _run_cached("stage5", lambda: run_stage5(STAGE4_WINNERS, SPLIT_CONTEXTS),
                              summarize_fn=_summarize_stage5)
STAGE7_DEPLOYMENT = _run_cached("stage7", lambda: run_stage7(STAGE4_WINNERS, SPLIT_CONTEXTS),
                                 summarize_fn=_summarize_stage7)
FINAL_REPORT = _run_cached("stage8", lambda: run_stage8(
    STAGE1_RESULTS, STAGE2_RESULTS, STAGE3_RESULTS, STAGE4_WINNERS,
    STAGE5_RESULTS, STAGE7_DEPLOYMENT), summarize_fn=_refresh_stage8_report)
STAGE8B_CSV_PATHS = _run_cached("stage8b_csv", lambda: run_stage8b_csv(STAGE1_RESULTS, STAGE3_RESULTS, STAGE4_WINNERS),
                                 summarize_fn=_refresh_stage8b_csv)

# %%
print("=" * 78)
print("Module 4 (Cross-Pipeline Validation) — COMPLETE.")
print(f"  Validation splits used : {VALIDATION_SPLITS}")
print(f"  Corrected winner family: {FINAL_REPORT['corrected_winner_family']}")
for _fam in FAMILIES:
    _ci = STAGE5_RESULTS[_fam]["ci"]
    print(f"  [{_fam:7s}] test {SELECTION_METRIC} = {_ci['mean']:.4f}  "
          f"95% CI [{_ci.get('ci_low', float('nan')):.4f}, {_ci.get('ci_high', float('nan')):.4f}]  (n={_ci['n']})  "
          f"{STAGE5_RESULTS[_fam].get('full_model_name')}")
print(f"  Report -> {VALIDATION_DIR / 'cross_pipeline_validation_report.json'}")
print(f"  CSV reports -> {VALIDATION_CSV_DIR}")
print(f"  Deployed models -> {VALIDATION_MODELS_DIR}")
print("=" * 78)


# %% [markdown]
# ## Section 14 — Full train / validation / test metric report, WITH confidence intervals [reporting] — requirement #2
#
# `train_*` = Stage 5's own refit-on-full-split-train, evaluated on that
# SAME training data (memorization diagnostic — not a generalization
# estimate). `validation_*` = Stage 3's cross-split, held-out-fold CV on
# TRAINING rows only (never touches test) — this is what Stage 4 actually
# selected the winning architecture on. `test_*` = Stage 5's held-out
# test partition(s) (see Section 8 for what selection could have seen). Every one
# of the 6 `REPORT_METRICS` columns gets this treatment at all 3 stages.
#
# **"Is it better CI or error?" — both, computed the same way, everywhere
# in this report.** Every `{stage}_{metric}` estimate in this table is a
# mean over `n` (up to `N_VALIDATION_SPLITS`=5) random re-splits of ONE
# labelled pool, which share rows — so the standard error is the Nadeau-
# Bengio corrected `std * sqrt(1/n + n_test/n_train)` (`_nb_sem`), not the
# naive `std / sqrt(n)` that assumes independent splits; and with
# n this small (5, occasionally fewer if a split failed) a normal-
# approximation CI is unreliable — so this table uses the Student-t
# distribution with `df = n-1` (`_metric_ci_across_splits` below,
# literally the same math `_repeated_cv_ci` already uses for the single
# `SELECTION_METRIC` scalar in Stage 5/6, just applied to every
# `REPORT_METRICS` column instead of one). The plain "±std" ("error") is
# also reported alongside, because it answers a different question: std
# describes how much an individual split's score varies, while the CI
# describes how precisely the MEAN across splits is known — both are
# useful, and neither substitutes for the other. This is deliberately
# NOT the bootstrap-CI approach `_pooled_oof_metrics`-style pooled-OOF
# estimates would use elsewhere in this pipeline (bootstrap resampling of
# one fixed set of predictions) — that CI answers "how much would this
# score vary if I re-sampled these same rows", a different and, for a
# 5-independent-split design, less directly relevant question than "how
# consistent was this architecture's performance across genuinely
# different splits", which is what Stage 3/5's per-split scores measure
# and what the t-distribution CI below is built to answer.

# %%
def build_train_val_test_report(stage4_winners: dict, stage5_results: dict, corrected_winner_family: str) -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        s4, s5, ci = stage4_winners[fam], stage5_results[fam], stage5_results[fam]["ci"]
        train_ci = _metric_ci_across_splits(s5.get("per_split_train_metrics") or {},
                                            per_split_ci=s5.get("per_split_train_ci"))
        validation_ci = _metric_ci_across_splits(s4.get("per_split_val_metrics") or {},
                                                 per_split_ci=s4.get("per_split_val_ci"))
        test_ci = _metric_ci_across_splits(s5.get("per_split_val_metrics") or {},
                                           per_split_ci=s5.get("per_split_test_ci"))
        row = {
            "family": fam, "family_label": FAMILY_LABELS[fam],
            "full_model_name": s5.get("full_model_name"),
            "is_corrected_winner": fam == corrected_winner_family,
            f"{SELECTION_METRIC}_validation_cv": s4["cross_split_mean"],
            f"{SELECTION_METRIC}_test": ci["mean"],
            f"{SELECTION_METRIC}_test_ci_low": ci.get("ci_low", float("nan")),
            f"{SELECTION_METRIC}_test_ci_high": ci.get("ci_high", float("nan")),
            "test_n_splits": ci["n"],
        }
        for m in REPORT_METRICS:
            for stage_label, stage_ci in (("train", train_ci), ("validation", validation_ci), ("test", test_ci)):
                mci = stage_ci.get(m, {})
                row[f"{stage_label}_{m}_mean"] = mci.get("mean", float("nan"))
                row[f"{stage_label}_{m}_std"] = mci.get("std", float("nan"))
                row[f"{stage_label}_{m}_ci_low"] = mci.get("ci_low", float("nan"))
                row[f"{stage_label}_{m}_ci_high"] = mci.get("ci_high", float("nan"))
                row[f"{stage_label}_{m}_n"] = mci.get("n", 0)
                row[f"{stage_label}_{m}_ci_method"] = mci.get("method", "none")
        row["overfit_gap_train_minus_validation"] = _overfit_gap({"train_metrics": s4["train_metrics"],
                                                                    "val_metrics": s4["val_metrics"]})
        row["overfit_gap_train_minus_test"] = _overfit_gap(s5)
        rows.append(row)
    return pd.DataFrame(rows)


def print_train_val_test_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("TRAIN / VALIDATION / TEST METRIC COMPARISON — mean ± std, 95% CI (Nadeau-Bengio t over splits;")
    print("  with one split: that split's stratified-bootstrap / Brodersen-posterior CI over its rows)")
    print("  TRAIN      = Stage 5 refit on the FULL split-train, evaluated on that SAME training data")
    print("               (memorization diagnostic — not a generalization estimate)")
    print("  VALIDATION = Stage 3 cross-split, held-out-fold CV on TRAINING rows only (never touches")
    print("               test) — this is what Stage 4 selected the winning architecture on")
    print("  TEST       = Stage 5 held-out test partition(s), never read by any selection step")
    print("=" * 100)
    for _, row in df.iterrows():
        marker = "   <== CORRECTED WINNER (Stage 8, Borda-ranked)" if row["is_corrected_winner"] else ""
        print(f"\n[{row['family']:7s}] {row['family_label']}  ({row.get('full_model_name')}){marker}")
        print(f"  {'metric':16s}{'train':>26s}{'validation':>26s}{'test':>26s}")
        for m in REPORT_METRICS:
            tr_m, tr_s = row[f"train_{m}_mean"], row[f"train_{m}_std"]
            va_m, va_s = row[f"validation_{m}_mean"], row[f"validation_{m}_std"]
            te_m, te_s = row[f"test_{m}_mean"], row[f"test_{m}_std"]
            print(f"  {m:16s}{tr_m:8.4f}±{tr_s:<6.4f}   {va_m:8.4f}±{va_s:<6.4f}   {te_m:8.4f}±{te_s:<6.4f}")
            print(f"  {'':16s}CI[{row[f'train_{m}_ci_low']:.4f},{row[f'train_{m}_ci_high']:.4f}]"
                  f"      CI[{row[f'validation_{m}_ci_low']:.4f},{row[f'validation_{m}_ci_high']:.4f}]"
                  f"      CI[{row[f'test_{m}_ci_low']:.4f},{row[f'test_{m}_ci_high']:.4f}]")
        print(f"  {'overfit gap (mean of QUALITY_METRICS, clipped at 0)':55s}"
              f"train-vs-val={row['overfit_gap_train_minus_validation']:.4f}   "
              f"train-vs-test={row['overfit_gap_train_minus_test']:.4f}")
        print(f"  {SELECTION_METRIC:16s}validation(cv)={row[f'{SELECTION_METRIC}_validation_cv']:.4f}   "
              f"test={row[f'{SELECTION_METRIC}_test']:.4f}  "
              f"95% CI [{row[f'{SELECTION_METRIC}_test_ci_low']:.4f}, {row[f'{SELECTION_METRIC}_test_ci_high']:.4f}]  "
              f"(n_splits={row['test_n_splits']})")
    print("\n" + "=" * 100)


TRAIN_VAL_TEST_REPORT = build_train_val_test_report(STAGE4_WINNERS, STAGE5_RESULTS, FINAL_REPORT["corrected_winner_family"])
print_train_val_test_report(TRAIN_VAL_TEST_REPORT)
_tvt_csv_path = VALIDATION_CSV_DIR / "train_validation_test_metric_comparison_with_ci.csv"
TRAIN_VAL_TEST_REPORT.to_csv(_tvt_csv_path, index=False)
_module_logger.notice(f"  Train/validation/test CI report -> {_tvt_csv_path}")


# %% [markdown]
#

# %% [markdown]
# ## Section 15 — Stage 9: unassociated-source inference — MOVED
#
# This section (plus 15a/15b/15c: per-family predictors, the
# confidence-threshold/high-confidence exports, and the Stage 9 driver)
# lives in its own module, `fermi_inference.py`. It is
# not an import of this file (that would re-run the whole 7-stage
# validation driver, since this file has no `if __name__` guard) — it is a
# self-contained script that re-derives `SPLIT0`/`VALIDATION_MODELS_DIR`/
# etc. with the same formulas used here and loads Stage 7's deployed
# `.joblib` models plus Stage 8's persisted
# `cross_pipeline_validation_report.json` straight off disk. Run this file
# first (through Stage 8, at least), then run `fermi_inference.py`.

# %% [markdown]
# ## Section 16 — Per-class metrics export [reporting, new — suggested addition]
#
# Precision/recall/F1/support PER FINE CLASS for each family's Stage-5
# winner, mean ± std aggregated across the same `VALIDATION_SPLITS` Stage 5
# already evaluated on (Stage 5's cached `predictions_by_split`, no
# refitting). The macro-averaged `QUALITY_METRICS`/`REPORT_METRICS`
# columns everywhere else in this file hide which SPECIFIC rare classes
# are dragging a low macro score down; this table makes that visible.

# %%
def _per_class_report_dict(y_true_enc: np.ndarray, y_pred_enc: np.ndarray, label_encoder) -> dict:
    y_true_str = label_encoder.inverse_transform(y_true_enc)
    y_pred_str = label_encoder.inverse_transform(y_pred_enc)
    return classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0)


def export_per_class_metrics_csv(stage5_results: dict, split_contexts: dict) -> dict:
    logger = get_section_logger("stage9b_perclass", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSECTION 16 — per-class metrics export\n{'=' * 70}")
    out_paths = {}
    for fam in FAMILIES:
        pred_cache = stage5_results[fam].get("predictions_by_split", {}) or {}
        per_split_reports = {}
        for split_id, pair in pred_cache.items():
            y_true_enc, y_pred_enc = pair
            if y_true_enc is None or y_pred_enc is None or len(y_true_enc) == 0:
                continue
            fine_ctx, merged_ctx, tl_ctx = split_contexts[split_id]
            label_encoder = (fine_ctx.label_encoder if fam == "fine"
                              else merged_ctx.fine_label_encoder if fam == "merged"
                              else tl_ctx.fine_label_encoder)
            try:
                per_split_reports[split_id] = _per_class_report_dict(y_true_enc, y_pred_enc, label_encoder)
            except Exception as e:
                logger.warning(f"  ⚠ [{fam}|{split_id}] per-class report failed: {type(e).__name__}: {e}")
        if not per_split_reports:
            logger.warning(f"  ⚠ [{fam}] no valid split predictions available for per-class metrics — skipping.")
            continue
        all_classes = sorted({c for rep in per_split_reports.values() for c in rep
                               if c not in ("accuracy", "macro avg", "weighted avg")})
        rows = []
        for cls in all_classes:
            precisions = [rep[cls]["precision"] for rep in per_split_reports.values() if cls in rep]
            recalls = [rep[cls]["recall"] for rep in per_split_reports.values() if cls in rep]
            f1s = [rep[cls]["f1-score"] for rep in per_split_reports.values() if cls in rep]
            supports = [rep[cls]["support"] for rep in per_split_reports.values() if cls in rep]
            rows.append({
                "family": fam, "fine_class": cls,
                "precision_mean": float(np.mean(precisions)) if precisions else float("nan"),
                "precision_std": float(np.std(precisions, ddof=1)) if len(precisions) >= 2 else float("nan"),
                "recall_mean": float(np.mean(recalls)) if recalls else float("nan"),
                "recall_std": float(np.std(recalls, ddof=1)) if len(recalls) >= 2 else float("nan"),
                "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
                "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) >= 2 else float("nan"),
                "support_mean": float(np.mean(supports)) if supports else float("nan"),
                "n_splits": len(precisions),
            })
        df = pd.DataFrame(rows).sort_values("f1_mean")
        # Per class and split: recall k/n with its exact Clopper-Pearson interval — with 1-3 test rows
        # per rare class the point value alone says almost nothing.
        recall_rows = []
        for split_id, rep in per_split_reports.items():
            for cls in all_classes:
                if cls not in rep:
                    continue
                n = int(round(rep[cls]["support"]))
                k = int(round(rep[cls]["recall"] * n))
                lo, hi = clopper_pearson(k, n)
                recall_rows.append({"family": fam, "split_id": split_id, "fine_class": cls, "n_test": n,
                                    "k_correct": k, "recall": k / n if n else float("nan"),
                                    "recall_ci_low": lo, "recall_ci_high": hi})
        recall_path = VALIDATION_CSV_DIR / f"per_class_recall_ci_{fam}.csv"
        pd.DataFrame(recall_rows).to_csv(recall_path, index=False)
        logger.notice(f"  [{fam}] per-class recall with Clopper-Pearson 95% CI, per split -> {recall_path}")
        path = VALIDATION_CSV_DIR / f"per_class_metrics_{fam}.csv"
        df.to_csv(path, index=False)
        out_paths[fam] = str(path)
        logger.notice(f"  [{fam}] per-class metrics ({len(df)} fine class(es), "
                       f"{len(per_split_reports)} split(s) pooled) -> {path}")
    return out_paths


def _summarize_stage9b_perclass(out_paths: dict) -> None:
    logger = get_section_logger("stage9b_perclass", VALIDATION_LOGS_DIR)
    for fam, p in out_paths.items():
        logger.notice(f"  [{fam}] per-class metrics -> {p}")


PER_CLASS_METRICS_PATHS = _run_cached("stage9b_perclass",
                                       lambda: export_per_class_metrics_csv(STAGE5_RESULTS, SPLIT_CONTEXTS),
                                       summarize_fn=_summarize_stage9b_perclass)


# %% [markdown]
# ## Section 16a — Shift-aware test metrics [reporting]
#
# Every Stage-5 number describes ASSOCIATED sources, but inference targets
# UNASSOCIATED ones: far more often at |b| < 10° and at low significance
# (Module 1's KS+PSI check flags about half the features). This section
# re-reads Stage 5's cached test predictions (no classifier is refit) and
# reports them (a) per stratum: |b| below/above 10°, and Signif_Avg tertiles
# whose edges come from the unassociated sources; (b) importance-weighted
# toward the unassociated feature distribution. A domain classifier
# (HistGBC, this split's associated TRAIN rows vs the unassociated sources,
# so it never sees the rows it weights) gives w = p(unassoc|x)/p(assoc|x)
# (Shimodaira 2000; Sugiyama et al. 2007, JMLR 8:985; applied to Fermi-LAT
# by Malyshev 2023, RASTI 2:735). Weights are clipped at their 99th
# percentile (Cortes et al. 2010) and Kish's effective sample size is
# reported next to them.
#
# Limits, to state with the numbers: valid only under covariate shift (same
# p(class | x) in both populations); unassociated sources whose true class
# is outside the taxonomy (bcu/spp-like, spurious/diffuse) are not
# represented at all; and inference-time confidence cuts remain calibrated
# to the ASSOCIATED class mix (no prior re-estimation, Saerens et al. 2002,
# because the class-weighted, multi-node models have no single training prior).

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SHIFT_GLAT_THRESHOLD_DEG = 10.0
SHIFT_WEIGHT_CLIP_QUANTILE = 0.99
SHIFT_MIN_STRATUM_ROWS = 10
UNKNOWN_SOURCES_CSV = (_cfg("DATA_PATH", Path("./data/processed/without_optional_features/"))
                       / "unknown_sources_full_processed.csv")


def _shift_metrics(y_true, y_pred, w=None) -> dict:
    return {"bal_accuracy": float(balanced_accuracy_score(y_true, y_pred, sample_weight=w)),
            "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=w)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", sample_weight=w, zero_division=0))}


def _ci_columns(ci: dict) -> dict:
    """Flattens `bootstrap_metric_ci` output into `{metric, metric_std, metric_ci_low, metric_ci_high}` columns."""
    return {f"{m}{suffix}": v[key] for m, v in ci.items()
            for suffix, key in (("", "value"), ("_std", "std"), ("_ci_low", "ci_low"), ("_ci_high", "ci_high"))}


def _importance_weights(x_assoc_train: pd.DataFrame, x_unassoc: pd.DataFrame, x_eval: pd.DataFrame) -> np.ndarray:
    """w(x) = p(unassoc|x)/p(assoc|x) * n_assoc/n_unassoc, clipped at the 99th percentile."""
    x_dom = pd.concat([x_assoc_train, x_unassoc], ignore_index=True)
    y_dom = np.r_[np.zeros(len(x_assoc_train)), np.ones(len(x_unassoc))]
    dom = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=SEED)
    dom.fit(x_dom, y_dom)
    p = np.clip(dom.predict_proba(x_eval)[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1 - p) * (len(x_assoc_train) / len(x_unassoc))
    return np.minimum(w, np.quantile(w, SHIFT_WEIGHT_CLIP_QUANTILE))


def export_shift_aware_metrics(stage5_results: dict, split_contexts: dict) -> dict:
    logger = get_section_logger("stage9e_shift", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSECTION 16a — shift-aware test metrics\n{'=' * 70}")
    df_unk = pd.read_csv(UNKNOWN_SOURCES_CSV)
    signif_edges = df_unk["Signif_Avg"].quantile([1 / 3, 2 / 3]).to_numpy()
    rows = []
    for split_id in VALIDATION_SPLITS:
        fine_ctx, merged_ctx, tl_ctx = split_contexts[split_id]
        x_train = fine_ctx.x_train_by_tag["ALL_FEATURES"]
        x_unk = df_unk.reindex(columns=x_train.columns)
        for fam, ctx in (("fine", fine_ctx), ("merged", merged_ctx), ("3level", tl_ctx)):
            if fam not in FAMILIES:
                continue
            y_true, y_pred = (stage5_results[fam].get("predictions_by_split") or {}).get(split_id, (None, None))
            x_test = ctx.x_test_by_tag["ALL_FEATURES"]
            if y_true is None or len(y_true) != len(x_test):
                logger.warning(f"  ⚠ [{fam}|{split_id}] no test predictions aligned with the test frame — skipped.")
                continue
            y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
            abs_b, signif = x_test["GLAT"].abs().to_numpy(), x_test["Signif_Avg"].to_numpy()
            strata = {
                "all": np.ones(len(y_true), dtype=bool),
                f"|b|<{SHIFT_GLAT_THRESHOLD_DEG:g}": abs_b < SHIFT_GLAT_THRESHOLD_DEG,
                f"|b|>={SHIFT_GLAT_THRESHOLD_DEG:g}": abs_b >= SHIFT_GLAT_THRESHOLD_DEG,
                f"Signif_Avg<{signif_edges[0]:.1f}": signif < signif_edges[0],
                f"Signif_Avg {signif_edges[0]:.1f}-{signif_edges[1]:.1f}":
                    (signif >= signif_edges[0]) & (signif < signif_edges[1]),
                f"Signif_Avg>={signif_edges[1]:.1f}": signif >= signif_edges[1],
            }
            for name, mask in strata.items():
                if mask.sum() < SHIFT_MIN_STRATUM_ROWS or len(np.unique(y_true[mask])) < 2:
                    continue
                rows.append({"family": fam, "split_id": split_id, "stratum": name, "n": int(mask.sum()),
                             "n_classes": int(len(np.unique(y_true[mask]))), "ess": float(mask.sum()),
                             **_ci_columns(bootstrap_metric_ci(y_true[mask], y_pred[mask], _shift_metrics))})
            w = _importance_weights(x_train, x_unk, x_test)
            rows.append({"family": fam, "split_id": split_id, "stratum": "importance_weighted_to_unassociated",
                         "n": len(y_true), "n_classes": int(len(np.unique(y_true))),
                         "ess": float(w.sum() ** 2 / (w ** 2).sum()),
                         **_ci_columns(bootstrap_metric_ci(y_true, y_pred, _shift_metrics, weights=w))})

    per_split = pd.DataFrame(rows)
    if per_split.empty:
        logger.warning("  ⚠ no shift-aware metrics could be computed.")
        return {}
    metric_cols = ["bal_accuracy", "accuracy", "macro_f1"]
    summary = (per_split.groupby(["family", "stratum"], sort=False)
               .agg(n_splits=("split_id", "nunique"), n_mean=("n", "mean"), ess_mean=("ess", "mean"),
                    **{f"{m}_{s}": (m, s) for m in metric_cols for s in ("mean", "std")})
               .reset_index())
    per_split_path = VALIDATION_CSV_DIR / "shift_aware_test_metrics_per_split.csv"
    summary_path = VALIDATION_CSV_DIR / "shift_aware_test_metrics.csv"
    per_split.to_csv(per_split_path, index=False)
    summary.to_csv(summary_path, index=False)
    for r in summary.itertuples():
        logger.notice(f"  [{r.family:6s}] {r.stratum:38s} n≈{r.n_mean:6.0f} ESS≈{r.ess_mean:6.0f}  "
                      f"bal_acc={r.bal_accuracy_mean:.3f}±{r.bal_accuracy_std:.3f} (across splits; "
                      f"per-split row-level CIs in {per_split_path.name})  "
                      f"acc={r.accuracy_mean:.3f}±{r.accuracy_std:.3f}")
    return {"summary_path": str(summary_path), "per_split_path": str(per_split_path), "summary": summary}


SHIFT_AWARE_METRICS = _run_cached("stage9e_shift",
                                  lambda: export_shift_aware_metrics(STAGE5_RESULTS, SPLIT_CONTEXTS))


# %% [markdown]
# ## Section 16b — Calibration / Brier-score check [reporting, new — suggested addition]
#
# Opt-in, gated by `RUN_CALIBRATION_CHECK_NOW` (Section 0, default
# `False`) — mirrors the opt-in pattern used for Stage 2/6 and SHAP in the
# DEAD CODE section. This is the one section past
# Stage 5 that REFITS (Stage 4's winner, per family, per split) rather than
# re-reading Stage 5's cache, so it's the standout time cost among the
# "reporting" sections; nothing downstream of it in this file consumes its
# output, so skipping it is free otherwise.
#
# "0.8 confidence" (requirement #5's own inference-time confidence score,
# now computed by fermi_inference.py) is only a meaningful filter if
# the model is actually
# CALIBRATED — i.e. among sources predicted at ~0.8 confidence, roughly 80%
# really are correct. This section checks that on the labeled TEST
# partitions (Stage 4's winning architecture, refit per split — reusing
# `_fit_predict_merged_core`/`_fit_predict_3level_core`'s `return_proba=True`
# path so this doesn't need a third reimplementation of the chain-routing
# math): a multiclass Brier score (mean squared distance between the
# predicted probability vector and the one-hot true label, lower is
# better — 0 is a perfect, fully-confident-and-correct forecaster) per
# split, mean ± std across splits; and a 10-bin reliability table
# (mean predicted confidence vs. empirical accuracy in that confidence
# bin) — the latter is what actually answers "is 0.8 confidence really
# ~80% correct for this family".

# %%
def _brier_with_ci(y_true_enc: np.ndarray, proba: np.ndarray, n_boot: int = 1000) -> dict:
    """Multiclass Brier score of one split plus a bootstrap 95% CI over its rows."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true_enc)), y_true_enc] = 1.0
    per_row = np.sum((proba - onehot) ** 2, axis=1)
    rng = np.random.default_rng(SEED)
    boots = [per_row[rng.integers(0, len(per_row), len(per_row))].mean() for _ in range(n_boot)]
    return {"value": float(per_row.mean()), "std": float(np.std(boots, ddof=1)),
            "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975))}


def multiclass_brier_score(y_true_enc: np.ndarray, proba: np.ndarray) -> float:
    """Mean, over samples, of the squared Euclidean distance between the
    predicted probability vector and the one-hot true-label vector — the
    standard multiclass generalization of the binary Brier score. 0 is
    perfect (fully confident and correct every time); 2 is the worst
    possible (fully confident and wrong every time)."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true_enc)), y_true_enc] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _reliability_table(fam: str, confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences >= lo) & (confidences < hi if hi < 1.0 else confidences <= hi)
        n = int(mask.sum())
        rows.append({"family": fam, "bin_low": lo, "bin_high": hi, "n": n,
                      "mean_predicted_confidence": float(confidences[mask].mean()) if n else float("nan"),
                      "empirical_accuracy": float(corrects[mask].mean()) if n else float("nan"),
                      "calibration_gap": (float(confidences[mask].mean() - corrects[mask].mean())
                                          if n else float("nan"))})
    return pd.DataFrame(rows)


def _ece(confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted mean |accuracy - confidence| over equal-width
    confidence bins (Naeini, Cooper & Hauskrecht 2015, AAAI; Guo et al. 2017)."""
    bins = np.minimum((confidences * n_bins).astype(int), n_bins - 1)
    return float(sum(np.sum(bins == b) / len(bins) * abs(corrects[bins == b].mean() - confidences[bins == b].mean())
                     for b in range(n_bins) if np.any(bins == b)))


def run_calibration_check(stage4_winners: dict, split_contexts: dict) -> dict:
    """Per family and split, the probabilities of the winner on the held-out TEST rows: Brier score
    (with a bootstrap CI), negative log-likelihood, ECE and a reliability table. For `fine` this is
    done twice — RAW (class-weighted model) and CALIBRATED (`fit_calibrator`, fit on out-of-fold
    predictions of that split's TRAINING rows only). Confidence = the probability given to the
    predicted label (the raw argmax in both cases, since calibration does not change the label)."""
    logger = get_section_logger("stage9c_calibration", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSECTION 16b — calibration / Brier-score check\n{'=' * 70}")
    out, per_split_rows = {}, []
    for fam in FAMILIES:
        acc = {}  # variant -> {"brier": {split: ci}, "conf": [...], "correct": [...], "nll": [...], "temperature": [...]}
        for split_id in pbar(VALIDATION_SPLITS, desc=f"[calibration|{fam}]"):
            fine_ctx, merged_ctx, tl_ctx = split_contexts[split_id]
            winner = _winner_for(stage4_winners, fam, split_id)
            try:
                variants, temperature = {}, float("nan")
                if fam == "fine":
                    cfg = winner["cfg"]
                    x_tr, y_tr = fine_ctx.x_train_by_tag[cfg["tag"]], fine_ctx.y_train_enc
                    model = fit_component(cfg, MODEL_SPECS, get_builder, x_tr, y_tr)
                    variants["raw"] = np.asarray(model.predict_proba(fine_ctx.x_test_by_tag[cfg["tag"]]))
                    calibrator = fit_calibrator(cfg, MODEL_SPECS, get_builder, x_tr, y_tr)
                    variants["calibrated"] = calibrator.transform(variants["raw"])
                    temperature = calibrator.temperature_
                    y_true = fine_ctx.y_test_enc
                elif fam == "merged":
                    branch_ds = _m2_build_branch_datasets(merged_ctx)
                    _, variants["raw"] = _fit_predict_merged_on_test(
                        merged_ctx, winner["router_cfg"], winner["branch_cfgs"], merged_ctx.router_meta,
                        branch_ds, return_proba=True)
                    y_true = merged_ctx.y_test_fine_enc
                else:
                    branch_ds = build_branch_datasets(tl_ctx)
                    _, variants["raw"] = _fit_predict_3level_on_test(
                        tl_ctx, winner["gate_cfg"], winner["subgate_cfg"], winner["nonblazar_cfg"],
                        winner["branch_cfgs"], tl_ctx.gate_meta, tl_ctx.subgate_meta, tl_ctx.nonblazar_meta,
                        branch_ds, return_proba=True)
                    y_true = tl_ctx.y_test_fine_enc
                y_true = np.asarray(y_true)
                pred = variants["raw"].argmax(axis=1)
                for name, proba in variants.items():
                    conf = proba[np.arange(len(pred)), pred]
                    nll = float(-np.mean(np.log(np.clip(proba[np.arange(len(y_true)), y_true], 1e-12, None))))
                    a = acc.setdefault(name, {"brier": {}, "conf": [], "correct": [], "nll": [], "ece": []})
                    a["brier"][split_id] = _brier_with_ci(y_true, proba)
                    a["conf"].append(conf)
                    a["correct"].append((pred == y_true).astype(float))
                    a["nll"].append(nll)
                    a["ece"].append(_ece(conf, (pred == y_true).astype(float)))
                    per_split_rows.append({"family": fam, "variant": name, "split_id": split_id,
                                           "temperature": temperature if name == "calibrated" else float("nan"),
                                           "nll": nll, "ece": a["ece"][-1],
                                           **{f"brier_{k}": x for k, x in a["brier"][split_id].items()}})
            except Exception as e:
                logger.warning(f"  ⚠ [{fam}|{split_id}] calibration check failed: {type(e).__name__}: {e}")

        if not acc:
            logger.warning(f"  ⚠ [{fam}] no split produced a usable calibration check — skipping.")
            out[fam] = None
            continue
        out[fam] = {}
        for name, a in acc.items():
            briers = [b["value"] for b in a["brier"].values()]
            confidences, corrects = np.concatenate(a["conf"]), np.concatenate(a["correct"])
            reliability_path = VALIDATION_CSV_DIR / f"calibration_reliability_{fam}_{name}.csv"
            _reliability_table(fam, confidences, corrects).to_csv(reliability_path, index=False)
            out[fam][name] = {
                "brier_score_mean": float(np.mean(briers)),
                "brier_score_std": (float(np.std(briers, ddof=1)) if len(briers) >= 2
                                    else next(iter(a["brier"].values()))["std"]),
                "nll_mean": float(np.mean(a["nll"])), "ece_mean": float(np.mean(a["ece"])),
                "n_splits": len(briers), "reliability_path": str(reliability_path),
            }
            logger.notice(f"  [{fam}|{name}] Brier = {out[fam][name]['brier_score_mean']:.4f} "
                          f"± {out[fam][name]['brier_score_std']:.4f}  NLL = {out[fam][name]['nll_mean']:.4f}  "
                          f"ECE = {out[fam][name]['ece_mean']:.4f}  (n={len(briers)} split(s), lower is better)  "
                          f"reliability -> {reliability_path}")

    pd.DataFrame(per_split_rows).to_csv(VALIDATION_CSV_DIR / "calibration_per_split_ci.csv", index=False)
    summary_path = VALIDATION_CSV_DIR / "calibration_summary.csv"
    pd.DataFrame([{"family": fam, "variant": name, **m} for fam, v in out.items() if v
                  for name, m in v.items()]).to_csv(summary_path, index=False)
    logger.notice(f"  Calibration summary (raw vs calibrated) -> {summary_path}")
    return out


def _summarize_stage9c_calibration(out: dict) -> None:
    logger = get_section_logger("stage9c_calibration", VALIDATION_LOGS_DIR)
    for fam, variants in out.items():
        for name, v in (variants or {}).items():
            logger.notice(f"  [{fam}|{name}] Brier = {v['brier_score_mean']:.4f} ± {v['brier_score_std']:.4f}  "
                          f"ECE = {v['ece_mean']:.4f}  (n={v['n_splits']} split(s))  reliability -> {v['reliability_path']}")


if RUN_CALIBRATION_CHECK_NOW:
    CALIBRATION_CHECK = _run_cached("stage9c_calibration",
                                     lambda: run_calibration_check(STAGE4_WINNERS, SPLIT_CONTEXTS),
                                     summarize_fn=_summarize_stage9c_calibration)
else:
    CALIBRATION_CHECK = None
    _module_logger.notice("  Section 16b skipped (RUN_CALIBRATION_CHECK_NOW=False in Section 0) — "
                           "set True to run the calibration/Brier-score check (refits Stage 4's "
                           "winners per split; the most expensive reporting-only stage past Stage 5 "
                           "in this file). Nothing downstream reads CALIBRATION_CHECK, so this is "
                           "otherwise free to skip.")

# %% [markdown]
# ## Section 16c — Rare-class hierarchical partial-pooling shrinkage
#
# Deliberately reporting-only: a closed-form empirical-
# Bayes reanalysis of per-fine-class confusion counts ALREADY on disk —
# Stage 5's cached `predictions_by_split`, the identical source Section 16
# already reads, pooled across `VALIDATION_SPLITS`. No new CV folds, no
# new model refits; nothing upstream of Stage 5 is touched, and nothing
# downstream of it (Borda ranking, `SELECTION_METRIC`, Stage 1/3/4/8's
# architecture selection) reads this section's output. That is not a
# scope compromise — the method has no meaningful "processing" or "train"
# stage form to begin with, since it operates purely on already-computed
# counts.
#
# **Method** — Beta-Binomial partial pooling, empirical Bayes via method-
# of-moments (James-Stein-type shrinkage, the fast closed-form alternative
# to MCMC/PyMC): each fine
# class's raw recall `k/n` is treated as a noisy draw from a Beta prior
# whose mean and concentration are estimated from its sibling classes,
# then combined with `(k, n)` into a shrunk posterior mean + credible
# interval. Grouping is read directly off the taxonomy already in scope
# (`TAXONOMY`/`SUPERCLASS_ORDER`/`BLAZAR_SUPERCLASSES`) — never invented:
#
# - **`fine`** — 1 level: every fine class pooled toward one grand mean
#   (no natural sub-grouping beyond the taxonomy itself, per §2).
# - **`merged`** — 2 levels: fine class nested inside its `SUPERCLASS_ORDER`
#   group (7 groups) — e.g. a rare class like `nov` shrinks toward
#   `BINARY_SYSTEMS`'s own (already-shrunk) mean, not straight to the
#   grand mean across pulsars and blazars.
# - **`3level`** — 3 levels: fine class nested inside superclass nested
#   inside the gate's BLAZAR/OTHER split — the deepest, most taxonomically
#   faithful of the three, matching `fermi_3levels.py`'s own routing.
#
# **Thin-group variance (§5)**: a group's own between-class concentration
# is estimated from its member classes only when it has >=3 (a fixed
# `_mom_beta_concentration` threshold); otherwise the concentration is
# BORROWED from the 7-superclass level — the level with the most groups —
# per §5's own mitigation ("borrowing the variance scale from a level
# with more groups... rather than letting the branch-local fit estimate
# it from almost nothing"). The gate level (`3level`'s top, only 2 groups)
# always borrows this way; its own 2-point concentration estimate is not
# trusted for anything.
#
# **Zero held-out data (§6, irreducible)**: a class with `n=0` pooled test
# rows (e.g. a superclass never held out at all in some split) collapses
# EXACTLY to its group's shrunk prior mean, with that prior's own
# (typically wide) credible interval — flagged via `zero_held_out=True` in
# the CSV, never silently presented as a validated per-class number. This
# is the honest thing to report, not new information recovered from
# nowhere: the physical sample floor (e.g. `nov`'s 8 total sources in
# 4FGL-DR4) is untouched by any statistical reweighting downstream of it.
#
# **What this is NOT**: a multi-prior sensitivity sweep (§6's "stated
# modeling choice... not fully identifiable from ~7 superclass groups"
# caveat) is disclosed in `rare_class_shrinkage_caveats.json` rather than
# implemented as a 2-3-prior comparison — out of scope for this
# reporting-only pass; and it is never read by any Borda ranking or
# `SELECTION_METRIC` computation anywhere in this file (§7).

# %%
from scipy.stats import beta as _beta_dist

SHRINKAGE_MIN_CONCENTRATION = 1.0  # floor on any estimated Beta concentration M — keeps a proper
                                    # (non-degenerate) prior even when the method-of-moments
                                    # variance estimate is noisy, non-positive, or undefined.
SHRINKAGE_MIN_GROUPS_FOR_OWN_VARIANCE = 3  # below this, borrow the concentration (see §5 above)
SHRINKAGE_CI = 0.95


def _pooled_class_counts(stage5_results: dict, split_contexts: dict, fam: str) -> dict:
    """{fine_class_str: {"k": correct, "n": total}} pooled across every
    valid VALIDATION_SPLITS test partition for family `fam` — the SAME
    Stage-5-cached predictions_by_split Section 16 already reads (no new
    refits, no new folds: a pure post-hoc reanalysis of counts already on
    disk)."""
    pred_cache = stage5_results[fam].get("predictions_by_split", {}) or {}
    counts: dict = {}
    for split_id, pair in pred_cache.items():
        y_true_enc, y_pred_enc = pair
        if y_true_enc is None or y_pred_enc is None or len(y_true_enc) == 0:
            continue
        fine_ctx, merged_ctx, tl_ctx = split_contexts[split_id]
        label_encoder = (fine_ctx.label_encoder if fam == "fine"
                          else merged_ctx.fine_label_encoder if fam == "merged"
                          else tl_ctx.fine_label_encoder)
        y_true_str = label_encoder.inverse_transform(y_true_enc)
        y_pred_str = label_encoder.inverse_transform(y_pred_enc)
        for t, p in zip(y_true_str, y_pred_str):
            c = counts.setdefault(t, {"k": 0, "n": 0})
            c["n"] += 1
            c["k"] += int(t == p)
    return counts


def _mom_beta_concentration(rates: np.ndarray, weights: np.ndarray, mu: float) -> tuple:
    """Method-of-moments Beta concentration M = alpha+beta from a set of
    group-level rates (each a k_g/n_g proportion), given the weighted
    grand mean `mu`. For a Beta-Binomial random-effects model,
    Var(p) ~= mu*(1-mu)/(M+1), so M = mu*(1-mu)/V_between - 1, where
    `V_between` is the weighted between-group variance of the raw rates
    MINUS the expected within-group binomial sampling variance (a
    DerSimonian-Laird-style moment correction). Returns `(M, low_confidence)`
    — `low_confidence=True` when fewer than
    `SHRINKAGE_MIN_GROUPS_FOR_OWN_VARIANCE` groups fed the estimate (§5's
    "thin-group variance estimation" caveat) or the moment estimate was
    non-positive/undefined — either way signaling the caller to borrow `M`
    from a level with more groups instead of trusting this one."""
    mask = np.isfinite(rates) & (weights > 0)
    rates, weights = rates[mask], weights[mask]
    n_groups = len(rates)
    low_conf = n_groups < SHRINKAGE_MIN_GROUPS_FOR_OWN_VARIANCE
    if n_groups < 2 or not np.isfinite(mu) or mu <= 0 or mu >= 1:
        return SHRINKAGE_MIN_CONCENTRATION, True
    w_sum = float(weights.sum())
    weighted_var = float(np.sum(weights * (rates - mu) ** 2) / w_sum)
    expected_within = float(np.sum(weights * (mu * (1 - mu) / np.maximum(weights, 1))) / w_sum)
    v_between = weighted_var - expected_within
    if not np.isfinite(v_between) or v_between <= 1e-6:
        # No between-group spread detectable beyond sampling noise (or too little data to tell) —
        # a weak-but-proper prior beats an undefined/infinite concentration.
        return max(SHRINKAGE_MIN_CONCENTRATION, 10.0), True
    m = mu * (1 - mu) / v_between - 1.0
    return max(SHRINKAGE_MIN_CONCENTRATION, float(m)), low_conf


def _shrink_beta_binomial(k: int, n: int, mu: float, m: float, confidence: float = SHRINKAGE_CI) -> dict:
    """Posterior mean + credible interval of a Beta(mu*m, (1-mu)*m) prior
    updated with (k successes / n trials) — the closed-form empirical-
    Bayes (James-Stein-type) shrinkage estimate. `n=0` (a class never held out at all in this family's
    pooled test data — the "Rare-Superclass Blind Spot", see the caveats below) collapses
    EXACTLY to the prior mean `mu` with the prior's own (typically wide)
    credible interval — the honest thing to report, not new information
    recovered from nowhere."""
    alpha0, beta0 = mu * m, (1.0 - mu) * m
    alpha_post, beta_post = alpha0 + k, beta0 + (n - k)
    mean_post = alpha_post / (alpha_post + beta_post)
    lo = float(_beta_dist.ppf((1 - confidence) / 2, alpha_post, beta_post))
    hi = float(_beta_dist.ppf(1 - (1 - confidence) / 2, alpha_post, beta_post))
    return {"shrunk_recall": float(mean_post), "ci_low": lo, "ci_high": hi,
            "prior_mean": float(mu), "concentration_m": float(m)}


def _group_pooled_rate(members: dict) -> tuple:
    """(k, n, rate) pooled over a dict of {leaf: {"k","n"}} members."""
    k = sum(v["k"] for v in members.values())
    n = sum(v["n"] for v in members.values())
    return k, n, (k / n if n else float("nan"))


def _shrink_flat(counts: dict) -> list:
    """`fine` family — 1 level: every fine class pooled toward one grand
    mean (§2: "the weakest-motivated case of the three, since there's no
    natural sub-grouping to exploit beyond the taxonomy itself")."""
    classes = sorted(counts)
    ks = np.array([counts[c]["k"] for c in classes], dtype=float)
    ns = np.array([counts[c]["n"] for c in classes], dtype=float)
    mu0 = float(ks.sum() / ns.sum()) if ns.sum() else float("nan")
    rates = np.divide(ks, ns, out=np.full_like(ks, np.nan), where=ns > 0)
    m0, low_conf = _mom_beta_concentration(rates, ns, mu0)
    rows = []
    for c in classes:
        k, n = counts[c]["k"], counts[c]["n"]
        res = _shrink_beta_binomial(k, n, mu0, m0)
        rows.append({"fine_class": c, "n_pooled": n, "k_correct": k,
                      "raw_recall": (k / n) if n else float("nan"), **res,
                      "group_path": "grand_mean",
                      "low_confidence_variance_estimate": bool(low_conf),
                      "zero_held_out": n == 0})
    return rows


def _shrink_two_level(counts: dict) -> list:
    """`merged` family — 2 levels: fine class nested inside `SUPERCLASS_ORDER`
    (7 groups, read directly off `TAXONOMY` — never invented, per §2)."""
    fine_to_super = dict(TAXONOMY.items())
    groups = {sc: {c: counts[c] for c in counts if fine_to_super.get(c) == sc} for sc in SUPERCLASS_ORDER}

    # Top level: grand mean + between-superclass concentration — the 7-superclass level is also
    # what every thin branch below borrows its own variance scale from (§5).
    sc_names = [sc for sc in SUPERCLASS_ORDER if groups[sc]]
    sc_pooled = {sc: _group_pooled_rate(groups[sc]) for sc in sc_names}
    sc_n_arr = np.array([sc_pooled[sc][1] for sc in sc_names], dtype=float)
    sc_rate_arr = np.array([sc_pooled[sc][2] for sc in sc_names], dtype=float)
    n_root_total = sum(v[1] for v in sc_pooled.values())
    mu_root = float(sum(v[0] for v in sc_pooled.values()) / n_root_total) if n_root_total else float("nan")
    m_borrow, root_low_conf = _mom_beta_concentration(sc_rate_arr, sc_n_arr, mu_root)

    rows = []
    for sc in SUPERCLASS_ORDER:
        members = groups[sc]
        if not members:
            continue
        k_g, n_g, _ = sc_pooled[sc]
        mu_g = _shrink_beta_binomial(k_g, n_g, mu_root, m_borrow)["shrunk_recall"]

        member_classes = sorted(members)
        m_rates = np.array([(members[c]["k"] / members[c]["n"]) if members[c]["n"] else np.nan
                             for c in member_classes], dtype=float)
        m_ns = np.array([members[c]["n"] for c in member_classes], dtype=float)
        m_within, within_low_conf = _mom_beta_concentration(m_rates, m_ns, mu_g)
        if within_low_conf:
            m_within = m_borrow  # too few fine classes in this branch to trust its own spread — borrow

        for c in member_classes:
            k, n = members[c]["k"], members[c]["n"]
            res = _shrink_beta_binomial(k, n, mu_g, m_within)
            rows.append({"fine_class": c, "n_pooled": n, "k_correct": k,
                          "raw_recall": (k / n) if n else float("nan"), **res,
                          "group_path": f"grand_mean->{sc}",
                          "low_confidence_variance_estimate": bool(within_low_conf or root_low_conf),
                          "zero_held_out": n == 0})

    mapped = {c for sc in SUPERCLASS_ORDER for c in groups[sc]}
    for c, v in counts.items():
        if c in mapped:
            continue
        res = _shrink_beta_binomial(v["k"], v["n"], mu_root, m_borrow)
        rows.append({"fine_class": c, "n_pooled": v["n"], "k_correct": v["k"],
                      "raw_recall": (v["k"] / v["n"]) if v["n"] else float("nan"), **res,
                      "group_path": "grand_mean->UNMAPPED(not in TAXONOMY/SUPERCLASS_ORDER)",
                      "low_confidence_variance_estimate": True, "zero_held_out": v["n"] == 0})
    return rows


def _shrink_three_level(counts: dict) -> list:
    """`3level` family — 3 levels: fine class nested inside superclass
    nested inside the gate's BLAZAR/OTHER split — `TAXONOMY`/
    `SUPERCLASS_ORDER`/`BLAZAR_SUPERCLASSES`, the identical taxonomy
    `fermi_3levels.py`'s own gate routes on, never re-derived."""
    fine_to_super = dict(TAXONOMY.items())
    groups = {sc: {c: counts[c] for c in counts if fine_to_super.get(c) == sc} for sc in SUPERCLASS_ORDER}
    gate_groups = {"BLAZAR": [sc for sc in SUPERCLASS_ORDER if sc in BLAZAR_SUPERCLASSES],
                   "OTHER": [sc for sc in SUPERCLASS_ORDER if sc not in BLAZAR_SUPERCLASSES]}

    # Level with the most groups to source the variance scale from (§5) — the 7 superclasses,
    # computed identically to the 2-level case and reused at both levels below.
    sc_names = [sc for sc in SUPERCLASS_ORDER if groups[sc]]
    sc_pooled = {sc: _group_pooled_rate(groups[sc]) for sc in sc_names}
    sc_n_arr = np.array([sc_pooled[sc][1] for sc in sc_names], dtype=float)
    sc_rate_arr = np.array([sc_pooled[sc][2] for sc in sc_names], dtype=float)
    n_super_total = sum(v[1] for v in sc_pooled.values())
    mu_superlevel = float(sum(v[0] for v in sc_pooled.values()) / n_super_total) if n_super_total else float("nan")
    m_borrow, _ = _mom_beta_concentration(sc_rate_arr, sc_n_arr, mu_superlevel)

    # Gate level (>=1, <=2 groups — §6's disclosed subjectivity: a concentration estimated from
    # this few points is not trustworthy on its own, so it's computed only for a defensible grand
    # mean and ALWAYS borrows m_borrow rather than its own moment estimate.
    gate_names = [g for g in ("BLAZAR", "OTHER") if any(groups[sc] for sc in gate_groups[g])]
    gate_pooled = {g: _group_pooled_rate({c: v for sc in gate_groups[g] for c, v in groups[sc].items()})
                   for g in gate_names}
    n_gate_total = sum(v[1] for v in gate_pooled.values())
    mu_root = float(sum(v[0] for v in gate_pooled.values()) / n_gate_total) if n_gate_total else float("nan")
    m_root = m_borrow

    rows = []
    for g in gate_names:
        k_gate, n_gate, _ = gate_pooled[g]
        mu_gate = _shrink_beta_binomial(k_gate, n_gate, mu_root, m_root)["shrunk_recall"]
        for sc in gate_groups[g]:
            members = groups[sc]
            if not members:
                continue
            k_g, n_g, _ = sc_pooled[sc]
            mu_g = _shrink_beta_binomial(k_g, n_g, mu_gate, m_borrow)["shrunk_recall"]

            member_classes = sorted(members)
            m_rates = np.array([(members[c]["k"] / members[c]["n"]) if members[c]["n"] else np.nan
                                 for c in member_classes], dtype=float)
            m_ns = np.array([members[c]["n"] for c in member_classes], dtype=float)
            m_within, within_low_conf = _mom_beta_concentration(m_rates, m_ns, mu_g)
            if within_low_conf:
                m_within = m_borrow

            for c in member_classes:
                k, n = members[c]["k"], members[c]["n"]
                res = _shrink_beta_binomial(k, n, mu_g, m_within)
                rows.append({"fine_class": c, "n_pooled": n, "k_correct": k,
                              "raw_recall": (k / n) if n else float("nan"), **res,
                              "group_path": f"grand_mean->{g}->{sc}",
                              "low_confidence_variance_estimate": True,  # gate level (<=2 groups) never trusted alone
                              "zero_held_out": n == 0})

    mapped = {c for sc in SUPERCLASS_ORDER for c in groups[sc]}
    for c, v in counts.items():
        if c in mapped:
            continue
        res = _shrink_beta_binomial(v["k"], v["n"], mu_root, m_root)
        rows.append({"fine_class": c, "n_pooled": v["n"], "k_correct": v["k"],
                      "raw_recall": (v["k"] / v["n"]) if v["n"] else float("nan"), **res,
                      "group_path": "grand_mean->UNMAPPED(not in TAXONOMY/SUPERCLASS_ORDER)",
                      "low_confidence_variance_estimate": True, "zero_held_out": v["n"] == 0})
    return rows


RARE_CLASS_SHRINKAGE_CAVEATS = {
    "source": "Caveats NOT solvable by this method, irreducible because the classes are rare per se "
              "(recorded here, not just asserted in a docstring)",
    "zero_held_out_is_a_hard_floor": (
        "If a class/superclass is never held out at all across the pooled validation splits, "
        "shrinkage collapses that estimate entirely to the prior/hyperprior mean. That's the honest "
        "thing to report — a wide-credible-interval population-level guess instead of NaN or a "
        "spurious in-sample number — but it is not new information recovered from nowhere."),
    "physical_sample_size_is_untouched": (
        "E.g. 'nov' (novae) has 8 sources in the whole 4FGL-DR4 catalog before any split. Whatever "
        "the estimation method, the ceiling on how precisely its true recall can ever be known is "
        "set by how many such sources exist in the catalog, not by CV design or Bayesian modeling "
        "choices downstream of that fact."),
    "oracle_branch_training_is_untouched": (
        "Branches still train only on rows whose TRUE superclass matches ('Known Limitation: Oracle "
        "Branch Training') — this shrinkage improves how a branch's own small-sample performance is "
        "REPORTED; it does nothing about the router's/gate's misrouting rate at deployment."),
    "prior_choice_is_a_real_disclosed_subjectivity": (
        "How much a rare class shrinks toward its immediate superclass mean vs. a shared cross-"
        "superclass mean depends on hyperprior/concentration choices that are themselves not fully "
        "identifiable from ~7 superclass groups. This is a stated modeling choice, not something a "
        "full sensitivity sweep across 2-3 reasonable prior specifications was run to confirm in this "
        "reporting-only pass — treat the reported credible intervals accordingly."),
    "it_changes_what_is_reported_not_what_is_true": (
        "A shrunk estimate is a better-calibrated GUESS about a rare class's true recall under a "
        "stated statistical model — it is still a guess. The underlying epistemic problem (very "
        "little test evidence exists for classes like 'nov'/'gc'/'sey') persists at whatever "
        "confidence level this report states honestly."),
    "reporting_only_scope": (
        "These shrunk estimates augment/replace the "
        "raw per-class numbers shown in REPORTS ONLY. They are not read by QUALITY_METRICS, "
        "SELECTION_METRIC, or any Borda ranking in this file (Stage 1/3/4/8 are untouched). Using "
        "them as an actual selection criterion would be a larger methodological change needing its "
        "own validation (e.g. checking whether it changes which architecture chain-CV picks) before "
        "being trusted for that role."),
}


def export_rare_class_shrinkage_csv(stage5_results: dict, split_contexts: dict) -> dict:
    logger = get_section_logger("stage9d_shrinkage", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSECTION 16c — rare-class hierarchical partial-pooling shrinkage "
                  f"[reporting-only]\n{'=' * 70}")
    dispatch = {"fine": _shrink_flat, "merged": _shrink_two_level, "3level": _shrink_three_level}
    out_paths = {}
    for fam in FAMILIES:
        counts = _pooled_class_counts(stage5_results, split_contexts, fam)
        if not counts:
            logger.warning(f"  ⚠ [{fam}] no pooled test predictions available — skipping shrinkage.")
            continue
        rows = dispatch[fam](counts)
        for r in rows:
            r["family"] = fam
        df = pd.DataFrame(rows).sort_values("raw_recall")
        path = VALIDATION_CSV_DIR / f"rare_class_shrinkage_{fam}.csv"
        df.to_csv(path, index=False)
        out_paths[fam] = str(path)
        n_zero = int(df["zero_held_out"].sum())
        logger.notice(f"  [{fam}] {len(df)} fine class(es) shrunk ({n_zero} with zero held-out rows "
                       "across all pooled splits — those collapse to their group's prior mean) "
                       f"-> {path}")

    caveats_path = VALIDATION_DIR / "rare_class_shrinkage_caveats.json"
    with open(caveats_path, "w") as f:
        json.dump(RARE_CLASS_SHRINKAGE_CAVEATS, f, indent=2)
    out_paths["caveats"] = str(caveats_path)
    logger.notice(f"  Caveats (disclosed, not glossed over) -> {caveats_path}")
    logger.notice("  NOTE — reporting-only: these shrunk estimates are NOT fed "
                   "back into QUALITY_METRICS/SELECTION_METRIC or any Borda ranking anywhere in this "
                   "file (Stage 1/3/4/8 are untouched); they only replace/augment the raw per-class "
                   "numbers shown in reports.")
    return out_paths


def _summarize_stage9d_shrinkage(out_paths: dict) -> None:
    logger = get_section_logger("stage9d_shrinkage", VALIDATION_LOGS_DIR)
    for key, p in out_paths.items():
        logger.notice(f"  [{key}] -> {p}")


RARE_CLASS_SHRINKAGE_PATHS = _run_cached("stage9d_shrinkage",
                                          lambda: export_rare_class_shrinkage_csv(STAGE5_RESULTS, SPLIT_CONTEXTS),
                                          summarize_fn=_summarize_stage9d_shrinkage)

# %% [markdown]
# ## DEAD CODE — Stage 2, Stage 6, and SHAP
#
# Everything below is retired from the active pipeline: Stage 1, 3, 4, 5,
# 7, 8, 9 are the 7 stages this file (plus `fermi_inference.py` for
# Stage 9) actually runs. The three pieces below — Stage 2 (winner's-curse
# nested-CV diagnostic), Stage 6 (family-vs-family significance testing),
# and Section 18's SHAP explainability — are kept exactly as fully-written,
# runnable Python, purely for reference and manual reuse. None of them is
# reachable from `STAGE_DEPENDENCIES`, the Section 13 driver, or Stage 8's
# persisted report (`run_stage8` no longer even takes a `stage6` argument).
# Every `RUN_..._NOW` flag below now DEFAULTS TO FALSE (all three used to
# default True for Stage 2/SHAP, and Stage 6 used to run unconditionally —
# that automatic wiring is exactly what "dead code" retires here). To
# resurrect any one of them: flip its flag to True and re-run that cell;
# nothing else in this file needs to change, since each was already
# decoupled from the other active stages' own checkpoints.

# %% [markdown]
# ### DEAD CODE — Stage 2: winner's-curse diagnostic
#
# Nested CV around fine's entire Stage-0/1/2 selection procedure
# (`N_NESTED_DIAG_REPEATS x N_OUTER_NESTED` outer folds) — by far the most
# expensive stage in this file. Never consumed by the real Stage 3-8
# selection/estimation; `run_stage8` tolerates `stage2=None` (its default,
# now that this is dead code) and reports `"status": "not_run"` for it.

# %%
N_NESTED_DIAG_REPEATS = 3
N_OUTER_NESTED = 2

STAGE2_MODEL_SUBSET = [m for m in ALL_MODEL_NAMES if not m.startswith("MLP")]
STAGE2_COARSE_SURVIVAL_FRACTION = 0.10
STAGE2_COARSE_MIN_SURVIVORS = 2
STAGE2_INNER_N_SPLITS = 3
# Off by default — Stage 2 does not run as part of a plain top-to-bottom execution of this file;
# set it to True (or "RUN_WINNERS_CURSE_DIAGNOSTIC_NOW": True in run_fermi.py) to run it.
RUN_WINNERS_CURSE_DIAGNOSTIC_NOW = _cfg("RUN_WINNERS_CURSE_DIAGNOSTIC_NOW", False)


def run_stage2(fine_ctx) -> dict:
    logger = get_section_logger("stage2_diagnostic", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSTAGE 2 — winner's-curse diagnostic, fine only, split 0\n"
                   f"{N_NESTED_DIAG_REPEATS} outer-fold seeds x {N_OUTER_NESTED} folds each\n{'=' * 70}")

    def _inner_safe_run(fn, *a, context="", **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            logger.warning(f"  ⚠ stage2 inner selection call failed [{context}]: {type(e).__name__}: {e}")
            return None

    per_repeat = []
    for r in pbar(range(N_NESTED_DIAG_REPEATS), desc="[stage2] outer-fold-seed repeats"):
        seed_r = SEED + 9000 + r
        classes, counts = np.unique(fine_ctx.y_train_enc, return_counts=True)
        rare_mask = np.isin(fine_ctx.y_train_enc, classes[counts < N_OUTER_NESTED])
        strat_idx, rare_idx = np.where(~rare_mask)[0], np.where(rare_mask)[0]
        y_strat = fine_ctx.y_train_enc[strat_idx]
        strat_min = int(np.unique(y_strat, return_counts=True)[1].min()) if len(strat_idx) else 0
        n_eff = max(2, min(N_OUTER_NESTED, strat_min)) if strat_min >= 2 else 2
        skf = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed_r)

        fold_naive, fold_corrected = [], []
        for tr_local, va_local in skf.split(strat_idx, y_strat):
            outer_tr = np.concatenate([strat_idx[tr_local], rare_idx])
            outer_te = strat_idx[va_local]
            x_outer_tr_by_tag = {t: x.iloc[outer_tr].reset_index(drop=True) for t, x in fine_ctx.x_train_by_tag.items()}
            x_outer_te_by_tag = {t: x.iloc[outer_te].reset_index(drop=True) for t, x in fine_ctx.x_train_by_tag.items()}
            y_outer_tr, y_outer_te = fine_ctx.y_train_enc[outer_tr], fine_ctx.y_train_enc[outer_te]

            try:
                winner_info = _select_winners(
                    x_outer_tr_by_tag, y_outer_tr, fine_ctx.tags, STAGE2_MODEL_SUBSET, _inner_safe_run,
                    MODEL_SPECS, get_builder, SELECTION_METRIC, "stage2_diag", "fine_nested_diag",
                    STAGE2_INNER_N_SPLITS, SCALER_VARIANTS, DEFAULT_SCALER_NAME, PCA_BASE_TAG, PCA_VARIANTS,
                    WEIGHTING_MODES, COARSE_VAL_FRACTION, STAGE2_COARSE_SURVIVAL_FRACTION,
                    STAGE2_COARSE_MIN_SURVIVORS, None, logs_dir=None)
            except Exception as e:
                logger.warning(f"  ⚠ stage2 repeat {r} fold failed entirely: {type(e).__name__}: {e}")
                continue
            if not winner_info:
                continue
            best_model_name, best_win = max(winner_info.items(), key=lambda kv: kv[1].get("cv_score", -np.inf))
            best_cfg = _leaderboard_row_to_cfg({**best_win, "model_name": best_model_name})
            naive_score = best_win.get("cv_score", float("nan"))

            model = fit_component(best_cfg, MODEL_SPECS, get_builder, x_outer_tr_by_tag[best_cfg["tag"]], y_outer_tr)
            corrected_score = _score_by_name(SELECTION_METRIC, y_outer_te, model.predict(x_outer_te_by_tag[best_cfg["tag"]]))
            fold_naive.append(naive_score)
            fold_corrected.append(corrected_score)

        if fold_naive:
            naive_mean, corrected_mean = float(np.mean(fold_naive)), float(np.mean(fold_corrected))
            per_repeat.append({"repeat": r, "seed": seed_r, "naive_mean": naive_mean,
                                "corrected_mean": corrected_mean, "gap": naive_mean - corrected_mean,
                                "n_outer_folds": len(fold_naive)})
            logger.notice(f"  [repeat {r}] naive={naive_mean:.4f}  corrected={corrected_mean:.4f}  "
                           f"gap={naive_mean - corrected_mean:.4f}")

    if per_repeat:
        gaps = [p["gap"] for p in per_repeat]
        summary = {"mean_gap": float(np.mean(gaps)), "std_gap": float(np.std(gaps)),
                   "mean_naive": float(np.mean([p["naive_mean"] for p in per_repeat])),
                   "mean_corrected": float(np.mean([p["corrected_mean"] for p in per_repeat])),
                   "n_repeats_completed": len(per_repeat)}
    else:
        summary = {"mean_gap": float("nan"), "std_gap": float("nan"), "mean_naive": float("nan"),
                   "mean_corrected": float("nan"), "n_repeats_completed": 0}
    logger.notice(f"  -> mean optimism gap across {summary['n_repeats_completed']} repeat(s): "
                  f"{summary['mean_gap']:.4f} ± {summary['std_gap']:.4f}")
    return {"per_repeat": per_repeat, "summary": summary}


if RUN_WINNERS_CURSE_DIAGNOSTIC_NOW:
    STAGE2_RESULTS = _run_cached("stage2", lambda: run_stage2(FINE_CTX_SPLIT0))
    _module_logger.notice("  Stage 2 diagnostic complete. Note: it is no longer folded into Stage 8's "
                           "persisted report automatically (run_stage8 dropped its old re-fold-in "
                           "behavior along with Stage 6 when both were retired to DEAD CODE) — inspect "
                           "STAGE2_RESULTS directly, or pass it into run_stage8(..., stage2=STAGE2_RESULTS, ...) "
                           "yourself and re-run Stage 8 if you want it in the JSON report.")
else:
    _module_logger.notice("  DEAD CODE: Stage 2 skipped (RUN_WINNERS_CURSE_DIAGNOSTIC_NOW=False) — "
                           "set it True and re-run this cell to run the winner's-curse diagnostic.")


# %% [markdown]
# ### DEAD CODE — Stage 6: family-vs-family significance testing
#
# Primary test: Nadeau-Bengio corrected resampled paired t-test on the 5
# split-level `SELECTION_METRIC` differences (from Stage 5 — no
# recomputation). Supplementary: split-0-only exact McNemar + a
# class-stratified paired bootstrap on the common test rows, using Stage
# 5's cached split-0 predictions. Each pair also gets a Holm-Bonferroni-
# adjusted p-value (across the 3 non-independent pairwise comparisons) and
# a paired Cohen's d. `_repeated_cv_ci` — the one function from this
# original section Stage 5 actually depends on — lives in Section 8 now,
# not here; everything below is genuinely unused by the active pipeline.

# %%
def nadeau_bengio_corrected_ttest(diffs: Sequence[float], n_train: float, n_test: float,
                                   confidence: float = 0.95) -> dict:
    diffs = np.asarray([d for d in diffs if np.isfinite(d)], dtype=float)
    n = len(diffs)
    if n < 2:
        return {"mean_diff": float(diffs.mean()) if n else float("nan"), "t_stat": float("nan"),
                "p_value": float("nan"), "df": n - 1, "n": n, "note": "fewer than 2 valid split differences"}
    mean_diff = float(diffs.mean())
    var_diff = float(diffs.var(ddof=1))
    rho = (n_test / n_train) if n_train else float("nan")
    corrected_var = var_diff * (1.0 / n + rho)
    if not np.isfinite(corrected_var) or corrected_var <= 0:
        return {"mean_diff": mean_diff, "t_stat": float("nan"), "p_value": float("nan"), "df": n - 1, "n": n,
                "note": "corrected variance non-positive/undefined (near-identical per-split scores)"}
    t_stat = mean_diff / np.sqrt(corrected_var)
    df = n - 1
    p_value = float(2 * (1 - _t_dist.cdf(abs(t_stat), df)))
    tmult = float(_t_dist.ppf(0.5 + confidence / 2, df=df))
    margin = tmult * np.sqrt(corrected_var)
    return {"mean_diff": mean_diff, "t_stat": float(t_stat), "p_value": p_value, "df": df, "n": n,
            "ci_low": mean_diff - margin, "ci_high": mean_diff + margin, "rho_test_train_ratio": rho}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    correct_a = np.asarray(pred_a) == np.asarray(y_true)
    correct_b = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n_discordant = n01 + n10
    if n_discordant == 0:
        return {"n01": n01, "n10": n10, "n_discordant": 0, "p_value": 1.0}
    k = min(n01, n10)
    p_value = float(binomtest(k, n_discordant, p=0.5, alternative="two-sided").pvalue)
    return {"n01": n01, "n10": n10, "n_discordant": n_discordant, "p_value": p_value}


def stratified_paired_bootstrap(y_true, pred_a, pred_b, metric_name: Optional[str] = None,
                                 n_bootstrap: int = 1000, seed: int = SEED, confidence: float = 0.95) -> dict:
    metric_name = metric_name or SELECTION_METRIC
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    class_idx = {c: np.where(y_true == c)[0] for c in classes}
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([rng.choice(class_idx[c], size=len(class_idx[c]), replace=True) for c in classes])
        score_a = _score_by_name(metric_name, y_true[idx], pred_a[idx])
        score_b = _score_by_name(metric_name, y_true[idx], pred_b[idx])
        diffs.append(score_a - score_b)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    p_value = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return {"mean_diff": float(diffs.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": p_value, "n_bootstrap": n_bootstrap}


def cohens_d_paired(diffs: Sequence[float]) -> float:
    """Paired Cohen's d on the same split-level differences the
    Nadeau-Bengio test consumes: d = mean(diffs) / std(diffs, ddof=1).
    Conventional |d| bands (Cohen 1988): ~0.2 small, ~0.5 medium, ~0.8
    large — offered as a rough read only; with n=5 splits this estimate
    itself carries wide uncertainty, which is exactly why it is reported
    alongside, not instead of, the Nadeau-Bengio CI on mean_diff."""
    arr = np.asarray([d for d in diffs if np.isfinite(d)], dtype=float)
    if len(arr) < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(arr.mean() / sd)


def holm_bonferroni(p_values: dict) -> dict:
    """Holm-Bonferroni step-down correction for a family of
    pairwise tests. `p_values`: {label: p_value}. Returns, for every label,
    {"p_raw", "p_adjusted", "reject_at_0.05"} — controls the family-wise
    error rate across all labels jointly (less conservative than plain
    Bonferroni, still exact). Labels with a non-finite p_value are passed
    through untested (adjusted=nan, reject=False) and excluded from the
    step-down ordering of the finite ones."""
    finite_items = [(k, v) for k, v in p_values.items() if np.isfinite(v)]
    m = len(finite_items)
    out = {k: {"p_raw": float(v), "p_adjusted": float("nan"), "reject_at_0.05": False}
           for k, v in p_values.items()}
    if m == 0:
        return out
    order = sorted(range(m), key=lambda i: finite_items[i][1])
    running_max = 0.0
    for rank, idx in enumerate(order):
        k, p = finite_items[idx]
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        out[k]["p_adjusted"] = float(running_max)
        out[k]["reject_at_0.05"] = bool(running_max < 0.05)
    return out


def run_stage6(stage5_results: dict) -> dict:
    logger = get_section_logger("stage6_significance", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSTAGE 6 — family-vs-family significance testing [estimation]\n{'=' * 70}")
    pairs = [("fine", "merged"), ("fine", "3level"), ("merged", "3level")]
    out = {"nadeau_bengio": {}, "mcnemar_split0": {}, "bootstrap_split0": {},
           "cohens_d": {}, "holm_bonferroni": {}}

    nb_p_values = {}
    for fam_a, fam_b in pairs:
        pair_key = f"{fam_a}_vs_{fam_b}"
        diffs = []
        for split_id in VALIDATION_SPLITS:
            sa, sb = (stage5_results[fam_a]["per_split_scores"].get(split_id),
                      stage5_results[fam_b]["per_split_scores"].get(split_id))
            if sa is not None and sb is not None and np.isfinite(sa) and np.isfinite(sb):
                diffs.append(sa - sb)
        row_counts = SPLIT_ROW_COUNTS.get(SPLIT0, {})
        n_train, n_test = row_counts.get("n_train"), row_counts.get("n_test")
        result = nadeau_bengio_corrected_ttest(diffs, n_train or 1, n_test or 1)
        out["nadeau_bengio"][pair_key] = result
        nb_p_values[pair_key] = result.get("p_value", float("nan"))
        d = cohens_d_paired(diffs)
        out["cohens_d"][pair_key] = {"d": d, "n": len(diffs),
                                      "magnitude": _cohens_d_magnitude(d)}
        logger.notice(f"  [Nadeau-Bengio] {fam_a} vs {fam_b}: mean_diff={result['mean_diff']:.4f}  "
                      f"p={result.get('p_value', float('nan')):.4f}  n={result['n']}  "
                      f"Cohen's d={d:.3f} ({_cohens_d_magnitude(d)})")

        y_true_a, pred_a = stage5_results[fam_a]["predictions_by_split"].get(SPLIT0, (None, None))
        y_true_b, pred_b = stage5_results[fam_b]["predictions_by_split"].get(SPLIT0, (None, None))
        if y_true_a is not None and y_true_b is not None and len(y_true_a) == len(y_true_b):
            mc = mcnemar_exact(y_true_a, pred_a, pred_b)
            bs = stratified_paired_bootstrap(y_true_a, pred_a, pred_b, seed=SEED)
            out["mcnemar_split0"][pair_key] = mc
            out["bootstrap_split0"][pair_key] = bs
            logger.notice(f"  [split-0 McNemar] {fam_a} vs {fam_b}: p={mc['p_value']:.4f}  "
                          f"(n01={mc['n01']}, n10={mc['n10']})")
        else:
            logger.warning(f"  ⚠ [split-0 supplementary] {fam_a} vs {fam_b}: predictions not directly "
                           "comparable (row-count mismatch or missing) — skipped.")

    out["holm_bonferroni"] = holm_bonferroni(nb_p_values)
    for pair_key, hb in out["holm_bonferroni"].items():
        logger.notice(f"  [Holm-Bonferroni] {pair_key}: p_raw={hb['p_raw']:.4f} -> "
                      f"p_adjusted={hb['p_adjusted']:.4f}  reject@0.05={hb['reject_at_0.05']}")
    return out


def _cohens_d_magnitude(d: float) -> str:
    if not np.isfinite(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# Off by default — Stage 6 sits outside the Section 13 driver and is opt-in, like Stage 2 and SHAP.
RUN_STAGE6_SIGNIFICANCE_NOW = _cfg("RUN_STAGE6_SIGNIFICANCE_NOW", False)

if RUN_STAGE6_SIGNIFICANCE_NOW:
    STAGE6_RESULTS = _run_cached("stage6", lambda: run_stage6(STAGE5_RESULTS))
    _module_logger.notice("  Stage 6 significance testing complete. Note: it is no longer folded "
                           "into Stage 8's persisted report (run_stage8 dropped its \"stage6\" "
                           "parameter entirely) — inspect STAGE6_RESULTS directly.")
else:
    STAGE6_RESULTS = None
    _module_logger.notice("  DEAD CODE: Stage 6 skipped (RUN_STAGE6_SIGNIFICANCE_NOW=False) — set it "
                           "True and re-run this cell to run family-vs-family significance testing.")

# %% [markdown]
# ### DEAD CODE — SHAP explainability (was Section 18)
#
# Explains ONLY each family's Stage-7 DEPLOYED winner (never a Stage 1/3
# candidate), one hierarchy node at a time. Tree-based winners
# (`SHAP_TREE_MODEL_NAMES`) get `shap.TreeExplainer` (exact, no sampling);
# every other model type gets a sampling-based `shap.Explainer` with
# background/eval sets capped at `SHAP_MAX_BACKGROUND_SAMPLES`/
# `SHAP_MAX_EVAL_SAMPLES`. Each node's result is still cached to its own
# `VALIDATION_CHECKPOINTS_DIR/shap_<node>.joblib` (independent of the
# Section 0 stage-checkpoint machinery), and mean-|SHAP| feature-
# importance tables still go to
# `VALIDATION_CSV_DIR/shap_importance_<node>.csv` per node — the only
# change from the original Section 18 is that it no longer runs by default
# and its config knobs now live down here with the code that reads them.

# %%
RUN_SHAP_EXPLAINABILITY_NOW = _cfg("RUN_SHAP_EXPLAINABILITY_NOW", False)  # off by default, see the DEAD CODE header above
SHAP_MAX_BACKGROUND_SAMPLES = 100   # k-means-summarized or randomly sampled background for non-tree explainers
SHAP_MAX_EVAL_SAMPLES = 200         # rows actually explained per node — mean|SHAP| stabilizes well below 1000
SHAP_TREE_MODEL_NAMES = {"RandomForest", "GBC", "HistGBC", "CatBoost", "DecisionTree"}


def _shap_checkpoint_path(node_label: str) -> Path:
    return VALIDATION_CHECKPOINTS_DIR / f"shap_{_safe_filename(node_label)}.joblib"


def _build_shap_explainer(model, model_name: str, x_background: pd.DataFrame):
    """Returns (explainer, kind). TreeExplainer for SHAP_TREE_MODEL_NAMES
    (exact); otherwise a sampling-based Explainer around predict_proba
    (falling back to predict if predict_proba is unavailable), background
    capped at SHAP_MAX_BACKGROUND_SAMPLES rows."""
    import shap
    underlying = getattr(model, "model_", model)  # unwrap this pipeline's own custom estimator wrappers, if any
    if model_name in SHAP_TREE_MODEL_NAMES:
        try:
            return shap.TreeExplainer(underlying), "tree"
        except Exception as e:
            _module_logger.warning(f"  ⚠ TreeExplainer failed for '{model_name}' ({type(e).__name__}: {e}) — "
                                    "falling back to the sampling-based Explainer.")
    bg = (x_background if len(x_background) <= SHAP_MAX_BACKGROUND_SAMPLES
          else x_background.sample(SHAP_MAX_BACKGROUND_SAMPLES, random_state=SEED))
    predict_fn = underlying.predict_proba if hasattr(underlying, "predict_proba") else underlying.predict
    return shap.Explainer(predict_fn, bg), "sampling"


def compute_node_shap(node_label: str, model, model_name: str, x_train: pd.DataFrame,
                       x_eval: pd.DataFrame, force: bool = False) -> Optional[dict]:
    logger = get_section_logger("stage18_shap", VALIDATION_LOGS_DIR)
    ckpt_path = _shap_checkpoint_path(node_label)
    if ckpt_path.exists() and not force:
        try:
            payload = joblib.load(ckpt_path)
            logger.notice(f"  -> [{node_label}] SHAP loaded from checkpoint -> {ckpt_path}")
            return payload
        except Exception as e:
            logger.warning(f"  ⚠ [{node_label}] failed to load SHAP checkpoint "
                           f"({type(e).__name__}: {e}) — recomputing.")

    try:
        import shap  # noqa: F401 — availability probe; _build_shap_explainer does the real import
    except ImportError:
        logger.warning("  ⚠ 'shap' package not installed — skipping SHAP (DEAD CODE) entirely. "
                        "Install with `pip install shap` to enable it.")
        return None
    if x_train is None or x_eval is None or len(x_train) == 0 or len(x_eval) == 0:
        logger.warning(f"  ⚠ [{node_label}] empty train/eval frame — skipping.")
        return None

    x_eval_capped = (x_eval if len(x_eval) <= SHAP_MAX_EVAL_SAMPLES
                      else x_eval.sample(SHAP_MAX_EVAL_SAMPLES, random_state=SEED))
    try:
        explainer, kind = _build_shap_explainer(model, model_name, x_train)
        shap_values = explainer(x_eval_capped)
        values = np.asarray(shap_values.values)
        # Multiclass tree/sampling explainers commonly return (n_samples, n_features, n_classes);
        # binary/regression-style explainers return (n_samples, n_features) — average over whichever
        # axes are present beyond the feature axis so every node collapses to one importance-per-feature.
        if values.ndim == 3:
            mean_abs = np.mean(np.abs(values), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(values), axis=0)
    except Exception as e:
        logger.error(f"  ⚠ [{node_label}] SHAP computation FAILED ({model_name}, {type(e).__name__}: {e}) — "
                      "skipping this node; other nodes are unaffected.")
        return None

    importance_df = pd.DataFrame({"feature": list(x_eval_capped.columns),
                                   "mean_abs_shap": np.asarray(mean_abs).ravel()}
                                  ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    payload = {"node_label": node_label, "model_name": model_name, "explainer_kind": kind,
               "n_background_rows": min(len(x_train), SHAP_MAX_BACKGROUND_SAMPLES),
               "n_eval_samples": len(x_eval_capped), "importance": importance_df}
    joblib.dump(payload, ckpt_path)
    csv_path = VALIDATION_CSV_DIR / f"shap_importance_{_safe_filename(node_label)}.csv"
    importance_df.to_csv(csv_path, index=False)
    logger.notice(f"  [{node_label}] SHAP ({kind} explainer, {model_name}, n_eval={len(x_eval_capped)}) "
                   f"-> {csv_path}  (checkpoint -> {ckpt_path})")
    logger.notice(f"    top 5 features: {importance_df.head(5)['feature'].tolist()}")
    return payload


def _pool_tag_merged(ctx, tag: str, local_meta: Optional[dict]) -> pd.DataFrame:
    tr = _m2_resolve_full_length_tag_frame(ctx, tag, local_meta)
    te = _resolve_test_frame_merged(ctx, tag, local_meta)
    return pd.concat([tr, te], ignore_index=True)


def _pool_tag_3level(ctx, tag: str, local_meta: Optional[dict]) -> pd.DataFrame:
    tr = (ctx.x_train_by_tag[tag] if tag in ctx.x_train_by_tag
          else ctx.x_train_by_tag["ALL_FEATURES"][local_meta["feature_sets"][tag]])
    te = _resolve_test_frame_3level(ctx, tag, local_meta)
    return pd.concat([tr, te], ignore_index=True)


def run_stage18_shap(stage7_deployment: dict, split_contexts: dict, force: bool = False) -> dict:
    logger = get_section_logger("stage18_shap", VALIDATION_LOGS_DIR)
    logger.notice(f"\n{'=' * 70}\nSECTION 18 — SHAP explainability [optional]\n{'=' * 70}")
    fine_ctx, merged_ctx, tl_ctx = split_contexts[SPLIT0]
    out = {}

    # --- fine: one flat model -------------------------------------------------------------
    deployed_f = joblib.load(stage7_deployment["fine"]["model_path"])
    cfg_f = deployed_f["cfg"]
    # Pool train+test, matching what Stage 7 actually fit the deployed fine model on
    # (`x_pool = pd.concat([fine_ctx.x_train_by_tag[...], fine_ctx.x_test_by_tag[...]])` in
    # `run_stage7`) — using train-only rows here would make fine's SHAP background/eval set
    # inconsistent with merged/3level's nodes below, which already pool correctly.
    x_fine_pool = pd.concat([fine_ctx.x_train_by_tag[cfg_f["tag"]], fine_ctx.x_test_by_tag[cfg_f["tag"]]],
                             ignore_index=True)[deployed_f["features"]]
    out["fine_model"] = compute_node_shap("fine_model", deployed_f["model"], cfg_f["model_name"],
                                           x_fine_pool, x_fine_pool, force=force)

    # --- merged: router + each trained branch --------------------------------------------
    deployed_m = joblib.load(stage7_deployment["merged"]["model_path"])
    router_cfg_m, router_meta_m = deployed_m["router_cfg"], deployed_m["router_meta"]
    x_router_pool = _pool_tag_merged(merged_ctx, router_cfg_m["tag"], router_meta_m)
    out["merged_router"] = compute_node_shap("merged_router", deployed_m["router_model"],
                                              router_cfg_m["model_name"], x_router_pool, x_router_pool, force=force)

    y_super_pool_str = np.concatenate([merged_ctx.y_train_super_str, merged_ctx.y_test_super_str])
    for sc in SUPERCLASS_ORDER:
        bm = deployed_m["branch_models"].get(sc)
        bcfg = deployed_m["branch_cfgs"].get(sc)
        if bm is None or bcfg is None:
            continue
        sc_mask = y_super_pool_str == sc
        local_meta = deployed_m["branch_meta"].get(sc)
        try:
            x_branch_pool = _pool_tag_merged(merged_ctx, bcfg["tag"], local_meta)[sc_mask].reset_index(drop=True)
        except KeyError as e:
            logger.warning(f"  ⚠ [merged_branch_{sc}] {e} — skipping this branch's SHAP.")
            continue
        out[f"merged_branch_{sc}"] = compute_node_shap(f"merged_branch_{sc}", bm["model"], bcfg["model_name"],
                                                         x_branch_pool, x_branch_pool, force=force)

    # --- 3level: gate + blazar_subgate + nonblazar_router + each trained branch ----------
    deployed_3 = joblib.load(stage7_deployment["3level"]["model_path"])
    gate_cfg, gate_meta = deployed_3["gate_cfg"], deployed_3["gate_meta"]
    subgate_cfg, subgate_meta = deployed_3["subgate_cfg"], deployed_3["subgate_meta"]
    nonblazar_cfg, nonblazar_meta = deployed_3["nonblazar_cfg"], deployed_3["nonblazar_meta"]
    y_super_pool_str3 = np.concatenate([tl_ctx.y_train_super_str, tl_ctx.y_test_super_str])
    blazar_mask_pool = np.isin(y_super_pool_str3, list(BLAZAR_SUPERCLASSES))

    x_gate_pool = _pool_tag_3level(tl_ctx, gate_cfg["tag"], gate_meta)
    out["3level_gate"] = compute_node_shap("3level_gate", deployed_3["gate_model"], gate_cfg["model_name"],
                                            x_gate_pool, x_gate_pool, force=force)

    x_sub_pool = _pool_tag_3level(tl_ctx, subgate_cfg["tag"], subgate_meta)[blazar_mask_pool].reset_index(drop=True)
    out["3level_blazar_subgate"] = compute_node_shap("3level_blazar_subgate", deployed_3["subgate_model"],
                                                       subgate_cfg["model_name"], x_sub_pool, x_sub_pool, force=force)

    x_nb_pool = _pool_tag_3level(tl_ctx, nonblazar_cfg["tag"], nonblazar_meta)[~blazar_mask_pool].reset_index(drop=True)
    out["3level_nonblazar_router"] = compute_node_shap("3level_nonblazar_router", deployed_3["nonblazar_model"],
                                                         nonblazar_cfg["model_name"], x_nb_pool, x_nb_pool, force=force)

    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        bm = deployed_3["branch_models"].get(sc)
        bcfg = deployed_3["branch_cfgs"].get(sc)
        if bm is None or bcfg is None:
            continue
        sc_mask = y_super_pool_str3 == sc
        local_meta = deployed_3["branch_meta"].get(sc)
        try:
            x_branch_pool = _pool_tag_3level(tl_ctx, bcfg["tag"], local_meta)[sc_mask].reset_index(drop=True)
        except KeyError as e:
            logger.warning(f"  ⚠ [3level_branch_{sc}] {e} — skipping this branch's SHAP.")
            continue
        out[f"3level_branch_{sc}"] = compute_node_shap(f"3level_branch_{sc}", bm["model"], bcfg["model_name"],
                                                          x_branch_pool, x_branch_pool, force=force)

    n_ok = sum(1 for v in out.values() if v is not None)
    logger.notice(f"  SHAP (DEAD CODE) complete: {n_ok}/{len(out)} node(s) explained successfully.")
    return out


if RUN_SHAP_EXPLAINABILITY_NOW:
    SHAP_RESULTS = run_stage18_shap(STAGE7_DEPLOYMENT, SPLIT_CONTEXTS)
else:
    _module_logger.notice("  DEAD CODE: SHAP skipped (RUN_SHAP_EXPLAINABILITY_NOW=False) — set it "
                           "True (and `pip install shap` if needed) and re-run this cell to compute "
                           "per-node SHAP feature importances for every deployed winner.")
