# %% [markdown]
# # fermi_3levels.py

# %% [markdown]
# ## Hierarchical 3-Level Classification (Module 3-alt of 4)
#
# Level 1 **GATE** (BLAZAR vs OTHER) -> Level 2 {**BLAZAR_SUBGATE**
# (BLAZAR_HE vs BLAZAR_LE, direct fsrq/bll passthrough) |
# **NONBLAZAR_ROUTER** (5-way over the non-blazar superclasses)} ->
# Level 3 (per-non-blazar-superclass fine sub-classifiers). Runs against
# the shared kernel (`fermi_common.py`, already executed earlier in
# this kernel — `SEED`, `VERSION_TAG`, `SELECTION_METRIC`, `USE_GPU`,
# `MODEL_SPECS`, `ALL_MODEL_NAMES`, `PCA_BASE_TAG`, `TAXONOMY`,
# `SUPERCLASS_ORDER`, `select_and_refit_val_only`, `fit_component`,
# `get_builder`, `evaluate_model`, `plot_confusion_matrix`,
# `cv_score_estimator`, `make_run_id`, `io_for`, `get_section_logger`,
# `save_result_multi`, `publish_split_summary`, `pbar`, `_safe_filename`,
# `SPLIT_ID_DEFAULT`, `_BASE_SAFE_RUN`, etc. are simply in-scope, not
# imported cross-file) and against `fermi_processing.py`'s real,
# per-split output (already executed earlier in this kernel too).
#
# ### Physical motivation
# Setup-A/B leaderboards show fsrq/ssrq/css are empirically hard to
# separate from bll using available 4FGL features even after the physical
# fsrq/ssrq/css merge — that binary decision gets drowned out when it has
# to compete inside a crowded 7-way superclass router dominated by two
# large blazar classes. This module isolates that fsrq-vs-bll decision
# into its own dedicated sub-gate, trained only on blazar rows, rather than
# folding it into the router `fermi_merged.py` uses instead. Whether the
# resulting 3-level tree actually beats the 2-level tree end to end, on
# held-out test data, is answered by `fermi_validation.py` (no test set is
# read in this file).
#
# ## How it works
#
# - **Input layout.** Module 1 writes per split to
#   `fermi_module1_processing_{VERSION_TAG}/{split_id}/`; the feature
#   metadata's top-level key is `"setups"`. `discover_splits()` reads
#   Module 1's `multi_split_manifest.json` (entries keyed
#   `split_idx`/`split_seed`/`split_dir`, not `id`), and the whole procedure
#   is wrapped in `run_3level_module_for_split(split_id)`.
# - **Feature selection is not recomputed here.** The gate / blazar
#   sub-gate / non-blazar router / Level-3 branch feature sets come from
#   Module 1's `feature_metadata.json`
#   (`hierarchical_feature_selection.3level.{level1_gate,
#   level2_blazar_subgate,level2_nonblazar_router,level3_branches}`), sliced
#   out of the already-loaded `ALL_FEATURES` pool by `_apply_local_feature_sets`
#   — a column selection, not a second stability-selection run. Module 1
#   also decides how large a branch must be for local selection.
# - **Combo screening.** `run_gate_subgate_router_screening` sends every
#   one of the `TOP_N_GATES x TOP_N_BLAZAR_SUBGATES x TOP_N_NONBLAZAR_ROUTERS`
#   candidates (a small grid) straight to the expensive chain-CV, branch
#   configs held fixed — no coarse pre-filter at the combo level, same as
#   `fermi_merged.py`'s Section 9.
# - **Screening drives the deployed system.** Persistence is deferred until
#   *after* combo screening: the baseline and every screened combo are
#   compared on one common scale (`end_to_end_val_{SELECTION_METRIC}`), and
#   whichever wins — refit on the full training split if needed — is what
#   gets persisted and reported as `generalization_estimate`.
# - **Soft vs. hard routing is disclosed.** The chain-CV's soft probability
#   blend reverts to hard top-1 routing at any stage whose winning model has
#   no `predict_proba` (the adaptive-LOO `DecisionTree`, the GPU Torch MLP,
#   and the default `probability=False` SVM). Rather than restricting model
#   candidacy (which would change the leaderboards), `_predict_proba_safe`
#   fills a `hard_routing_log` dict on every chain-CV pass, and
#   `run_3level_module_for_split` writes it into
#   `generalization_estimate.hard_routed_stages_oof` plus a plain-language
#   `soft_routing_disclosure` string in the metrics JSON, so the paper can
#   state per split whether the adopted configuration's end-to-end metric is
#   a uniform soft blend or partially hard-routed.
# - **Namespacing.** `fermi_merged.py`'s Level-2 branches and this module's
#   Level-3 branches share superclass names and the same metric-suffixed
#   `OUTPUT_ROOT` (see Section 0), so Level-3 branches are named
#   `branch_{sc}_L3` to avoid writing to the same results/log files. Every
#   other `label_set` here (`gate`, `blazar_subgate`, `nonblazar_router`,
#   `threelevel`) is already distinct.
# - **Cross-split summary** via this module's own
#   `build_all_splits_summary_3level`.
# - **Logging / I/O** via `fermi_common.py`: `io_for(OUTPUT_ROOT, split_id)`
#   (per-split logs/results/plots), `get_section_logger` (per-node
#   `NOTICE`-floored console + full-detail log file), `save_result_multi`
#   via a `make_safe_run` closure (per-node results files),
#   `publish_split_summary` at the end of each split's run, `pbar()` around
#   every branch/split driver loop, and a resumable, fault-isolated
#   split-driver loop.
#
# ## Jupytext
# This file is written in jupytext "percent" format (`# %%` / `# %%
# [markdown]` cell markers) for clean notebook conversion, matching
# `fermi_processing.py`, `fermi_fine.py`, and `fermi_merged.py`.

# %% [markdown]
# ## Selection-bias disclosure (restated from Module 1 and `fermi_merged.py`)
#
# Every model-selection decision in this file — the gate's, blazar
# sub-gate's, non-blazar router's, and each branch's winning
# model/feature-set/weighting, and the winning
# whole-system gate/sub-gate/router combination out of the screened
# candidates — is chosen by comparing validation scores across many
# candidates. This is the standard, accepted cost of any search over many
# candidates using held-out data rather than a formally nested
# CV-within-CV scheme (prohibitively expensive here, for the same reason
# Module 1 gives for not nesting feature selection inside model
# selection): the reported validation number for whichever candidate wins
# carries a small, well-characterized optimism from having been picked,
# among many, on the same data used to score it (Cawley & Talbot, 2010,
# *Journal of Machine Learning Research* 11:2079-2107). This affects only
# the internal validation numbers used to *select* a winner — it does not
# affect any held-out test-set metric, since no test set is read anywhere
# in this file (see "no test set" note in Section 9's markdown).

# %%
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score

# %% [markdown]
# ## Section 0 — Multi-split discovery
#
# Identical to `fermi_merged.py`'s own Section 0 — reads Module 1's
# real `multi_split_manifest.json` (`"splits"` is a list of
# `{"split_idx", "split_seed", "split_dir", ...}` dicts, no `"id"` key;
# the split identifier is the basename of `"split_dir"`).
#
# `OUTPUT_ROOT` is deliberately the SAME metric-suffixed tree
# `fermi_merged.py` writes to — both hierarchical pipelines'
# gate/router/branch model selection genuinely depends on
# `SELECTION_METRIC`, so both modules' outputs are metric-suffixed the
# same way, and both benefit from landing under one
# `results_aggregated/{split_id}/` folder per split for easy
# side-by-side comparison. See the "Namespacing" note in the header for how
# this file avoids the resulting `label_set` namespace collision on branches.

# %%
PROCESSING_ROOT = _cfg("PROCESSING_ROOT", Path(f"./fermi_module1_processing_{VERSION_TAG}/"))
OUTPUT_ROOT = _cfg("OUTPUT_ROOT", Path(f"./outputs_{VERSION_TAG}_{SELECTION_METRIC}/"))

ENABLE_MULTI_SPLIT_VALIDATION = _cfg("MODULE_MULTI_SPLIT", False)  # mirrors Module 1's / fermi_merged.py's own default-off switch


def discover_splits(processing_root: Path) -> list:
    """Reads Module 1's multi_split_manifest.json for the list of split ids
    it produced. Schema verified against fermi_processing.py's actual
    process_split()/driver-loop source: manifest["splits"] is a list of
    {"split_idx", "split_seed", "split_dir", "n_train_A", ...} dicts (no
    "id" key) — the split id is the basename of "split_dir"."""
    manifest_path = processing_root / "multi_split_manifest.json"
    if not manifest_path.exists():
        print(f"  ⚠ {manifest_path} not found — assuming only "
              f"'{SPLIT_ID_DEFAULT}' exists (a single-split Module 1 run).")
        return [SPLIT_ID_DEFAULT]
    with open(manifest_path) as f:
        manifest = json.load(f)
    raw = manifest.get("splits")
    if not raw:
        print(f"  ⚠ {manifest_path} exists but has no 'splits' entries — "
              f"assuming only '{SPLIT_ID_DEFAULT}' exists.")
        return [SPLIT_ID_DEFAULT]
    split_ids = [Path(s["split_dir"]).name if isinstance(s, dict) and "split_dir" in s
                 else str(s) for s in raw]
    return split_ids or [SPLIT_ID_DEFAULT]


_DISCOVERED_SPLITS = discover_splits(PROCESSING_ROOT)
SPLITS_TO_RUN = _DISCOVERED_SPLITS if ENABLE_MULTI_SPLIT_VALIDATION else _DISCOVERED_SPLITS[:1]

print(f"Config ready — SEED={SEED}  VERSION_TAG={VERSION_TAG}  USE_GPU={USE_GPU}  "
      f"SELECTION_METRIC={SELECTION_METRIC}")
print(f"  Module 1 output root -> {PROCESSING_ROOT.resolve()}")
print(f"  Discovered splits    -> {_DISCOVERED_SPLITS}")
print(f"  ENABLE_MULTI_SPLIT_VALIDATION={ENABLE_MULTI_SPLIT_VALIDATION}  -> running: {SPLITS_TO_RUN}")
print(f"  Outputs (metric-suffix, shared with fermi_merged.py) -> {OUTPUT_ROOT.resolve()}")

# %% [markdown]
# ## Section 1 — Config
#
# No local `MODEL_SPECS` override — same posture as `fermi_merged.py`
# and the original `fermi_3levels.py`: this module uses
# `fermi_common`'s shared registry as-is. `BRANCH_MIN_TRAIN_FOR_MODEL_SWEEP`
# (=10, Level-3 branch reliability-tier threshold) is a distinct concept
# from Module 1's own feature-selection floor
# (`BRANCH_MIN_TRAIN_FOR_SELECTION_3LEVEL`, baked into which branches even
# appear in `hierarchical_feature_selection.3level.level3_branches`): the
# former governs whether THIS file's model-selection funnel is
# trustworthy at a branch's sample size, the latter governs whether
# Module 1 trusted its OWN feature-importance ranking at that sample
# size — kept as two separate constants deliberately, mirroring
# `fermi_merged.py`'s own Section 1 markdown on this exact point.
#
# `TOP_N_GATES`/`TOP_N_BLAZAR_SUBGATES`/`TOP_N_NONBLAZAR_ROUTERS` still
# govern Section 10's combo screening.

# %%
MERGE_FINE_CLASSES = {"ssrq": "fsrq", "css": "fsrq"}  # consistency-guard only — see Section 2

FINE_SETUP_KEY = "B"
THREE_LEVEL_LABEL_SET = "threelevel"  # chain-CV / combo-screening label_set prefix (distinct from "merged")

BRANCH_MIN_TRAIN_FOR_MODEL_SWEEP = 10
TOP_N_GATES = 3
TOP_N_BLAZAR_SUBGATES = 3
TOP_N_NONBLAZAR_ROUTERS = 3
HISTGBC_MAX_ITER_CEILING = 1000

SUPER_LE = LabelEncoder().fit(SUPERCLASS_ORDER)
BLAZAR_SUPERCLASSES = {"BLAZAR_HE", "BLAZAR_LE"}
NONBLAZAR_SUPERCLASS_ORDER = [sc for sc in SUPERCLASS_ORDER if sc not in BLAZAR_SUPERCLASSES]
GATE_LE = LabelEncoder().fit(["BLAZAR", "OTHER"])
BLAZAR_SUBGATE_LE = LabelEncoder().fit(sorted(BLAZAR_SUPERCLASSES))
NONBLAZAR_ROUTER_LE = LabelEncoder().fit(NONBLAZAR_SUPERCLASS_ORDER)


def build_gate_labels(y_super_str: np.ndarray) -> np.ndarray:
    """BLAZAR vs OTHER string labels derived from a superclass string array."""
    return np.where(np.isin(y_super_str, list(BLAZAR_SUPERCLASSES)), "BLAZAR", "OTHER")




# %% [markdown]
# ## Section 1c — K-Fold-only DecisionTree override [perf guard]
#
# `DTreeCVPrunedClassifier`'s adaptive LOO (left on until now, per this
# module's "no local MODEL_SPECS override" posture) compounds the same
# way it does in `fermi_merged.py`: DecisionTree is swept across
# every tag/scaler/weighting variant for the gate (Sec 4), sub-gate
# (Sec 5), non-blazar router (Sec 6), AND every Level-3 branch (Sec 7),
# then — for winners — refit again per fold in the baseline chain-CV
# (Sec 9) and again per fold per candidate in the
# TOP_N_GATES x TOP_N_BLAZAR_SUBGATES x TOP_N_NONBLAZAR_ROUTERS combo
# screening grid (Sec 10). Local copy only —
# `fermi_common.MODEL_SPECS` stays untouched for other modules.
# Deliberately NOT applied to the Sec 7 reliability "fallback"/
# "unvalidatable" branch tiers, which keep calling bare
# `get_builder("DecisionTree")` for the genuine adaptive-LOO estimator —
# same bypass convention as `fermi_merged.py`.

# %%
def _build_dtree_kfold_only(pca_variance=None, scaler_cls=None, stopping_metric=None):
    """loo_min_class_count/loo_min_branch_train=-1 -> adaptive-LOO can
    never fire; always plain StratifiedKFold. scaler_cls accepted-and-
    ignored (trees never take a scaler)."""
    return DTreeCVPrunedClassifier(
        random_state=SEED,
        loo_min_class_count=-1,
        loo_min_branch_train=-1,
        stopping_metric=stopping_metric,
    )


THREELEVEL_MODEL_SPECS = {k: dict(v) for k, v in MODEL_SPECS.items()}
THREELEVEL_MODEL_SPECS["DecisionTree"] = dict(
    build=_build_dtree_kfold_only, build_gpu=_build_dtree_kfold_only,
    weight_support="sample_weight", supports_pca=False, scale_sensitive=False,
)


def _get_builder(model_name: str):
    return get_builder(model_name, THREELEVEL_MODEL_SPECS)


# %% [markdown]
# ## Section 2 — Load Setup B + hierarchical local-selection metadata, per split
#
# Reads Module 1's exported per-tag CSVs for Setup B (fine, `bcu`
# excluded), derives the per-row superclass target from `TAXONOMY`, and
# reads Module 1's already-computed gate/blazar_subgate/nonblazar_router/
# Level-3-branch local feature-selection results directly from
# `feature_metadata.json` — no recomputation. Global
# Setup-B tags come from `feature_metadata["setups"]["B"]["feature_sets"]`
# (that block only carries `n_features`/`achieved_power`, not the column
# list itself — the column list is simply whatever columns survive after
# dropping `label`/`label_encoded`/`source_name` from that tag's own
# exported CSV, exactly as `fermi_merged.py`'s own Section 2 markdown
# notes; no separate lookup is needed). The gate's/sub-gate's/router's/
# each branch's LOCAL tags carry real feature-name lists in
# `hierarchical_feature_selection.3level.{level1_gate,
# level2_blazar_subgate,level2_nonblazar_router,level3_branches}` — those
# are sliced directly out of the always-loaded `ALL_FEATURES` pool by
# `_apply_local_feature_sets` below, at the point each node actually
# needs them (the gate needs them against the full training set; the
# sub-gate/router/branches need them against their own row-masked
# subset, so they can't be sliced once, eagerly, at context-build time).
#
# The fsrq/ssrq/css merge is NOT reapplied here — Module 1 already
# applied it once, globally, before any split (`fermi_processing.py`
# Section 1). This function instead asserts Module 1's own
# `physical_merge` block matches what this file expects and that no
# `ssrq`/`css` label is actually present in the loaded data, failing
# loudly rather than silently re-deriving something Module 1 already
# decided — the identical consistency-guard pattern
# `fermi_merged.py`'s `build_merged_context` uses (its own Punch
# item 4).

# %%
@dataclass
class ThreeLevelContext:
    """One Setup-B context for one split: global feature-set tags (for the
    gate sweep and as the starting tag list every other node extends with
    its own local tags), the shared fine/superclass label arrays, and
    Module 1's gate/blazar_subgate/nonblazar_router/branch local-selection
    metadata (raw dicts — each node's own pipeline function is responsible
    for slicing ALL_FEATURES by these, against that node's own row mask)."""
    split_id: str
    x_train_by_tag: dict
    x_test_by_tag: dict
    y_train_fine_enc: np.ndarray
    y_test_fine_enc: np.ndarray
    y_train_fine_str: np.ndarray
    y_test_fine_str: np.ndarray
    y_train_super_str: np.ndarray
    y_test_super_str: np.ndarray
    y_train_super_enc: np.ndarray
    y_test_super_enc: np.ndarray
    fine_label_encoder: LabelEncoder
    fine_class_names: np.ndarray
    tags: list = field(default_factory=list)
    gate_meta: Optional[dict] = None
    subgate_meta: Optional[dict] = None
    nonblazar_meta: Optional[dict] = None
    branch_meta: dict = field(default_factory=dict)


def _load_tag_split_with_label(source_dir: Path, tag: str, split: str):
    path = source_dir / "datasets" / f"{FINE_SETUP_KEY}_{split}_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected Module-1 export missing: {path}")
    df = pd.read_csv(path)
    y_str = df["label"].astype(str).to_numpy()
    source_name = df["source_name"].to_numpy() if "source_name" in df.columns else None
    x = df.drop(columns=["label", "label_encoded", "source_name"], errors="ignore")
    return x, y_str, source_name


def _assert_row_alignment(x_by_tag: dict, sn_by_tag: dict, tags: list, setup_label: str, split: str):
    """[Ported from fermi_fine.py / fermi_merged.py §5.3 — a real,
    previously-latent mismatch risk] Every tag's export for one
    setup/split must share identical row order before a single boolean
    mask is later applied positionally across every tag's frame. Verified
    via source_name when available; falls back to a shape-only check with
    a loud warning against an older export that predates that column."""
    ref_tag = tags[0]
    ref_sn = sn_by_tag.get(ref_tag)
    for tag in tags[1:]:
        sn = sn_by_tag.get(tag)
        if ref_sn is not None and sn is not None:
            if not np.array_equal(sn, ref_sn):
                raise ValueError(
                    f"{setup_label} [{split}]: tag '{tag}' rows are not row-aligned with "
                    f"tag '{ref_tag}' (source_name order/content differs) — every tag's "
                    "export for one setup/split must share identical row order."
                )
        elif len(x_by_tag[tag]) != len(x_by_tag[ref_tag]):
            raise ValueError(f"{setup_label} [{split}] tag '{tag}': no 'source_name' column to verify "
                              f"row alignment, and row COUNT mismatches tag '{ref_tag}' "
                              f"({len(x_by_tag[tag])} vs {len(x_by_tag[ref_tag])}) — cannot proceed safely.")


def _apply_local_feature_sets(x_all_features_masked: pd.DataFrame, local_meta: Optional[dict],
                               x_train_by_tag: dict, tags_list: list) -> None:
    """Slices Module 1's already-computed local feature
    sets straight out of the already-loaded ALL_FEATURES pool (masked to
    whatever row subset this node trains on). No compute_selection_engine
    call — Module 1 already ran that stability-selection engine once,
    against this exact row subset and target; recomputing it here would
    duplicate real cost and risk drifting out of sync with what Module 1
    persisted. `local_meta` is `None`
    when Module 1 judged this decision's row count too small to trust
    local selection at all (see `BRANCH_MIN_TRAIN_FOR_SELECTION_3LEVEL`
    in `fermi_processing.py`) — the node then sweeps only the global
    Setup-B tags."""
    if local_meta is None:
        return
    for local_tag, feats in local_meta["feature_sets"].items():
        x_train_by_tag[local_tag] = x_all_features_masked[feats].reset_index(drop=True)
        tags_list.append(local_tag)


def build_3level_context(source_dir: Path, feature_metadata: dict, split_id: str) -> ThreeLevelContext:
    setup_meta = feature_metadata["setups"][FINE_SETUP_KEY]
    all_tags = list(setup_meta["feature_sets"].keys())
    if "ALL_FEATURES" not in all_tags:
        raise ValueError(
            f"[{split_id}] Setup {FINE_SETUP_KEY}'s feature_metadata has no 'ALL_FEATURES' tag — "
            "required to build gate/blazar_subgate/nonblazar_router/branch local-selection tag frames."
        )
    if PCA_BASE_TAG not in all_tags:
        raise ValueError(f"[{split_id}] PCA_BASE_TAG='{PCA_BASE_TAG}' not among exported tags {all_tags}")

    x_train_by_tag, x_test_by_tag = {}, {}
    sn_train_by_tag, sn_test_by_tag = {}, {}
    y_train_fine_str = y_test_fine_str = None
    for tag in all_tags:
        x_tr, y_tr_str, sn_tr = _load_tag_split_with_label(source_dir, tag, "train")
        x_te, y_te_str, sn_te = _load_tag_split_with_label(source_dir, tag, "test")
        x_train_by_tag[tag], x_test_by_tag[tag] = x_tr, x_te
        sn_train_by_tag[tag], sn_test_by_tag[tag] = sn_tr, sn_te
        if y_train_fine_str is None:
            y_train_fine_str, y_test_fine_str = y_tr_str, y_te_str
    _assert_row_alignment(x_train_by_tag, sn_train_by_tag, all_tags, f"Setup {FINE_SETUP_KEY}", f"{split_id}/train")
    _assert_row_alignment(x_test_by_tag, sn_test_by_tag, all_tags, f"Setup {FINE_SETUP_KEY}", f"{split_id}/test")

    # Physical-merge consistency guard, NOT a re-merge — Module 1 already
    # merged fsrq/ssrq/css before any split/selection (Section 2 markdown).
    merge_block = feature_metadata.get("physical_merge", {})
    if not merge_block.get("applied"):
        raise ValueError(
            f"[{split_id}] Module 1's feature_metadata reports physical_merge.applied=False — "
            "this module assumes Module 1 already merged fsrq/ssrq/css; re-check upstream config."
        )
    if merge_block.get("merge_map") != MERGE_FINE_CLASSES:
        print(f"  ⚠ [{split_id}] Module 1's merge_map={merge_block.get('merge_map')} differs from "
              f"this module's expected {MERGE_FINE_CLASSES} — proceeding, but this file's own "
              "assumptions about which classes were merged may now be stale.")
    stale_labels = (set(np.unique(y_train_fine_str)) | set(np.unique(y_test_fine_str))) & set(MERGE_FINE_CLASSES)
    if stale_labels:
        raise ValueError(
            f"[{split_id}] Found un-merged label(s) {sorted(stale_labels)} in Module 1's export despite "
            "physical_merge.applied=True — Module 1's merge and this file's expectation have drifted "
            "out of sync. Do not silently re-merge; investigate the upstream mismatch first."
        )

    le_fine = LabelEncoder().fit(y_train_fine_str)
    unseen_test = set(np.unique(y_test_fine_str)) - set(le_fine.classes_)
    if unseen_test:
        raise ValueError(
            f"[{split_id}] Setup {FINE_SETUP_KEY} test split contains fine class(es) {sorted(unseen_test)} "
            "unseen in train — Module 1's own safe_encode() should already prevent this; investigate "
            "upstream before proceeding."
        )
    y_train_fine_enc = le_fine.transform(y_train_fine_str)
    y_test_fine_enc = le_fine.transform(y_test_fine_str)

    unmapped_train = sorted(set(y_train_fine_str) - set(TAXONOMY))
    unmapped_test = sorted(set(y_test_fine_str) - set(TAXONOMY))
    if unmapped_train or unmapped_test:
        raise ValueError(f"[{split_id}] Fine classes with no TAXONOMY entry: "
                          f"train={unmapped_train}, test={unmapped_test}")
    y_train_super_str = np.array([TAXONOMY[c] for c in y_train_fine_str])
    y_test_super_str = np.array([TAXONOMY[c] for c in y_test_fine_str])
    y_train_super_enc = SUPER_LE.transform(y_train_super_str)
    y_test_super_enc = SUPER_LE.transform(y_test_super_str)

    hfs = feature_metadata.get("hierarchical_feature_selection", {}).get("3level", {})
    gate_meta = hfs.get("level1_gate")
    subgate_meta = hfs.get("level2_blazar_subgate")
    nonblazar_meta = hfs.get("level2_nonblazar_router")
    branch_meta = hfs.get("level3_branches", {}) or {}
    if not hfs:
        print(f"  ⚠ [{split_id}] Module 1's feature_metadata has no "
              "hierarchical_feature_selection.3level block — every node below sweeps only "
              "Setup B's global tags (no local selection).")

    ctx_tags = [t for t in all_tags if t != "ALL_FEATURES"]

    return ThreeLevelContext(
        split_id=split_id, x_train_by_tag=x_train_by_tag, x_test_by_tag=x_test_by_tag,
        y_train_fine_enc=y_train_fine_enc, y_test_fine_enc=y_test_fine_enc,
        y_train_fine_str=y_train_fine_str, y_test_fine_str=y_test_fine_str,
        y_train_super_str=y_train_super_str, y_test_super_str=y_test_super_str,
        y_train_super_enc=y_train_super_enc, y_test_super_enc=y_test_super_enc,
        fine_label_encoder=le_fine, fine_class_names=le_fine.classes_, tags=ctx_tags,
        gate_meta=gate_meta, subgate_meta=subgate_meta, nonblazar_meta=nonblazar_meta, branch_meta=branch_meta,
    )


def plot_taxonomy_tree(fine_class_names, plots_dir: Path, logger=None):
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.axis("off")
    root_x, root_y = 0.04, 0.5
    ax.text(root_x, root_y, "4FGL\nsource", ha="center", va="center", fontsize=11,
            fontweight="bold", bbox=dict(boxstyle="round", fc="lightgray"))

    gate_x, gate_y = 0.20, 0.5
    ax.plot([root_x + 0.04, gate_x - 0.04], [root_y, gate_y], color="gray", linewidth=0.8)
    ax.text(gate_x, gate_y, "GATE\nBLAZAR/OTHER", ha="center", va="center", fontsize=9,
            fontweight="bold", bbox=dict(boxstyle="round", fc="#ffd9b3"))

    subgate_x, subgate_y = 0.40, 0.80
    ax.plot([gate_x + 0.06, subgate_x - 0.05], [gate_y + 0.02, subgate_y], color="gray", linewidth=0.8)
    ax.text(subgate_x, subgate_y, "SUBGATE\nHE/LE", ha="center", va="center", fontsize=9,
            fontweight="bold", bbox=dict(boxstyle="round", fc="#cfe8ff"))
    for fc, fy in zip(("fsrq", "bll"), (0.88, 0.72)):
        fx = 0.60
        ax.plot([subgate_x + 0.05, fx - 0.03], [subgate_y - (0.80 - fy), fy], color="lightgray", linewidth=0.6)
        ax.text(fx, fy, fc, ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round", fc="#fff2cc"))

    router_x, router_y = 0.40, 0.20
    ax.plot([gate_x + 0.06, router_x - 0.05], [gate_y - 0.02, router_y], color="gray", linewidth=0.8)
    ax.text(router_x, router_y, "NONBLAZAR\nROUTER (5-way)", ha="center", va="center", fontsize=9,
            fontweight="bold", bbox=dict(boxstyle="round", fc="#cfe8ff"))
    branch_ys = np.linspace(0.36, 0.02, len(NONBLAZAR_SUPERCLASS_ORDER))
    for sc, by in zip(NONBLAZAR_SUPERCLASS_ORDER, branch_ys):
        bx = 0.62
        ax.plot([router_x + 0.07, bx - 0.03], [router_y, by], color="gray", linewidth=0.7)
        ax.text(bx, by, sc, ha="center", va="center", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round", fc="#d9f2d9"))
        fines = sorted(c for c, s in TAXONOMY.items() if s == sc and c in fine_class_names)
        fine_ys = (np.linspace(by + 0.025 * (len(fines) - 1), by - 0.025 * (len(fines) - 1), len(fines))
                   if len(fines) > 1 else [by])
        for fc, fy in zip(fines, fine_ys):
            fx = 0.86
            ax.plot([bx + 0.05, fx - 0.03], [by, fy], color="lightgray", linewidth=0.6)
            ax.text(fx, fy, fc, ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle="round", fc="#fff2cc"))

    ax.set_title("3-Level Taxonomy: GATE → {SUBGATE (HE/LE) | NONBLAZAR ROUTER (5-way)} "
                 "→ Level-3 fine classes", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = plots_dir / "taxonomy_tree_3level.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    msg = f"  Saved -> {out_path}"
    (logger.notice(msg) if logger is not None else print(msg))


def _leaderboard_row_to_cfg(row: dict) -> dict:
    pca_variance = row.get("pca_variance")
    if pca_variance is None or (isinstance(pca_variance, float) and np.isnan(pca_variance)):
        pca_variance = None
    scaler_name = row.get("scaler_name")
    if scaler_name is None or (isinstance(scaler_name, float) and np.isnan(scaler_name)):
        scaler_name = None
    return {"model_name": row["model_name"], "tag": row["tag"], "pca_variance": pca_variance,
            "scaler_name": scaler_name, "weighting": row["weighting"]}


def _resolve_full_length_tag_frame(ctx: ThreeLevelContext, tag: str, local_meta: Optional[dict]):
    """Full-length (every row of train) frame for `tag`, resolved from
    ctx.x_train_by_tag directly for a global Setup-B tag, or sliced fresh
    from ALL_FEATURES using `local_meta`'s feature list for a gate-/
    subgate-/router-/branch-local tag. Chain-CV indexes by POSITION
    (tr_idx/eval_idx), so it always needs the full row range, never a
    pre-masked subset."""
    if tag in ctx.x_train_by_tag:
        return ctx.x_train_by_tag[tag]
    if local_meta and tag in local_meta.get("feature_sets", {}):
        return ctx.x_train_by_tag["ALL_FEATURES"][local_meta["feature_sets"][tag]]
    raise KeyError(f"Tag '{tag}' is neither a global Setup-B tag nor a known local-selection tag.")


# %% [markdown]
# ## Section 3 — Per-node `safe_run`
#
# One closure per (split, label_set), writing to both the split's full
# aggregator and that label_set's own per-node results file, with
# failures logged at `ERROR` to that node's own log file — identical
# pattern to `fermi_merged.py`'s `make_safe_run`.

# %%
def make_safe_run(io_bundle: dict, label_set: str):
    node_results_path = io_bundle["results_dir"] / f"{_safe_filename(label_set)}.json"
    logger = get_section_logger(label_set, io_bundle["logs_dir"])

    def safe_run(run_fn, *args, context="", **kwargs):
        return _BASE_SAFE_RUN(
            run_fn, *args,
            results_path=[io_bundle["split_results_path"], node_results_path],
            context=context, logger=logger, **kwargs,
        )
    return safe_run


# %% [markdown]
# ## Section 4 — Level 1: GATE (BLAZAR vs OTHER)
#
# Trained on the FULL Setup-B split — every row is either blazar or
# non-blazar. Thin wrapper around `select_and_refit_val_only`, exactly
# like the gate/sub-gate/router pipelines below and
# `fermi_merged.py`'s single router pipeline.

# %%
def run_gate_pipeline(ctx: ThreeLevelContext, io_bundle: dict):
    logger = get_section_logger("gate", io_bundle["logs_dir"])
    safe_run_gate = make_safe_run(io_bundle, "gate")
    node_results_path = io_bundle["results_dir"] / "gate.json"

    y_train_enc = GATE_LE.transform(build_gate_labels(ctx.y_train_super_str))
    class_names = GATE_LE.classes_

    x_train_by_tag = {t: x for t, x in ctx.x_train_by_tag.items() if t != "ALL_FEATURES"}
    tags = list(ctx.tags)
    _apply_local_feature_sets(ctx.x_train_by_tag["ALL_FEATURES"], ctx.gate_meta, x_train_by_tag, tags)
    if PCA_BASE_TAG not in x_train_by_tag:
        # PCA_BASE_TAG may itself be "ALL_FEATURES" (fermi_common.py
        # config) — the filter above always strips it, and _apply_local_
        # feature_sets only restores it if gate_meta happens to define a
        # local tag with that exact name. Stage 0 of select_and_refit_val_
        # only unconditionally indexes x_train_by_tag[PCA_BASE_TAG], so
        # without this guard it KeyErrors before the first candidate scores.
        x_train_by_tag[PCA_BASE_TAG] = ctx.x_train_by_tag["ALL_FEATURES"]
    if ctx.gate_meta is None:
        logger.warning("  ⚠ [gate] no Module-1 local feature selection for this node — "
                        "sweeping Setup B's global tags only.")

    winner_info, leaderboard = select_and_refit_val_only(
        x_train_by_tag, y_train_enc, class_names, "gate", tags, ALL_MODEL_NAMES,
        safe_run_gate, io_bundle["split_results_path"], context_prefix=f"{ctx.split_id}|gate",
        val_split_report_path=io_bundle["split_dir"] / "val_split_report_gate.json",
        logs_dir=io_bundle["logs_dir"], extra_results_paths=[node_results_path],
        histgbc_max_iter_ceiling=HISTGBC_MAX_ITER_CEILING,
        model_specs=THREELEVEL_MODEL_SPECS, get_builder_fn=_get_builder,
    )
    if not winner_info:
        raise RuntimeError(f"[{ctx.split_id}] LEVEL 1 GATE: every model failed selection.")

    winner_info_path = io_bundle["split_dir"] / "winner_info_L1_gate.json"
    with open(winner_info_path, "w") as f:
        json.dump(winner_info, f, indent=2, default=str)
    logger.notice(f"  Saved -> {winner_info_path}")

    row = leaderboard.iloc[0].to_dict()
    cfg = _leaderboard_row_to_cfg(row)
    logger.notice(f"→ Selected GATE: {cfg['model_name']} | val_{SELECTION_METRIC}="
                  f"{row[f'val_{SELECTION_METRIC}']:.4f}")

    gate_model = fit_component(cfg, THREELEVEL_MODEL_SPECS, _get_builder, x_train_by_tag[cfg["tag"]], y_train_enc)
    gate_info = dict(cfg)
    gate_info.update({k: row[k] for k in row if k.startswith("val_") or k.startswith("train_")})
    return gate_model, gate_info, leaderboard, winner_info, x_train_by_tag


# %% [markdown]
# ## Section 5 — Level 2a: BLAZAR SUB-GATE (BLAZAR_HE vs BLAZAR_LE)
#
# Trained ONLY on rows whose true superclass is a blazar class. Its
# prediction IS the fine-grain answer (fsrq / bll passthrough, post
# physical merge) — there is no Level-3 model for this branch. Selects
# on `SELECTION_METRIC` uniformly with every other node in this module.

# %%
def run_blazar_subgate_pipeline(ctx: ThreeLevelContext, io_bundle: dict):
    logger = get_section_logger("blazar_subgate", io_bundle["logs_dir"])
    safe_run_sg = make_safe_run(io_bundle, "blazar_subgate")
    node_results_path = io_bundle["results_dir"] / "blazar_subgate.json"

    tr_mask = np.isin(ctx.y_train_super_str, list(BLAZAR_SUPERCLASSES))
    y_train_enc = BLAZAR_SUBGATE_LE.transform(ctx.y_train_super_str[tr_mask])
    class_names = BLAZAR_SUBGATE_LE.classes_
    logger.notice(f"[{ctx.split_id}] [blazar_subgate] train={int(tr_mask.sum())}  classes={list(class_names)}")

    x_train_by_tag = {t: x[tr_mask].reset_index(drop=True)
                       for t, x in ctx.x_train_by_tag.items() if t != "ALL_FEATURES"}
    tags = list(ctx.tags)
    x_all_masked = ctx.x_train_by_tag["ALL_FEATURES"][tr_mask].reset_index(drop=True)
    _apply_local_feature_sets(x_all_masked, ctx.subgate_meta, x_train_by_tag, tags)
    if PCA_BASE_TAG not in x_train_by_tag:
        x_train_by_tag[PCA_BASE_TAG] = x_all_masked
    if ctx.subgate_meta is None:
        logger.warning("  ⚠ [blazar_subgate] no Module-1 local feature selection for this node — "
                        "sweeping Setup B's global tags only.")

    winner_info, leaderboard = select_and_refit_val_only(
        x_train_by_tag, y_train_enc, class_names, "blazar_subgate", tags, ALL_MODEL_NAMES,
        safe_run_sg, io_bundle["split_results_path"], context_prefix=f"{ctx.split_id}|blazar_subgate",
        val_split_report_path=io_bundle["split_dir"] / "val_split_report_blazar_subgate.json",
        logs_dir=io_bundle["logs_dir"], extra_results_paths=[node_results_path],
        histgbc_max_iter_ceiling=HISTGBC_MAX_ITER_CEILING,
        model_specs=THREELEVEL_MODEL_SPECS, get_builder_fn=_get_builder,
    )
    if not winner_info:
        raise RuntimeError(f"[{ctx.split_id}] LEVEL 2a BLAZAR SUB-GATE: every model failed selection.")

    winner_info_path = io_bundle["split_dir"] / "winner_info_L2_blazar_subgate.json"
    with open(winner_info_path, "w") as f:
        json.dump(winner_info, f, indent=2, default=str)
    logger.notice(f"  Saved -> {winner_info_path}")

    row = leaderboard.iloc[0].to_dict()
    cfg = _leaderboard_row_to_cfg(row)
    logger.notice(f"→ Selected BLAZAR SUB-GATE: {cfg['model_name']} | val_{SELECTION_METRIC}="
                  f"{row[f'val_{SELECTION_METRIC}']:.4f}")

    subgate_model = fit_component(cfg, THREELEVEL_MODEL_SPECS, _get_builder, x_train_by_tag[cfg["tag"]], y_train_enc)
    subgate_info = dict(cfg)
    subgate_info.update({k: row[k] for k in row if k.startswith("val_") or k.startswith("train_")})
    return subgate_model, subgate_info, leaderboard, winner_info, x_train_by_tag


# %% [markdown]
# ## Section 6 — Level 2b: NON-BLAZAR ROUTER (5-way)
#
# Trained ONLY on rows whose true superclass is NOT a blazar class —
# removes blazar row-count dominance from this competition entirely.

# %%
def run_nonblazar_router_pipeline(ctx: ThreeLevelContext, io_bundle: dict):
    logger = get_section_logger("nonblazar_router", io_bundle["logs_dir"])
    safe_run_nb = make_safe_run(io_bundle, "nonblazar_router")
    node_results_path = io_bundle["results_dir"] / "nonblazar_router.json"

    tr_mask = ~np.isin(ctx.y_train_super_str, list(BLAZAR_SUPERCLASSES))
    y_train_enc = NONBLAZAR_ROUTER_LE.transform(ctx.y_train_super_str[tr_mask])
    class_names = NONBLAZAR_ROUTER_LE.classes_
    logger.notice(f"[{ctx.split_id}] [nonblazar_router] train={int(tr_mask.sum())}  classes={list(class_names)}")

    x_train_by_tag = {t: x[tr_mask].reset_index(drop=True)
                       for t, x in ctx.x_train_by_tag.items() if t != "ALL_FEATURES"}
    tags = list(ctx.tags)
    x_all_masked = ctx.x_train_by_tag["ALL_FEATURES"][tr_mask].reset_index(drop=True)
    _apply_local_feature_sets(x_all_masked, ctx.nonblazar_meta, x_train_by_tag, tags)
    if PCA_BASE_TAG not in x_train_by_tag:
        x_train_by_tag[PCA_BASE_TAG] = x_all_masked
    if ctx.nonblazar_meta is None:
        logger.warning("  ⚠ [nonblazar_router] no Module-1 local feature selection for this node — "
                        "sweeping Setup B's global tags only.")

    winner_info, leaderboard = select_and_refit_val_only(
        x_train_by_tag, y_train_enc, class_names, "nonblazar_router", tags, ALL_MODEL_NAMES,
        safe_run_nb, io_bundle["split_results_path"], context_prefix=f"{ctx.split_id}|nonblazar_router",
        val_split_report_path=io_bundle["split_dir"] / "val_split_report_nonblazar_router.json",
        logs_dir=io_bundle["logs_dir"], extra_results_paths=[node_results_path],
        histgbc_max_iter_ceiling=HISTGBC_MAX_ITER_CEILING,
        model_specs=THREELEVEL_MODEL_SPECS, get_builder_fn=_get_builder,
    )
    if not winner_info:
        raise RuntimeError(f"[{ctx.split_id}] LEVEL 2b NON-BLAZAR ROUTER: every model failed selection.")

    winner_info_path = io_bundle["split_dir"] / "winner_info_L2_nonblazar_router.json"
    with open(winner_info_path, "w") as f:
        json.dump(winner_info, f, indent=2, default=str)
    logger.notice(f"  Saved -> {winner_info_path}")

    row = leaderboard.iloc[0].to_dict()
    cfg = _leaderboard_row_to_cfg(row)
    logger.notice(f"→ Selected NON-BLAZAR ROUTER: {cfg['model_name']} | val_{SELECTION_METRIC}="
                  f"{row[f'val_{SELECTION_METRIC}']:.4f}")

    router_model = fit_component(cfg,THREELEVEL_MODEL_SPECS, _get_builder, x_train_by_tag[cfg["tag"]], y_train_enc)
    router_info = dict(cfg)
    router_info.update({k: row[k] for k in row if k.startswith("val_") or k.startswith("train_")})
    return router_model, router_info, leaderboard, winner_info, x_train_by_tag


# %% [markdown]
# ## Section 7 — Level 3: Per-Non-Blazar-Superclass Branches (val-only, no test)
#
# Same three-tier reliability system as `fermi_merged.py`'s Level-2
# branches: "sweep"
# (`n_train >= BRANCH_MIN_TRAIN_FOR_MODEL_SWEEP`, the full multi-model
# funnel), "fallback" (fewer rows but not every class a singleton — one
# robust adaptive-LOO DecisionTree, one honest pooled-OOF estimate),
# "unvalidatable" (every class a singleton — no CV scheme can ever hold
# one out; fit-on-all-data, TRAIN-only diagnostics, explicitly not
# claimed as a generalization estimate). Only `NONBLAZAR_SUPERCLASS_ORDER`
# branches are built — blazar superclasses are handled entirely by
# `BLAZAR_SUBGATE_MODEL`'s fine-class passthrough (Section 5).
#
# Every Level-3 branch's `label_set` carries an `_L3`
# suffix (`branch_{sc}_L3`) to avoid colliding with `fermi_merged.py`'s
# Level-2 branches, which use the plain `branch_{sc}` name for the same
# superclass strings under the same shared, metric-suffixed `OUTPUT_ROOT`.

# %%
def build_branch_datasets(ctx: ThreeLevelContext, route_str_train: np.ndarray = None) -> dict:
    """`route_str_train` defaults to `ctx.y_train_super_str` (oracle
    routing — the only routing this module ever selects branch
    architecture against; Section 9's combo screening reuses these exact
    branch configs unchanged, never re-derives them — branches are held
    fixed across every gate/sub-gate/router combination)."""
    route_str_train = ctx.y_train_super_str if route_str_train is None else route_str_train
    branches = {}
    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        tr_mask = route_str_train == sc
        fine_tr = ctx.y_train_fine_str[tr_mask]
        if len(set(fine_tr)) < 2:
            branches[sc] = None
            continue
        le_branch = LabelEncoder().fit(fine_tr)
        y_tr_enc = le_branch.transform(fine_tr)

        x_train_by_tag = {tag: ctx.x_train_by_tag[tag][tr_mask].reset_index(drop=True) for tag in ctx.tags}
        branch_tags = list(ctx.tags)
        local_meta = ctx.branch_meta.get(sc)
        x_all_masked = None
        if local_meta is not None:
            x_all_masked = ctx.x_train_by_tag["ALL_FEATURES"][tr_mask].reset_index(drop=True)
            _apply_local_feature_sets(x_all_masked, local_meta, x_train_by_tag, branch_tags)
        if PCA_BASE_TAG not in x_train_by_tag:
            if x_all_masked is None:
                x_all_masked = ctx.x_train_by_tag["ALL_FEATURES"][tr_mask].reset_index(drop=True)
            x_train_by_tag[PCA_BASE_TAG] = x_all_masked

        branches[sc] = {
            "x_train_by_tag": x_train_by_tag, "tags": branch_tags,
            "y_train_enc": y_tr_enc, "class_names": le_branch.classes_,
            "label_encoder": le_branch, "n_train": int(tr_mask.sum()), "local_meta": local_meta,
        }
    return branches


def _branch_validation_tier(branch: dict):
    y = branch["y_train_enc"]
    classes, counts = np.unique(y, return_counts=True)
    n_singletons = int((counts < 2).sum())
    diag = {"n_train": branch["n_train"], "n_classes": int(len(classes)),
            "n_singleton_classes": n_singletons, "singleton_fraction": n_singletons / len(classes)}
    if n_singletons == len(classes):
        return "unvalidatable", diag
    if branch["n_train"] < BRANCH_MIN_TRAIN_FOR_MODEL_SWEEP:
        return "fallback", diag
    return "sweep", diag


def _fit_reliability_fallback_branch(sc: str, branch: dict, label_set: str, diag: dict, io_bundle: dict):
    logger = get_section_logger(label_set, io_bundle["logs_dir"])
    tag = branch["tags"][0]
    x_tr, y_tr = branch["x_train_by_tag"][tag], branch["y_train_enc"]

    cv_result = cv_score_estimator(get_builder("DecisionTree"), {}, x_tr, y_tr, "none",
                                    "sample_weight", SELECTION_METRIC, n_splits=5, seed=SEED,
                                    collect_oof=True, show_progress=False)
    zero_n = cv_result["n_structural_zero_recall_classes"]
    logger.notice(
        f"  [{label_set}] fallback DecisionTree pooled-OOF {cv_result['cv_mode']} "
        f"({cv_result['n_folds']} fold(s), {cv_result['oof_n']} pooled predictions): "
        f"{SELECTION_METRIC}={cv_result['mean']:.4f} (bootstrap std={cv_result['std']:.4f})."
        + (f" {zero_n}/{diag['n_classes']} singleton class(es) get a deterministic 0% recall under "
           "LOO (structural floor, not evidence against this model)." if zero_n else ""))

    model = get_builder("DecisionTree")()
    model.fit(x_tr, y_tr)
    train_metrics = evaluate_model(f"{label_set} [fallback DTree] [TRAIN]", y_tr, model.predict(x_tr),
                                    branch["class_names"], label_set=label_set, split="train", verbose=False)

    cfg = {"model_name": "DecisionTree", "tag": tag, "pca_variance": None, "scaler_name": None, "weighting": "none"}
    branch_result = {
        "model": model, "label_encoder": branch["label_encoder"], "model_name": "DecisionTree",
        "weighting": "none", "tag": tag, "pca_variance": None, "scaler_name": None,
        "val_accuracy": cv_result.get("accuracy_mean", cv_result["mean"]),
        "val_bal_accuracy": cv_result.get("bal_accuracy_mean", cv_result["mean"]),
        "val_macro_f1": cv_result.get("macro_f1_mean", float("nan")),
        "val_kappa": cv_result.get("kappa_mean", float("nan")),
        "train_accuracy": train_metrics["accuracy"], "train_bal_accuracy": train_metrics["bal_accuracy"],
        "train_macro_f1": train_metrics["macro_f1"], "train_kappa": train_metrics["kappa"],
        "n_classes": len(branch["class_names"]), "reliability_fallback": True, "reliability_diag": diag,
    }
    run_id = make_run_id(label_set, "DecisionTree", tag, "final_val_reliability_fallback_none")
    node_results_path = io_bundle["results_dir"] / f"{_safe_filename(label_set)}.json"
    save_result_multi(run_id, {
        "model_name": "DecisionTree", "feature_set": tag, "tag": tag,
        "variant": "final_val_reliability_fallback_none", "label_set": label_set,
        "n_features": x_tr.shape[1], "weighting": "none",
        "val_accuracy": branch_result["val_accuracy"], "val_bal_accuracy": branch_result["val_bal_accuracy"],
        "val_macro_f1": branch_result["val_macro_f1"], "val_kappa": branch_result["val_kappa"],
        "train_accuracy": branch_result["train_accuracy"], "train_bal_accuracy": branch_result["train_bal_accuracy"],
        "train_macro_f1": branch_result["train_macro_f1"], "train_kappa": branch_result["train_kappa"],
        "reliability_fallback": True, "reliability_diag": diag,
    }, [io_bundle["split_results_path"], node_results_path], logger=logger)
    return branch_result, None, cfg


def _fit_unvalidatable_branch(sc: str, branch: dict, label_set: str, diag: dict, io_bundle: dict):
    logger = get_section_logger(label_set, io_bundle["logs_dir"])
    tag = branch["tags"][0]
    x_tr, y_tr = branch["x_train_by_tag"][tag], branch["y_train_enc"]

    model = get_builder("DecisionTree")()
    model.fit(x_tr, y_tr)
    train_metrics = evaluate_model(f"{label_set} [unvalidatable — TRAIN fit only]", y_tr, model.predict(x_tr),
                                    branch["class_names"], label_set=label_set, split="train", verbose=False)
    logger.notice(f"  [{label_set}] train_bal_accuracy={train_metrics['bal_accuracy']:.4f} (memorization, "
                  "NOT a generalization estimate — every class has exactly 1 training example; collect "
                  ">=2 examples/class, or fold this superclass into a coarser catalog level, before "
                  "trusting this branch's contribution to end-to-end balanced accuracy).")

    cfg = {"model_name": "DecisionTree", "tag": tag, "pca_variance": None, "scaler_name": None, "weighting": "none"}
    branch_result = {
        "model": model, "label_encoder": branch["label_encoder"], "model_name": "DecisionTree",
        "weighting": "none", "tag": tag, "pca_variance": None, "scaler_name": None,
        "val_accuracy": float("nan"), "val_bal_accuracy": float("nan"),
        "val_macro_f1": float("nan"), "val_kappa": float("nan"),
        "train_accuracy": train_metrics["accuracy"], "train_bal_accuracy": train_metrics["bal_accuracy"],
        "train_macro_f1": train_metrics["macro_f1"], "train_kappa": train_metrics["kappa"],
        "n_classes": len(branch["class_names"]), "unvalidatable": True, "reliability_diag": diag,
    }
    run_id = make_run_id(label_set, "DecisionTree", tag, "final_unvalidatable_no_cv")
    node_results_path = io_bundle["results_dir"] / f"{_safe_filename(label_set)}.json"
    save_result_multi(run_id, {
        "model_name": "DecisionTree", "feature_set": tag, "tag": tag,
        "variant": "final_unvalidatable_no_cv", "label_set": label_set,
        "n_features": x_tr.shape[1], "weighting": "none",
        "val_accuracy": None, "val_bal_accuracy": None, "val_macro_f1": None, "val_kappa": None,
        "train_accuracy": branch_result["train_accuracy"], "train_bal_accuracy": branch_result["train_bal_accuracy"],
        "train_macro_f1": branch_result["train_macro_f1"], "train_kappa": branch_result["train_kappa"],
        "unvalidatable": True, "reliability_diag": diag,
    }, [io_bundle["split_results_path"], node_results_path], logger=logger)
    return branch_result, None, cfg


def run_level3_branch(sc: str, branch: dict, io_bundle: dict, split_id: str):
    """Trains one Level-3 branch under its reliability tier (see Section 7)."""
    label_set = f"branch_{sc}_L3"
    logger = get_section_logger(label_set, io_bundle["logs_dir"])
    logger.notice(f"[{split_id}] LEVEL 3 [{label_set}] — {len(branch['class_names'])} fine classes, "
                  f"train={branch['n_train']} (val-only, no test)")

    tier, diag = _branch_validation_tier(branch)
    if tier == "unvalidatable":
        logger.warning(f"  ⚠ [{label_set}] n_train={diag['n_train']} == n_classes={diag['n_classes']} "
                        "(every class has exactly 1 example) — no CV scheme can produce a held-out "
                        "estimate. Fitting on all data, TRAIN-only diagnostics; NOT reporting a "
                        "val_bal_accuracy for this branch.")
        return _fit_unvalidatable_branch(sc, branch, label_set, diag, io_bundle)
    if tier == "fallback":
        logger.notice(f"  ℹ [{label_set}] n_train={diag['n_train']} < {BRANCH_MIN_TRAIN_FOR_MODEL_SWEEP} — "
                       f"skipping the {len(ALL_MODEL_NAMES)}-model x scaler x feature-set x weighting sweep "
                       "for COMPUTE BUDGET only (the pooled-OOF metric is honest, not noise-limited). "
                       "Using a single robust adaptive-LOO DecisionTree.")
        return _fit_reliability_fallback_branch(sc, branch, label_set, diag, io_bundle)

    safe_run_branch = make_safe_run(io_bundle, label_set)
    node_results_path = io_bundle["results_dir"] / f"{_safe_filename(label_set)}.json"
    winner_info, leaderboard = select_and_refit_val_only(
        branch["x_train_by_tag"], branch["y_train_enc"], branch["class_names"], label_set,
        branch["tags"], ALL_MODEL_NAMES, safe_run_branch, io_bundle["split_results_path"],
        context_prefix=f"{split_id}|{label_set}",
        val_split_report_path=io_bundle["split_dir"] / f"val_split_report_{_safe_filename(label_set)}.json",
        logs_dir=io_bundle["logs_dir"], extra_results_paths=[node_results_path],
        histgbc_max_iter_ceiling=HISTGBC_MAX_ITER_CEILING,
        model_specs=THREELEVEL_MODEL_SPECS, get_builder_fn=_get_builder,
    )
    if not winner_info:
        logger.warning(f"  ⚠ {label_set}: every model failed selection — branch skipped "
                        "(falls back to majority-class at inference).")
        return None, None, None

    row = leaderboard.iloc[0].to_dict()
    cfg = _leaderboard_row_to_cfg(row)
    logger.notice(f"→ Branch winner [{label_set}]: {cfg['model_name']} / {cfg['tag']} / {cfg['weighting']}  "
                  f"val_{SELECTION_METRIC}={row[f'val_{SELECTION_METRIC}']:.4f}")

    model = fit_component(cfg, THREELEVEL_MODEL_SPECS, _get_builder, branch["x_train_by_tag"][cfg["tag"]], branch["y_train_enc"])
    branch_result = {
        "model": model, "label_encoder": branch["label_encoder"], "model_name": cfg["model_name"],
        "weighting": cfg["weighting"], "tag": cfg["tag"], "pca_variance": cfg["pca_variance"],
        "scaler_name": cfg["scaler_name"],
        "val_accuracy": row["val_accuracy"], "val_bal_accuracy": row["val_bal_accuracy"],
        "val_macro_f1": row["val_macro_f1"], "val_kappa": row["val_kappa"],
        "train_accuracy": row["train_accuracy"], "train_bal_accuracy": row["train_bal_accuracy"],
        "train_macro_f1": row["train_macro_f1"], "train_kappa": row["train_kappa"],
        "n_classes": len(branch["class_names"]),
    }
    return branch_result, leaderboard, cfg


def train_branch_models(ctx: ThreeLevelContext, io_bundle: dict):
    branch_datasets = build_branch_datasets(ctx)
    branch_models, branch_leaderboards, branch_cfgs = {}, {}, {}
    for sc in pbar(NONBLAZAR_SUPERCLASS_ORDER, desc=f"[{ctx.split_id}] Level-3 branches"):
        branch = branch_datasets.get(sc)
        if branch is None:
            logger = get_section_logger(f"branch_{sc}_L3", io_bundle["logs_dir"])
            logger.notice(f"LEVEL 3 [branch_{sc}_L3]: single fine class — no sub-classifier trained.")
            branch_models[sc] = branch_leaderboards[sc] = branch_cfgs[sc] = None
            continue
        bm, lb, cfg = run_level3_branch(sc, branch, io_bundle, ctx.split_id)
        branch_models[sc], branch_leaderboards[sc], branch_cfgs[sc] = bm, lb, cfg
    return branch_datasets, branch_models, branch_leaderboards, branch_cfgs


# %% [markdown]
# ## Section 8 — Persist Fitted GATE/SUBGATE/ROUTER + Branch Models
#
# Writes the actual fitted estimators, label encoders, and each
# component's exact feature-column list — read directly off the
# DataFrame each component was fit on, never a separate metadata lookup
# that could drift out of sync (the same philosophy `fermi_merged.py`'s
# `persist_hierarchical_models` uses) — to `{split_dir}/inference_3level/`
# (a DIFFERENT subfolder from the 2-level module's `inference_2level/`,
# even though both live under the same metric-suffixed `OUTPUT_ROOT`).
# Called once, AFTER Section 9 decides the final
# gate/sub-gate/router combination — see the driver function near the
# bottom of this file — so what gets persisted here always reflects
# whichever combo actually won.

# %%
def persist_3level_models(gate_model, gate_cfg, gate_x_by_tag,
                           subgate_model, subgate_cfg, subgate_x_by_tag,
                           nonblazar_model, nonblazar_cfg, nonblazar_x_by_tag,
                           branch_models: dict, ctx: ThreeLevelContext, branch_datasets: dict,
                           io_bundle: dict, adopted_from: str):
    inference_dir = io_bundle["split_dir"] / "inference_3level"
    inference_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version_tag": VERSION_TAG, "selection_metric": SELECTION_METRIC, "split_id": ctx.split_id,
        "fine_setup_used_for_all_levels": FINE_SETUP_KEY, "adopted_from": adopted_from,
        "gate": None, "blazar_subgate": None, "nonblazar_router": None,
        "branches": {}, "blazar_fine_passthrough": {},
    }

    def _dump_stage(name, model, cfg, x_by_tag, class_list):
        path = inference_dir / f"{name}_model.joblib"
        joblib.dump(model, path)
        return {
            "model_name": cfg["model_name"], "tag": cfg["tag"],
            "pca_variance": cfg.get("pca_variance"), "scaler_name": cfg.get("scaler_name"),
            "features": list(x_by_tag[cfg["tag"]].columns),
            "model_path": str(path.relative_to(io_bundle["split_dir"])),
            "label_encoder_classes": list(class_list),
        }

    manifest["gate"] = _dump_stage("gate", gate_model, gate_cfg, gate_x_by_tag, GATE_LE.classes_)
    manifest["blazar_subgate"] = _dump_stage("blazar_subgate", subgate_model, subgate_cfg,
                                              subgate_x_by_tag, BLAZAR_SUBGATE_LE.classes_)
    manifest["nonblazar_router"] = _dump_stage("nonblazar_router", nonblazar_model, nonblazar_cfg,
                                                nonblazar_x_by_tag, NONBLAZAR_ROUTER_LE.classes_)

    for sc in BLAZAR_SUPERCLASSES:
        fines = sorted(c for c, s in TAXONOMY.items() if s == sc and c in ctx.fine_class_names)
        if len(fines) != 1:
            raise ValueError(f"Expected exactly one fine class for '{sc}' post-merge, found {fines}")
        manifest["blazar_fine_passthrough"][sc] = fines[0]

    branch_fallback_fine = {}
    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        tr_mask_sc = ctx.y_train_super_str == sc
        fine_tr_sc = ctx.y_train_fine_str[tr_mask_sc]
        if len(fine_tr_sc) > 0:
            vals, counts = np.unique(fine_tr_sc, return_counts=True)
            branch_fallback_fine[sc] = str(vals[np.argmax(counts)])

    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        bm = branch_models.get(sc)
        if bm is None:
            manifest["branches"][sc] = {"status": "passthrough_or_failed",
                                         "fallback_fine_class": branch_fallback_fine.get(sc)}
            continue
        model_path = inference_dir / f"branch_{_safe_filename(sc)}_model.joblib"
        le_path = inference_dir / f"branch_{_safe_filename(sc)}_label_encoder.joblib"
        joblib.dump(bm["model"], model_path)
        joblib.dump(bm["label_encoder"], le_path)
        branch_frame = branch_datasets[sc]["x_train_by_tag"][bm["tag"]]
        manifest["branches"][sc] = {
            "status": "trained", "model_name": bm["model_name"], "tag": bm["tag"],
            "features": list(branch_frame.columns),
            "model_path": str(model_path.relative_to(io_bundle["split_dir"])),
            "label_encoder_path": str(le_path.relative_to(io_bundle["split_dir"])),
            "fine_classes": list(bm["label_encoder"].classes_),
            "fallback_fine_class": branch_fallback_fine.get(sc),
        }

    manifest_path = inference_dir / "hierarchical_inference_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    n_trained = sum(1 for b in manifest["branches"].values() if b.get("status") == "trained")
    logger = get_section_logger("_module", io_bundle["logs_dir"])
    logger.notice(f"  Persisted gate/subgate/nonblazar_router + {n_trained} branch model(s) "
                  f"[{adopted_from}] -> {inference_dir}")
    logger.notice(f"  Manifest -> {manifest_path}")
    return manifest


# %% [markdown]
# ## Section 9 — End-to-end CHAIN cross-validation (gate -> {subgate|router} -> branch), NO TEST
#
# K=5 StratifiedKFold over the full FINE_CTX train split, stratified on
# the 7-way superclass label so gate/sub-gate/router/branch routing all
# see representative folds. Every fold refits the gate, sub-gate, router,
# and every Level-3 branch **from scratch on that fold's training rows
# only**, using each component's already-selected configuration (no
# re-sweep inside the CV loop — see the selection-bias disclosure above
# for why that matters), routes the fold's held-out rows through the
# full chain, and pools out-of-fold fine-class predictions across all
# folds before scoring — the same pooled-OOF philosophy
# `cv_score_estimator` uses at the single-model level (Section 1.3c of
# `fermi_common.py`), applied here to the whole tree at once.
#
# Routing is **soft**: *P(fine class c) = P(BLAZAR)·P(subclass|BLAZAR)*
# for blazar subclasses, *P(OTHER)·P(superclass|OTHER)·P(c|superclass)*
# otherwise — every branch contributes to every row's final answer,
# weighted by the gate's and router's confidence, rather than committing
# irrevocably to one path per row. This blend is only
# genuinely uniform when every stage's winning model exposes
# `predict_proba`; `_predict_proba_safe` falls back to a hard one-hot
# decision for any stage whose model doesn't (a custom decision tree, the
# GPU Torch MLP, and the default `probability=False` SVM), and now
# records exactly which stage(s) that happened for via `hard_routing_log`
# — surfaced in the persisted metrics (Section 12) rather than left
# silent.
#
# **No test set is ever read anywhere in this module** — every number
# this file produces (this section's chain-CV included) is a
# cross-validation or train-refit estimate computed entirely from the
# training split, by design: it keeps the test set clean for whichever
# downstream evaluation (Module 4) is meant to report the paper's actual,
# once-only, held-out generalization number. This module's own
# `generalization_estimate` is therefore a model-*selection* metric, not
# a claim of final test accuracy.

# %%
def _predict_proba_safe(model, X, n_classes, stage_name: Optional[str] = None,
                         hard_routing_log: Optional[dict] = None):
    """Falls back to a one-hot matrix
    from .predict() when a model lacks predict_proba — reproduces hard
    top-1 routing for that ONE stage while every other stage in the same
    chain-CV pass stays soft-blended. `hard_routing_log`, when given,
    records which (stage_name -> model class) pairs actually fell back
    this fold, so the module's own metrics disclose exactly how
    consistently the "soft probability blend" held rather than leaving it
    a silent, undisclosed inconsistency.

    Coercion is duck-typed (`hasattr(..., "to_numpy"/"values")`)
    rather than `isinstance(proba, pd.DataFrame)` — the GPU builders in
    this module (_CumlSampleWeightAdapter, _CumlSVCSampleWeightAdapter)
    can return a cuDF DataFrame, which is not a pd.DataFrame subclass and
    was previously slipping past this guard unconverted, crashing the
    downstream p_branch[:, k] / p_gate[:, k] positional indexing with a
    pandas-style KeyError instead of triggering the intended fallback."""
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if isinstance(proba, np.ndarray):
                pass
            elif hasattr(proba, "to_numpy"):     # pandas.DataFrame / cudf.DataFrame
                proba = proba.to_numpy()
            elif hasattr(proba, "values"):        # fallback for older DataFrame-likes
                proba = np.asarray(proba.values)
            else:
                proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] == n_classes:
                return proba
        except Exception:
            pass
    if hard_routing_log is not None and stage_name is not None:
        hard_routing_log[stage_name] = type(model).__name__
    pred_idx = np.asarray(model.predict(X)).reshape(-1)
    onehot = np.zeros((len(pred_idx), n_classes))
    onehot[np.arange(len(pred_idx)), pred_idx.astype(int)] = 1.0
    return onehot


def _fit_and_route_chain_fold(ctx: ThreeLevelContext, tr_idx: np.ndarray, eval_idx: np.ndarray,
                               gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs: dict,
                               gate_meta, subgate_meta, nonblazar_meta, branch_datasets: dict,
                               seed=SEED, hard_routing_log: Optional[dict] = None):
    y_super_full, y_fine_full = ctx.y_train_super_str, ctx.y_train_fine_str
    y_super_tr = y_super_full[tr_idx]

    x_gate_full = _resolve_full_length_tag_frame(ctx, gate_cfg["tag"], gate_meta)
    gate_fold = fit_component(gate_cfg, THREELEVEL_MODEL_SPECS, _get_builder, x_gate_full.iloc[tr_idx],
                               GATE_LE.transform(build_gate_labels(y_super_tr)), seed)

    blazar_mask_tr = np.isin(y_super_tr, list(BLAZAR_SUPERCLASSES))
    x_sub_full = _resolve_full_length_tag_frame(ctx, subgate_cfg["tag"], subgate_meta)
    subgate_fold = fit_component(subgate_cfg, THREELEVEL_MODEL_SPECS, _get_builder,
                                  x_sub_full.iloc[tr_idx][blazar_mask_tr],
                                  BLAZAR_SUBGATE_LE.transform(y_super_tr[blazar_mask_tr]), seed)

    nonblazar_mask_tr = ~blazar_mask_tr
    x_nb_full = _resolve_full_length_tag_frame(ctx, nonblazar_cfg["tag"], nonblazar_meta)
    nonblazar_fold = fit_component(nonblazar_cfg, THREELEVEL_MODEL_SPECS, _get_builder,
                                    x_nb_full.iloc[tr_idx][nonblazar_mask_tr],
                                    NONBLAZAR_ROUTER_LE.transform(y_super_tr[nonblazar_mask_tr]), seed)

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
        le_branch = LabelEncoder().fit(fine_tr_sc)
        branch_local_meta = branch_datasets.get(sc, {}).get("local_meta") if branch_datasets.get(sc) else None
        x_branch_full = _resolve_full_length_tag_frame(ctx, cfg["tag"], branch_local_meta)
        model = fit_component(cfg, THREELEVEL_MODEL_SPECS, _get_builder, x_branch_full.iloc[tr_idx][sc_mask_tr],
                               le_branch.transform(fine_tr_sc), seed)
        branch_folds[sc] = (model, le_branch)

    x_gate_eval = x_gate_full.iloc[eval_idx]
    p_gate = _predict_proba_safe(gate_fold, x_gate_eval, len(GATE_LE.classes_), "gate", hard_routing_log)
    p_blazar = p_gate[:, list(GATE_LE.classes_).index("BLAZAR")]
    p_other = p_gate[:, list(GATE_LE.classes_).index("OTHER")]

    p_subgate = _predict_proba_safe(subgate_fold, x_sub_full.iloc[eval_idx], len(BLAZAR_SUBGATE_LE.classes_),
                                     "blazar_subgate", hard_routing_log)
    p_router = _predict_proba_safe(nonblazar_fold, x_nb_full.iloc[eval_idx], len(NONBLAZAR_ROUTER_LE.classes_),
                                    "nonblazar_router", hard_routing_log)

    all_superclasses = list(SUPER_LE.classes_)
    super_idx = {s: i for i, s in enumerate(all_superclasses)}
    n_eval = len(eval_idx)
    p_super = np.zeros((n_eval, len(all_superclasses)))
    for j, sc in enumerate(BLAZAR_SUBGATE_LE.classes_):
        p_super[:, super_idx[sc]] = p_blazar * p_subgate[:, j]
    for j, sc in enumerate(NONBLAZAR_ROUTER_LE.classes_):
        p_super[:, super_idx[sc]] = p_other * p_router[:, j]
    pred_super_eval = np.array(all_superclasses)[p_super.argmax(axis=1)]

    all_fine_classes = list(ctx.fine_class_names)
    fine_idx = {c: i for i, c in enumerate(all_fine_classes)}
    p_fine = np.zeros((n_eval, len(all_fine_classes)))

    for sc in BLAZAR_SUPERCLASSES:
        fines = sorted(c for c, s in TAXONOMY.items() if s == sc and c in ctx.fine_class_names)
        if len(fines) != 1:
            raise ValueError(f"Expected exactly one fine class for '{sc}' post-merge, found {fines}")
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
        x_branch_full = _resolve_full_length_tag_frame(ctx, branch_cfgs[sc]["tag"], branch_local_meta)
        x_branch_eval = x_branch_full.iloc[eval_idx]
        p_branch = _predict_proba_safe(model, x_branch_eval, len(le_branch.classes_), f"branch_{sc}",
                                        hard_routing_log)
        for k, fc in enumerate(le_branch.classes_):
            p_fine[:, fine_idx[fc]] += p_sc * p_branch[:, k]

    pred_fine_eval = np.array(all_fine_classes)[p_fine.argmax(axis=1)]
    return pred_fine_eval, pred_super_eval


def _end_to_end_chain_cv(ctx: ThreeLevelContext, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs: dict,
                          gate_meta, subgate_meta, nonblazar_meta, branch_datasets: dict, io_bundle: dict,
                          label_set: str, n_splits=5, seed=SEED):
    logger = get_section_logger(label_set, io_bundle["logs_dir"])
    y_super_str, y_fine_str = ctx.y_train_super_str, ctx.y_train_fine_str
    n = len(y_super_str)

    classes, counts = np.unique(y_super_str, return_counts=True)
    rare_mask = np.isin(y_super_str, classes[counts < n_splits])
    strat_idx, rare_idx = np.where(~rare_mask)[0], np.where(rare_mask)[0]
    y_strat = y_super_str[strat_idx]
    strat_min = int(np.unique(y_strat, return_counts=True)[1].min()) if len(strat_idx) else 0
    n_eff = max(2, min(n_splits, strat_min)) if strat_min >= 2 else 2
    skf = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)
    cv_mode = f"StratifiedKFold(n_splits={n_eff}, {len(rare_idx)} rare-superclass row(s) train-only)"
    logger.notice(f"  [{label_set}] chain CV: {cv_mode}")

    oof_fine_pred = np.empty(n, dtype=object)
    oof_super_pred = np.empty(n, dtype=object)
    oof_touched = np.zeros(n, dtype=bool)
    hard_routing_log: dict = {}

    for fold_no, (tr_local, va_local) in enumerate(skf.split(strat_idx, y_strat), start=1):
        tr_idx = np.concatenate([strat_idx[tr_local], rare_idx])
        va_idx = strat_idx[va_local]
        pred_fine_va, pred_super_va = _fit_and_route_chain_fold(
            ctx, tr_idx, va_idx, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs,
            gate_meta, subgate_meta, nonblazar_meta, branch_datasets, seed=seed,
            hard_routing_log=hard_routing_log)
        oof_super_pred[va_idx], oof_fine_pred[va_idx], oof_touched[va_idx] = pred_super_va, pred_fine_va, True
        logger.info(f"    fold {fold_no}/{n_eff} done ({len(va_idx)} held-out rows)")

    touched = oof_touched
    y_true_fine_enc = ctx.fine_label_encoder.transform(y_fine_str[touched])
    y_pred_fine_enc = ctx.fine_label_encoder.transform(oof_fine_pred[touched])
    y_true_super_enc = SUPER_LE.transform(y_super_str[touched])
    y_pred_super_enc = SUPER_LE.transform(oof_super_pred[touched])

    end_to_end_val = evaluate_model(f"End-to-end 3-level CHAIN CV [{label_set}]", y_true_fine_enc,
                                     y_pred_fine_enc, ctx.fine_label_encoder.classes_,
                                     label_set=label_set, split="val", logger=logger)

    all_idx = np.arange(n)
    train_hard_routing_log: dict = {}
    pred_fine_tr, _ = _fit_and_route_chain_fold(
        ctx, all_idx, all_idx, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs,
        gate_meta, subgate_meta, nonblazar_meta, branch_datasets, seed=seed,
        hard_routing_log=train_hard_routing_log)
    y_true_fine_tr_enc = ctx.fine_label_encoder.transform(y_fine_str)
    y_pred_fine_tr_enc = ctx.fine_label_encoder.transform(pred_fine_tr)
    end_to_end_train = evaluate_model(f"End-to-end 3-level CHAIN CV [{label_set}] TRAIN REFIT",
                                       y_true_fine_tr_enc, y_pred_fine_tr_enc,
                                       ctx.fine_label_encoder.classes_, label_set=label_set,
                                       split="train", verbose=False, logger=logger)

    if hard_routing_log:
        logger.notice(f"  [{label_set}] {len(hard_routing_log)} stage(s) fell back to HARD "
                      f"top-1 routing at least once during chain-CV (no predict_proba): "
                      f"{hard_routing_log} — every other stage stayed soft-blended.")
    else:
        logger.notice(f"  [{label_set}] every stage exposed predict_proba — the end-to-end "
                      "metric below is a genuine, uniform soft-probability blend across all three levels.")

    # Per-fine-class recall diagnostic — distinguishes the strict, official
    # balanced-accuracy metric (which counts every class equally, including
    # near-zero-support ones under OOF) from a macro recall excluding
    # sample-starved classes, mirroring fermi_merged.py's own diagnostic.
    fine_names = np.asarray(ctx.fine_label_encoder.classes_)
    labels_arr = np.arange(len(fine_names))
    per_class_recall = recall_score(y_true_fine_enc, y_pred_fine_enc, labels=labels_arr,
                                     average=None, zero_division=0)
    support_by_label = dict(zip(*np.unique(y_true_fine_enc, return_counts=True)))
    diag_rows = sorted([(fine_names[i], float(per_class_recall[i]), int(support_by_label.get(i, 0)))
                         for i in labels_arr], key=lambda r: r[1])
    for name, rec, n_oof in diag_rows:
        flag = "  <-- n_oof<10, near-uninformative under this CV, not evidence of a bad model" if n_oof < 10 else ""
        logger.info(f"    {name:>10s}: recall={rec:.3f}  n_oof={n_oof}{flag}")
    _well_sampled = [r for _, r, n_oof in diag_rows if n_oof >= 10]
    macro_recall_excl_rare = float(np.mean(_well_sampled)) if _well_sampled else float("nan")
    logger.notice(f"  [{label_set}] macro recall excl. {len(diag_rows) - len(_well_sampled)} sample-starved "
                  f"class(es): {macro_recall_excl_rare:.4f}  vs. strict balanced_accuracy="
                  f"{end_to_end_val['bal_accuracy']:.4f}")

    plot_confusion_matrix(f"end_to_end_fine_3level_chainCV_{_safe_filename(label_set)}", y_true_fine_enc,
                           y_pred_fine_enc, ctx.fine_label_encoder.classes_, label_set=label_set,
                           plots_dir=io_bundle["plots_dir"], logger=logger)
    routing_val = evaluate_model(f"Full 3-level routing CHAIN CV [{label_set}]", y_true_super_enc,
                                  y_pred_super_enc, SUPER_LE.classes_, label_set=label_set,
                                  split="val", logger=logger)
    return {"end_to_end_val": end_to_end_val, "end_to_end_train": end_to_end_train,
            "routing_val": routing_val, "cv_mode": cv_mode, "n_folds": n_eff,
            "n_oof_rows": int(touched.sum()), "hard_routed_stages_oof": hard_routing_log,
            "macro_recall_excl_rare_classes": macro_recall_excl_rare}


# %% [markdown]
# ## Section 10 — GATE x SUBGATE x ROUTER combo screening (branches held fixed)
#
# `TOP_N_GATES x TOP_N_BLAZAR_SUBGATES x TOP_N_NONBLAZAR_ROUTERS`
# already-CV-selected candidates are crossed into a grid; every candidate
# goes straight to the expensive chain-CV, with no coarse single-split
# pre-filter, mirroring `fermi_merged.py`'s own Section 9
# (same rationale: the search space here is small enough, and branch
# configs stay fixed throughout, that the coarse-then-fine funnel's
# noise/prune-risk isn't worth it at the combo level — unlike fine
# Setup-A/B or gate/subgate/router/branch selection, where the funnel
# earns its keep on much larger candidate counts).

# %%
def run_gate_subgate_router_screening(ctx: ThreeLevelContext, gate_leaderboard: pd.DataFrame,
                                       subgate_leaderboard: pd.DataFrame, nonblazar_leaderboard: pd.DataFrame,
                                       branch_cfgs: dict, branch_datasets: dict,
                                       gate_meta, subgate_meta, nonblazar_meta, io_bundle: dict):
    logger = get_section_logger(THREE_LEVEL_LABEL_SET, io_bundle["logs_dir"])

    top_gate_rows = (gate_leaderboard.sort_values(f"val_{SELECTION_METRIC}", ascending=False)
                                      .head(TOP_N_GATES).to_dict("records"))
    top_subgate_rows = (subgate_leaderboard.sort_values(f"val_{SELECTION_METRIC}", ascending=False)
                                            .head(TOP_N_BLAZAR_SUBGATES).to_dict("records"))
    top_nonblazar_rows = (nonblazar_leaderboard.sort_values(f"val_{SELECTION_METRIC}", ascending=False)
                                                .head(TOP_N_NONBLAZAR_ROUTERS).to_dict("records"))
    n_combos = len(top_gate_rows) * len(top_subgate_rows) * len(top_nonblazar_rows)
    logger.notice(f"[{THREE_LEVEL_LABEL_SET}_screen] FULL pass (no coarse filter) — {len(top_gate_rows)} "
                  f"gate(s) x {len(top_subgate_rows)} subgate(s) x {len(top_nonblazar_rows)} router(s) = "
                  f"{n_combos} combo(s), branch configs held fixed, ALL going directly to expensive chain CV")

    combo_lookup = {}
    hierarchical_eval_by_combo = {}
    for grow in top_gate_rows:
        for srow in top_subgate_rows:
            for nrow in top_nonblazar_rows:
                combo_name = f"{grow['model_name']}+{srow['model_name']}+{nrow['model_name']}"
                gate_cfg, subgate_cfg, nonblazar_cfg = (_leaderboard_row_to_cfg(grow),
                                                          _leaderboard_row_to_cfg(srow),
                                                          _leaderboard_row_to_cfg(nrow))
                combo_lookup[combo_name] = {"combo_name": combo_name, "gate_cfg": gate_cfg,
                                             "subgate_cfg": subgate_cfg, "nonblazar_cfg": nonblazar_cfg}
                logger.notice(f"→ [{combo_name}] expensive K-fold/LOO chain CV...")
                ev = _end_to_end_chain_cv(ctx, gate_cfg, subgate_cfg, nonblazar_cfg, branch_cfgs,
                                           gate_meta, subgate_meta, nonblazar_meta, branch_datasets, io_bundle,
                                           label_set=f"{THREE_LEVEL_LABEL_SET}__{_safe_filename(combo_name)}")
                hierarchical_eval_by_combo[combo_name] = ev

    metric_col = f"end_to_end_val_{SELECTION_METRIC}"
    rows = [{
        "combo": name, "gate_model": combo_lookup[name]["gate_cfg"]["model_name"],
        "subgate_model": combo_lookup[name]["subgate_cfg"]["model_name"],
        "router_model": combo_lookup[name]["nonblazar_cfg"]["model_name"],
        "routing_val_accuracy": ev["routing_val"]["accuracy"],
        "end_to_end_val_accuracy": ev["end_to_end_val"]["accuracy"],
        "end_to_end_val_bal_accuracy": ev["end_to_end_val"]["bal_accuracy"],
        "end_to_end_val_macro_f1": ev["end_to_end_val"]["macro_f1"],
        "end_to_end_val_kappa": ev["end_to_end_val"]["kappa"],
        metric_col: ev["end_to_end_val"][SELECTION_METRIC],
        "n_folds": ev["n_folds"], "n_oof_rows": ev["n_oof_rows"], "cv_mode": ev["cv_mode"],
        "hard_routed_stages_oof": ev["hard_routed_stages_oof"],
    } for name, ev in hierarchical_eval_by_combo.items()]
    combo_comparison = pd.DataFrame(rows).sort_values(metric_col, ascending=False) if rows else pd.DataFrame()
    if not combo_comparison.empty:
        out_csv = io_bundle["split_dir"] / "gate_subgate_router_combo_comparison.csv"
        combo_comparison.to_csv(out_csv, index=False)
        logger.notice(f"  Saved -> {out_csv}  ({len(combo_comparison)}/{n_combos} combo(s), full set, "
                      "no coarse filter)")

    return hierarchical_eval_by_combo, combo_lookup, combo_comparison


def _refit_gate_subgate_router_full(ctx: ThreeLevelContext, gate_cfg: dict, subgate_cfg: dict,
                                     nonblazar_cfg: dict, seed=SEED):
    """Full-training-split refit of one gate+subgate+router combo — needed
    only when a Section-10 screening candidate beats the Section 4-6
    baseline (whose own gate/subgate/router models already ARE full
    refits, from select_and_refit_val_only's own final pass). Branch
    models never need refitting here — they're held fixed across every
    combo (Section 7's `build_branch_datasets`)."""
    tr_mask_blazar = np.isin(ctx.y_train_super_str, list(BLAZAR_SUPERCLASSES))
    tr_mask_nonblazar = ~tr_mask_blazar

    x_gate_full = _resolve_full_length_tag_frame(ctx, gate_cfg["tag"], ctx.gate_meta)
    gate_model = fit_component(gate_cfg, THREELEVEL_MODEL_SPECS, _get_builder, x_gate_full,
                                GATE_LE.transform(build_gate_labels(ctx.y_train_super_str)), seed)

    x_sub_full = _resolve_full_length_tag_frame(ctx, subgate_cfg["tag"], ctx.subgate_meta)
    x_sub_blazar = x_sub_full[tr_mask_blazar].reset_index(drop=True)
    subgate_model = fit_component(subgate_cfg, THREELEVEL_MODEL_SPECS, _get_builder, x_sub_blazar,
                                   BLAZAR_SUBGATE_LE.transform(ctx.y_train_super_str[tr_mask_blazar]), seed)

    x_nb_full = _resolve_full_length_tag_frame(ctx, nonblazar_cfg["tag"], ctx.nonblazar_meta)
    x_nb_nonblazar = x_nb_full[tr_mask_nonblazar].reset_index(drop=True)
    nonblazar_model = fit_component(nonblazar_cfg, THREELEVEL_MODEL_SPECS, _get_builder, x_nb_nonblazar,
                                     NONBLAZAR_ROUTER_LE.transform(ctx.y_train_super_str[tr_mask_nonblazar]), seed)

    gate_x_by_tag = {gate_cfg["tag"]: x_gate_full}
    subgate_x_by_tag = {subgate_cfg["tag"]: x_sub_blazar}
    nonblazar_x_by_tag = {nonblazar_cfg["tag"]: x_nb_nonblazar}
    return (gate_model, gate_x_by_tag), (subgate_model, subgate_x_by_tag), (nonblazar_model, nonblazar_x_by_tag)


# %% [markdown]
# ## Section 11 — MLP architecture comparison (gate + subgate + router + every branch)

# %%
def compare_mlp_architectures(leaderboards_by_label_set: dict, io_bundle: dict, split_id: str):
    logger = get_section_logger(THREE_LEVEL_LABEL_SET, io_bundle["logs_dir"])
    sort_col = f"val_{SELECTION_METRIC}"
    rows = []
    for label_set, board in leaderboards_by_label_set.items():
        if board is None or board.empty:
            continue
        mlp_rows = board[board["model_name"].isin(MLP_ARCHITECTURES.keys())].copy()
        if mlp_rows.empty:
            continue
        mlp_rows["label_set"] = label_set
        mlp_rows["n_hidden_layers"] = mlp_rows["model_name"].map(lambda m: len(MLP_ARCHITECTURES[m]))
        mlp_rows["n_total_neurons"] = mlp_rows["model_name"].map(lambda m: int(sum(MLP_ARCHITECTURES[m])))
        rows.append(mlp_rows)
    comparison = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if comparison.empty:
        logger.warning("  ⚠ No MLP_* rows found in any leaderboard — skipping comparison.")
        return comparison

    comparison = comparison.sort_values(["label_set", sort_col], ascending=[True, False])
    for label_set in leaderboards_by_label_set:
        sub = comparison[comparison["label_set"] == label_set]
        if sub.empty:
            continue
        best_row = sub.iloc[0]
        logger.notice(f"→ {label_set} best MLP topology: {best_row['model_name']} "
                      f"({MLP_ARCHITECTURES[best_row['model_name']]}) {sort_col}={best_row[sort_col]:.4f}")
        fig, ax = plt.subplots(figsize=(9, max(4, len(sub) * 0.5)))
        sub_sorted = sub.sort_values(sort_col)
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(sub_sorted)))
        bars = ax.barh(sub_sorted["model_name"], sub_sorted[sort_col], color=colors, edgecolor="k", linewidth=0.4)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
        ax.set_xlabel(f"{sort_col} (best feature-set/weighting config, validation)")
        ax.set_title(f"MLP Topology Comparison — {label_set} [{split_id}]")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        out_path = io_bundle["plots_dir"] / f"mlp_architecture_comparison_{_safe_filename(label_set)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.notice(f"  Saved -> {out_path}")

    out_csv = io_bundle["split_dir"] / "mlp_architecture_comparison_3level.csv"
    comparison.to_csv(out_csv, index=False)
    logger.notice(f"  Saved -> {out_csv}")
    return comparison


# %% [markdown]
# ## Section 12 — Run one split end to end (`run_3level_module_for_split`)
#
# Orchestrates one split: builds the context (Section 2), runs the gate
# (Section 4), the blazar sub-gate (Section 5), the non-blazar router
# (Section 6), the Level-3 branches (Section 7), the baseline chain-CV
# (Section 9), the gate/subgate/router combo screening (Section 10),
# resolves the final winning combo, refits/persists it (Section 8), runs
# the MLP comparison (Section 11), writes the structured metrics JSON
# (with `generalization_estimate` and `hard_routed_stages_oof` sourced
# from whichever combo actually won), and publishes
# this split's final artifacts.

# %%
def run_3level_module_for_split(split_id: str) -> dict:
    source_dir = PROCESSING_ROOT / split_id
    feature_metadata_path = source_dir / "diagnostics" / "feature_metadata.json"
    if not feature_metadata_path.exists():
        raise FileNotFoundError(
            f"{feature_metadata_path} not found — run fermi_processing.py (Module 1) first, "
            f"or check that split '{split_id}' actually exists under PROCESSING_ROOT."
        )
    with open(feature_metadata_path) as f:
        feature_metadata = json.load(f)

    io_bundle = io_for(OUTPUT_ROOT, split_id)
    module_logger = get_section_logger("_module", io_bundle["logs_dir"])
    module_logger.notice(f"\n{'=' * 78}\nMODULE 3-ALT (Hierarchical 3-Level) — split '{split_id}'\n{'=' * 78}")
    module_logger.notice(f"  datasets -> {(source_dir / 'datasets').resolve()}")
    module_logger.notice(f"  results  -> {io_bundle['results_dir'].resolve()}")
    module_logger.notice(f"  logs     -> {io_bundle['logs_dir'].resolve()}")
    module_logger.notice(f"  plots    -> {io_bundle['plots_dir'].resolve()}")

    ctx = build_3level_context(source_dir, feature_metadata, split_id)
    module_logger.notice(f"Setup {FINE_SETUP_KEY} loaded: {len(ctx.fine_class_names)} fine classes, "
                          f"{len(SUPERCLASS_ORDER)} superclasses, tags={ctx.tags}, "
                          f"train={len(ctx.y_train_fine_enc):>5}, test={len(ctx.y_test_fine_enc):>5}")
    for sc in SUPERCLASS_ORDER:
        n_fine = len({c for c, s in TAXONOMY.items() if s == sc} & set(ctx.fine_class_names))
        module_logger.notice(f"  {sc:16s}: {n_fine} fine class(es) present"
                              f"{'  [BLAZAR]' if sc in BLAZAR_SUPERCLASSES else ''}")
    plot_taxonomy_tree(ctx.fine_class_names, io_bundle["plots_dir"], logger=module_logger)

    # --- Level 1: GATE -------------------------------------------------------
    gate_model, gate_info, gate_leaderboard, gate_winner_info, gate_x_by_tag = run_gate_pipeline(ctx, io_bundle)

    # --- Level 2a: BLAZAR SUB-GATE --------------------------------------------
    subgate_model, subgate_info, subgate_leaderboard, subgate_winner_info, subgate_x_by_tag = \
        run_blazar_subgate_pipeline(ctx, io_bundle)

    # --- Level 2b: NON-BLAZAR ROUTER -------------------------------------------
    nonblazar_model, nonblazar_info, nonblazar_leaderboard, nonblazar_winner_info, nonblazar_x_by_tag = \
        run_nonblazar_router_pipeline(ctx, io_bundle)

    # --- Level 3: branches -----------------------------------------------------
    branch_datasets, branch_models, branch_leaderboards, branch_cfgs = train_branch_models(ctx, io_bundle)

    # --- Baseline end-to-end chain-CV (Sections 4-6's winners) ------------------
    baseline_eval = _end_to_end_chain_cv(ctx, gate_info, subgate_info, nonblazar_info, branch_cfgs,
                                          ctx.gate_meta, ctx.subgate_meta, ctx.nonblazar_meta, branch_datasets,
                                          io_bundle, label_set=THREE_LEVEL_LABEL_SET)

    # --- GATE x SUBGATE x ROUTER combo screening (branches held fixed) ----------
    hierarchical_eval_by_combo, combo_lookup, combo_comparison = run_gate_subgate_router_screening(
        ctx, gate_leaderboard, subgate_leaderboard, nonblazar_leaderboard, branch_cfgs, branch_datasets,
        ctx.gate_meta, ctx.subgate_meta, ctx.nonblazar_meta, io_bundle)
    

    # --- Top-15 (of 27) gate x subgate x router combo summary (viz + CSV) ---
    # Branches are held FIXED across all 27 screened combos (by design —
    # see Section 9's markdown) — `branch_cfgs` below is that one fixed
    # composition, same object already used for the baseline chain-CV
    # call above. Also adds end_to_end_TRAIN_* alongside VAL_*, neither
    # currently persisted together anywhere upstream.
    TOP_K_SUMMARY_3LEVEL = 15
    _branches_str = "/".join(f"{sc}={(branch_cfgs.get(sc) or {}).get('model_name', 'passthrough')}"
                              for sc in NONBLAZAR_SUPERCLASS_ORDER)
    _rows = []
    for _combo_name, _ev in hierarchical_eval_by_combo.items():
        _entry = combo_lookup[_combo_name]
        _gate_cfg, _subgate_cfg, _router_cfg = _entry["gate_cfg"], _entry["subgate_cfg"], _entry["nonblazar_cfg"]
        _rows.append({
            "combo": _combo_name,
            "full_model_name": (f"gate:{_gate_cfg['model_name']}[{_gate_cfg['tag']}] + "
                                 f"subgate:{_subgate_cfg['model_name']}[{_subgate_cfg['tag']}] + "
                                 f"router:{_router_cfg['model_name']}[{_router_cfg['tag']}] + "
                                 f"branches:{_branches_str}"),
            "gate_model": _gate_cfg["model_name"], "subgate_model": _subgate_cfg["model_name"],
            "nonblazar_router_model": _router_cfg["model_name"],
            "end_to_end_val_accuracy": _ev["end_to_end_val"]["accuracy"],
            "end_to_end_val_bal_accuracy": _ev["end_to_end_val"]["bal_accuracy"],
            "end_to_end_val_macro_f1": _ev["end_to_end_val"]["macro_f1"],
            "end_to_end_val_kappa": _ev["end_to_end_val"]["kappa"],
            "end_to_end_train_accuracy": _ev["end_to_end_train"]["accuracy"],
            "end_to_end_train_bal_accuracy": _ev["end_to_end_train"]["bal_accuracy"],
            "end_to_end_train_macro_f1": _ev["end_to_end_train"]["macro_f1"],
            "end_to_end_train_kappa": _ev["end_to_end_train"]["kappa"],
            "n_folds": _ev["n_folds"], "cv_mode": _ev["cv_mode"],
        })
    _summary = (pd.DataFrame(_rows).sort_values(f"end_to_end_val_{SELECTION_METRIC}", ascending=False)
                .head(TOP_K_SUMMARY_3LEVEL).reset_index(drop=True))
    _summary.insert(0, "rank", np.arange(1, len(_summary) + 1))

    module_logger.notice(f"\nTOP-{len(_summary)} (of {len(hierarchical_eval_by_combo)}) GATE x SUBGATE x ROUTER "
                          f"COMBOS — split '{split_id}' (sorted by end_to_end_val_{SELECTION_METRIC})")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        module_logger.notice(_summary.to_string(index=False))

    top15_csv = io_bundle["split_dir"] / f"top{TOP_K_SUMMARY_3LEVEL}_summary_3level.csv"
    _summary.to_csv(top15_csv, index=False)
    module_logger.notice(f"  Saved -> {top15_csv}")
    

    # --- resolve the FINAL combo across baseline + screened survivors ---
    best_candidate = ("section4_6_baseline", baseline_eval, gate_info, subgate_info, nonblazar_info,
                       baseline_eval["end_to_end_val"][SELECTION_METRIC])
    for combo_name, ev in hierarchical_eval_by_combo.items():
        score = ev["end_to_end_val"][SELECTION_METRIC]
        if score > best_candidate[5]:
            entry = combo_lookup[combo_name]
            best_candidate = (f"combo_screening:{combo_name}", ev, entry["gate_cfg"], entry["subgate_cfg"],
                               entry["nonblazar_cfg"], score)
    adopted_from, final_eval, final_gate_cfg, final_subgate_cfg, final_nonblazar_cfg, final_score = best_candidate

    if adopted_from == "section4_6_baseline":
        final_gate_model, final_gate_x_by_tag = gate_model, gate_x_by_tag
        final_subgate_model, final_subgate_x_by_tag = subgate_model, subgate_x_by_tag
        final_nonblazar_model, final_nonblazar_x_by_tag = nonblazar_model, nonblazar_x_by_tag
        module_logger.notice(f"[wiring fix] combo screening did not beat the gate/subgate/router baseline "
                              f"({final_score:.4f}) — baseline combo adopted, unchanged.")
    else:
        module_logger.notice(f"[wiring fix] combo screening candidate '{adopted_from}' beat the baseline "
                              f"({final_score:.4f} vs. {baseline_eval['end_to_end_val'][SELECTION_METRIC]:.4f}) — "
                              "refitting gate/subgate/router on the full training split and adopting it "
                              "for persistence/reporting.")
        (final_gate_model, final_gate_x_by_tag), (final_subgate_model, final_subgate_x_by_tag), \
            (final_nonblazar_model, final_nonblazar_x_by_tag) = _refit_gate_subgate_router_full(
                ctx, final_gate_cfg, final_subgate_cfg, final_nonblazar_cfg)

    hierarchical_inference_manifest = persist_3level_models(
        final_gate_model, final_gate_cfg, final_gate_x_by_tag,
        final_subgate_model, final_subgate_cfg, final_subgate_x_by_tag,
        final_nonblazar_model, final_nonblazar_cfg, final_nonblazar_x_by_tag,
        branch_models, ctx, branch_datasets, io_bundle, adopted_from,
    )

    # --- MLP comparison ----------------------------------------------------------
    leaderboards_by_label_set = {"gate": gate_leaderboard, "blazar_subgate": subgate_leaderboard,
                                  "nonblazar_router": nonblazar_leaderboard}
    leaderboards_by_label_set.update({f"branch_{sc}_L3": lb for sc, lb in branch_leaderboards.items()
                                       if lb is not None})
    mlp_comparison = compare_mlp_architectures(leaderboards_by_label_set, io_bundle, split_id)

    # --- Structured metrics persistence -------------------------------------------
    def _branch_metric_note(sc, bm):
        if bm is not None:
            return {k: v for k, v in bm.items() if k not in ("model", "label_encoder")}
        if branch_datasets.get(sc) is None:
            return {"note": "single fine class — no sub-classifier needed"}
        return {"note": "every model/config failed for this branch — falls back to majority training-set "
                         "fine class at inference"}

    metrics = {
        "seed": SEED, "version_tag": VERSION_TAG, "use_gpu": USE_GPU, "split_id": split_id,
        "selection_metric": SELECTION_METRIC, "fine_setup_used_for_all_levels": FINE_SETUP_KEY,
        "hierarchy": "3-level: gate(BLAZAR/OTHER) -> blazar_subgate(BLAZAR_HE/LE, direct fine passthrough) "
                     "| nonblazar_router(5-way) -> level3 branch models",
        "blazar_superclasses": sorted(BLAZAR_SUPERCLASSES),
        "nonblazar_superclass_order": NONBLAZAR_SUPERCLASS_ORDER,
        "level1_gate_local_feature_selection_source": "fermi_processing.py feature_metadata.json "
            "(hierarchical_feature_selection.3level.level1_gate)",
        "level1_gate_local_feature_selection": ctx.gate_meta,
        "level2_blazar_subgate_local_feature_selection_source": "fermi_processing.py feature_metadata.json "
            "(hierarchical_feature_selection.3level.level2_blazar_subgate)",
        "level2_blazar_subgate_local_feature_selection": ctx.subgate_meta,
        "level2_nonblazar_router_local_feature_selection_source": "fermi_processing.py feature_metadata.json "
            "(hierarchical_feature_selection.3level.level2_nonblazar_router)",
        "level2_nonblazar_router_local_feature_selection": ctx.nonblazar_meta,
        "level3_tags_swept": ctx.tags,
        "level3_local_feature_selection_source": "fermi_processing.py feature_metadata.json "
            "(hierarchical_feature_selection.3level.level3_branches)",
        "level3_local_feature_selection": ctx.branch_meta,
        "superclass_order": SUPERCLASS_ORDER, "taxonomy": TAXONOMY,
        "scaler_variants_swept": list(SCALER_VARIANTS.keys()), "default_scaler": DEFAULT_SCALER_NAME,
        "pca_variants_swept": list(PCA_VARIANTS.keys()),
        "mlp_architectures_swept": {k: list(v) for k, v in MLP_ARCHITECTURES.items()},
        "level1_gate": gate_info, "level1_gate_leaderboard": gate_leaderboard.to_dict("records"),
        "level2_blazar_subgate": subgate_info,
        "level2_blazar_subgate_leaderboard": subgate_leaderboard.to_dict("records"),
        "level2_nonblazar_router": nonblazar_info,
        "level2_nonblazar_router_leaderboard": nonblazar_leaderboard.to_dict("records"),
        "level3_branches": {sc: _branch_metric_note(sc, bm) for sc, bm in branch_models.items()},
        "hierarchical_evaluation_baseline": baseline_eval,
        "hierarchical_evaluation_by_combo": hierarchical_eval_by_combo,
        "combo_comparison_csv": str(io_bundle["split_dir"] / "gate_subgate_router_combo_comparison.csv"),
        "adopted_combo": adopted_from,
        "generalization_estimate": {
            "source": f"end_to_end_chain_cv_oof ({adopted_from})",
            "accuracy": final_eval["end_to_end_val"]["accuracy"],
            "bal_accuracy": final_eval["end_to_end_val"]["bal_accuracy"],
            "macro_f1": final_eval["end_to_end_val"]["macro_f1"],
            "kappa": final_eval["end_to_end_val"]["kappa"],
            "n_folds": final_eval["n_folds"], "n_oof_rows": final_eval["n_oof_rows"],
            "hard_routed_stages_oof": final_eval.get("hard_routed_stages_oof", {}),
        },
        "soft_routing_disclosure": (
            "End-to-end fine-class probabilities are a soft blend across gate -> {blazar_subgate | "
            "nonblazar_router} -> branch, weighted by each upstream stage's predicted probability, EXCEPT "
            "for a stage whose winning model exposes no predict_proba (the adaptive-LOO DecisionTree, the "
            "GPU Torch MLP, and the default probability=False SVM are the three such families in this "
            "pipeline's MODEL_SPECS) — that one stage falls back to a hard top-1 decision for that pass. "
            "See generalization_estimate.hard_routed_stages_oof above for exactly which stage(s), if any, "
            "fell back for the adopted configuration."
        ),
        "selection_bias_disclosure": (
            "Gate, blazar sub-gate, non-blazar router, branch, and (if adopted_combo != "
            "'section4_6_baseline') whole-system configuration were all chosen by comparing validation "
            "scores across many candidates on the same data used to score them — a small, well-characterized "
            "optimism (Cawley & Talbot, 2010, JMLR 11:2079-2107) affecting these validation numbers, not any "
            "held-out test-set metric, since no test set is read anywhere in this file."
        ),
        "inference_note": (
            "Unassociated-source inference (fermi_inference.py) uses the same soft-probability "
            "gate/subgate/router/branch blend this file's chain-CV validates; the 3-level vs 2-level "
            "comparison on held-out test data is made by fermi_validation.py."
        ),
    }
    metrics_path = io_bundle["split_dir"] / "3level_classification_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    module_logger.notice(f"Hierarchical metrics (3-level) written -> {metrics_path}")

    module_logger.notice("=" * 70)
    module_logger.notice(f"Module 3-alt (Hierarchical 3-Level) complete — split '{split_id}', "
                          "VALIDATION ONLY, NO TEST TOUCHED.")
    module_logger.notice(f"  Selection metric        : {SELECTION_METRIC}")
    module_logger.notice(f"  Level-1 GATE            : {gate_info['model_name']} "
                          f"(val_{SELECTION_METRIC}={gate_info.get(f'val_{SELECTION_METRIC}', float('nan')):.4f})")
    module_logger.notice(f"  Level-2a BLAZAR SUBGATE  : {subgate_info['model_name']} "
                          f"(val_{SELECTION_METRIC}="
                          f"{subgate_info.get(f'val_{SELECTION_METRIC}', float('nan')):.4f})")
    module_logger.notice(f"  Level-2b NONBLAZAR ROUTER: {nonblazar_info['model_name']} "
                          f"(val_{SELECTION_METRIC}="
                          f"{nonblazar_info.get(f'val_{SELECTION_METRIC}', float('nan')):.4f})")
    n_trained = sum(1 for b in branch_models.values() if b is not None)
    module_logger.notice(f"  Level-3 branches        : {n_trained}/{len(NONBLAZAR_SUPERCLASS_ORDER)} trained")
    module_logger.notice(f"  Adopted combo           : {adopted_from}")
    module_logger.notice(f"  End-to-end val {SELECTION_METRIC}    : {final_score:.4f}")
    module_logger.notice(f"  Metrics file            : {metrics_path}")
    module_logger.notice("=" * 70)

    summary_files = [
        io_bundle["metadata_manifest_path"], metrics_path,
        io_bundle["split_dir"] / "gate_subgate_router_combo_comparison.csv",
        io_bundle["split_dir"] / "mlp_architecture_comparison_3level.csv",
        io_bundle["split_dir"] / "winner_info_L1_gate.json",
        io_bundle["split_dir"] / "winner_info_L2_blazar_subgate.json",
        io_bundle["split_dir"] / "winner_info_L2_nonblazar_router.json",
        io_bundle["split_dir"] / "inference_3level" / "hierarchical_inference_manifest.json",
    ]
    publish_split_summary(split_id, OUTPUT_ROOT, summary_files)

    return {
        "split_id": split_id, "gate_info": gate_info, "subgate_info": subgate_info,
        "nonblazar_info": nonblazar_info, "branch_models": branch_models,
        "baseline_eval": baseline_eval, "final_eval": final_eval, "adopted_from": adopted_from,
        "final_score": final_score, "mlp_comparison": mlp_comparison,
    }


# %% [markdown]
# ## Section 13 — Cross-split summary
#
# `fermi_merged.py`'s own Section 12 (Cell B) calls
# `build_all_splits_summary(OUTPUT_ROOT, ALL_SPLITS_RESULTS)` — a name
# that does not exist anywhere in this pipeline (checked against
# `fermi_common.py` and against `fermi_merged.py`'s own source),
# which would raise `NameError` the first time that cell is reached.
# `fermi_fine.py` instead defines its own small, self-contained,
# module-specific summary builder (`build_all_splits_summary_fine`). This
# file follows that working pattern — `build_all_splits_summary_3level`
# — rather than the broken call.

# %%
def build_all_splits_summary_3level(output_root: Path, all_splits_results: dict):
    """One row per split: gate/subgate/router winners, which combo was
    adopted, and the final end-to-end validation score — the table that
    answers "how much does the winning 3-level configuration move across
    independent draws of the data," the entire point of Module 1's
    multi-split design once ENABLE_MULTI_SPLIT_VALIDATION is turned on."""
    rows = []
    for split_id, result in all_splits_results.items():
        rows.append({
            "split_id": split_id,
            "gate_model": result["gate_info"]["model_name"],
            "blazar_subgate_model": result["subgate_info"]["model_name"],
            "nonblazar_router_model": result["nonblazar_info"]["model_name"],
            "adopted_from": result["adopted_from"],
            f"end_to_end_val_{SELECTION_METRIC}": result["final_score"],
            "end_to_end_val_accuracy": result["final_eval"]["end_to_end_val"]["accuracy"],
            "end_to_end_val_bal_accuracy": result["final_eval"]["end_to_end_val"]["bal_accuracy"],
            "end_to_end_val_macro_f1": result["final_eval"]["end_to_end_val"]["macro_f1"],
            "end_to_end_val_kappa": result["final_eval"]["end_to_end_val"]["kappa"],
        })
    if not rows:
        print("  ⚠ build_all_splits_summary_3level: nothing to summarize.")
        return None
    combined = pd.DataFrame(rows)
    out_dir = output_root / "results_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_splits_summary_3level.csv"
    combined.to_csv(out_path, index=False)
    print(f"  ✓ all_splits_summary_3level -> {out_path}  ({len(combined)} split(s))")
    return out_path


def _reconstruct_result_from_metrics(split_id: str, metrics_path: Path) -> dict:
    """Read back an already-completed split's 3level_classification_metrics.json
    instead of recomputing it. Shape-compatible with
    run_3level_module_for_split()'s return value for everything
    build_all_splits_summary_3level() actually reads — mirrors
    fermi_merged.py's own helper of the same name (same resumable
    driver-loop design)."""
    with open(metrics_path) as f:
        mm = json.load(f)
    gen = mm["generalization_estimate"]
    return {
        "split_id": split_id,
        "gate_info": mm["level1_gate"], "subgate_info": mm["level2_blazar_subgate"],
        "nonblazar_info": mm["level2_nonblazar_router"],
        "adopted_from": mm["adopted_combo"], "final_score": gen[SELECTION_METRIC],
        "final_eval": {"end_to_end_val": {
            "accuracy": gen["accuracy"], "bal_accuracy": gen["bal_accuracy"],
            "macro_f1": gen["macro_f1"], "kappa": gen["kappa"],
        }},
        "branch_models": {sc: (None if "note" in v else True) for sc, v in mm["level3_branches"].items()},
    }


# %% [markdown]
# ### Section 13 — driver loop (CELL A)
#
# Resumable and fault-isolated, same mechanism as `fermi_merged.py`'s
# driver loop (only the completion-marker filename changes): a split whose
# `results_aggregated/{split_id}/3level_classification_metrics.json`
# already exists is read back, not recomputed; any other split's crash is
# caught, its traceback written to
# `results_aggregated/{split_id}/CRASH_traceback.txt`, and the loop moves
# on to the next split instead of losing every already-completed split's
# work. Re-running this cell after fixing whatever broke is always safe.

# %%
import traceback

ALL_SPLITS_RESULTS = {}
FAILED_SPLITS = []
for _split_id in pbar(SPLITS_TO_RUN, desc="3-level module — splits"):
    _done_path = OUTPUT_ROOT / "results_aggregated" / _split_id / "3level_classification_metrics.json"
    if _done_path.exists():
        print(f"  ⏭ [{_split_id}] already complete — reading {_done_path}, no recompute.")
        ALL_SPLITS_RESULTS[_split_id] = _reconstruct_result_from_metrics(_split_id, _done_path)
        continue
    try:
        ALL_SPLITS_RESULTS[_split_id] = run_3level_module_for_split(_split_id)
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

# %% [markdown]
# ### Section 13 — summary build (CELL B)
#
# Separate cell; run only once `FAILED_SPLITS` is empty (or intentionally
# accepted as-is).

# %%
ALL_SPLITS_SUMMARY_PATH = build_all_splits_summary_3level(OUTPUT_ROOT, ALL_SPLITS_RESULTS)

print("=" * 78)
print("Module 3-alt (Hierarchical 3-Level) — ALL REQUESTED SPLITS COMPLETE.")
print(f"  Splits run           : {SPLITS_TO_RUN}")
print(f"  Cross-split summary  : {ALL_SPLITS_SUMMARY_PATH}")
print(f"  Per-split artifacts  : {OUTPUT_ROOT / 'results_aggregated'}/{{split_id}}/")
print("=" * 78)

# %%
import glob, pandas as pd

def reprint_top_summaries(output_root, split_ids):
    for split_id in split_ids:
        split_dir = output_root / split_id
        for csv_path in sorted(split_dir.glob("top*_summary*.csv")):
            df = pd.read_csv(csv_path)
            print(f"\nTOP-{len(df)} — {csv_path.name}  [{split_id}]")
            with pd.option_context("display.max_columns", None, "display.width", 240):
                print(df.to_string(index=False))

reprint_top_summaries(OUTPUT_ROOT, SPLITS_TO_RUN)

# %% [markdown]
#
