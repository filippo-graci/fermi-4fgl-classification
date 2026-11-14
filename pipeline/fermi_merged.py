# %% [markdown]
# # fermi_merged.py

# %% [markdown]
# ##  Hierarchical Merged Classification (Module 3 of 4)
#
# Two-level pipeline: a Level-1 superclass router feeds soft routing
# probabilities into seven independently-specialized Level-2 per-superclass
# branch classifiers. Runs against the shared kernel (`fermi_common.py`,
# already executed earlier in this kernel — `SEED`, `VERSION_TAG`,
# `SELECTION_METRIC`, `USE_GPU`, `MODEL_SPECS`, `ALL_MODEL_NAMES`,
# `PCA_BASE_TAG`, `TAXONOMY`, `SUPERCLASS_ORDER`, `select_and_refit_val_only`,
# `fit_component`, `get_builder`, `carve_train_core_val_split`,
# `coarse_filter`, `_score_by_name`, `COARSE_VAL_FRACTION`,
# `COARSE_MIN_SURVIVORS`, `evaluate_model`, `plot_confusion_matrix`,
# `plot_comparison_table`, `io_for`, `get_section_logger`,
# `save_result_multi`, `publish_split_summary`, `pbar`, `_safe_filename`,
# `SPLIT_ID_DEFAULT`, etc. are simply in-scope, not imported cross-file) and
# against `fermi_processing.py`'s real, per-split output (already
# executed earlier in this kernel too).
#
# ## How it works
#
# - **Input layout.** Module 1 writes per split to
#   `fermi_module1_processing_{VERSION_TAG}/{split_id}/`. `discover_splits()`
#   reads Module 1's `multi_split_manifest.json` (entries keyed
#   `split_idx`/`split_seed`/`split_dir`, not `id`), and the whole procedure
#   is wrapped in `run_merged_module_for_split(split_id)`.
# - **Feature selection is not recomputed here.** `build_merged_context`
#   reads Module 1's `router_meta`/`merged_branch_meta` from that split's
#   `feature_metadata.json`
#   (`hierarchical_feature_selection.merged_2level.{level1_router,
#   level2_branches}`) and slices the already-loaded `ALL_FEATURES` pool by
#   those feature-name lists — a column selection, not a second
#   stability-selection run.
# - **Combo screening drives the deployed system.** Section 8's
#   TOP_N_ROUTERS x TOP_N_BRANCH_COMBOS screening is needed to reach the best
#   model performance, so persistence is deferred until *after* it runs: the
#   baseline (Section 4/6) combo and every Section-8 survivor are compared on
#   one common scale (`end_to_end_val_{SELECTION_METRIC}`), the best is refit
#   on the full training split, and *that* combo is persisted for inference
#   and reported as `generalization_estimate`. `_end_to_end_chain_cv`
#   computes the honest soft-blend (router probabilities x branch
#   probabilities) estimate.
# - **Physical merge.** Module 1 applies the fsrq/ssrq/css merge once,
#   globally, before any split or feature selection (`fermi_processing.py`
#   Section 1). `build_merged_context` reads Module 1's
#   `feature_metadata["physical_merge"]` block and asserts that (a) the merge
#   was applied with the class map this file expects and (b) no `ssrq`/`css`
#   label appears in the loaded data — a fail-loud consistency guard, not a
#   second transformation that could drift out of sync.
# - **Selection optimism** (Cawley & Talbot 2010, *JMLR* 11:2079-2107) is
#   disclosed in the markdown cell below, covering the choice of a winner
#   among ~15 candidates per component (router, each branch) and of a whole-
#   system configuration among the screened combos.
# - **Logging / I/O** via `fermi_common.py`: `io_for(OUTPUT_ROOT, split_id)`
#   (per-split logs/results/plots), `get_section_logger` (per-branch
#   `NOTICE`-floored console + full-detail log file),
#   `save_result_multi`/`make_safe_run` (per-branch results files),
#   `publish_split_summary` at the end of each split's run, and `pbar()`
#   around the branch-training and split-driver loops.
#
# ## Jupytext
# This file is written in jupytext "percent" format (`# %%` / `# %%
# [markdown]` cell markers) for clean notebook conversion, matching
# `fermi_processing.py`.

# %% [markdown]
# ## Selection-bias disclosure (restated from Module 1, extended here)
#
# Every model-selection decision in this file — the router's winning
# model/feature-set/weighting, each branch's winning configuration, and the
# winning whole-system router/branch-combination out
# of Section 8's screened candidates — is chosen by comparing validation
# scores across many candidates. This is the standard, accepted cost of any
# search over many candidates using held-out data rather than a formally
# nested CV-within-CV scheme (prohibitively expensive here, for the same
# reason Module 1 gives for not nesting feature selection inside
# model selection): the reported validation number for whichever candidate
# wins carries a small, well-characterized optimism from having been picked,
# among many, on the same data used to score it (Cawley & Talbot, 2010,
# *Journal of Machine Learning Research* 11:2079-2107). This affects only
# the internal validation numbers used to *select* a winner — it does not
# affect any held-out test-set metric, since no test set is read anywhere in
# this file. State this plainly in the paper's methods text, extended from
# Module 1's equivalent sentence to cover this module's router/branch/
# whole-system selection explicitly.

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
# Reads `fermi_processing.py`'s real `multi_split_manifest.json` —
# verified directly against that file's own source (not just its prose
# methodology write-up, which is what `fermi_fine.py`'s own
# `discover_splits` was written against, and why that one is defensive about
# a schema it never actually confirmed). The manifest's `"splits"` key is a
# list of dicts shaped `{"split_idx", "split_seed", "split_dir", ...}` — no
# `"id"` key — so the split identifier used everywhere downstream
# (`split_00_primary`, `split_01`, ...) is recovered from the last path
# component of `"split_dir"`.

# %%
PROCESSING_ROOT = _cfg("PROCESSING_ROOT", Path(f"./fermi_module1_processing_{VERSION_TAG}/"))
OUTPUT_ROOT = _cfg("OUTPUT_ROOT", Path(f"./outputs_{VERSION_TAG}_{SELECTION_METRIC}/"))

ENABLE_MULTI_SPLIT_VALIDATION = _cfg("MODULE_MULTI_SPLIT", False)  # mirrors Module 1's own default-off switch


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
print(f"  Outputs (metric-suffix) -> {OUTPUT_ROOT.resolve()}")

# %% [markdown]
# ## Section 1 — Config
#
# This module deliberately keeps NO local `MODEL_SPECS` override — unlike
# `fermi_fine.py`'s K-Fold-only `DecisionTree` override, the merged
# pipeline uses `fermi_common`'s adaptive-LOO `DTreeCVPrunedClassifier`
# and every other shared default as-is. `HISTGBC_MAX_ITER_CEILING=1000`
# here is deliberately different from the fine module's `2000` — a
# disclosed divergence.
#
# `BRANCH_MIN_TRAIN_FOR_MODEL_SWEEP` (reliability tier threshold, =10) is a
# distinct concept from Module 1's own feature-selection floor
# (`BRANCH_MIN_TRAIN_FOR_SELECTION_MERGED`, =40, baked into which branches
# even appear in `merged_branch_meta`): the former governs whether THIS
# file's model-selection funnel is trustworthy at a branch's sample size,
# the latter governs whether Module 1 trusted its OWN feature-importance
# ranking at that sample size. Kept as two separate constants deliberately.

# %%
MERGE_FINE_CLASSES = {"ssrq": "fsrq", "css": "fsrq"}  # consistency-guard only — see Section 2

FINE_SETUP_KEY = "B"
MERGED_LABEL_SET = "merged"

BRANCH_MIN_TRAIN_FOR_MODEL_SWEEP = 10
TOP_N_ROUTERS = 3
TOP_N_BRANCH_COMBOS = 5
COMBO_COARSE_SURVIVAL_FRACTION = 0.10
HISTGBC_MAX_ITER_CEILING = 1000

SUPER_LE = LabelEncoder().fit(SUPERCLASS_ORDER)


# %% [markdown]
# ## Section 1c — K-Fold-only DecisionTree override [perf guard]

# %%
# %% [markdown]
# 
#
# `DTreeCVPrunedClassifier`'s adaptive LOO (left on here per Section 1's
# "no local MODEL_SPECS override") compounds badly: DecisionTree is swept
# across every tag/scaler/weighting variant for the router (Sec 4) AND
# every branch (Sec 5/6), then — if it wins — refit again per fold in the
# baseline chain-CV (Sec 8) and again per fold per candidate in the
# TOP_N_ROUTERS x TOP_N_BRANCH_COMBOS screening grid (Sec 9). Local copy
# only — fermi_common.MODEL_SPECS stays untouched for other modules.

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


MERGED_MODEL_SPECS = {k: dict(v) for k, v in MODEL_SPECS.items()}
MERGED_MODEL_SPECS["DecisionTree"] = dict(
    build=_build_dtree_kfold_only, build_gpu=_build_dtree_kfold_only,
    weight_support="sample_weight", supports_pca=False, scale_sensitive=False,
)


def _get_builder(model_name: str):
    return get_builder(model_name, MERGED_MODEL_SPECS)


# %% [markdown]
# ## Section 2 — Load Setup B + hierarchical local-selection metadata, per split
#
# Reads Module 1's exported per-tag CSVs for Setup B (fine, `bcu` excluded),
# derives the per-row superclass target from `TAXONOMY`, and reads Module
# 1's already-computed router/branch local feature-selection results
# directly from `feature_metadata.json` — no
# recomputation. Global Setup B tags come from
# `feature_metadata["setups"]["B"]["feature_sets"]` (that block only carries
# `n_features`/`achieved_power`, not the column list itself — the column
# list is simply whatever columns survive after dropping
# `label`/`label_encoded`/`source_name` from that tag's own exported CSV,
# so no separate lookup is needed for global tags). The router's and each
# branch's LOCAL tags (`hierarchical_feature_selection.merged_2level.
# {level1_router,level2_branches}`), by contrast, carry real feature-name
# lists — those are sliced directly out of the always-loaded `ALL_FEATURES`
# pool.
#
# The fsrq/ssrq/css merge is NOT reapplied here — Module
# 1 already applied it once, globally, before any split. This function
# instead asserts Module 1's own `physical_merge` block matches what this
# file expects and that no `ssrq`/`css` label is actually present in the
# loaded data, failing loudly rather than silently re-deriving something
# Module 1 already decided.

# %%
@dataclass
class MergedContext:
    """One Setup-B context for one split: global + router-local feature-set
    tags (for the Level-1 router sweep), the shared fine/superclass label
    arrays, and Module 1's router/branch local-selection metadata (used to
    build each branch's own tag frames later, in build_branch_datasets)."""
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
    tags: list
    router_meta: Optional[dict]
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
    """[Ported from fermi_fine.py §5.3 — a real, previously-latent
    mismatch risk] Every tag's export for one setup/split must share
    identical row order before a single boolean mask is later applied
    positionally across every tag's frame. Verified via source_name when
    available; falls back to a shape-only check with a loud warning
    against an older export that predates that column."""
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
            print(f"  ⚠ {setup_label} [{split}] tag '{tag}': no 'source_name' column to verify "
                  f"row alignment, and row COUNT mismatches tag '{ref_tag}' "
                  f"({len(x_by_tag[tag])} vs {len(x_by_tag[ref_tag])}).")
            raise ValueError(f"{setup_label} [{split}] tag '{tag}': row count mismatch — "
                              "cannot proceed safely without source_name to verify alignment.")


def build_merged_context(source_dir: Path, feature_metadata: dict, split_id: str) -> MergedContext:
    setup_meta = feature_metadata["setups"][FINE_SETUP_KEY]
    all_tags = list(setup_meta["feature_sets"].keys())
    if "ALL_FEATURES" not in all_tags:
        raise ValueError(
            f"[{split_id}] Setup {FINE_SETUP_KEY}'s feature_metadata has no 'ALL_FEATURES' tag — "
            "required to build router/branch local-selection tag frames."
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

    # Consistency guard, NOT a re-merge — Module 1 already
    # merged fsrq/ssrq/css before any split/selection.
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

    # Router local tags — direct read of Module 1's already-
    # computed result, no compute_selection_engine call.
    hfs = feature_metadata.get("hierarchical_feature_selection", {}).get("merged_2level", {})
    router_meta = hfs.get("level1_router")
    ctx_tags = [t for t in all_tags if t != "ALL_FEATURES"]
    if router_meta:
        for local_tag, feats in router_meta["feature_sets"].items():
            x_train_by_tag[local_tag] = x_train_by_tag["ALL_FEATURES"][feats].reset_index(drop=True)
            x_test_by_tag[local_tag] = x_test_by_tag["ALL_FEATURES"][feats].reset_index(drop=True)
            ctx_tags.append(local_tag)
    else:
        print(f"  ⚠ [{split_id}] Module 1 has no level1_router local-selection metadata for this "
              "split — router sweep uses only Setup B's global tags.")

    branch_meta = hfs.get("level2_branches", {}) or {}

    return MergedContext(
        split_id=split_id, x_train_by_tag=x_train_by_tag, x_test_by_tag=x_test_by_tag,
        y_train_fine_enc=y_train_fine_enc, y_test_fine_enc=y_test_fine_enc,
        y_train_fine_str=y_train_fine_str, y_test_fine_str=y_test_fine_str,
        y_train_super_str=y_train_super_str, y_test_super_str=y_test_super_str,
        y_train_super_enc=y_train_super_enc, y_test_super_enc=y_test_super_enc,
        fine_label_encoder=le_fine, fine_class_names=le_fine.classes_, tags=ctx_tags,
        router_meta=router_meta, branch_meta=branch_meta,
    )


def plot_taxonomy_tree(fine_class_names, plots_dir: Path, logger=None):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")
    root_x, root_y = 0.06, 0.5
    ax.text(root_x, root_y, "4FGL\nsource", ha="center", va="center", fontsize=11,
            fontweight="bold", bbox=dict(boxstyle="round", fc="lightgray"))
    n_branches = len(SUPERCLASS_ORDER)
    branch_ys = np.linspace(0.92, 0.08, n_branches)
    for sc, by in zip(SUPERCLASS_ORDER, branch_ys):
        bx = 0.36
        ax.plot([root_x + 0.05, bx - 0.03], [root_y, by], color="gray", linewidth=0.8)
        ax.text(bx, by, sc, ha="center", va="center", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round", fc="#cfe8ff"))
        fines = sorted(c for c, s in TAXONOMY.items() if s == sc and c in fine_class_names)
        fine_ys = (np.linspace(by + 0.03 * (len(fines) - 1), by - 0.03 * (len(fines) - 1), len(fines))
                   if len(fines) > 1 else [by])
        for fc, fy in zip(fines, fine_ys):
            fx = 0.72
            ax.plot([bx + 0.05, fx - 0.03], [by, fy], color="lightgray", linewidth=0.6)
            ax.text(fx, fy, fc, ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round", fc="#fff2cc"))
    ax.set_title("Hierarchical Taxonomy: Level 1 (superclass, blue) → Level 2 (fine class, yellow)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = plots_dir / "taxonomy_tree.png"
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


def _resolve_full_length_tag_frame(ctx: MergedContext, tag: str, local_meta: Optional[dict]):
    """Resolves a tag's FULL-length (not superclass-masked) feature frame —
    needed by chain-CV, which slices arbitrary fold indices that don't
    respect any one branch's row mask. Global/router-local tags are already
    in ctx.x_train_by_tag at full length; a branch-local tag is sliced
    fresh from ALL_FEATURES using that branch's own local_meta."""
    if tag in ctx.x_train_by_tag:
        return ctx.x_train_by_tag[tag]
    if local_meta and tag in local_meta.get("feature_sets", {}):
        return ctx.x_train_by_tag["ALL_FEATURES"][local_meta["feature_sets"][tag]]
    raise KeyError(f"Tag '{tag}' is neither a global/router-local tag nor a known branch-local tag.")


# %% [markdown]
# ## Section 3 — Per-branch `safe_run`
#
# One closure per (split, label_set), writing to both the split's full
# aggregator and that label_set's own per-branch results file, with
# failures logged at `ERROR` to that branch's own log file — mirrors
# `fermi_fine.py`'s `make_safe_run` exactly.

# %%
def make_safe_run(io_bundle: dict, label_set: str):
    branch_results_path = io_bundle["results_dir"] / f"{_safe_filename(label_set)}.json"
    logger = get_section_logger(label_set, io_bundle["logs_dir"])

    def safe_run(run_fn, *args, context="", **kwargs):
        return _BASE_SAFE_RUN(
            run_fn, *args,
            results_path=[io_bundle["split_results_path"], branch_results_path],
            context=context, logger=logger, **kwargs,
        )
    return safe_run


# %% [markdown]
# ## Section 4 — Level 1: Merged (Superclass) Router — val-only, no test
#
# Thin wrapper around `select_and_refit_val_only`. No local `MODEL_SPECS`
# override — the shared defaults are exactly right here (see Section 1).

# %%
def run_level1_pipeline(ctx: MergedContext, io_bundle: dict):
    logger = get_section_logger(MERGED_LABEL_SET, io_bundle["logs_dir"])
    safe_run_l1 = make_safe_run(io_bundle, MERGED_LABEL_SET)
    branch_results_path = io_bundle["results_dir"] / f"{_safe_filename(MERGED_LABEL_SET)}.json"

    winner_info, leaderboard = select_and_refit_val_only(
        ctx.x_train_by_tag, ctx.y_train_super_enc, SUPER_LE.classes_, MERGED_LABEL_SET, ctx.tags,
        ALL_MODEL_NAMES, safe_run_l1, io_bundle["split_results_path"],
        context_prefix=f"{ctx.split_id}|L1-router",
        val_split_report_path=io_bundle["split_dir"] / "val_split_report_L1_router.json",
        logs_dir=io_bundle["logs_dir"], extra_results_paths=[branch_results_path],
        histgbc_max_iter_ceiling=HISTGBC_MAX_ITER_CEILING,
        model_specs=MERGED_MODEL_SPECS, get_builder_fn=_get_builder,
    )
    if not winner_info:
        raise RuntimeError(f"[{ctx.split_id}] LEVEL 1: every model failed selection.")

    winner_info_path = io_bundle["split_dir"] / f"winner_info_L1_{MERGED_LABEL_SET}.json"
    with open(winner_info_path, "w") as f:
        json.dump(winner_info, f, indent=2, default=str)
    logger.notice(f"  Saved -> {winner_info_path}")

    row = leaderboard.iloc[0].to_dict()
    cfg = _leaderboard_row_to_cfg(row)
    logger.notice(f"→ Selected ROUTER: {cfg['model_name']} | val_{SELECTION_METRIC}="
                  f"{row[f'val_{SELECTION_METRIC}']:.4f}")

    router_model = fit_component(cfg, MERGED_MODEL_SPECS, _get_builder, ctx.x_train_by_tag[cfg["tag"]],
                                  ctx.y_train_super_enc)
    router_info = dict(cfg)
    router_info.update({k: row[k] for k in row if k.startswith("val_") or k.startswith("train_")})
    return router_model, router_info, leaderboard, winner_info


# %% [markdown]
# ## Section 4b — Router OOF misrouting diagnostic (NOT used for training)
#
# Branches deliberately train on ORACLE routing
# (see Section 5/6 markdown for why), so this K-fold OOF pass is run purely
# for its printed misrouting-rate diagnostic — the array itself is never
# consumed downstream.

# %%
def _compute_oof_router_routing(ctx: MergedContext, router_cfg, router_meta, io_bundle,
                                 n_splits=5, seed=SEED):
    logger = get_section_logger(MERGED_LABEL_SET, io_bundle["logs_dir"])
    y_super_str = ctx.y_train_super_str
    n = len(y_super_str)
    classes, counts = np.unique(y_super_str, return_counts=True)
    rare_mask = np.isin(y_super_str, classes[counts < n_splits])
    strat_idx, rare_idx = np.where(~rare_mask)[0], np.where(rare_mask)[0]
    y_strat = y_super_str[strat_idx]
    strat_min = int(np.unique(y_strat, return_counts=True)[1].min()) if len(strat_idx) else 0
    n_eff = max(2, min(n_splits, strat_min)) if strat_min >= 2 else 2
    skf = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)

    x_router_full = _resolve_full_length_tag_frame(ctx, router_cfg["tag"], router_meta)
    oof_super_pred = np.empty(n, dtype=object)
    oof_touched = np.zeros(n, dtype=bool)

    for fold_no, (tr_local, va_local) in enumerate(skf.split(strat_idx, y_strat), start=1):
        tr_idx = np.concatenate([strat_idx[tr_local], rare_idx])
        va_idx = strat_idx[va_local]
        router_fold = fit_component(router_cfg, MERGED_MODEL_SPECS, _get_builder,
                                    x_router_full.iloc[tr_idx],
                                    SUPER_LE.transform(y_super_str[tr_idx]), seed)
        pred_super_va = SUPER_LE.inverse_transform(router_fold.predict(x_router_full.iloc[va_idx]))
        oof_super_pred[va_idx], oof_touched[va_idx] = pred_super_va, True
        logger.info(f"    [oof_routing] fold {fold_no}/{n_eff} done ({len(va_idx)} rows)")

    oof_super_pred[~oof_touched] = y_super_str[~oof_touched]
    n_misrouted = int((oof_super_pred != y_super_str).sum())
    logger.notice(f"  [oof_routing] {n_misrouted}/{n} rows ({n_misrouted/n:.2%}) misrouted vs. true "
                  "superclass under router OOF CV — diagnostic only; branches below train on oracle "
                  "routing, NOT this OOF-routed array.")


# %% [markdown]
# ## Section 5/6 — Level 2: Per-superclass branches — val-only, no test
#
# `build_branch_datasets` does not call
# `compute_selection_engine`: each branch's LOCAL feature-set tags are read
# straight from `ctx.branch_meta[sc]["feature_sets"]` (already computed and
# filtered by Module 1's own `BRANCH_MIN_TRAIN_FOR_SELECTION_MERGED` floor —
# a branch simply absent from `ctx.branch_meta` means Module 1 judged it too
# small to trust local selection at all, in which case this branch falls
# back to global+router tags only). Three-tier reliability system (shared
# with the 3-level sibling file): "sweep" (≥`BRANCH_MIN_TRAIN_FOR_MODEL_SWEEP` rows,
# the full multi-model funnel), "fallback" (fewer rows but not every class a
# singleton — one robust adaptive-LOO DecisionTree, one honest pooled-OOF
# estimate), "unvalidatable" (every class a singleton — no CV scheme can
# ever hold one out; fit-on-all-data, TRAIN-only diagnostics, explicitly not
# claimed as a generalization estimate).

# %%
def build_branch_datasets(ctx: MergedContext, route_str_train: np.ndarray = None) -> dict:
    route_str_train = ctx.y_train_super_str if route_str_train is None else route_str_train
    branches = {}
    for sc in SUPERCLASS_ORDER:
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
        x_tr_all = None
        if local_meta is not None:
            x_tr_all = ctx.x_train_by_tag["ALL_FEATURES"][tr_mask].reset_index(drop=True)
            for local_tag, feats in local_meta["feature_sets"].items():
                x_train_by_tag[local_tag] = x_tr_all[feats].reset_index(drop=True)
                branch_tags.append(local_tag)
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
    branch_results_path = io_bundle["results_dir"] / f"{_safe_filename(label_set)}.json"
    save_result_multi(run_id, {
        "model_name": "DecisionTree", "feature_set": tag, "tag": tag,
        "variant": "final_val_reliability_fallback_none", "label_set": label_set,
        "n_features": x_tr.shape[1], "weighting": "none",
        "val_accuracy": branch_result["val_accuracy"], "val_bal_accuracy": branch_result["val_bal_accuracy"],
        "val_macro_f1": branch_result["val_macro_f1"], "val_kappa": branch_result["val_kappa"],
        "train_accuracy": branch_result["train_accuracy"], "train_bal_accuracy": branch_result["train_bal_accuracy"],
        "train_macro_f1": branch_result["train_macro_f1"], "train_kappa": branch_result["train_kappa"],
        "reliability_fallback": True, "reliability_diag": diag,
    }, [io_bundle["split_results_path"], branch_results_path], logger=logger)
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
                  "NOT a generalization estimate — every class has exactly 1 training example).")

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
    branch_results_path = io_bundle["results_dir"] / f"{_safe_filename(label_set)}.json"
    save_result_multi(run_id, {
        "model_name": "DecisionTree", "feature_set": tag, "tag": tag,
        "variant": "final_unvalidatable_no_cv", "label_set": label_set,
        "n_features": x_tr.shape[1], "weighting": "none",
        "val_accuracy": None, "val_bal_accuracy": None, "val_macro_f1": None, "val_kappa": None,
        "train_accuracy": branch_result["train_accuracy"], "train_bal_accuracy": branch_result["train_bal_accuracy"],
        "train_macro_f1": branch_result["train_macro_f1"], "train_kappa": branch_result["train_kappa"],
        "unvalidatable": True, "reliability_diag": diag,
    }, [io_bundle["split_results_path"], branch_results_path], logger=logger)
    return branch_result, None, cfg


def run_level2_branch(sc: str, branch: dict, io_bundle: dict, split_id: str):
    label_set = f"branch_{sc}"
    logger = get_section_logger(label_set, io_bundle["logs_dir"])
    logger.notice(f"[{split_id}] LEVEL 2 [{label_set}] — {len(branch['class_names'])} fine classes, "
                  f"train={branch['n_train']} (val-only, no test)")

    tier, diag = _branch_validation_tier(branch)
    if tier == "unvalidatable":
        logger.warning(f"  ⚠ [{label_set}] n_train={diag['n_train']} == n_classes={diag['n_classes']} "
                        "(every class a singleton) — fitting on all data, TRAIN-only diagnostics.")
        return _fit_unvalidatable_branch(sc, branch, label_set, diag, io_bundle)
    if tier == "fallback":
        logger.notice(f"  ℹ [{label_set}] n_train={diag['n_train']} < {BRANCH_MIN_TRAIN_FOR_MODEL_SWEEP} — "
                       "skipping the multi-model sweep (winner's-curse risk this small); single robust "
                       "adaptive-LOO DecisionTree instead.")
        return _fit_reliability_fallback_branch(sc, branch, label_set, diag, io_bundle)

    safe_run_branch = make_safe_run(io_bundle, label_set)
    branch_results_path = io_bundle["results_dir"] / f"{_safe_filename(label_set)}.json"
    winner_info, leaderboard = select_and_refit_val_only(
        branch["x_train_by_tag"], branch["y_train_enc"], branch["class_names"], label_set,
        branch["tags"], ALL_MODEL_NAMES, safe_run_branch, io_bundle["split_results_path"],
        context_prefix=f"{split_id}|{label_set}",
        val_split_report_path=io_bundle["split_dir"] / f"val_split_report_{_safe_filename(label_set)}.json",
        logs_dir=io_bundle["logs_dir"], extra_results_paths=[branch_results_path],
        histgbc_max_iter_ceiling=HISTGBC_MAX_ITER_CEILING,
        model_specs=MERGED_MODEL_SPECS, get_builder_fn=_get_builder,
    )
    if not winner_info:
        logger.warning(f"  ⚠ {label_set}: every model failed selection — branch skipped "
                        "(falls back to majority-class at inference).")
        return None, None, None

    row = leaderboard.iloc[0].to_dict()
    cfg = _leaderboard_row_to_cfg(row)
    logger.notice(f"→ Branch winner [{label_set}]: {cfg['model_name']} / {cfg['tag']} / {cfg['weighting']} "
                  f"val_{SELECTION_METRIC}={row[f'val_{SELECTION_METRIC}']:.4f}")

    model = fit_component(cfg, MERGED_MODEL_SPECS, _get_builder, branch["x_train_by_tag"][cfg["tag"]],
                           branch["y_train_enc"])
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


def train_branch_models(ctx: MergedContext, io_bundle: dict):
    branch_datasets = build_branch_datasets(ctx)
    branch_models, branch_leaderboards, branch_cfgs = {}, {}, {}
    for sc in pbar(SUPERCLASS_ORDER, desc=f"[{ctx.split_id}] Level-2 branches"):
        branch = branch_datasets.get(sc)
        if branch is None:
            branch_models[sc] = branch_leaderboards[sc] = branch_cfgs[sc] = None
            continue
        bm, lb, cfg = run_level2_branch(sc, branch, io_bundle, ctx.split_id)
        branch_models[sc], branch_leaderboards[sc], branch_cfgs[sc] = bm, lb, cfg
    return branch_datasets, branch_models, branch_leaderboards, branch_cfgs


# %% [markdown]
# ## Section 7 — Persist fitted router + branch models for inference
#
# Writes the actual fitted estimators, label encoders, and each component's
# exact feature-column list (read directly off the DataFrame it was fit on
# — always exactly right, no separate metadata lookup that could drift out
# of sync) to `{split_dir}/inference_2level/`, plus a manifest JSON.
# Called once, AFTER Section 8 decides the final
# router/branch combination — see the driver function at the bottom of this
# file — so what gets persisted here is always whichever combo actually won
# (the Section 4/6 baseline, or a Section-8 screening survivor).

# %%
def persist_hierarchical_models(router_model, router_cfg, branch_models: dict, ctx: MergedContext,
                                 branch_datasets: dict, io_bundle: dict, adopted_from: str):
    inference_dir = io_bundle["split_dir"] / "inference_2level"
    inference_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version_tag": VERSION_TAG, "selection_metric": SELECTION_METRIC, "split_id": ctx.split_id,
        "fine_setup_used_for_both_levels": FINE_SETUP_KEY, "superclass_order": SUPERCLASS_ORDER,
        "taxonomy": TAXONOMY, "adopted_from": adopted_from, "router": None, "branches": {},
    }

    router_path = inference_dir / "router_model.joblib"
    joblib.dump(router_model, router_path)
    manifest["router"] = {
        "model_name": router_cfg["model_name"], "tag": router_cfg["tag"],
        "pca_variance": router_cfg.get("pca_variance"), "scaler_name": router_cfg.get("scaler_name"),
        "features": list(ctx.x_train_by_tag[router_cfg["tag"]].columns),
        "model_path": str(router_path.relative_to(io_bundle["split_dir"])),
        "super_label_encoder_classes": list(SUPER_LE.classes_),
    }

    branch_fallback_fine = {}
    for sc in SUPERCLASS_ORDER:
        tr_mask = ctx.y_train_super_str == sc
        fine_tr_sc = ctx.y_train_fine_str[tr_mask]
        if len(fine_tr_sc):
            vals, counts = np.unique(fine_tr_sc, return_counts=True)
            branch_fallback_fine[sc] = str(vals[np.argmax(counts)])

    for sc in SUPERCLASS_ORDER:
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
    logger.notice(f"  Persisted router + {n_trained} branch model(s) [{adopted_from}] -> {inference_dir}")
    logger.notice(f"  Manifest -> {manifest_path}")
    return manifest


# %% [markdown]
# ## Section 8 — End-to-end CHAIN cross-validation (router → branch), NO TEST
#
# The soft-voting combination: *P(fine class c) = Σ_superclass P(superclass)
# × P(fine class c | superclass)*. Every branch contributes to every
# prediction, weighted by the router's confidence that a row belongs to
# that branch's superclass — this is what reconciles oracle-routed branch
# training (Section 5/6) with a fair, honest end-to-end evaluation. K=5
# StratifiedKFold; every fold refits the router AND every branch on that
# fold's training rows only, using each component's already-selected
# configuration (no re-sweep inside the CV loop).

# %%
def _predict_proba_safe(model, X, n_classes):
    if hasattr(model, "predict_proba"):
        try:
            return _coerce_proba_to_ndarray(model.predict_proba(X), n_classes)
        except Exception:
            pass
    pred_idx = np.asarray(model.predict(X))
    onehot = np.zeros((len(pred_idx), n_classes))
    onehot[np.arange(len(pred_idx)), pred_idx] = 1.0
    return onehot


def _coerce_proba_to_ndarray(proba, n_classes):
    """predict_proba() on GPU-backed branch/router estimators
    (_LocalLabelReencodeClassifier -> XGBoost/LightGBM device='cuda',
    _CumlSampleWeightAdapter/_CumlSVCSampleWeightAdapter -> cuML) can come
    back as a pandas or cuDF DataFrame instead of a plain ndarray,
    depending on installed version. Positional [:, k] soft-voting in
    _fit_and_route_chain_fold requires a real 2-D ndarray with columns in
    local-label order 0..n_classes-1."""
    if isinstance(proba, np.ndarray):
        arr = proba
    elif hasattr(proba, "to_numpy"):        # pandas.DataFrame / cudf.DataFrame
        arr = proba.to_numpy()
    elif hasattr(proba, "values"):          # fallback for older DataFrame-likes
        arr = np.asarray(proba.values)
    else:
        arr = np.asarray(proba)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != n_classes:
        raise ValueError(
            f"predict_proba returned {arr.shape[1]} columns, expected n_classes={n_classes} "
            f"(model={type(proba).__name__})"
        )
    return arr


def _fit_and_route_chain_fold(ctx: MergedContext, tr_idx, eval_idx, router_cfg, branch_cfgs: dict,
                               router_meta, branch_datasets: dict, seed=SEED):
    y_super_full, y_fine_full = ctx.y_train_super_str, ctx.y_train_fine_str
    y_super_tr = y_super_full[tr_idx]

    x_router_full = _resolve_full_length_tag_frame(ctx, router_cfg["tag"], router_meta)
    router_fold = fit_component(router_cfg, MERGED_MODEL_SPECS, _get_builder, x_router_full.iloc[tr_idx],
                                 SUPER_LE.transform(y_super_tr), seed)

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
        le_branch = LabelEncoder().fit(fine_tr_sc)
        branch_local_meta = branch_datasets.get(sc, {}).get("local_meta") if branch_datasets.get(sc) else None
        x_branch_full = _resolve_full_length_tag_frame(ctx, cfg["tag"], branch_local_meta)
        model = fit_component(cfg, MERGED_MODEL_SPECS, _get_builder, x_branch_full.iloc[tr_idx][sc_mask_tr],
                               le_branch.transform(fine_tr_sc), seed)
        branch_folds[sc] = (model, le_branch)

    x_router_eval = x_router_full.iloc[eval_idx]
    all_superclasses = list(SUPER_LE.classes_)
    super_idx = {s: i for i, s in enumerate(all_superclasses)}
    p_super = _predict_proba_safe(router_fold, x_router_eval, len(all_superclasses))
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
        x_branch_full = _resolve_full_length_tag_frame(ctx, branch_cfgs[sc]["tag"], branch_local_meta)
        x_branch_eval = x_branch_full.iloc[eval_idx]
        p_branch = _predict_proba_safe(model, x_branch_eval, len(le_branch.classes_))
        for k, fc in enumerate(le_branch.classes_):
            p_fine[:, fine_idx[fc]] += p_sc * p_branch[:, k]

    pred_fine_eval = np.array(all_fine_classes)[p_fine.argmax(axis=1)]
    return pred_fine_eval, pred_super_eval


def _end_to_end_chain_cv(ctx: MergedContext, router_cfg, branch_cfgs: dict, router_meta,
                          branch_datasets: dict, io_bundle: dict, label_set: str,
                          n_splits=5, seed=SEED):
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

    for fold_no, (tr_local, va_local) in enumerate(skf.split(strat_idx, y_strat), start=1):
        tr_idx = np.concatenate([strat_idx[tr_local], rare_idx])
        va_idx = strat_idx[va_local]
        pred_fine_va, pred_super_va = _fit_and_route_chain_fold(
            ctx, tr_idx, va_idx, router_cfg, branch_cfgs, router_meta, branch_datasets, seed=seed)
        oof_super_pred[va_idx], oof_fine_pred[va_idx], oof_touched[va_idx] = pred_super_va, pred_fine_va, True
        logger.info(f"    fold {fold_no}/{n_eff} done ({len(va_idx)} held-out rows)")

    touched = oof_touched
    y_true_fine_enc = ctx.fine_label_encoder.transform(y_fine_str[touched])
    y_pred_fine_enc = ctx.fine_label_encoder.transform(oof_fine_pred[touched])
    y_true_super_enc = SUPER_LE.transform(y_super_str[touched])
    y_pred_super_enc = SUPER_LE.transform(oof_super_pred[touched])

    end_to_end_val = evaluate_model(f"End-to-end 2-level CHAIN CV [{label_set}]", y_true_fine_enc,
                                     y_pred_fine_enc, ctx.fine_label_encoder.classes_,
                                     label_set=label_set, split="val", logger=logger)

    all_idx = np.arange(n)
    pred_fine_tr, _ = _fit_and_route_chain_fold(ctx, all_idx, all_idx, router_cfg, branch_cfgs,
                                                 router_meta, branch_datasets, seed=seed)
    y_true_fine_tr_enc = ctx.fine_label_encoder.transform(y_fine_str)
    y_pred_fine_tr_enc = ctx.fine_label_encoder.transform(pred_fine_tr)
    end_to_end_train = evaluate_model(f"End-to-end 2-level CHAIN CV [{label_set}] TRAIN REFIT",
                                       y_true_fine_tr_enc, y_pred_fine_tr_enc,
                                       ctx.fine_label_encoder.classes_, label_set=label_set,
                                       split="train", verbose=False, logger=logger)

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

    plot_confusion_matrix(f"end_to_end_fine_2level_chainCV_{_safe_filename(label_set)}", y_true_fine_enc,
                           y_pred_fine_enc, ctx.fine_label_encoder.classes_, label_set=label_set,
                           plots_dir=io_bundle["plots_dir"], logger=logger)
    routing_val = evaluate_model(f"Level-1 routing CHAIN CV [{label_set}]", y_true_super_enc,
                                  y_pred_super_enc, SUPER_LE.classes_, label_set=label_set,
                                  split="val", logger=logger)
    return {"end_to_end_val": end_to_end_val, "end_to_end_train": end_to_end_train,
            "routing_val": routing_val, "cv_mode": cv_mode, "n_folds": n_eff,
            "n_oof_rows": int(touched.sum())}


# %% [markdown]
# ## Section 9 — Router x branch-combination screening
#
# TOP_N_ROUTERS candidates
# x TOP_N_BRANCH_COMBOS rank-variants of the full per-superclass branch
# combination are crossed into a grid; each cell gets one cheap
# single-split coarse score first, only the top decile (floored at
# `COARSE_MIN_SURVIVORS`) proceeds to the expensive K-fold chain CV. The
# baseline (Section 4/6 winners) is added to the same comparison as one more
# candidate, and the overall best by `end_to_end_val_{SELECTION_METRIC}` is
# what gets refit on the full training split and persisted.

# %%
def _build_branch_cfg_variants(branch_leaderboards: dict, n_variants: int):
    """variant k: for each superclass, rank-k model from
    branch_leaderboards[sc] (clamped to that branch's own leaderboard
    length; None/empty branches stay None at every k). k=0 reproduces the
    Section 4/6 branch winners exactly."""
    variants = []
    for k in range(n_variants):
        combo = {}
        for sc in SUPERCLASS_ORDER:
            lb = branch_leaderboards.get(sc)
            if lb is None or lb.empty:
                combo[sc] = None
                continue
            row = lb.iloc[min(k, len(lb) - 1)].to_dict()
            combo[sc] = _leaderboard_row_to_cfg(row)
        variants.append(combo)
    return variants


def _refit_combo_full(ctx: MergedContext, router_cfg: dict, branch_cfgs: dict, branch_datasets: dict, seed=SEED):
    """Full-training-split refit of one router+branch combo — needed only
    for a Section-8-selected combo that was never fit on the FULL split
    before (the baseline combo's models already ARE full refits, from
    Sections 4/6's own select_and_refit_val_only pass)."""
    router_model = fit_component(router_cfg, MERGED_MODEL_SPECS, _get_builder,
                                  ctx.x_train_by_tag[router_cfg["tag"]], ctx.y_train_super_enc, seed)
    branch_models = {}
    for sc in SUPERCLASS_ORDER:
        cfg = branch_cfgs.get(sc)
        branch = branch_datasets.get(sc)
        if cfg is None or branch is None:
            branch_models[sc] = None
            continue
        model = fit_component(cfg, MERGED_MODEL_SPECS, _get_builder, branch["x_train_by_tag"][cfg["tag"]],
                               branch["y_train_enc"], seed)
        branch_models[sc] = {"model": model, "label_encoder": branch["label_encoder"],
                              "model_name": cfg["model_name"], "weighting": cfg["weighting"],
                              "tag": cfg["tag"], "pca_variance": cfg.get("pca_variance"),
                              "scaler_name": cfg.get("scaler_name"), "n_classes": len(branch["class_names"])}
    return router_model, branch_models


def run_router_branch_screening(ctx: MergedContext, level1_leaderboard: pd.DataFrame,
                                 branch_leaderboards: dict, io_bundle: dict):
    logger = get_section_logger(MERGED_LABEL_SET, io_bundle["logs_dir"])

    # [no coarse pre-filter] every TOP_N_ROUTERS x TOP_N_BRANCH_COMBOS candidate
    # goes straight to the expensive chain-CV — no cheap single-split scoring
    # pass, no coarse_filter cut. All candidates are trained and logged.
    top_router_rows = (level1_leaderboard.sort_values(f"val_{SELECTION_METRIC}", ascending=False)
                                          .head(TOP_N_ROUTERS).to_dict("records"))
    branch_cfg_variants = _build_branch_cfg_variants(branch_leaderboards, TOP_N_BRANCH_COMBOS)
    logger.notice(f"[router_screen] FULL pass (no coarse filter) — {len(top_router_rows)} router(s) x "
                  f"{len(branch_cfg_variants)} branch-combo variant(s) = "
                  f"{len(top_router_rows) * len(branch_cfg_variants)} candidate(s), ALL going directly "
                  "to expensive chain CV")

    combo_lookup = {}
    hierarchical_eval_by_router = {}
    branch_datasets_for_screen = build_branch_datasets(ctx)  # oracle-routed, tag frames only (no fitting)
    for rrow in top_router_rows:
        router_cfg = _leaderboard_row_to_cfg(rrow)
        for bi, branch_cfgs_variant in enumerate(branch_cfg_variants):
            combo_name = f"{router_cfg['model_name']}__branch{bi}"
            combo_lookup[combo_name] = {"combo_name": combo_name, "router_cfg": router_cfg,
                                         "branch_cfgs": branch_cfgs_variant, "branch_combo_idx": bi}
            logger.notice(f"→ [{combo_name}] expensive K-fold chain CV...")
            ev = _end_to_end_chain_cv(ctx, router_cfg, branch_cfgs_variant, ctx.router_meta,
                                       branch_datasets_for_screen, io_bundle,
                                       label_set=f"{MERGED_LABEL_SET}__{_safe_filename(combo_name)}")
            hierarchical_eval_by_router[combo_name] = ev

    metric_col = f"end_to_end_val_{SELECTION_METRIC}"
    rows = [{
        "combo": name, "router": combo_lookup[name]["router_cfg"]["model_name"],
        "branch_combo_idx": combo_lookup[name]["branch_combo_idx"],
        "routing_val_accuracy": ev["routing_val"]["accuracy"],
        "end_to_end_val_accuracy": ev["end_to_end_val"]["accuracy"],
        "end_to_end_val_bal_accuracy": ev["end_to_end_val"]["bal_accuracy"],
        "end_to_end_val_macro_f1": ev["end_to_end_val"]["macro_f1"],
        "end_to_end_val_kappa": ev["end_to_end_val"]["kappa"],
        metric_col: ev["end_to_end_val"][SELECTION_METRIC],
        "n_folds": ev["n_folds"], "cv_mode": ev["cv_mode"],
    } for name, ev in hierarchical_eval_by_router.items()]
    router_comparison = (pd.DataFrame(rows).sort_values(metric_col, ascending=False)
                          if rows else pd.DataFrame())
    if not router_comparison.empty:
        out_csv = io_bundle["split_dir"] / "router_branch_comparison.csv"
        router_comparison.to_csv(out_csv, index=False)
        logger.notice(f"  Saved -> {out_csv}  ({len(router_comparison)}/{len(top_router_rows) * len(branch_cfg_variants)} "
                       "candidate(s), full set, no coarse filter)")

    # [NEW] Full-detail file: every combo's router model name AND every
    # superclass branch's model name, one row per combo, easy to scan —
    # separate from the summary CSV above so nothing downstream that reads
    # the original schema breaks.
    full_rows = []
    for name, ev in hierarchical_eval_by_router.items():
        entry = combo_lookup[name]
        row = {
            "combo": name,
            "router_model": entry["router_cfg"]["model_name"],
            "router_tag": entry["router_cfg"]["tag"],
            "router_scaler": entry["router_cfg"]["scaler_name"],
            "router_pca_variance": entry["router_cfg"]["pca_variance"],
            "router_weighting": entry["router_cfg"]["weighting"],
            "branch_combo_idx": entry["branch_combo_idx"],
            "routing_val_accuracy": ev["routing_val"]["accuracy"],
            "end_to_end_val_accuracy": ev["end_to_end_val"]["accuracy"],
            "end_to_end_val_bal_accuracy": ev["end_to_end_val"]["bal_accuracy"],
            "end_to_end_val_macro_f1": ev["end_to_end_val"]["macro_f1"],
            "end_to_end_val_kappa": ev["end_to_end_val"]["kappa"],
            metric_col: ev["end_to_end_val"][SELECTION_METRIC],
            "n_folds": ev["n_folds"], "cv_mode": ev["cv_mode"],
        }
        for sc in SUPERCLASS_ORDER:
            bcfg = entry["branch_cfgs"].get(sc)
            row[f"branch_{sc}_model"] = bcfg["model_name"] if bcfg else None
            row[f"branch_{sc}_tag"] = bcfg["tag"] if bcfg else None
        full_rows.append(row)

    router_comparison_full = (pd.DataFrame(full_rows).sort_values(metric_col, ascending=False)
                               if full_rows else pd.DataFrame())
    if not router_comparison_full.empty:
        out_csv_full = io_bundle["split_dir"] / "router_branch_comparison_full.csv"
        router_comparison_full.to_csv(out_csv_full, index=False)
        out_json_full = io_bundle["split_dir"] / "router_branch_comparison_full.json"
        with open(out_json_full, "w") as f:
            json.dump(full_rows, f, indent=2, default=str)
        logger.notice(f"  Saved -> {out_csv_full}  (router + every branch's model name, all "
                       f"{len(router_comparison_full)} combo(s))")

    return hierarchical_eval_by_router, combo_lookup, router_comparison


# %% [markdown]
# ## Section 10 — MLP architecture comparison
#
# Restricted to the six `MLP_ARCHITECTURES`
# topologies across the router leaderboard and every branch's leaderboard.

# %%
def compare_mlp_architectures(leaderboards_by_label_set: dict, io_bundle: dict, split_id: str):
    logger = get_section_logger(MERGED_LABEL_SET, io_bundle["logs_dir"])
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

    out_csv = io_bundle["split_dir"] / "mlp_architecture_comparison_merged.csv"
    comparison.to_csv(out_csv, index=False)
    logger.notice(f"  Saved -> {out_csv}")
    return comparison


# %% [markdown]
# ## Section 11 — Run one split end to end (`run_merged_module_for_split`)
#
# Orchestrates one split: builds the context (Section 2), runs Level-1
# (Section 4) and Level-2 (Section 5/6), runs the baseline chain-CV
# (Section 8), runs the Section-8-numbered screening (Section 9), resolves
# the final winning combo, refits/persists it (Section 7), writes the
# structured metrics JSON (with `generalization_estimate` sourced from
# whichever combo actually won), runs the MLP comparison,
# and publishes this split's final artifacts.

# %%
def run_merged_module_for_split(split_id: str) -> dict:
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
    module_logger.notice(f"\n{'=' * 78}\nMODULE 3 (Hierarchical Merged, 2-level) — split '{split_id}'\n{'=' * 78}")
    module_logger.notice(f"  datasets -> {(source_dir / 'datasets').resolve()}")
    module_logger.notice(f"  results  -> {io_bundle['results_dir'].resolve()}")
    module_logger.notice(f"  logs     -> {io_bundle['logs_dir'].resolve()}")
    module_logger.notice(f"  plots    -> {io_bundle['plots_dir'].resolve()}")

    ctx = build_merged_context(source_dir, feature_metadata, split_id)
    module_logger.notice(f"Setup {FINE_SETUP_KEY} loaded: {len(ctx.fine_class_names)} fine classes, "
                          f"{len(SUPERCLASS_ORDER)} superclasses, tags={ctx.tags}, "
                          f"train={len(ctx.y_train_fine_enc):>5}, test={len(ctx.y_test_fine_enc):>5}")
    plot_taxonomy_tree(ctx.fine_class_names, io_bundle["plots_dir"], logger=module_logger)

    # --- Level 1: router ----------------------------------------------------
    router_model, router_info, level1_leaderboard, level1_winner_info = run_level1_pipeline(ctx, io_bundle)
    _compute_oof_router_routing(ctx, router_info, ctx.router_meta, io_bundle)

    # --- Level 2: branches ---------------------------------------------------
    # Router (above) trains on the full multi-superclass
    # set; everything below sees only a branch's routed subset, where the
    # smallest class can be tiny even if the branch's total row count isn't.
    # Reset to False before this function returns (see below) so it doesn't
    # leak into the next split's Level-1 router fit.
    global KNN_PER_CLASS_CLAMP
    KNN_PER_CLASS_CLAMP = True
    branch_datasets, branch_models, branch_leaderboards, branch_cfgs = train_branch_models(ctx, io_bundle)

    # --- Baseline end-to-end chain-CV (Section 4/6 winners) ------------------
    baseline_eval = _end_to_end_chain_cv(ctx, router_info, branch_cfgs, ctx.router_meta, branch_datasets,
                                          io_bundle, label_set=MERGED_LABEL_SET)

    # --- router x branch-combination screening ---
    hierarchical_eval_by_router, combo_lookup, router_comparison = run_router_branch_screening(
        ctx, level1_leaderboard, branch_leaderboards, io_bundle)
    
    # --- Top-15 whole-system candidate summary (visualization + CSV) ---
    # TOP_N_ROUTERS x TOP_N_BRANCH_COMBOS = 3 x 5 = 15 by construction, so
    # this is every screened combo, sorted — not a further cut. Unlike
    # router_branch_comparison_full.csv (Section 9, above), this also
    # carries end_to_end_TRAIN_* alongside end_to_end_VAL_*, since both
    # were already computed by run_router_branch_screening's chain-CV but
    # only the val side was previously persisted.
    _rows = []
    for _combo_name, _ev in hierarchical_eval_by_router.items():
        _entry = combo_lookup[_combo_name]
        _router_cfg, _branch_cfgs_variant = _entry["router_cfg"], _entry["branch_cfgs"]
        _branches_str = "/".join(f"{sc}={(_branch_cfgs_variant.get(sc) or {}).get('model_name', 'passthrough')}"
                                  for sc in SUPERCLASS_ORDER)
        _rows.append({
            "combo": _combo_name,
            "full_model_name": (f"router:{_router_cfg['model_name']}[{_router_cfg['tag']}/"
                                 f"{_router_cfg['weighting']}] + branches:{_branches_str}"),
            "router_model": _router_cfg["model_name"], "router_tag": _router_cfg["tag"],
            "router_weighting": _router_cfg["weighting"], "branch_combo_idx": _entry["branch_combo_idx"],
            "routing_val_accuracy": _ev["routing_val"]["accuracy"],
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
    _summary = pd.DataFrame(_rows).sort_values(f"end_to_end_val_{SELECTION_METRIC}", ascending=False).reset_index(drop=True)
    _summary.insert(0, "rank", np.arange(1, len(_summary) + 1))

    module_logger.notice(f"\nTOP-{len(_summary)} WHOLE-SYSTEM MERGED CANDIDATES — split '{split_id}' "
                          f"(sorted by end_to_end_val_{SELECTION_METRIC})")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        module_logger.notice(_summary.to_string(index=False))

    top15_csv = io_bundle["split_dir"] / f"top{len(_summary)}_summary_merged.csv"
    _summary.to_csv(top15_csv, index=False)
    module_logger.notice(f"  Saved -> {top15_csv}")
    
    

    # --- Resolve the FINAL combo across baseline + screened survivors ---
    metric_col = f"end_to_end_val_{SELECTION_METRIC}"
    best_candidate = ("section4_6_baseline", baseline_eval, router_info, branch_cfgs,
                       baseline_eval["end_to_end_val"][SELECTION_METRIC])
    for combo_name, ev in hierarchical_eval_by_router.items():
        score = ev["end_to_end_val"][SELECTION_METRIC]
        if score > best_candidate[4]:
            best_candidate = (f"section8_screening:{combo_name}", ev,
                               combo_lookup[combo_name]["router_cfg"],
                               combo_lookup[combo_name]["branch_cfgs"], score)
    adopted_from, final_eval, final_router_cfg, final_branch_cfgs, final_score = best_candidate

    if adopted_from == "section4_6_baseline":
        final_router_model, final_branch_models = router_model, branch_models
        module_logger.notice(f"[wiring fix] Section 8 screening did not beat the Section 4/6 baseline "
                              f"({final_score:.4f}) — baseline combo adopted, unchanged.")
    else:
        module_logger.notice(f"[wiring fix] Section 8 screening candidate '{adopted_from}' beat the "
                              f"Section 4/6 baseline ({final_score:.4f} vs. "
                              f"{baseline_eval['end_to_end_val'][SELECTION_METRIC]:.4f}) — refitting on the "
                              "full training split and adopting it for persistence/reporting.")
        final_router_model, final_branch_models = _refit_combo_full(
            ctx, final_router_cfg, final_branch_cfgs, branch_datasets)

    hierarchical_inference_manifest = persist_hierarchical_models(
        final_router_model, final_router_cfg, final_branch_models, ctx, branch_datasets, io_bundle, adopted_from)

    # --- MLP comparison --------------------------------------------------------
    leaderboards_by_label_set = {MERGED_LABEL_SET: level1_leaderboard}
    leaderboards_by_label_set.update({f"branch_{sc}": lb for sc, lb in branch_leaderboards.items() if lb is not None})
    mlp_comparison = compare_mlp_architectures(leaderboards_by_label_set, io_bundle, split_id)

    # --- Structured metrics persistence -----------------------------------------
    def _branch_metric_note(sc, bm):
        if bm is not None:
            return {k: v for k, v in bm.items() if k not in ("model", "label_encoder")}
        if branch_datasets.get(sc) is None:
            return {"note": "single fine class — no sub-classifier needed"}
        return {"note": "every model/config failed for this branch — falls back to majority training-set "
                         "fine class at inference"}

    merged_metrics = {
        "seed": SEED, "version_tag": VERSION_TAG, "use_gpu": USE_GPU, "split_id": split_id,
        "selection_metric": SELECTION_METRIC, "fine_setup_used_for_both_levels": FINE_SETUP_KEY,
        # No local recompute — this IS Module 1's own metadata, echoed for provenance.
        "level1_local_feature_selection_source": "fermi_processing.py feature_metadata.json "
                                                  "(hierarchical_feature_selection.merged_2level.level1_router)",
        "level1_local_feature_selection": ctx.router_meta,
        "level2_tags_swept": ctx.tags,
        "level2_local_feature_selection_source": "fermi_processing.py feature_metadata.json "
                                                  "(hierarchical_feature_selection.merged_2level.level2_branches)",
        "level2_local_feature_selection": ctx.branch_meta,
        "superclass_order": SUPERCLASS_ORDER, "taxonomy": TAXONOMY,
        "scaler_variants_swept": list(SCALER_VARIANTS.keys()), "default_scaler": DEFAULT_SCALER_NAME,
        "pca_variants_swept": list(PCA_VARIANTS.keys()),
        "mlp_architectures_swept": {k: list(v) for k, v in MLP_ARCHITECTURES.items()},
        "level1_router": router_info, "level1_leaderboard": level1_leaderboard.to_dict("records"),
        "level2_branches": {sc: _branch_metric_note(sc, bm) for sc, bm in branch_models.items()},
        "hierarchical_evaluation_baseline": baseline_eval,
        "hierarchical_evaluation_by_router": hierarchical_eval_by_router,
        "router_comparison_csv": str(io_bundle["split_dir"] / "router_branch_comparison.csv"),
        # generalization_estimate comes from whichever combo actually won
        # (baseline OR a screening survivor).
        "adopted_combo": adopted_from,
        "generalization_estimate": {
            "source": f"end_to_end_chain_cv_oof ({adopted_from})",
            "accuracy": final_eval["end_to_end_val"]["accuracy"],
            "bal_accuracy": final_eval["end_to_end_val"]["bal_accuracy"],
            "macro_f1": final_eval["end_to_end_val"]["macro_f1"],
            "kappa": final_eval["end_to_end_val"]["kappa"],
            "n_folds": final_eval["n_folds"], "n_oof_rows": final_eval["n_oof_rows"],
        },
        "selection_bias_disclosure": (
            "Router, branch, and (if adopted_combo != 'section4_6_baseline') whole-system configuration "
            "were all chosen by comparing validation scores across many candidates on the same data used "
            "to score them — a small, well-characterized optimism (Cawley & Talbot, 2010, JMLR "
            "11:2079-2107) affecting these validation numbers, not any held-out test-set metric, since no "
            "test set is read anywhere in this file."
        ),
        "inference_note": (
            "Unassociated-source inference (fermi_inference.py) uses the same soft-probability "
            "router/branch blend this file's chain-CV validates."
        ),
    }
    metrics_path = io_bundle["split_dir"] / "merged_classification_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(merged_metrics, f, indent=2, default=str)
    module_logger.notice(f"Hierarchical metrics written -> {metrics_path}")

    # --- Summary -----------------------------------------------------------------
    module_logger.notice("=" * 70)
    module_logger.notice(f"Module 3 (Hierarchical Merged, 2-level) complete — split '{split_id}', "
                          "VALIDATION ONLY, NO TEST TOUCHED.")
    module_logger.notice(f"  Selection metric      : {SELECTION_METRIC}")
    module_logger.notice(f"  Level-1 router         : {router_info['model_name']} "
                          f"(val_{SELECTION_METRIC}={router_info.get(f'val_{SELECTION_METRIC}', float('nan')):.4f})")
    n_trained = sum(1 for b in branch_models.values() if b is not None)
    module_logger.notice(f"  Level-2 branches       : {n_trained}/{len(SUPERCLASS_ORDER)} trained")
    module_logger.notice(f"  Adopted combo          : {adopted_from}")
    module_logger.notice(f"  End-to-end val {SELECTION_METRIC} : {final_score:.4f}")
    module_logger.notice(f"  Metrics file           : {metrics_path}")
    module_logger.notice("=" * 70)

    summary_files = [
        io_bundle["metadata_manifest_path"], metrics_path,
        io_bundle["split_dir"] / "router_branch_comparison.csv",
        io_bundle["split_dir"] / "mlp_architecture_comparison_merged.csv",
        io_bundle["split_dir"] / f"winner_info_L1_{MERGED_LABEL_SET}.json",
        io_bundle["split_dir"] / "inference_2level" / "hierarchical_inference_manifest.json",
    ]
    publish_split_summary(split_id, OUTPUT_ROOT, summary_files)

    KNN_PER_CLASS_CLAMP = False # reset before next split's Level-1 router fit

    return {
        "split_id": split_id, "router_info": router_info, "branch_models": branch_models,
        "baseline_eval": baseline_eval, "final_eval": final_eval, "adopted_from": adopted_from,
        "final_score": final_score, "mlp_comparison": mlp_comparison,
    }


# %% [markdown]
# ## Section 12 — Driver loop

# %%
def build_all_splits_summary_merged(output_root: Path, all_splits_results: dict):
    """One row per split: router winner, adopted combo, end-to-end val
    score — mirrors build_all_splits_summary_fine / _3level's pattern."""
    rows = []
    for split_id, result in all_splits_results.items():
        rows.append({
            "split_id": split_id,
            "router_model": result["router_info"]["model_name"],
            "adopted_from": result["adopted_from"],
            f"end_to_end_val_{SELECTION_METRIC}": result["final_score"],
            "end_to_end_val_accuracy": result["final_eval"]["end_to_end_val"]["accuracy"],
            "end_to_end_val_bal_accuracy": result["final_eval"]["end_to_end_val"]["bal_accuracy"],
            "end_to_end_val_macro_f1": result["final_eval"]["end_to_end_val"]["macro_f1"],
            "end_to_end_val_kappa": result["final_eval"]["end_to_end_val"]["kappa"],
        })
    if not rows:
        print("  ⚠ build_all_splits_summary_merged: nothing to summarize.")
        return None
    combined = pd.DataFrame(rows)
    out_dir = output_root / "results_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_splits_summary_merged.csv"
    combined.to_csv(out_path, index=False)
    print(f"  ✓ all_splits_summary_merged -> {out_path}  ({len(combined)} split(s))")
    return out_path


def build_all_splits_summary_merged(output_root: Path, all_splits_results: dict):
    """One row per split: router winner, adopted combo, end-to-end val
    score — mirrors build_all_splits_summary_fine / _3level's pattern."""
    rows = []
    for split_id, result in all_splits_results.items():
        rows.append({
            "split_id": split_id,
            "router_model": result["router_info"]["model_name"],
            "adopted_from": result["adopted_from"],
            f"end_to_end_val_{SELECTION_METRIC}": result["final_score"],
            "end_to_end_val_accuracy": result["final_eval"]["end_to_end_val"]["accuracy"],
            "end_to_end_val_bal_accuracy": result["final_eval"]["end_to_end_val"]["bal_accuracy"],
            "end_to_end_val_macro_f1": result["final_eval"]["end_to_end_val"]["macro_f1"],
            "end_to_end_val_kappa": result["final_eval"]["end_to_end_val"]["kappa"],
        })
    if not rows:
        print("  ⚠ build_all_splits_summary_merged: nothing to summarize.")
        return None
    combined = pd.DataFrame(rows)
    out_dir = output_root / "results_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_splits_summary_merged.csv"
    combined.to_csv(out_path, index=False)
    print(f"  ✓ all_splits_summary_merged -> {out_path}  ({len(combined)} split(s))")
    return out_path




def _reconstruct_result_from_metrics(split_id: str, metrics_path: Path) -> dict:
    """Read back an already-completed split's merged_classification_metrics.json
    instead of recomputing it. Shape-compatible with run_merged_module_for_split()'s
    return value for everything build_all_splits_summary() actually reads."""
    with open(metrics_path) as f:
        mm = json.load(f)
    gen = mm["generalization_estimate"]
    return {
        "split_id": split_id,
        "router_info": mm["level1_router"],
        "adopted_from": mm["adopted_combo"],
        "final_score": gen[SELECTION_METRIC],
        "final_eval": {"end_to_end_val": {
            "accuracy": gen["accuracy"], "bal_accuracy": gen["bal_accuracy"],
            "macro_f1": gen["macro_f1"], "kappa": gen["kappa"],
        }},
        "branch_models": {sc: (None if "note" in v else True)
                           for sc, v in mm["level2_branches"].items()},
    }


# %% [markdown]
# ### Section 12 — driver loop (CELL A)

# %%
# %%
import traceback

ALL_SPLITS_RESULTS = {}
FAILED_SPLITS = []
for _split_id in pbar(SPLITS_TO_RUN, desc="Merged module — splits"):
    _done_path = OUTPUT_ROOT / "results_aggregated" / _split_id / "merged_classification_metrics.json"
    if _done_path.exists():
        print(f"  ⏭ [{_split_id}] already complete — reading {_done_path}, no recompute.")
        ALL_SPLITS_RESULTS[_split_id] = _reconstruct_result_from_metrics(_split_id, _done_path)
        continue
    try:
        ALL_SPLITS_RESULTS[_split_id] = run_merged_module_for_split(_split_id)
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
# ### Section 12 — summary build (CELL B)

# %%
# %%
ALL_SPLITS_SUMMARY_PATH = build_all_splits_summary_merged(OUTPUT_ROOT, ALL_SPLITS_RESULTS)

print("=" * 78)
print("Module 3 (Hierarchical Merged, 2-level) — ALL REQUESTED SPLITS COMPLETE.")
print(f"  Splits run          : {SPLITS_TO_RUN}")
print(f"  Cross-split summary  : {ALL_SPLITS_SUMMARY_PATH}")
print(f"  Per-split artifacts  : {OUTPUT_ROOT / 'results_aggregated'}/{{split_id}}/")
print("=" * 78)


# %%
def backfill_top15_merged(output_root, split_ids, metric=SELECTION_METRIC):
    for split_id in split_ids:
        mpath = output_root / "results_aggregated" / split_id / "merged_classification_metrics.json"
        if not mpath.exists():
            print(f"  ⚠ [{split_id}] no metrics json yet — needs a real run."); continue
        mm = json.load(open(mpath))
        by_router = mm.get("hierarchical_evaluation_by_router", {})
        if not by_router:
            print(f"  ⚠ [{split_id}] hierarchical_evaluation_by_router is empty — no screened combos "
                  "to summarize for this split (baseline-only run). Skipping.")
            continue
        rows = [{"combo": name, "router_model": name.split("__branch")[0],
                 "end_to_end_val_accuracy": ev["end_to_end_val"]["accuracy"],
                 "end_to_end_val_bal_accuracy": ev["end_to_end_val"]["bal_accuracy"],
                 "end_to_end_val_macro_f1": ev["end_to_end_val"]["macro_f1"],
                 "end_to_end_val_kappa": ev["end_to_end_val"]["kappa"],
                 f"end_to_end_val_{metric}": ev["end_to_end_val"][metric],
                 "n_folds": ev["n_folds"], "cv_mode": ev["cv_mode"]}
                for name, ev in by_router.items()]
        df = pd.DataFrame(rows).sort_values(f"end_to_end_val_{metric}", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        print(f"\nTOP-{len(df)} WHOLE-SYSTEM MERGED CANDIDATES — split '{split_id}' [backfilled]")
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(df.to_string(index=False))
        out_csv = output_root / split_id / f"top{len(df)}_summary_merged.csv"
        df.to_csv(out_csv, index=False)
        print(f"  ✓ -> {out_csv}")

backfill_top15_merged(OUTPUT_ROOT, SPLITS_TO_RUN)

# %%
import os
print("cwd:", os.getcwd())
print("OUTPUT_ROOT resolved:", OUTPUT_ROOT.resolve())
print("SPLITS_TO_RUN:", SPLITS_TO_RUN)
print("ALL_SPLITS_RESULTS keys:", list(ALL_SPLITS_RESULTS.keys()))
print("FAILED_SPLITS:", FAILED_SPLITS)

for name in ("merged_classification_metrics.json", "CRASH_traceback.txt"):
    hits = list(OUTPUT_ROOT.rglob(name))
    print(f"{name}: {hits if hits else 'NONE found anywhere under OUTPUT_ROOT'}")
