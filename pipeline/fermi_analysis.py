# %% [markdown]
# # Module 6: Post-hoc Analysis & Reporting

# %% [markdown]
#
#
# Runs against the hierarchy builders (`fermi_fine.py`, `fermi_merged.py`,
# `fermi_3levels.py`, `fermi_processing.py`, `fermi_common.py`) and the
# validation results (`fermi_validation.py`).
#
# Node-level diagnostics:
#   - Feature importance (permutation-based quick screen)
#   - Cumulative feature-power curves
#   - Per-class feature distributions
#   - Unsupervised clustering diagnostics
#   - Legacy per-node metric comparison bars
#   - Best-tree-per-node + tree-family gallery (DecisionTree/RandomForest/
#     GBC/HistGBC/CatBoost, with linear-coefficient and 2D decision-boundary
#     companions for non-tree models)
#
# Validation-level reports (from `fermi_validation.py`'s artifacts):
#   - Stage 5/6 cross-family comparison (CI forest plot + significance matrix)
#   - Train/validation/test three-way metric report
#   - Rare-class shrinkage (empirical-Bayes partial pooling) forest plots
#   - Per-class metrics table/heatmap
#   - Calibration reliability diagrams (guarded: RUN_CALIBRATION_CHECK_NOW
#     may be False, in which case CALIBRATION_CHECK is None)
#   - Unassociated-source inference confidence/agreement summaries
#   - End-to-end architecture summary table
#   - SHAP-based explainability (guarded: RUN_SHAP_EXPLAINABILITY_NOW may be
#     False, in which case SHAP_RESULTS is never assigned as a global at all)
#
# `AnalysisNode` carries an explicit
# `model_specs` dict populated per-hierarchy (FINE_MODEL_SPECS /
# MERGED_MODEL_SPECS / THREELEVEL_MODEL_SPECS). All builder lookups call the
# common `get_builder(model_name, model_specs=node.model_specs)` explicitly.
# This is required because `fermi_fine.py`, `fermi_merged.py`, and
# `fermi_3levels.py` EACH define a bare top-level `_get_builder(name)`
# closure bound to their own MODEL_SPECS dict; since all three run in the
# same shared kernel with 3-level last, any bare `_get_builder(...)` or
# bare `get_builder(name)` call silently resolves to THREELEVEL_MODEL_SPECS
# regardless of which hierarchy actually owns the node. Every call site in
# this module is therefore explicit.

# %%
from __future__ import annotations

import gc
import itertools
import json
import re
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Section 0: Output layout, constants, defensive fallbacks

# %%
ANALYSIS_OUT_DIR = OUTPUT_ROOT / "analysis"
ANALYSIS_FEATIMP_DIR = ANALYSIS_OUT_DIR / "feature_importance"
ANALYSIS_POWERCURVE_DIR = ANALYSIS_OUT_DIR / "power_curves"
ANALYSIS_DIST_DIR = ANALYSIS_OUT_DIR / "distributions"
ANALYSIS_CLUSTER_DIR = ANALYSIS_OUT_DIR / "clustering"
ANALYSIS_VIOLIN_DIR = ANALYSIS_OUT_DIR / "violin"
ANALYSIS_TREEGALLERY_DIR = ANALYSIS_OUT_DIR / "tree_gallery"
ANALYSIS_METRICCMP_DIR = ANALYSIS_OUT_DIR / "metric_comparison"

# validation-level sections
ANALYSIS_FAMILY_DIR = ANALYSIS_OUT_DIR / "family_comparison"
ANALYSIS_TVT_DIR = ANALYSIS_OUT_DIR / "train_val_test"
ANALYSIS_SHRINKAGE_DIR = ANALYSIS_OUT_DIR / "rare_class_shrinkage"
ANALYSIS_PERCLASS_DIR = ANALYSIS_OUT_DIR / "per_class_metrics"
ANALYSIS_CALIBRATION_DIR = ANALYSIS_OUT_DIR / "calibration"
ANALYSIS_INFERENCE_DIR = ANALYSIS_OUT_DIR / "inference_summary"
ANALYSIS_ARCH_DIR = ANALYSIS_OUT_DIR / "architecture_summary"
ANALYSIS_SHAP_DIR = ANALYSIS_OUT_DIR / "shap_explainability"

for _d in (
    ANALYSIS_OUT_DIR, ANALYSIS_FEATIMP_DIR, ANALYSIS_POWERCURVE_DIR, ANALYSIS_DIST_DIR,
    ANALYSIS_CLUSTER_DIR, ANALYSIS_VIOLIN_DIR, ANALYSIS_TREEGALLERY_DIR, ANALYSIS_METRICCMP_DIR,
    ANALYSIS_FAMILY_DIR, ANALYSIS_TVT_DIR, ANALYSIS_SHRINKAGE_DIR, ANALYSIS_PERCLASS_DIR,
    ANALYSIS_CALIBRATION_DIR, ANALYSIS_INFERENCE_DIR, ANALYSIS_ARCH_DIR, ANALYSIS_SHAP_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

ANALYSIS_LOGGER = get_section_logger("analysis", ANALYSIS_OUT_DIR / "logs")

ALL_FEATURES_TAG = "ALL_FEATURES"
MAX_MODELS_PER_PANEL = 8
FEATIMP_TOP_N = 30
CLUSTER_TOP_N_FEATURES = 6
CLUSTER_K_RANGE = range(2, 9)
SILHOUETTE_SAMPLE_CAP = 3000  # bounds silhouette_score's O(n^2) distance matrix
PERM_IMPORTANCE_N_REPEATS = 10
MIN_REAL_ROWS_FOR_VIOLIN = 5

# leaderboard column names — confirmed against
# the same fermi_common.py this file already shares a kernel with.
VAL_METRIC_COLS = ["val_accuracy", "val_bal_accuracy", "val_macro_f1", "val_kappa"]
METRIC_TITLES = {
    "val_accuracy": "Validation accuracy",
    "val_bal_accuracy": "Validation balanced accuracy",
    "val_macro_f1": "Validation macro-F1",
    "val_kappa": "Validation Cohen's \u03ba",
}
MLP_MODEL_NAMES = [m for m in MODEL_SPECS if m in MLP_ARCHITECTURES or "MLP" in m.upper()]

# --- Defensive fallbacks for names optionally supplied by the untracked
# viz helper file. Defensive pattern for TREE_MODEL_NAMES / dtreeviz, plus a
# local implementation of superclass_colors (a bare try/except ->
# palette=None would silently disable the palette).
if "TREE_MODEL_NAMES" not in globals():
    TREE_MODEL_NAMES = {"DecisionTree", "RandomForest", "GBC", "HistGBC", "CatBoost"}

if "SUPERCLASS_ORDER" in globals():
    _SC_ORDER_FOR_PALETTE = list(SUPERCLASS_ORDER)
else:
    _SC_ORDER_FOR_PALETTE = []

if "superclass_colors" not in globals():
    _SUPERCLASS_PALETTE_CACHE: dict[str, str] = {}

    def superclass_colors(classes: Sequence[str] | None = None) -> dict[str, str]:
        """Deterministic superclass -> hex color mapping.

        Falls back to a fixed, colorblind-friendly seaborn palette keyed by
        SUPERCLASS_ORDER (extended with any classes not present there, in
        sorted order, so the mapping is stable across calls/processes).
        """
        base_order = _SC_ORDER_FOR_PALETTE or sorted(set(classes or []))
        extra = sorted(set(classes or []) - set(base_order)) if classes else []
        full_order = list(base_order) + [c for c in extra if c not in base_order]
        if not full_order:
            return {}
        missing = [c for c in full_order if c not in _SUPERCLASS_PALETTE_CACHE]
        if missing:
            pal = sns.color_palette("colorblind", n_colors=len(full_order))
            hexes = [matplotlib.colors.to_hex(c) for c in pal]
            for name, hx in zip(full_order, hexes):
                _SUPERCLASS_PALETTE_CACHE.setdefault(name, hx)
        return {c: _SUPERCLASS_PALETTE_CACHE[c] for c in (classes or full_order)}

try:
    import dtreeviz as _DTREEVIZ_MOD  # noqa: F401
except Exception:
    _DTREEVIZ_MOD = None


# %% [markdown]
# ## Section 1: AnalysisNode — uniform view over every node of every hierarchy
#
# `model_specs` is the structural fix for the three-way `_get_builder`
# shadowing hazard (fine/merged/3-level each define a bare `_get_builder`
# closure bound to their own MODEL_SPECS dict; 3-level runs last in the
# shared kernel and silently wins any bare call). Every downstream lookup
# in this module uses `get_builder(model_name, model_specs=node.model_specs)`
# explicitly instead.

# %%
@dataclass
class AnalysisNode:
    key: str                              # unique id, e.g. "fine_A", "branch_bll_L3"
    hierarchy: str                        # "fine" | "merged" | "3level"
    label: str                            # display name
    x_train_by_tag: dict[str, pd.DataFrame]
    y_train_enc: np.ndarray
    class_names: list[str]
    tags: list[str]
    model_specs: dict[str, Any] # explicit, never resolved via bare _get_builder
    x_test_by_tag: dict[str, pd.DataFrame] = field(default_factory=dict)
    y_test_enc: Optional[np.ndarray] = None
    label_encoder: Optional[Any] = None
    superclass: Optional[str] = None      # set for branch/leaf nodes, used for palette coloring
    split_id: Optional[Any] = None
    label_set: Optional[str] = None # the on-disk label_set string this
                                           # node's leaderboard rows were saved under — NOT the same
                                           # as `key` (e.g. merged branch label_set="branch_{sc}" vs
                                           # key="branch_{sc}_L2"; 3-level gate label_set="gate" vs
                                           # key="3level_gate"). Required by _attach_leaderboard.
    leaderboard: Optional[pd.DataFrame] = None  # lazily attached by _attach_leaderboard

    def x_all(self, tag: str | None = None) -> pd.DataFrame:
        t = tag or ALL_FEATURES_TAG
        if t in self.x_train_by_tag:
            return self.x_train_by_tag[t]
        # defensive fallback: any tag present
        if self.x_train_by_tag:
            return next(iter(self.x_train_by_tag.values()))
        raise KeyError(f"AnalysisNode {self.key!r}: no feature tags available")


def _candidate_output_dirs(hierarchy: str) -> list[Path]:
    """Best-effort resolution of a hierarchy's own results directory. The
    exact per-hierarchy output-dir constant name isn't part of the confirmed
    API surface used elsewhere in this module, so several naming
    conventions are tried in order and the first one `load_all_results`
    finds anything under wins (see `_attach_leaderboard`)."""
    named_globals = [
        f"{hierarchy.upper()}_OUTPUT_DIR", f"{hierarchy.upper()}_RESULTS_DIR",
        f"{hierarchy.upper()}_DIR",
    ]
    candidates = [globals()[g] for g in named_globals if g in globals()]
    candidates += [
        OUTPUT_ROOT / hierarchy,
        OUTPUT_ROOT / f"{hierarchy}_results",
    ]
    return candidates


def _attach_leaderboard(node: AnalysisNode, split_ids: Optional[Sequence[str]] = None) -> AnalysisNode:
    """[Fixed] Loads this node's own persisted leaderboard rows, pooled
    across every split in `split_ids` (default: VALIDATION_SPLITS, the
    kernel global fermi_validation.py sets up) — never mixed with
    another node's rows. fine/merged/3-level all call
    `io_for(OUTPUT_ROOT, split_id)` directly (confirmed against
    fermi_fine.py's run_fine_module_for_split and fermi_3levels.py's
    Section 0 markdown on why merged/3-level share OUTPUT_ROOT) — there is
    NO per-hierarchy output subdirectory, so every hierarchy's leaderboards
    for one split live together under OUTPUT_ROOT/{split_id}/results/.
    load_all_results tags each row with `label_set`; this filters to
    node.label_set."""
    split_ids = split_ids if split_ids is not None else (
        VALIDATION_SPLITS if "VALIDATION_SPLITS" in globals() else [node.split_id]
    )
    if node.label_set is None:
        ANALYSIS_LOGGER.notice(f"[{node.key}] no label_set set on this node — cannot resolve a leaderboard")
        return node
    frames = []
    for split_id in split_ids:
        try:
            df = load_all_results(OUTPUT_ROOT, split_id=split_id)
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] load_all_results failed for split={split_id!r}: {exc}")
            continue
        if df is None or df.empty or "label_set" not in df.columns:
            continue
        sub = df[df["label_set"] == node.label_set].copy()
        if not sub.empty:
            sub["split_id"] = split_id
            frames.append(sub)
    if not frames:
        ANALYSIS_LOGGER.notice(
            f"[{node.key}] no leaderboard rows found for label_set={node.label_set!r} across {len(split_ids)} split(s)"
        )
        return node
    node.leaderboard = pd.concat(frames, ignore_index=True)
    return node


def _node_from_branch_dict(
    branch: dict,
    *,
    hierarchy: str,
    key: str,
    label: str,
    model_specs: dict[str, Any],
    label_set: Optional[str] = None,
    superclass: Optional[str] = None,
    split_id: Optional[Any] = None,
) -> Optional[AnalysisNode]:
    """Wraps a branch dict, as produced by `build_branch_datasets` /
    `_m2_build_branch_datasets`, into an AnalysisNode. Returns None (logging a
    warning) rather than raising, so a single malformed/empty branch cannot
    abort an entire hierarchy's node-building pass."""
    try:
        x_train_by_tag = dict(branch["x_train_by_tag"])
        y_train_enc = np.asarray(branch["y_train_enc"])
        if y_train_enc.size == 0:
            ANALYSIS_LOGGER.notice(f"[{key}] empty branch (0 train rows) — skipped")
            return None
        label_encoder = branch.get("label_encoder")
        class_names = branch.get("class_names")  # a numpy array: `or` would test its truth value
        class_names = list(class_names if class_names is not None else getattr(label_encoder, "classes_", []))
        tags = list(branch.get("tags") or x_train_by_tag.keys())
        return AnalysisNode(
            key=key,
            hierarchy=hierarchy,
            label=label,
            x_train_by_tag=x_train_by_tag,
            x_test_by_tag=dict(branch.get("x_test_by_tag") or {}),
            y_train_enc=y_train_enc,
            y_test_enc=(np.asarray(branch["y_test_enc"]) if branch.get("y_test_enc") is not None else None),
            label_encoder=label_encoder,
            class_names=class_names,
            tags=tags,
            model_specs=model_specs,
            label_set=label_set,
            superclass=superclass,
            split_id=split_id,
        )
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[{key}] failed to wrap branch dict: {exc}\n{traceback.format_exc()}")
        return None


def _safe_build_nodes(builder_fn: Callable[[], list[AnalysisNode]], label: str) -> list[AnalysisNode]:
    """Runs a per-hierarchy node builder under a try/except, logging and
    degrading to an empty node list on failure rather than aborting the
    whole analysis run."""
    try:
        nodes = builder_fn()
        ANALYSIS_LOGGER.info(f"[{label}] built {len(nodes)} node(s)")
        return nodes
    except Exception as exc:
        ANALYSIS_LOGGER.error(f"[{label}] node construction failed: {exc}\n{traceback.format_exc()}")
        return []


# %% [markdown]
# ### Section 1a: Fine-hierarchy nodes

# %%
def _load_feature_metadata(split_id: str) -> dict:
    """Mirrors validation's _val_load_feature_metadata — analysis has no
    module that exports FEATURE_METADATA as a kernel global, so it must
    load it itself, per split, the same way fermi_fine.py's
    run_fine_module_for_split does."""
    path = PROCESSING_ROOT / split_id / "diagnostics" / "feature_metadata.json"
    with open(path) as f:
        return json.load(f)



def _build_fine_nodes(split_id: Any = SPLIT0) -> list[AnalysisNode]:
    nodes: list[AnalysisNode] = []
    feature_metadata = _load_feature_metadata(split_id)
    source_dir = PROCESSING_ROOT / split_id
    for setup_key in FINE_SETUP_KEYS:
        try:
            ctx = build_fine_context(source_dir, feature_metadata, setup_key, split_id)
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[fine_{setup_key}] build_fine_context failed: {exc}")
            continue
        if ctx.y_train_enc is None or len(ctx.y_train_enc) == 0:
            continue
        nodes.append(AnalysisNode(
            key=f"fine_{setup_key}",
            hierarchy="fine",
            label=f"Fine [{FINE_SETUP_LABELS.get(setup_key, setup_key)}]",
            x_train_by_tag=dict(ctx.x_train_by_tag),
            x_test_by_tag=dict(ctx.x_test_by_tag or {}),
            y_train_enc=np.asarray(ctx.y_train_enc),
            y_test_enc=(np.asarray(ctx.y_test_enc) if ctx.y_test_enc is not None else None),
            label_encoder=ctx.label_encoder,
            class_names=list(ctx.class_names),
            tags=list(ctx.tags),
            model_specs=FINE_MODEL_SPECS, # explicit, not via bare _get_builder
            label_set=ctx.label_set,
            split_id=split_id,
        ))
    return nodes


# %% [markdown]
# ### Section 1b: Merged-hierarchy nodes (superclass router + per-superclass branches)

# %%
def _build_merged_nodes(split_id: Any = SPLIT0) -> list[AnalysisNode]:
    nodes: list[AnalysisNode] = []
    try:
        ctx = build_merged_context(PROCESSING_ROOT / split_id, _load_feature_metadata(split_id), split_id)
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[merged_router] build_merged_context failed: {exc}")
        return nodes

    # Router node: predicts SUPERCLASS_ORDER from ALL_FEATURES(+local tags).
    x_router_by_tag = dict(ctx.x_train_by_tag)
    router_tags = [t for t in ctx.tags if t != ALL_FEATURES_TAG]
    if getattr(ctx, "router_meta", None) is not None:
        _apply_local_feature_sets(ctx.x_train_by_tag[ALL_FEATURES_TAG], ctx.router_meta, x_router_by_tag, router_tags)
    if ALL_FEATURES_TAG not in x_router_by_tag:
        x_router_by_tag[ALL_FEATURES_TAG] = ctx.x_train_by_tag[ALL_FEATURES_TAG]
    nodes.append(AnalysisNode(
        key="merged_router",
        hierarchy="merged",
        label="Merged [Superclass Router]",
        x_train_by_tag=x_router_by_tag,
        y_train_enc=np.asarray(ctx.y_train_super_enc),
        label_encoder=SUPER_LE,
        class_names=list(SUPERCLASS_ORDER),
        tags=router_tags + [ALL_FEATURES_TAG],
        model_specs=MERGED_MODEL_SPECS,
        label_set=MERGED_LABEL_SET, # = "merged"
        split_id=split_id,
    ))

    # Per-superclass branch nodes: predicts fine class within that superclass.

    # _m2_build_branch_datasets builds every superclass's branch
    # in ONE call (it has its own internal SUPERCLASS_ORDER loop) and
    # returns {sc: branch_dict_or_None} — it does not take a superclass as
    # a second positional arg. The old per-sc call `_m2_build_branch_datasets(ctx, sc)`
    # fed `sc` into route_str_train, turning `tr_mask = route_str_train == sc`
    # into a scalar bool and corrupting every branch (numpy.ndarray-in-set crash).
    # Per-superclass branch nodes: predicts fine class within that superclass.
    # _m2_build_branch_datasets builds every superclass's branch
    # in ONE call (it has its own internal SUPERCLASS_ORDER loop) and
    # returns {sc: branch_dict_or_None} — it does not take a superclass as
    # a second positional arg. The old per-sc call `_m2_build_branch_datasets(ctx, sc)`
    # fed `sc` into route_str_train, turning `tr_mask = route_str_train == sc`
    # into a scalar bool and corrupting every branch (numpy.ndarray-in-set crash).
    try:
        branches_by_sc = _m2_build_branch_datasets(ctx)
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[merged_branches] _m2_build_branch_datasets failed: {exc}")
        branches_by_sc = {}

    for sc in SUPERCLASS_ORDER:
        branch = branches_by_sc.get(sc)
        if branch is None:
            continue
        if ALL_FEATURES_TAG not in branch.get("x_train_by_tag", {}):
            tr_mask = np.asarray(ctx.y_train_super_str) == sc
            branch.setdefault("x_train_by_tag", {})[ALL_FEATURES_TAG] = (
                ctx.x_train_by_tag[ALL_FEATURES_TAG][tr_mask].reset_index(drop=True)
            )
        node = _node_from_branch_dict(
            branch,
            hierarchy="merged",
            key=f"branch_{sc}_L2",
            label=f"Merged [{sc} branch]",
            model_specs=MERGED_MODEL_SPECS,
            label_set=f"branch_{sc}", # — no "_L2" on disk
            superclass=sc,
            split_id=split_id,
        )
        if node is not None:
            nodes.append(node)
    return nodes


# %% [markdown]
# ### Section 1c: 3-level-hierarchy nodes (gate, blazar subgate, non-blazar router, per-superclass branches)

# %%
def _build_threelevel_nodes(split_id: Any = SPLIT0) -> list[AnalysisNode]:
    nodes: list[AnalysisNode] = []
    try:
        ctx = build_3level_context(PROCESSING_ROOT / split_id, _load_feature_metadata(split_id), split_id)
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[3level] build_3level_context failed: {exc}")
        return nodes

    # --- Gate node: blazar vs non-blazar vs ... (build_gate_labels) ---
    try:
        y_gate_enc = GATE_LE.transform(build_gate_labels(ctx.y_train_super_str))
        x_gate_by_tag = {t: x for t, x in ctx.x_train_by_tag.items() if t != ALL_FEATURES_TAG}
        gate_tags = [t for t in ctx.tags if t != ALL_FEATURES_TAG]
        if getattr(ctx, "gate_meta", None) is not None:
            _apply_local_feature_sets(ctx.x_train_by_tag[ALL_FEATURES_TAG], ctx.gate_meta, x_gate_by_tag, gate_tags)
        if ALL_FEATURES_TAG not in x_gate_by_tag:
            x_gate_by_tag[ALL_FEATURES_TAG] = ctx.x_train_by_tag[ALL_FEATURES_TAG]
        nodes.append(AnalysisNode(
            key="3level_gate",
            hierarchy="3level",
            label="3-Level [Gate]",
            x_train_by_tag=x_gate_by_tag,
            y_train_enc=y_gate_enc,
            label_encoder=GATE_LE,
            class_names=list(GATE_LE.classes_),
            tags=gate_tags + [ALL_FEATURES_TAG],
            model_specs=THREELEVEL_MODEL_SPECS,
            label_set="gate",
            split_id=split_id,
        ))
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[3level_gate] node construction failed: {exc}\n{traceback.format_exc()}")

    # --- Blazar subgate + non-blazar router (row-masked) ---
    # meta_attrs lists candidate ctx attribute names in priority order
    # (ctx.subgate_meta / ctx.nonblazar_meta first; extra aliases are tried
    # defensively in case ThreeLevelContext names them differently).
    for sub_key, le, meta_attrs, mask_fn, label in (
        ("blazar_subgate", BLAZAR_SUBGATE_LE, ("subgate_meta", "blazar_subgate_meta"),
         lambda s: np.isin(s, list(BLAZAR_SUPERCLASSES)), "3-Level [Blazar Subgate]"),
        ("nonblazar_router", NONBLAZAR_ROUTER_LE, ("nonblazar_meta", "nonblazar_router_meta"),
         lambda s: ~np.isin(s, list(BLAZAR_SUPERCLASSES)), "3-Level [Non-Blazar Router]"),
    ):
        try:
            tr_mask = mask_fn(np.asarray(ctx.y_train_super_str))
            if tr_mask.sum() == 0:
                continue
            y_sub_str = np.asarray(ctx.y_train_super_str)[tr_mask]
            y_sub_enc = le.transform(y_sub_str)
            x_sub_by_tag = {t: x[tr_mask].reset_index(drop=True) for t, x in ctx.x_train_by_tag.items() if t != ALL_FEATURES_TAG}
            sub_tags = [t for t in ctx.tags if t != ALL_FEATURES_TAG]
            meta_blob = next((getattr(ctx, a) for a in meta_attrs if getattr(ctx, a, None) is not None), None)
            x_all_masked = ctx.x_train_by_tag[ALL_FEATURES_TAG][tr_mask].reset_index(drop=True)
            if meta_blob is not None:
                _apply_local_feature_sets(x_all_masked, meta_blob, x_sub_by_tag, sub_tags)
            if ALL_FEATURES_TAG not in x_sub_by_tag:
                x_sub_by_tag[ALL_FEATURES_TAG] = x_all_masked
            nodes.append(AnalysisNode(
                key=f"3level_{sub_key}",
                hierarchy="3level",
                label=label,
                x_train_by_tag=x_sub_by_tag,
                y_train_enc=y_sub_enc,
                label_encoder=le,
                class_names=list(le.classes_),
                tags=sub_tags + [ALL_FEATURES_TAG],
                model_specs=THREELEVEL_MODEL_SPECS,
                label_set=sub_key, # sub_key IS the on-disk label_set
                split_id=split_id,
            ))
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[3level_{sub_key}] node construction failed: {exc}\n{traceback.format_exc()}")

    # --- Per-non-blazar-superclass branch nodes (fine class within superclass) ---

    # same call-signature bug as merged's branch loop — build_branch_datasets
    # builds every non-blazar superclass's branch in one call and returns a dict.
    try:
        branches_by_sc = build_branch_datasets(ctx)
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[3level_branches] build_branch_datasets failed: {exc}")
        branches_by_sc = {}

    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        branch = branches_by_sc.get(sc)
        if branch is None:
            continue
        node = _node_from_branch_dict(
            branch,
            hierarchy="3level",
            key=f"branch_{sc}_L3",
            label=f"3-Level [{sc} branch]",
            model_specs=THREELEVEL_MODEL_SPECS,
            label_set=f"branch_{sc}_L3",
            superclass=sc,
            split_id=split_id,
        )
        if node is not None:
            nodes.append(node)
    return nodes


# %% [markdown]
# ### Section 1d: Assemble all nodes, attach leaderboards

# %%
ALL_ANALYSIS_NODES: list[AnalysisNode] = (
    _safe_build_nodes(_build_fine_nodes, "fine")
    + (_safe_build_nodes(_build_merged_nodes, "merged") if "merged" in FAMILIES else [])
    + (_safe_build_nodes(_build_threelevel_nodes, "3level") if "3level" in FAMILIES else [])
)

for _node in ALL_ANALYSIS_NODES:
    _attach_leaderboard(_node)

ANALYSIS_LOGGER.info(f"Assembled {len(ALL_ANALYSIS_NODES)} analysis node(s) across all hierarchies.")


# %% [markdown]
# ## Section 2: Feature importance & cumulative power-curve suites (ported)

# %%
def _pct_from_tag(tag: str) -> Optional[str]:
    """Maps a feature-set tag back to the FEATURE_POWER_THRESHOLDS key it
    was built from. fermi_common.py's compute_feature_sets writes tags
    as f"{prefix}_{key}" (prefix may be empty) plus f"{key}_DECORR", where
    key in {"99","90","80"} — never a "PCT"-style name. Returns the
    threshold key itself (e.g. "99"), or None if `tag` isn't one."""
    base = tag[:-len("_DECORR")] if tag.endswith("_DECORR") else tag
    if base in FEATURE_POWER_THRESHOLDS:
        return base
    suffix = base.rsplit("_", 1)[-1]
    return suffix if suffix in FEATURE_POWER_THRESHOLDS else None


def _threshold_tags(node: AnalysisNode) -> list[str]:
    """Non-_DECORR feature-set tags on this node that correspond to a
    FEATURE_POWER_THRESHOLDS key, sorted ascending by that threshold's
    power fraction."""
    out = []
    for t in node.tags:
        if t.endswith("_DECORR"):
            continue
        key = _pct_from_tag(t)
        if key is not None:
            out.append((FEATURE_POWER_THRESHOLDS[key], t))
    out.sort(key=lambda pair: pair[0])
    return [t for _, t in out]


def _resolve_pct_tag(node: AnalysisNode, threshold_key: str) -> Optional[str]:
    """Resolves a FEATURE_POWER_THRESHOLDS key (e.g. "99") to the matching
    tag actually present on this node, or None."""
    for t in _threshold_tags(node):
        if _pct_from_tag(t) == threshold_key:
            return t
    return None


_PERM_IMPORTANCE_CACHE: dict[int, pd.DataFrame] = {}

def _quick_permutation_importance(
    x_df: pd.DataFrame, y_enc: np.ndarray, seed: int = SEED,
) -> pd.DataFrame:
    cache_key = (id(x_df), x_df.shape[0], int(y_enc.sum()) if len(y_enc) else 0)
    cached = _PERM_IMPORTANCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if x_df.shape[0] < 10 or x_df.shape[1] == 0 or len(np.unique(y_enc)) < 2:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    clf.fit(x_df.values, y_enc)
    result = permutation_importance(
        clf, x_df.values, y_enc,
        scoring="balanced_accuracy", n_repeats=PERM_IMPORTANCE_N_REPEATS,
        random_state=seed, n_jobs=min(4, (Path("/proc").exists() and __import__("os").cpu_count()) or 4),
    )
    df = pd.DataFrame({
        "feature": x_df.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    _PERM_IMPORTANCE_CACHE[cache_key] = df
    return df


def plot_feature_importance_bar(node: AnalysisNode, imp_df: pd.DataFrame, out_path: Path) -> None:
    if imp_df.empty:
        ANALYSIS_LOGGER.notice(f"[{node.key}] feature importance empty — skipping plot")
        return
    top = imp_df.head(FEATIMP_TOP_N)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(top))))
    ax.barh(top["feature"][::-1], top["importance_mean"][::-1],
            xerr=top["importance_std"][::-1], color="#4C72B0", ecolor="#333333", capsize=2)
    ax.set_xlabel("Permutation importance (Δ balanced accuracy)")
    ax.set_title(f"{node.label} — top {len(top)} features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_feature_importance_suite(nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for node in nodes:
        try:
            x_df = node.x_all(ALL_FEATURES_TAG)
            imp_df = _quick_permutation_importance(x_df, node.y_train_enc)
            out_path = ANALYSIS_FEATIMP_DIR / f"{node.key}_feature_importance.png"
            plot_feature_importance_bar(node, imp_df, out_path)
            imp_df.to_csv(ANALYSIS_FEATIMP_DIR / f"{node.key}_feature_importance.csv", index=False)
            results[node.key] = {"importance": imp_df, "plot_path": str(out_path)}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] feature importance suite failed: {exc}\n{traceback.format_exc()}")
            results[node.key] = {"error": str(exc)}
        finally:
            gc.collect()
    return results


# %%
def plot_cumulative_power_curve(node: AnalysisNode, scores: dict[float, float], out_path: Path) -> None:
    if not scores:
        ANALYSIS_LOGGER.notice(f"[{node.key}] power curve empty — skipping plot")
        return
    pcts = sorted(scores.keys())
    vals = [scores[p] for p in pcts]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pcts, vals, marker="o", color="#C44E52")
    ax.set_xlabel("Feature-set threshold (top-N% by importance)")
    ax.set_ylabel("Balanced accuracy (quick HGBC screen)")
    ax.set_title(f"{node.label} — cumulative feature power")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_power_curve_suite(nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for node in nodes:
        try:
            scores: dict[float, float] = {}
            for threshold_key in FEATURE_POWER_THRESHOLDS:
                tag = _resolve_pct_tag(node, threshold_key)
                if tag is None or tag not in node.x_train_by_tag:
                    continue
                x_df = node.x_train_by_tag[tag]
                if x_df.shape[0] < 10 or len(np.unique(node.y_train_enc)) < 2:
                    continue
                clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200)
                clf.fit(x_df.values, node.y_train_enc)
                pred = clf.predict(x_df.values)
                scores[FEATURE_POWER_THRESHOLDS[threshold_key] * 100] = balanced_accuracy_score(node.y_train_enc, pred)
            out_path = ANALYSIS_POWERCURVE_DIR / f"{node.key}_power_curve.png"
            plot_cumulative_power_curve(node, scores, out_path)
            results[node.key] = {"scores": scores, "plot_path": str(out_path)}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] power curve suite failed: {exc}\n{traceback.format_exc()}")
            results[node.key] = {"error": str(exc)}
        finally:
            gc.collect()
    return results


# %% [markdown]
# ## Section 3: Per-class feature distributions & unsupervised clustering (ported)

# %%
def run_distribution_suite(
    nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES,
    top_n_features: int = 8,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for node in nodes:
        try:
            x_df = node.x_all(ALL_FEATURES_TAG)
            y_str = np.asarray(node.class_names)[node.y_train_enc] if node.class_names else node.y_train_enc.astype(str)
            counts = pd.Series(y_str).value_counts()
            usable_classes = counts[counts >= MIN_REAL_ROWS_FOR_VIOLIN].index.tolist()
            if len(usable_classes) < 2 or x_df.shape[1] == 0:
                ANALYSIS_LOGGER.notice(f"[{node.key}] too few real rows/classes for violin suite — skipped")
                results[node.key] = {"skipped": True}
                continue

            imp_df = _quick_permutation_importance(x_df, node.y_train_enc)
            feats = (imp_df["feature"].head(top_n_features).tolist() if not imp_df.empty
                     else list(x_df.columns[:top_n_features]))

            try:
                palette = superclass_colors(usable_classes)
            except Exception:
                palette = None

            plot_df = x_df.copy()
            plot_df["__class__"] = y_str
            plot_df = plot_df[plot_df["__class__"].isin(usable_classes)]

            n_feats = len(feats)
            ncols = min(3, n_feats) or 1
            nrows = int(np.ceil(n_feats / ncols)) if n_feats else 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
            for i, feat in enumerate(feats):
                ax = axes[i // ncols][i % ncols]
                sns.violinplot(data=plot_df, x="__class__", y=feat, hue="__class__",
                                palette=palette, legend=False, ax=ax, cut=0)
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=45)
            for j in range(n_feats, nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")
            fig.suptitle(f"{node.label} — per-class feature distributions")
            fig.tight_layout()
            out_path = ANALYSIS_DIST_DIR / f"{node.key}_distributions.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results[node.key] = {"features": feats, "plot_path": str(out_path)}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] distribution suite failed: {exc}\n{traceback.format_exc()}")
            results[node.key] = {"error": str(exc)}
        finally:
            gc.collect()
    return results


# %%
def run_clustering_suite(
    nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for node in nodes:
        try:
            x_df = node.x_all(ALL_FEATURES_TAG)
            if x_df.shape[0] < max(CLUSTER_K_RANGE) + 1 or x_df.shape[1] == 0:
                results[node.key] = {"skipped": True}
                continue

            imp_df = _quick_permutation_importance(x_df, node.y_train_enc)
            feats = (imp_df["feature"].head(CLUSTER_TOP_N_FEATURES).tolist() if not imp_df.empty
                     else list(x_df.columns[:CLUSTER_TOP_N_FEATURES]))
            x_sub = x_df[feats].fillna(x_df[feats].median())
            x_scaled = StandardScaler().fit_transform(x_sub.values)

            sil_by_k: dict[int, float] = {}
            best_k, best_sil = None, -1.0
            for k in CLUSTER_K_RANGE:
                if k >= x_scaled.shape[0]:
                    continue
                km = KMeans(n_clusters=k, random_state=SEED, n_init=3)  # sweep phase: cheap n_init
                labels = km.fit_predict(x_scaled)
                if len(np.unique(labels)) < 2:
                    continue
                sil = silhouette_score(
                    x_scaled, labels,
                    sample_size=min(SILHOUETTE_SAMPLE_CAP, x_scaled.shape[0]),
                    random_state=SEED,
                )
                sil_by_k[k] = sil
                if sil > best_sil:
                    best_k, best_sil = k, sil

            best_labels = None
            if best_k is not None:
                # refit the winning k once at full n_init for the labels we actually plot/return
                km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
                best_labels = km_final.fit_predict(x_scaled)

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].plot(list(sil_by_k.keys()), list(sil_by_k.values()), marker="o", color="#55A868")
            axes[0].set_xlabel("k (n_clusters)")
            axes[0].set_ylabel("Silhouette score")
            axes[0].set_title(f"{node.label} — silhouette vs k")

            if best_labels is not None and x_scaled.shape[1] >= 2:
                sc = axes[1].scatter(x_scaled[:, 0], x_scaled[:, 1], c=best_labels, cmap="tab10", s=18, alpha=0.7)
                axes[1].set_xlabel(feats[0])
                axes[1].set_ylabel(feats[1])
                axes[1].set_title(f"best k={best_k} (silhouette={best_sil:.3f})")
                fig.colorbar(sc, ax=axes[1], label="cluster")
            else:
                axes[1].axis("off")

            fig.tight_layout()
            out_path = ANALYSIS_CLUSTER_DIR / f"{node.key}_clustering.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results[node.key] = {
                "features_used": feats, "silhouette_by_k": sil_by_k,
                "best_k": best_k, "best_silhouette": best_sil, "plot_path": str(out_path),
            }
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] clustering suite failed: {exc}\n{traceback.format_exc()}")
            results[node.key] = {"error": str(exc)}
        finally:
            gc.collect()
    return results


# %% [markdown]
# ## Section 3b Per-feature KDE distribution gallery
#
# One KDE-by-class plot PER FEATURE in each node's own resolved 80%-power
# feature set (every column, no top-N cap), saved as an individual file per
# feature per node — the "canonical feature-set gallery" view. Distinct
# from Section 3's `run_distribution_suite` above (one combined top-8-feature
# violin panel per node, off ALL_FEATURES_TAG).

# %%
def _class_labels_for_node(node: AnalysisNode) -> np.ndarray:
    if node.label_encoder is not None and hasattr(node.label_encoder, "inverse_transform"):
        try:
            return np.asarray(node.label_encoder.inverse_transform(node.y_train_enc.astype(int)))
        except Exception:
            pass
    name_by_idx = {i: name for i, name in enumerate(node.class_names)}
    return np.asarray([name_by_idx.get(int(v), str(v)) for v in node.y_train_enc])


def plot_feature_distribution_kde(
    x_df: pd.DataFrame, y_labels: Sequence[str], feature: str, out_path: Path, title: str,
    palette: Optional[dict] = None,
) -> None:
    if feature not in x_df.columns:
        raise KeyError(f"feature '{feature}' not in frame")
    plot_df = pd.DataFrame({feature: x_df[feature].values, "class": list(y_labels)}).dropna()
    if plot_df["class"].nunique() < 1 or plot_df.empty:
        raise ValueError(f"no usable data for feature '{feature}'")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    classes_present = sorted(plot_df["class"].unique().tolist())
    color_map = palette or dict(zip(classes_present, sns.color_palette("tab10", n_colors=len(classes_present))))
    for cls in classes_present:
        vals = plot_df.loc[plot_df["class"] == cls, feature]
        if vals.nunique() < 2 or len(vals) < 3:
            continue
        sns.kdeplot(vals, ax=ax, label=str(cls), color=color_map.get(cls), fill=True, alpha=0.18, linewidth=2.0)
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=13)
    ax.legend(title="Class", fontsize=9, title_fontsize=10, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_feature_distribution_gallery(
    nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES, pct_tag: str = "80",
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for node in nodes:
        tag = _resolve_pct_tag(node, pct_tag)
        if tag is None or tag not in node.x_train_by_tag:
            ANALYSIS_LOGGER.notice(f"[{node.key}] no resolvable {pct_tag}% tag among {node.tags} — skipped")
            results[node.key] = {"skipped": True}
            continue
        x_df = node.x_train_by_tag[tag]
        if x_df is None or x_df.empty:
            results[node.key] = {"skipped": True}
            continue
        y_labels = _class_labels_for_node(node)
        try:
            palette = superclass_colors(sorted(pd.unique(y_labels).tolist()))
        except Exception:
            palette = None
        node_dir = ANALYSIS_DIST_DIR / "feature_gallery" / node.hierarchy / node.key
        feature_status = {}
        for feature in x_df.columns:
            try:
                plot_feature_distribution_kde(
                    x_df, y_labels, feature, out_path=node_dir / f"{_safe_filename(feature)}.png",
                    title=f"{node.hierarchy}/{node.key} — {feature} (tag={tag})", palette=palette,
                )
                feature_status[feature] = "ok"
            except Exception as exc:
                ANALYSIS_LOGGER.warning(f"[{node.key}] feature-distribution ({feature}) failed: {exc}")
                feature_status[feature] = "failed"
        results[node.key] = {"tag": tag, "features": feature_status}
        gc.collect()
    return results


# %% [markdown]
# ## Section 3c

# %%

# %% [markdown]
# ## Section 3c Pairwise clustering sweep
#
# On ONE gallery node (merged router): every
# pairwise combination of the top-N important features, each with its own
# silhouette-optimal k and flux-aware log axes. Distinct from Section 3's
# `run_clustering_suite` above (every node, but only ever the first two
# features, no log-axis handling).

# %%
def _looks_like_flux(feature_name: str) -> bool:
    name = feature_name.lower()
    return ("flux" in name) or ("energy_flux" in name) or name.startswith("nuf")


def plot_cluster_scatter_pair(x_df: pd.DataFrame, feat_x: str, feat_y: str, out_path: Path, title: str) -> int:
    sub = x_df[[feat_x, feat_y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 10:
        raise ValueError(f"insufficient points ({len(sub)}) for clustering on {feat_x}/{feat_y}")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(sub.values)
    best_k, best_score = CLUSTER_K_RANGE[0], -1.0
    n = x_scaled.shape[0]
    for k in CLUSTER_K_RANGE:
        if k >= n:
            break
        km = KMeans(n_clusters=k, random_state=SEED, n_init=3)  # sweep phase: cheap n_init
        labels = km.fit_predict(x_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(
            x_scaled, labels,
            sample_size=min(SILHOUETTE_SAMPLE_CAP, n),
            random_state=SEED,
        )
        if score > best_score:
            best_k, best_score = k, score
    k_opt = best_k
    km = KMeans(n_clusters=k_opt, random_state=SEED, n_init=10)  # final fit: full n_init for the plotted labels
    cluster_labels = km.fit_predict(x_scaled)
    centroids_original = scaler.inverse_transform(km.cluster_centers_)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    palette = sns.color_palette("tab10", n_colors=k_opt)
    for c in range(k_opt):
        mask = cluster_labels == c
        ax.scatter(sub.values[mask, 0], sub.values[mask, 1], s=14, alpha=0.55, color=palette[c], label=f"cluster {c}", linewidths=0)
    ax.scatter(centroids_original[:, 0], centroids_original[:, 1], marker="X", s=180, color="black",
               edgecolor="white", linewidth=1.0, label="centroids", zorder=5)
    if _looks_like_flux(feat_x):
        ax.set_xscale("log")
    if _looks_like_flux(feat_y):
        ax.set_yscale("log")
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_title(f"{title}  (k_opt = {k_opt})", fontsize=12)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return k_opt


def run_clustering_pairwise_suite(
    node_key: str = "merged_router", top_n: int = CLUSTER_TOP_N_FEATURES,
) -> dict[str, Any]:
    node = next((n for n in ALL_ANALYSIS_NODES if n.key == node_key), None)
    if node is None:
        ANALYSIS_LOGGER.warning(f"run_clustering_pairwise_suite: node {node_key!r} not found — skipped")
        return {"skipped": True}

    tag = _resolve_pct_tag(node, "80")
    if tag is None or tag not in node.x_train_by_tag:
        ANALYSIS_LOGGER.warning(f"[cluster-pairwise] {node.key} has no 80% tag — skipped")
        return {"skipped": True}
    x_df = node.x_train_by_tag[tag]
    if x_df is None or x_df.empty:
        return {"skipped": True}

    try:
        imp_df = _quick_permutation_importance(x_df, node.y_train_enc)
        top_features = imp_df["feature"].head(top_n).tolist()
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[cluster-pairwise] importance ranking failed for {node.key}, "
                                 f"falling back to first {top_n} columns: {exc}")
        top_features = list(x_df.columns[:top_n])

    node_dir = ANALYSIS_CLUSTER_DIR / "pairwise" / node.hierarchy / node.key
    pairs: dict[str, Any] = {}
    for feat_x, feat_y in itertools.combinations(top_features, 2):
        pair_key = f"{feat_x}__vs__{feat_y}"
        try:
            k_opt = plot_cluster_scatter_pair(
                x_df, feat_x, feat_y,
                out_path=node_dir / f"{_safe_filename(feat_x)}__vs__{_safe_filename(feat_y)}.png",
                title=f"{node.hierarchy}/{node.key} — {feat_x} vs {feat_y}",
            )
            pairs[pair_key] = {"status": "ok", "k_opt": k_opt}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[cluster-pairwise] {feat_x} vs {feat_y} failed: {exc}")
            pairs[pair_key] = {"status": "failed", "k_opt": None}
        finally:
            gc.collect()
    return {"tag": tag, "pairs": pairs}


# %% [markdown]
# ## Section 4: Legacy per-node metric comparison (ported)

# %%
def run_metric_comparison_suite(
    nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES,
    metrics: Sequence[str] = QUALITY_METRICS,
) -> dict[str, Any]:
    """Bar-chart comparison of every model on a node's persisted leaderboard
    (from `_attach_leaderboard`, i.e. whatever the corresponding validation
    stage already wrote to disk) across QUALITY_METRICS. Diagnostic only —
    genuine held-out cross-family comparison lives in Section 5 below."""
    results: dict[str, Any] = {}
    for node in nodes:
        try:
            lb = node.leaderboard
            if lb is None or lb.empty:
                results[node.key] = {"skipped": True}
                continue
            present_metrics = [m for m in metrics if m in lb.columns]
            if not present_metrics or "model_name" not in lb.columns:
                results[node.key] = {"skipped": True}
                continue
            plot_df = lb.sort_values(present_metrics[0], ascending=False).head(MAX_MODELS_PER_PANEL)
            fig, ax = plt.subplots(figsize=(11, 5.5))
            x = np.arange(len(plot_df))
            width = 0.8 / len(present_metrics)
            for i, m in enumerate(present_metrics):
                ax.bar(x + i * width, plot_df[m].values, width=width, label=m)
            ax.set_xticks(x + width * (len(present_metrics) - 1) / 2)
            ax.set_xticklabels(plot_df["model_name"], rotation=45, ha="right")
            ax.set_ylabel("Score")
            ax.set_title(f"{node.label} — leaderboard metric comparison")
            ax.legend(fontsize=9)
            fig.tight_layout()
            out_path = ANALYSIS_METRICCMP_DIR / f"{node.key}_metric_comparison.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results[node.key] = {"plot_path": str(out_path), "n_models": len(plot_df)}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] metric comparison suite failed: {exc}\n{traceback.format_exc()}")
            results[node.key] = {"error": str(exc)}
        finally:
            gc.collect()
    return results


# %% [markdown]
# ## Section 4b

# %%

# %% [markdown]
# ## Section 4b Per-node leaderboard violin comparisons
#
# For every node, four leaderboard-driven comparisons (by_weighting,
# by_model_all, by_model_mlp_only, by_feature_selection), each paginated
# across VAL_METRIC_COLS. A category with >= MIN_REAL_ROWS_FOR_VIOLIN real
# rows (pooled across every split via `_attach_leaderboard`) gets a genuine
# violin + jittered strip; a sparser category gets a mean point + error bar.
# Never mixes rows across nodes — `node.leaderboard` is already filtered to
# this node's own label_set by `_attach_leaderboard`.

# %%
def _normalize_feature_tag(tag: Any) -> str:
    """Collapses a raw tag string down to a coarse feature-selection family for grouping."""
    if tag is None or (isinstance(tag, float) and np.isnan(tag)):
        return "unknown"
    tokens = set(str(tag).upper().split("_"))
    if "PHYSICS" in tokens and "CORE" in tokens:
        base = "PHYSICS_CORE"
    elif "80" in tokens:
        base = "80"
    elif "90" in tokens:
        base = "90"
    elif "99" in tokens:
        base = "99"
    elif tokens == set(ALL_FEATURES_TAG.upper().split("_")):
        base = "ALL_FEATURES"
    else:
        base = "other"
    return f"{base}_DECORR" if "DECORR" in tokens else base


def plot_metric_comparison(
    df: pd.DataFrame, group_col: str, out_dir: Path, prefix: str,
    max_per_panel: int = MAX_MODELS_PER_PANEL, metrics: Sequence[str] = VAL_METRIC_COLS,
    min_n_for_violin: int = MIN_REAL_ROWS_FOR_VIOLIN,
) -> list:
    saved_paths = []
    if df.empty or group_col not in df.columns:
        ANALYSIS_LOGGER.warning(f"[metric-cmp] '{prefix}' skipped: empty frame or missing column '{group_col}'")
        return saved_paths

    categories = sorted(df[group_col].dropna().unique().tolist())
    if not categories:
        return saved_paths
    pages = [categories[i:i + max_per_panel] for i in range(0, len(categories), max_per_panel)]

    for page_idx, page_categories in enumerate(pages):
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            continue
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(max(9, 0.9 * len(page_categories) + 4), 3.8 * n_metrics), squeeze=False)
        axes = axes[:, 0]

        for ax, metric in zip(axes, available_metrics):
            std_col = f"{metric}_std"
            violin_datasets, violin_positions = [], []
            for i, cat in enumerate(page_categories):
                vals = df.loc[df[group_col] == cat, metric].dropna().to_numpy()
                if len(vals) >= min_n_for_violin:
                    violin_datasets.append(vals)
                    violin_positions.append(i)

            if violin_datasets:
                vp = ax.violinplot(violin_datasets, positions=violin_positions, widths=0.7, showmeans=False, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor("#4C72B0")
                    body.set_alpha(0.45)
                rng = np.random.default_rng(SEED)
                for pos, vals in zip(violin_positions, violin_datasets):
                    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                    ax.scatter(np.full(len(vals), pos) + jitter, vals, s=10, color="black", alpha=0.45, zorder=3)

            for i, cat in enumerate(page_categories):
                vals = df.loc[df[group_col] == cat, metric].dropna()
                if len(vals) == 0 or len(vals) >= min_n_for_violin:
                    continue
                if len(vals) == 1:
                    cat_rows = df.loc[(df[group_col] == cat) & df[metric].notna()]
                    row = cat_rows.iloc[0]
                    err = float(row[std_col]) if std_col in df.columns and pd.notna(row.get(std_col)) else 0.0
                    mean = float(vals.iloc[0])
                else:
                    mean, err = float(vals.mean()), float(vals.std(ddof=1))
                ax.errorbar([i], [mean], yerr=[[err], [err]], fmt="o", color="#C44E52", capsize=4, markersize=6, zorder=4)

            n_by_cat = [int((df[group_col] == cat).sum()) for cat in page_categories]
            positions = np.arange(len(page_categories))
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{cat}\n(n={n})" for cat, n in zip(page_categories, n_by_cat)], rotation=35, ha="right", fontsize=8.5)
            ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=11)
            ax.set_xlabel("")
            ax.set_xlim(-0.6, len(page_categories) - 0.4)

        fig.suptitle(
            f"{prefix} — page {page_idx + 1}/{len(pages)} ({len(page_categories)} of {len(categories)} categories)\n"
            f"blue violin = \u2265{min_n_for_violin} real rows across splits; red point = fewer (errorbar = within-row CV-fold std)",
            fontsize=10.5,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out_path = out_dir / f"{_safe_filename(prefix)}_page{page_idx + 1:02d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


def run_metric_suites_for_node(node: AnalysisNode) -> dict[str, list]:
    """Runs all 4 comparisons for ONE node, off its already-pooled
    (`_attach_leaderboard`), already-node-filtered leaderboard."""
    out_dir = ANALYSIS_VIOLIN_DIR / node.hierarchy / node.key
    pooled = node.leaderboard
    if pooled is None or pooled.empty:
        ANALYSIS_LOGGER.notice(f"[{node.key}] no leaderboard rows — violin suite skipped")
        return {}
    tag_col = "tag" if "tag" in pooled.columns else ("feature_set" if "feature_set" in pooled.columns else None)
    if tag_col is not None:
        pooled = pooled.assign(feature_family=pooled[tag_col].map(_normalize_feature_tag))

    comparisons = [
        ("by_weighting", "weighting", pooled),
        ("by_model_all", "model_name", pooled),
        ("by_model_mlp_only", "model_name",
         pooled[pooled["model_name"].isin(MLP_MODEL_NAMES)] if "model_name" in pooled.columns else pooled.iloc[0:0]),
        ("by_feature_selection", "feature_family", pooled if "feature_family" in pooled.columns else pooled.iloc[0:0]),
    ]
    results = {}
    for name, group_col, sub_df in comparisons:
        try:
            results[name] = plot_metric_comparison(sub_df, group_col, out_dir, f"{node.hierarchy}_{node.key}_{name}")
        except Exception as exc:
            ANALYSIS_LOGGER.error(f"[metric-cmp] {node.hierarchy}/{node.key} comparison '{name}' failed: "
                                   f"{exc}\n{traceback.format_exc()}")
            results[name] = []
    return results


def run_leaderboard_violin_suite(nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for node in nodes:
        try:
            results[node.key] = run_metric_suites_for_node(node)
        except Exception as exc:
            ANALYSIS_LOGGER.error(f"[metric-cmp] {node.hierarchy}/{node.key} failed entirely: "
                                   f"{exc}\n{traceback.format_exc()}")
            results[node.key] = {"error": str(exc)}
        finally:
            gc.collect()
    return results


# %% [markdown]
# ## Section 4c Panoramic leaderboard violin comparisons
#
# `run_metric_suites_for_node` (above) is intentionally narrow — one
# node's own pooled-across-splits leaderboard, never mixed with another
# node's rows, so "by_weighting" for e.g. `branch_bll_L3` never drowns in
# `fine_B`'s row count. That narrowness is also why some categories (most
# visibly `weighting="none"`, which rarely wins Stage 2 for any given
# model/branch) end up with too few real rows to draw a genuine violin —
# per node there just aren't many chances for a sparse category to clear
# `MIN_REAL_ROWS_FOR_VIOLIN`.
#
# `run_panoramic_violin_suite` runs the exact same four comparisons
# (`plot_metric_comparison` is untouched) on ONE pooled frame built from
# EVERY node's leaderboard across EVERY hierarchy (fine + merged router/
# branches + 3-level gate/subgate/router/branches) — many more rows per
# category, at the honest cost of mixing scores earned on different
# label_sets/taxonomies/difficulty. This is a "how does this weighting
# mode / model family / feature selection tend to do across the whole
# pipeline" view, not a per-node leaderboard, so it complements
# `run_leaderboard_violin_suite` rather than replacing it.
#
# Also depends on `fermi_common.py`'s `_persist_weighting_sweep`
# (persists every Stage-2 weighting candidate a model was scored on, not
# just the winner) actually having run for a node's split — until that
# split is regenerated with the patched kernel, "none" still won't show
# up for that node/split's rows, panoramic or not; this suite just gives
# whatever real rows DO exist (across every node, old- or new-format) the
# best shot at clearing MIN_REAL_ROWS_FOR_VIOLIN.

# %%
def run_panoramic_violin_suite(nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES) -> dict[str, list]:
    """Pools `node.leaderboard` across every node passed in (no per-node
    filtering beyond what `_attach_leaderboard` already did) and runs the
    same by_weighting/by_model_all/by_model_mlp_only/by_feature_selection
    comparisons `run_metric_suites_for_node` runs per-node, off the pooled
    frame. Tags each pooled row with node_key/node_hierarchy first (not
    used for grouping here, but kept so a saved leaderboard CSV/pickle of
    the pooled frame stays traceable back to its source node)."""
    frames = []
    for node in nodes:
        lb = node.leaderboard
        if lb is None or lb.empty:
            continue
        tagged = lb.copy()
        tagged["node_key"] = node.key
        tagged["node_hierarchy"] = node.hierarchy
        frames.append(tagged)

    if not frames:
        ANALYSIS_LOGGER.notice("[panoramic] no leaderboard rows on any node — suite skipped")
        return {}

    pooled = pd.concat(frames, ignore_index=True)
    tag_col = "tag" if "tag" in pooled.columns else ("feature_set" if "feature_set" in pooled.columns else None)
    if tag_col is not None:
        pooled = pooled.assign(feature_family=pooled[tag_col].map(_normalize_feature_tag))

    out_dir = ANALYSIS_VIOLIN_DIR / "_panoramic"
    comparisons = [
        ("by_weighting", "weighting", pooled),
        ("by_model_all", "model_name", pooled),
        ("by_model_mlp_only", "model_name",
         pooled[pooled["model_name"].isin(MLP_MODEL_NAMES)] if "model_name" in pooled.columns else pooled.iloc[0:0]),
        ("by_feature_selection", "feature_family", pooled if "feature_family" in pooled.columns else pooled.iloc[0:0]),
    ]
    results = {}
    for name, group_col, sub_df in comparisons:
        try:
            results[name] = plot_metric_comparison(sub_df, group_col, out_dir, f"panoramic_{name}")
        except Exception as exc:
            ANALYSIS_LOGGER.error(f"[panoramic] comparison '{name}' failed: {exc}\n{traceback.format_exc()}")
            results[name] = []

    ANALYSIS_LOGGER.notice(f"[panoramic] pooled {len(pooled)} leaderboard row(s) "
                            f"across {len(frames)}/{len(nodes)} node(s) with data")
    return results


# %% [markdown]
# ## Section 5: Tree gallery
#
# `_fit_model_fresh` and `_build_estimator_for_leaderboard_row` take
# `node.model_specs` explicitly: a bare `get_builder(model_name)` would use
# whichever `_get_builder` closure was bound last in the shared kernel
# (3-level's, always).

# %%
NON_TREE_LINEAR_MODELS = ["LDA", "LogReg"]
NON_TREE_BOUNDARY_MODELS = ["KNN", "SVM"]
_NATIVE_NAN_MODELS = {"CatBoost", "HistGBC"}


def _median_impute(x_df: pd.DataFrame) -> pd.DataFrame:
    return x_df.fillna(x_df.median(numeric_only=True))


def _fit_model_fresh(node: AnalysisNode, model_name: str, x_df: pd.DataFrame, y_enc: np.ndarray):
    if model_name not in _NATIVE_NAN_MODELS:
        x_df = _median_impute(x_df)
    builder = get_builder(model_name, model_specs=node.model_specs)
    estimator = builder() if callable(builder) else builder
    estimator.fit(x_df.values, y_enc)
    return estimator, x_df


def _build_estimator_for_leaderboard_row(node: AnalysisNode, row: pd.Series):
    model_name = row.get("model_name")
    if model_name == "HistGBC":
        try:
            return StratifiedEarlyStoppingHGBC(random_state=SEED)
        except Exception:
            pass
    builder = get_builder(model_name, model_specs=node.model_specs)
    return builder() if callable(builder) else builder


def _best_row_for_node(node: AnalysisNode, model_pool: Sequence[str] = tuple(TREE_MODEL_NAMES)) -> Optional[pd.Series]:
    lb = node.leaderboard
    if lb is None or lb.empty or "model_name" not in lb.columns:
        return None
    sub = lb[lb["model_name"].isin(model_pool)]
    if sub.empty or SELECTION_METRIC not in sub.columns:
        return None
    return sub.sort_values(SELECTION_METRIC, ascending=False).iloc[0]


# %%
def _unwrap_tree(estimator):
    """Peels wrapper estimators (calibration/pipeline-style) down to the
    underlying tree-bearing object, if any wrapping is present."""
    for attr in ("estimator_", "base_estimator_", "estimator", "base_estimator"):
        inner = getattr(estimator, attr, None)
        if inner is not None and inner is not estimator:
            return _unwrap_tree(inner)
    return estimator


def _extract_one_tree(estimator, model_name: str):
    est = _unwrap_tree(estimator)
    if model_name == "DecisionTree" and hasattr(est, "tree_"):
        return {"kind": "sklearn_tree", "obj": est}
    if model_name == "RandomForest" and hasattr(est, "estimators_") and est.estimators_:
        return {"kind": "sklearn_tree", "obj": est.estimators_[0]}
    if model_name == "GBC" and hasattr(est, "estimators_") and est.estimators_.size:
        return {"kind": "sklearn_tree", "obj": est.estimators_[0, 0]}
    if model_name == "CatBoost":
        return {"kind": "catboost_native", "obj": est}
    if model_name == "HistGBC" and hasattr(est, "_predictors"):
        return {"kind": "histgbc_raw", "obj": est}
    return {"kind": "unavailable", "obj": None}


def _plot_histgbc_tree_manual(est, feature_names: Sequence[str], ax) -> bool:
    """HistGradientBoostingClassifier has no public tree export; this reads
    the first predictor's raw `.nodes` structured array directly (fields:
    value, count, feature_idx, threshold, left, right, is_leaf) and draws a
    simple depth-indented text tree onto `ax` (text-only render; a full
    graphical layout is out of scope for a diagnostic gallery panel)."""
    try:
        predictors = est._predictors
        tree0 = predictors[0][0] if isinstance(predictors[0], (list, tuple)) else predictors[0]
        nodes = tree0.nodes
    except Exception:
        return False

    lines = []

    def _walk(idx: int, depth: int):
        if depth > 6 or idx >= len(nodes):
            lines.append("  " * depth + "...")
            return
        node = nodes[idx]
        is_leaf = bool(node["is_leaf"])
        if is_leaf:
            lines.append("  " * depth + f"leaf: value={float(node['value']):.4f} n={int(node['count'])}")
        else:
            fname = feature_names[int(node["feature_idx"])] if int(node["feature_idx"]) < len(feature_names) else f"f{int(node['feature_idx'])}"
            lines.append("  " * depth + f"[{fname} <= {float(node['threshold']):.4g}]  n={int(node['count'])}")
            _walk(int(node["left"]), depth + 1)
            _walk(int(node["right"]), depth + 1)

    _walk(0, 0)
    text = "\n".join(lines[:40])
    ax.axis("off")
    ax.text(0.01, 0.99, text, family="monospace", fontsize=7, va="top", ha="left", transform=ax.transAxes)
    return True


def plot_one_tree(node: AnalysisNode, model_name: str, estimator, x_df: pd.DataFrame, out_path: Path,
                   extra_title: str = "") -> bool:
    extracted = _extract_one_tree(estimator, model_name)
    kind, obj = extracted["kind"], extracted["obj"]
    if kind == "unavailable":
        return False
    fig, ax = plt.subplots(figsize=(14, 9))
    try:
        if kind == "sklearn_tree":
            from sklearn.tree import plot_tree
            if _DTREEVIZ_MOD is not None:
                try:
                    plot_tree_splits_dtreeviz(obj, x_df, node.y_train_enc, node.class_names, out_path)
                    plt.close(fig)
                    return True
                except Exception:
                    pass
            plot_tree(obj, feature_names=list(x_df.columns), class_names=[str(c) for c in node.class_names],
                      filled=True, max_depth=3, fontsize=7, ax=ax)
        elif kind == "catboost_native":
            ax.axis("off")
            ax.text(0.5, 0.5, "CatBoost native tree export not available in this environment\n"
                              "(export via estimator.plot_tree(tree_idx=0) offline if needed).",
                    ha="center", va="center", wrap=True)
        elif kind == "histgbc_raw":
            ok = _plot_histgbc_tree_manual(obj, list(x_df.columns), ax)
            if not ok:
                ax.axis("off")
                ax.text(0.5, 0.5, "HistGBC raw tree structure unavailable.", ha="center", va="center")
        ax.set_title(f"{node.label} — {model_name} (one representative tree){extra_title}")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        return True
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[{node.key}] plot_one_tree({model_name}) failed: {exc}")
        return False
    finally:
        plt.close(fig)


def plot_linear_coefficients(node: AnalysisNode, model_name: str, estimator, x_df: pd.DataFrame, out_path: Path) -> None:
    est = _unwrap_tree(estimator)
    coef = getattr(est, "coef_", None)
    if coef is None:
        return
    coef = np.atleast_2d(coef)
    mean_abs = np.abs(coef).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:FEATIMP_TOP_N]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(order))))
    ax.barh(np.array(x_df.columns)[order][::-1], mean_abs[order][::-1], color="#8172B2")
    ax.set_xlabel("|coefficient| (mean across classes if multiclass)")
    ax.set_title(f"{node.label} — {model_name} linear coefficients")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decision_boundary_2d(node: AnalysisNode, model_name: str, x_df: pd.DataFrame, y_enc: np.ndarray, out_path: Path) -> None:
    imp_df = _quick_permutation_importance(x_df, y_enc)
    top2 = (imp_df["feature"].head(2).tolist() if not imp_df.empty else list(x_df.columns[:2]))
    if len(top2) < 2:
        return
    x2 = _median_impute(x_df[top2])
    builder = get_builder(model_name, model_specs=node.model_specs)
    est2 = builder() if callable(builder) else builder
    est2.fit(x2.values, y_enc)

    x_min, x_max = x2.iloc[:, 0].min() - 1, x2.iloc[:, 0].max() + 1
    y_min, y_max = x2.iloc[:, 1].min() - 1, x2.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    try:
        zz = est2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[{node.key}] decision boundary predict failed for {model_name}: {exc}")
        return

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="tab10")
    ax.scatter(x2.iloc[:, 0], x2.iloc[:, 1], c=y_enc, cmap="tab10", s=14, edgecolor="k", linewidth=0.2)
    ax.set_xlabel(top2[0])
    ax.set_ylabel(top2[1])
    ax.set_title(f"{node.label} — {model_name} decision boundary (top-2 features)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_best_tree_per_node_suite(nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for node in nodes:
        try:
            row = _best_row_for_node(node)
            if row is None:
                results[node.key] = {"skipped": True}
                continue
            model_name = row["model_name"]
            x_df = node.x_all(ALL_FEATURES_TAG)
            estimator = _build_estimator_for_leaderboard_row(node, row)
            x_fit = x_df if model_name in _NATIVE_NAN_MODELS else _median_impute(x_df)
            estimator.fit(x_fit.values, node.y_train_enc)
            out_path = ANALYSIS_TREEGALLERY_DIR / f"{node.key}_best_tree_{model_name}.png"
            ok = plot_one_tree(node, model_name, estimator, x_fit, out_path)
            results[node.key] = {"model_name": model_name, "plot_path": str(out_path) if ok else None}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] best-tree suite failed: {exc}\n{traceback.format_exc()}")
            results[node.key] = {"error": str(exc)}
        finally:
            gc.collect()
    return results


def run_tree_family_gallery(node_key: str = "merged_router") -> dict[str, Any]:
    """Full tree/linear/boundary-model family gallery on a single reference
    node (the merged superclass router by default), addressed by its stable
    node key rather than a bare global."""
    node = next((n for n in ALL_ANALYSIS_NODES if n.key == node_key), None)
    if node is None:
        ANALYSIS_LOGGER.warning(f"run_tree_family_gallery: node {node_key!r} not found — skipped")
        return {"skipped": True}

    x_df = node.x_all(ALL_FEATURES_TAG)
    out: dict[str, Any] = {"pruned_tree": None, "tree_models": {}, "linear_models": {},
                            "boundary_models": {}, "mlp_models": {}}

    # cost-complexity pruned tree — a dedicated item,
    # distinct from the regular DecisionTree family entry below.
    try:
        x_fit = _median_impute(x_df)
        pruned = DTreeCVPrunedClassifier(random_state=SEED)
        pruned.fit(x_fit.values, node.y_train_enc)
        ccp_alpha = float(getattr(pruned, "best_alpha_", float("nan")))
        n_leaves = int(pruned.model_.get_n_leaves()) if hasattr(pruned, "model_") else None
        depth = int(pruned.model_.get_depth()) if hasattr(pruned, "model_") else None
        extra_title = f" [PRUNED: ccp_alpha={ccp_alpha:.5f}, {n_leaves} leaves, depth={depth}]"
        out_path = ANALYSIS_TREEGALLERY_DIR / f"{node.key}_pruned_tree.png"
        ok = plot_one_tree(node, "DecisionTree", pruned, x_fit, out_path, extra_title=extra_title)
        out["pruned_tree"] = {
            "plot_path": str(out_path) if ok else None,
            "ccp_alpha": ccp_alpha, "n_leaves": n_leaves, "depth": depth,
        }
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"[{node.key}] pruned-tree gallery item failed: {exc}")
        out["pruned_tree"] = None
    finally:
        gc.collect()

    for model_name in sorted(TREE_MODEL_NAMES):
        try:
            estimator, x_fit = _fit_model_fresh(node, model_name, x_df, node.y_train_enc)
            out_path = ANALYSIS_TREEGALLERY_DIR / f"{node.key}_family_{model_name}.png"
            ok = plot_one_tree(node, model_name, estimator, x_fit, out_path)
            out["tree_models"][model_name] = str(out_path) if ok else None
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] family gallery ({model_name}) failed: {exc}")
            out["tree_models"][model_name] = None
        finally:
            gc.collect()

    for model_name in NON_TREE_LINEAR_MODELS:
        try:
            estimator, x_fit = _fit_model_fresh(node, model_name, x_df, node.y_train_enc)
            out_path = ANALYSIS_TREEGALLERY_DIR / f"{node.key}_linear_{model_name}.png"
            plot_linear_coefficients(node, model_name, estimator, x_fit, out_path)
            out["linear_models"][model_name] = str(out_path)
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] linear coef gallery ({model_name}) failed: {exc}")
            out["linear_models"][model_name] = None
        finally:
            gc.collect()

    for model_name in NON_TREE_BOUNDARY_MODELS:
        try:
            out_path = ANALYSIS_TREEGALLERY_DIR / f"{node.key}_boundary_{model_name}.png"
            plot_decision_boundary_2d(node, model_name, x_df, node.y_train_enc, out_path)
            out["boundary_models"][model_name] = str(out_path)
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] decision boundary gallery ({model_name}) failed: {exc}")
            out["boundary_models"][model_name] = None
        finally:
            gc.collect()

    # MLPs' substitute visualization — a fresh permutation-
    # importance panel, since MLPs have no tree/coefficients/natural boundary.
    for model_name in MLP_MODEL_NAMES:
        try:
            estimator, x_fit = _fit_model_fresh(node, model_name, x_df, node.y_train_enc)
            perm = permutation_importance(estimator, x_fit.values, node.y_train_enc, scoring="balanced_accuracy",
                                          n_repeats=PERM_IMPORTANCE_N_REPEATS, random_state=SEED, n_jobs=-1)
            imp_df = pd.DataFrame({
                "feature": x_fit.columns, "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }).sort_values("importance_mean", ascending=False, ignore_index=True)
            out_path = ANALYSIS_TREEGALLERY_DIR / f"{node.key}_mlp_importance_{model_name}.png"
            plot_feature_importance_bar(node, imp_df, out_path)
            out["mlp_models"][model_name] = str(out_path)
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node.key}] MLP importance gallery ({model_name}) failed: {exc}")
            out["mlp_models"][model_name] = None
        finally:
            gc.collect()

    return out


# %% [markdown]
# ## Section 6 Stage 5/6 cross-family comparison
#
# Reads `STAGE5_RESULTS` (per-family held-out val metrics + CI) and
# `STAGE6_RESULTS` (Nadeau-Bengio / McNemar / bootstrap / Cohen's d /
# Holm-Bonferroni pairwise significance) from `fermi_validation.py`'s
# module globals. Both dicts' exact nesting can vary slightly run-to-run
# (e.g. whether CI is a tuple or a dict), so extraction here is defensive
# rather than assuming one fixed shape.

# %%
def _fam_point_and_ci(fam: str, metric: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (point, lo, hi) for a family/metric from STAGE5_RESULTS,
    trying the CI dict first and falling back to per-split mean +/- std."""
    entry = STAGE5_RESULTS.get(fam, {}) if "STAGE5_RESULTS" in globals() else {}
    point = (entry.get("val_metrics") or {}).get(metric)
    ci_block = entry.get("ci") or {}
    ci_val = ci_block.get(metric)
    lo = hi = None
    if isinstance(ci_val, (tuple, list)) and len(ci_val) == 2:
        lo, hi = ci_val
    elif isinstance(ci_val, dict):
        lo = ci_val.get("lo", ci_val.get("lower", ci_val.get("ci_lo")))
        hi = ci_val.get("hi", ci_val.get("upper", ci_val.get("ci_hi")))
    if lo is None or hi is None:
        per_split = (entry.get("per_split_val_metrics") or {}).get(metric) or (entry.get("per_split_scores") or {}).get(metric)
        if per_split:
            arr = np.asarray(list(per_split.values()) if isinstance(per_split, dict) else per_split, dtype=float)
            if arr.size:
                point = point if point is not None else float(np.mean(arr))
                lo, hi = float(np.mean(arr) - np.std(arr)), float(np.mean(arr) + np.std(arr))
    return point, lo, hi


def plot_family_forest(metric: str, out_path: Path) -> Optional[pd.DataFrame]:
    rows = []
    for fam in FAMILIES:
        point, lo, hi = _fam_point_and_ci(fam, metric)
        if point is None:
            continue
        rows.append({"family": FAMILY_LABELS.get(fam, fam), "point": point, "lo": lo, "hi": hi})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 1.2 * len(df) + 1.5))
    y = np.arange(len(df))
    xerr = None
    if df["lo"].notna().all() and df["hi"].notna().all():
        xerr = np.vstack([df["point"] - df["lo"], df["hi"] - df["point"]])
    ax.errorbar(df["point"], y, xerr=xerr, fmt="o", color="#4C72B0", capsize=4, markersize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["family"])
    ax.set_xlabel(metric)
    ax.set_title(f"Held-out {metric} by hierarchy family (Stage 5)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return df


def _pairwise_pvalue_matrix(test_name: str) -> Optional[pd.DataFrame]:
    """Normalizes STAGE6_RESULTS[test_name] (keyed however the validation
    module keys family pairs — 'fine_vs_merged', ('fine','merged'), or a
    nested {fam: {fam: stat}} dict) into a symmetric family x family
    p-value DataFrame."""
    stage6_results = globals().get("STAGE6_RESULTS")
    if not stage6_results:
        return None
    block = stage6_results.get(test_name)
    if not block:
        return None
    mat = pd.DataFrame(index=FAMILIES, columns=FAMILIES, dtype=float)
    def _pval_from(v):
        if isinstance(v, dict):
            return v.get("p_value", v.get("p", v.get("pvalue")))
        if isinstance(v, (tuple, list)) and len(v) >= 2:
            return v[1]
        if isinstance(v, (int, float)):
            return float(v)
        return None

    if isinstance(block, dict):
        for k, v in block.items():
            pair = None
            if isinstance(k, tuple) and len(k) == 2:
                pair = k
            elif isinstance(k, str) and "_vs_" in k:
                pair = tuple(k.split("_vs_", 1))
            elif isinstance(k, str) and k in FAMILIES and isinstance(v, dict):
                for k2, v2 in v.items():
                    p = _pval_from(v2)
                    if p is not None and k in FAMILIES and k2 in FAMILIES:
                        mat.loc[k, k2] = p
                        mat.loc[k2, k] = p
                continue
            if pair and pair[0] in FAMILIES and pair[1] in FAMILIES:
                p = _pval_from(v)
                if p is not None:
                    mat.loc[pair[0], pair[1]] = p
                    mat.loc[pair[1], pair[0]] = p
    if mat.isna().all().all():
        return None
    return mat


def plot_significance_heatmap(out_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not globals().get("STAGE6_RESULTS"):
        payload["skipped"] = True
        payload["reason"] = "STAGE6_RESULTS is None (RUN_STAGE6_SIGNIFICANCE_NOW=False upstream)"
        return payload
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    any_plotted = False
    for ax, test_name, title in (
        (axes[0], "nadeau_bengio", "Nadeau-Bengio corrected t-test (p-value)"),
        (axes[1], "holm_bonferroni", "Holm-Bonferroni corrected (p-value)"),
    ):
        mat = _pairwise_pvalue_matrix(test_name)
        payload[test_name] = mat.to_dict() if mat is not None else None
        if mat is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{test_name}\n(not available)", ha="center", va="center")
            continue
        any_plotted = True
        sns.heatmap(mat.astype(float), annot=True, fmt=".3f", cmap="viridis_r", vmin=0, vmax=1,
                    ax=ax, cbar_kws={"label": "p-value"})
        ax.set_title(title)
    fig.tight_layout()
    if any_plotted:
        fig.savefig(out_path, dpi=150)
        payload["plot_path"] = str(out_path)
    plt.close(fig)
    return payload


def run_family_comparison_suite() -> dict[str, Any]:
    if "STAGE5_RESULTS" not in globals():
        ANALYSIS_LOGGER.notice("STAGE5_RESULTS not present — skipping family comparison suite")
        return {"skipped": True}
    results: dict[str, Any] = {"forest_plots": {}, "significance": {}}
    for metric in QUALITY_METRICS:
        try:
            out_path = ANALYSIS_FAMILY_DIR / f"family_forest_{metric}.png"
            df = plot_family_forest(metric, out_path)
            results["forest_plots"][metric] = {"table": df, "plot_path": str(out_path) if df is not None else None}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"family forest ({metric}) failed: {exc}\n{traceback.format_exc()}")
            results["forest_plots"][metric] = {"error": str(exc)}
    try:
        out_path = ANALYSIS_FAMILY_DIR / "family_significance_matrix.png"
        results["significance"] = plot_significance_heatmap(out_path)
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"significance heatmap failed: {exc}\n{traceback.format_exc()}")
        results["significance"] = {"error": str(exc)}
    return results


# %% [markdown]
# ## Section 7 Train / validation / test three-way report
#
# `TRAIN_VAL_TEST_REPORT` (a DataFrame built by fermi_validation.py's
# `build_train_val_test_report`, also persisted to
# `VALIDATION_CSV_DIR/"train_validation_test_metric_comparison_with_ci.csv"`)
# is read defensively: it is normalized to long form (family, metric, split,
# value[, ci_lo, ci_hi]) whether it originally ships wide (train_/val_/
# test_-prefixed columns) or already long (a `split` column present).

# %%
def _normalize_tvt_report(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    fam_col = cols_lower.get("family") or cols_lower.get("fam") or cols_lower.get("hierarchy")
    if fam_col is None:
        return None
    metric_col = cols_lower.get("metric")

    # Already long: (family, metric, split, value[, ci_lo, ci_hi]).
    if metric_col is not None and "split" in cols_lower and ("value" in cols_lower or "mean" in cols_lower):
        split_col = cols_lower["split"]
        val_col = cols_lower.get("value", cols_lower.get("mean"))
        lo_col = cols_lower.get("ci_lo", cols_lower.get("lower"))
        hi_col = cols_lower.get("ci_hi", cols_lower.get("upper"))
        out = df[[fam_col, metric_col, split_col, val_col] + ([lo_col] if lo_col else []) + ([hi_col] if hi_col else [])].copy()
        out.columns = ["family", "metric", "split"] + ["value"] + (["ci_lo"] if lo_col else []) + (["ci_hi"] if hi_col else [])
        return out

    # Real build_train_val_test_report schema: one row per family, columns
    # "{train,validation,test}_{metric}_mean"/"_ci_low"/"_ci_high". Discover
    # (split, metric) pairs from column names — don't assume REPORT_METRICS
    # is in scope.
    mean_pat = re.compile(r"^(train|validation|test)_(.+)_mean$")
    metrics_by_split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for c in df.columns:
        m = mean_pat.match(c)
        if m and m.group(1) in metrics_by_split:
            metrics_by_split[m.group(1)].append(m.group(2))

    long_rows = []
    for _, r in df.iterrows():
        for split, metrics in metrics_by_split.items():
            split_label = "val" if split == "validation" else split
            for metric in metrics:
                lo_col, hi_col = f"{split}_{metric}_ci_low", f"{split}_{metric}_ci_high"
                long_rows.append({
                    "family": r[fam_col], "metric": metric, "split": split_label,
                    "value": r[f"{split}_{metric}_mean"],
                    "ci_lo": r[lo_col] if lo_col in df.columns else None,
                    "ci_hi": r[hi_col] if hi_col in df.columns else None,
                })
    if long_rows:
        return pd.DataFrame(long_rows)

    # Fallback: single-metric wide form ("train_mean"/"train"/"train_value").
    if metric_col is None:
        return None
    long_rows = []
    for split in ("train", "val", "validation", "test"):
        val_col = next((c for k, c in cols_lower.items() if k == f"{split}_mean" or k == split or k == f"{split}_value"), None)
        if val_col is None:
            continue
        lo_col = next((c for k, c in cols_lower.items() if k == f"{split}_ci_lo" or k == f"{split}_lower"), None)
        hi_col = next((c for k, c in cols_lower.items() if k == f"{split}_ci_hi" or k == f"{split}_upper"), None)
        split_label = "val" if split == "validation" else split
        for _, r in df.iterrows():
            long_rows.append({
                "family": r[fam_col], "metric": r[metric_col], "split": split_label,
                "value": r[val_col],
                "ci_lo": r[lo_col] if lo_col else None,
                "ci_hi": r[hi_col] if hi_col else None,
            })
    return pd.DataFrame(long_rows) if long_rows else None


def run_train_val_test_suite() -> dict[str, Any]:
    if "TRAIN_VAL_TEST_REPORT" not in globals():
        ANALYSIS_LOGGER.notice("TRAIN_VAL_TEST_REPORT not present — skipping train/val/test suite")
        return {"skipped": True}
    long_df = _normalize_tvt_report(TRAIN_VAL_TEST_REPORT)
    if long_df is None:
        ANALYSIS_LOGGER.warning("TRAIN_VAL_TEST_REPORT present but could not be normalized — skipping")
        return {"skipped": True}

    results: dict[str, Any] = {"plots": {}}
    metrics_present = [m for m in QUALITY_METRICS if m in long_df["metric"].unique()]
    for metric in metrics_present or long_df["metric"].unique():
        try:
            sub = long_df[long_df["metric"] == metric]
            fams = sub["family"].unique()
            splits = [s for s in ("train", "val", "test") if s in sub["split"].unique()]
            fig, ax = plt.subplots(figsize=(1.6 * len(fams) + 3, 5))
            x = np.arange(len(fams))
            width = 0.8 / max(len(splits), 1)
            for i, split in enumerate(splits):
                s = sub[sub["split"] == split].set_index("family").reindex(fams)
                yerr = None
                if "ci_lo" in s.columns and s["ci_lo"].notna().any():
                    yerr = np.vstack([
                        (s["value"] - s["ci_lo"]).clip(lower=0).fillna(0),
                        (s["ci_hi"] - s["value"]).clip(lower=0).fillna(0),
                    ])
                ax.bar(x + i * width, s["value"].values, width=width, yerr=yerr, capsize=3, label=split)
            ax.set_xticks(x + width * (len(splits) - 1) / 2)
            ax.set_xticklabels([FAMILY_LABELS.get(f, f) for f in fams])
            ax.set_ylabel(metric)
            ax.set_title(f"Train / val / test — {metric}")
            ax.legend()
            fig.tight_layout()
            out_path = ANALYSIS_TVT_DIR / f"train_val_test_{metric}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results["plots"][metric] = str(out_path)
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"train/val/test plot ({metric}) failed: {exc}\n{traceback.format_exc()}")
            results["plots"][metric] = None
    results["table"] = long_df
    return results


# %% [markdown]
# ## Section 8 Rare-class shrinkage forest plots
#
# `RARE_CLASS_SHRINKAGE_PATHS` maps each family (plus a `"caveats"` key) to a
# CSV path. Column names are matched by pattern (raw vs shrunk/pooled
# estimate, CI bounds) rather than assumed verbatim, since the shrinkage
# export's exact column naming is an implementation detail of
# `export_rare_class_shrinkage_csv`.

# %%
def _match_col(columns: Sequence[str], *patterns: str) -> Optional[str]:
    low = {c.lower(): c for c in columns}
    for p in patterns:
        for k, orig in low.items():
            if p in k:
                return orig
    return None


def run_rare_class_shrinkage_suite() -> dict[str, Any]:
    if "RARE_CLASS_SHRINKAGE_PATHS" not in globals():
        ANALYSIS_LOGGER.notice("RARE_CLASS_SHRINKAGE_PATHS not present — skipping shrinkage suite")
        return {"skipped": True}
    results: dict[str, Any] = {}
    for fam in FAMILIES:
        path = RARE_CLASS_SHRINKAGE_PATHS.get(fam)
        if not path:
            continue
        try:
            df = pd.read_csv(path)
            class_col = _match_col(df.columns, "class", "label")
            raw_col = _match_col(df.columns, "raw_recall", "raw")
            shrunk_col = _match_col(df.columns, "shrunk", "pooled", "posterior")
            lo_col = _match_col(df.columns, "ci_lo", "lower")
            hi_col = _match_col(df.columns, "ci_hi", "upper")
            if class_col is None or shrunk_col is None:
                results[fam] = {"skipped": True, "reason": "columns not recognized"}
                continue
            df = df.sort_values(shrunk_col)
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df) + 1)))
            y = np.arange(len(df))
            if raw_col:
                ax.scatter(df[raw_col], y, color="#999999", marker="x", label="raw recall", zorder=3)
            xerr = None
            if lo_col and hi_col:
                xerr = np.vstack([
                    (df[shrunk_col] - df[lo_col]).clip(lower=0),
                    (df[hi_col] - df[shrunk_col]).clip(lower=0),
                ])
            ax.errorbar(df[shrunk_col], y, xerr=xerr, fmt="o", color="#C44E52",
                        capsize=3, label="shrunk (empirical-Bayes) recall", zorder=4)
            ax.set_yticks(y)
            ax.set_yticklabels(df[class_col])
            ax.set_xlabel("Recall")
            ax.set_title(f"{FAMILY_LABELS.get(fam, fam)} — rare-class shrinkage (reporting only)")
            ax.legend(fontsize=9)
            fig.tight_layout()
            out_path = ANALYSIS_SHRINKAGE_DIR / f"{fam}_rare_class_shrinkage.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results[fam] = {"plot_path": str(out_path), "n_classes": len(df)}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{fam}] rare-class shrinkage plot failed: {exc}\n{traceback.format_exc()}")
            results[fam] = {"error": str(exc)}
    caveats_path = RARE_CLASS_SHRINKAGE_PATHS.get("caveats")
    if caveats_path:
        results["caveats_path"] = str(caveats_path)
    return results


# %% [markdown]
# ## Section 9 Per-class metrics table / heatmap

# %%
def run_per_class_metrics_suite() -> dict[str, Any]:
    if "PER_CLASS_METRICS_PATHS" not in globals():
        ANALYSIS_LOGGER.notice("PER_CLASS_METRICS_PATHS not present — skipping per-class metrics suite")
        return {"skipped": True}
    results: dict[str, Any] = {}
    for fam in FAMILIES:
        path = PER_CLASS_METRICS_PATHS.get(fam)
        if not path:
            continue
        try:
            df = pd.read_csv(path)
            class_col = _match_col(df.columns, "class", "label") or df.columns[0]
            metric_cols = [c for c in df.columns if _match_col([c], "precision", "recall", "f1", "support")]
            if not metric_cols:
                metric_cols = [c for c in df.columns if c != class_col and pd.api.types.is_numeric_dtype(df[c])]
            heat_cols = [c for c in metric_cols if _match_col([c], "precision", "recall", "f1")] or metric_cols
            fig, ax = plt.subplots(figsize=(1.4 * len(heat_cols) + 3, 0.4 * len(df) + 2))
            sns.heatmap(df.set_index(class_col)[heat_cols].astype(float), annot=True, fmt=".2f",
                        cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "score"})
            ax.set_title(f"{FAMILY_LABELS.get(fam, fam)} — per-class metrics")
            fig.tight_layout()
            out_path = ANALYSIS_PERCLASS_DIR / f"{fam}_per_class_metrics.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results[fam] = {"table": df, "plot_path": str(out_path)}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{fam}] per-class metrics suite failed: {exc}\n{traceback.format_exc()}")
            results[fam] = {"error": str(exc)}
    return results


# %% [markdown]
# ## Section 10 Calibration reliability diagrams
#
# `RUN_CALIBRATION_CHECK_NOW` may be False, in which case validation still always
# assigns `CALIBRATION_CHECK = None` (confirmed — unlike SHAP_RESULTS, which
# is never assigned at all when its opt-in flag is False). This suite
# therefore only needs to guard against `None`/absence, not a NameError.

# %%
def run_calibration_suite() -> dict[str, Any]:
    """One reliability diagram per family, raw and (fine only) calibrated curves overlaid."""
    if "CALIBRATION_CHECK" not in globals() or CALIBRATION_CHECK is None:
        ANALYSIS_LOGGER.notice("CALIBRATION_CHECK is None/absent (RUN_CALIBRATION_CHECK_NOW=False) — skipping")
        return {"skipped": True}
    results: dict[str, Any] = {}
    for fam in FAMILIES:
        variants = CALIBRATION_CHECK.get(fam)
        if not variants:
            continue
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
            for name, entry in variants.items():
                rel_df = pd.read_csv(entry["reliability_path"])
                ax.plot(rel_df["mean_predicted_confidence"], rel_df["empirical_accuracy"], marker="o",
                        label=f"{name} (Brier {entry['brier_score_mean']:.3f}, ECE {entry['ece_mean']:.3f})")
            ax.set_xlabel("Confidence of the predicted label")
            ax.set_ylabel("Empirical accuracy")
            ax.set_title(f"{FAMILY_LABELS.get(fam, fam)} — reliability diagram (test rows)")
            ax.legend()
            fig.tight_layout()
            plot_path = ANALYSIS_CALIBRATION_DIR / f"{fam}_reliability.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            results[fam] = {**{name: {k: e.get(k) for k in ("brier_score_mean", "brier_score_std", "nll_mean",
                                                             "ece_mean", "n_splits")}
                               for name, e in variants.items()}, "plot_path": str(plot_path)}
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{fam}] calibration suite failed: {exc}\n{traceback.format_exc()}")
            results[fam] = {"error": str(exc)}
    return results


# %% [markdown]
# ## Section 11 Unassociated-source inference summary
#
# `STAGE9_INFERENCE` (Stage 9, deployment-stage inference over unassociated
# 4FGL sources) is read defensively for a per-family confidence-threshold
# summary (`CONFIDENCE_SUMMARY_THRESHOLDS`) and, where present, a
# cross-family agreement figure.

# %%
def run_inference_summary_suite() -> dict[str, Any]:
    stage9 = globals().get("STAGE9_INFERENCE") or globals().get("UNKNOWN_SOURCE_PREDICTIONS")
    if not stage9:
        ANALYSIS_LOGGER.notice(
            "Neither STAGE9_INFERENCE nor UNKNOWN_SOURCE_PREDICTIONS present — "
            "skipping inference summary suite"
        )
        return {"skipped": True}

    results: dict[str, Any] = {"per_family": {}}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    any_plotted = False

    for fam in FAMILIES:
        try:
            entry = stage9.get(fam) if isinstance(stage9.get(fam), dict) else None
            conf_summary: dict[float, int] = {}
            n_total = None
            high_conf_path = None

            if entry is not None:
                # legacy in-memory shape, if some future Stage 9 ever produces it directly
                conf_summary = entry.get("confidence_summary") or {}
                n_total = entry.get("n_total") or entry.get("n_sources")
                high_conf_path = entry.get("high_confidence_export_path")
            else:
                # actual fermi_inference.py shape: per-family CSV paths
                summary_path = (stage9.get("confidence_summary_paths") or {}).get(fam)
                if summary_path and Path(summary_path).exists():
                    summary_df = pd.read_csv(summary_path)
                    conf_summary = dict(zip(summary_df["confidence_threshold"], summary_df["n_sources"]))
                    if "n_total_predicted" in summary_df.columns and len(summary_df):
                        n_total = int(summary_df["n_total_predicted"].iloc[0])
                high_conf_path = (stage9.get("high_confidence_paths") or {}).get(fam)

            if conf_summary:
                xs = sorted(float(t) for t in conf_summary.keys())
                counts = [conf_summary[t] if t in conf_summary else conf_summary.get(str(t)) for t in xs]
                fracs = [c / n_total if (n_total and c is not None and c <= n_total) else c for c in counts]
                ax.plot(xs, fracs, marker="o", label=FAMILY_LABELS.get(fam, fam))
                any_plotted = True

            results["per_family"][fam] = {
                "n_total": n_total,
                "confidence_summary": conf_summary,
                "high_confidence_export_path": high_conf_path,
            }
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{fam}] inference summary extraction failed: {exc}\n{traceback.format_exc()}")
            results["per_family"][fam] = {"error": str(exc)}

    if any_plotted:
        hce_thresh = globals().get("HIGH_CONFIDENCE_EXPORT_THRESHOLD", 0.80)
        ax.axvline(hce_thresh, linestyle="--", color="gray", label=f"export threshold ({hce_thresh})")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Fraction of unassociated sources ≥ threshold")
        ax.set_title("Unassociated-source classification confidence (Stage 9)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out_path = ANALYSIS_INFERENCE_DIR / "inference_confidence_summary.png"
        fig.savefig(out_path, dpi=150)
        results["plot_path"] = str(out_path)
    plt.close(fig)

    cross_agree = stage9.get("cross_family_agreement")
    if cross_agree is None:
        agreement_path = stage9.get("agreement_path")
        if agreement_path and Path(agreement_path).exists():
            agree_df = pd.read_csv(agreement_path)
            cross_agree = {
                "n_all_agree": int(agree_df["all_families_agree"].sum()) if "all_families_agree" in agree_df.columns else None,
                "n_total": len(agree_df),
                "agreement_path": agreement_path,
            }
    if cross_agree is not None:
        results["cross_family_agreement"] = cross_agree

    return results


# %% [markdown]
# ## Section 12 End-to-end architecture summary table
#
# One row per family: the winning end-to-end model name (`full_model_name`),
# per-node model attribution (`node_models`), held-out headline metric, and
# the frozen deployment artifact path from `STAGE7_DEPLOYMENT`.

# %%
def build_architecture_summary_table() -> pd.DataFrame:
    rows = []
    stage5 = globals().get("STAGE5_RESULTS", {})
    stage7 = globals().get("STAGE7_DEPLOYMENT", {})
    for fam in FAMILIES:
        s5 = stage5.get(fam, {})
        s7 = stage7.get(fam, {})
        point, lo, hi = _fam_point_and_ci(fam, SELECTION_METRIC) if "STAGE5_RESULTS" in globals() else (None, None, None)
        node_models = s5.get("node_models") or {}
        rows.append({
            "family": FAMILY_LABELS.get(fam, fam),
            "full_model_name": s5.get("full_model_name"),
            f"held_out_{SELECTION_METRIC}": point,
            f"{SELECTION_METRIC}_ci_lo": lo,
            f"{SELECTION_METRIC}_ci_hi": hi,
            "n_nodes": len(node_models),
            "node_models": "; ".join(f"{k}={v}" for k, v in node_models.items()) if isinstance(node_models, dict) else node_models,
            "deployment_model_path": s7.get("model_path"),
            "n_pooled_rows": s7.get("n_pooled_rows"),
        })
    return pd.DataFrame(rows)


def run_architecture_summary_suite() -> dict[str, Any]:
    try:
        df = build_architecture_summary_table()
        out_csv = ANALYSIS_ARCH_DIR / "architecture_summary.csv"
        df.to_csv(out_csv, index=False)

        fig, ax = plt.subplots(figsize=(min(14, 2 + 1.2 * len(df.columns)), 1 + 0.6 * len(df)))
        ax.axis("off")
        display_df = df.copy()
        for c in display_df.columns:
            display_df[c] = display_df[c].apply(lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))
        tbl = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.6)
        ax.set_title("End-to-end architecture summary (Stage 5-7)", pad=20)
        fig.tight_layout()
        out_png = ANALYSIS_ARCH_DIR / "architecture_summary.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"table": df, "csv_path": str(out_csv), "plot_path": str(out_png)}
    except Exception as exc:
        ANALYSIS_LOGGER.warning(f"architecture summary suite failed: {exc}\n{traceback.format_exc()}")
        return {"error": str(exc)}


# %% [markdown]
# ## Section 13 SHAP explainability
#
# `RUN_SHAP_EXPLAINABILITY_NOW` may be False, in which case validation never
# assigns `SHAP_RESULTS` as a global at all (confirmed — a bare reference
# raises NameError). Access is therefore via `globals().get(...)`, not a
# direct name lookup. When SHAP wasn't run, this suite explicitly notes that
# Section 2's permutation-importance suite already covers explainability
# for this run, rather than silently producing nothing.

# %%
def run_shap_suite() -> dict[str, Any]:
    shap_results = globals().get("SHAP_RESULTS")
    if not shap_results:
        ANALYSIS_LOGGER.notice(
            "SHAP_RESULTS not assigned (RUN_SHAP_EXPLAINABILITY_NOW=False) — "
            "falling back to Section 2's permutation-importance suite for explainability."
        )
        return {"skipped": True, "fallback": "permutation_importance (Section 2)"}

    results: dict[str, Any] = {}
    for node_label, payload in shap_results.items():
        try:
            imp_df = payload.get("importance")
            if imp_df is None or (hasattr(imp_df, "empty") and imp_df.empty):
                results[node_label] = {"skipped": True}
                continue
            imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).head(FEATIMP_TOP_N)
            fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(imp_df))))
            ax.barh(imp_df["feature"][::-1], imp_df["mean_abs_shap"][::-1], color="#DD8452")
            ax.set_xlabel("Mean |SHAP value|")
            model_name = payload.get("model_name", "?")
            explainer_kind = payload.get("explainer_kind", "?")
            ax.set_title(f"{node_label} — SHAP importance ({model_name}, {explainer_kind})")
            fig.tight_layout()
            out_path = ANALYSIS_SHAP_DIR / f"{_safe_filename(node_label)}_shap.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            results[node_label] = {
                "plot_path": str(out_path),
                "n_background_rows": payload.get("n_background_rows"),
                "n_eval_samples": payload.get("n_eval_samples"),
            }
        except Exception as exc:
            ANALYSIS_LOGGER.warning(f"[{node_label}] SHAP plot failed: {exc}\n{traceback.format_exc()}")
            results[node_label] = {"error": str(exc)}
        finally:
            gc.collect()
    return results


# %% [markdown]
# ## Section 14: Driver

# %%
def run_analysis_pipeline(nodes: Sequence[AnalysisNode] = ALL_ANALYSIS_NODES) -> dict[str, Any]:
    summary: dict[str, Any] = {
        # node-level
        "feature_importance": None,
        "power_curves": None,
        "distributions": None,
        "feature_distribution_gallery": None,
        "clustering": None,
        "clustering_pairwise": None,
        "metric_comparisons": None,
        "leaderboard_violin_suite": None,
        "leaderboard_violin_suite_panoramic": None,
        "tree_gallery_per_node": None,
        "tree_gallery_family": None,
        # validation-level
        "family_comparison": None,
        "train_val_test": None,
        "rare_class_shrinkage": None,
        "per_class_metrics": None,
        "calibration": None,
        "inference_summary": None,
        "architecture_summary": None,
        "shap_explainability": None,
    }

    def _run(step_name: str, fn: Callable[[], Any]) -> None:
        ANALYSIS_LOGGER.info(f"=== analysis: running '{step_name}' ===")
        try:
            summary[step_name] = fn()
            ANALYSIS_LOGGER.info(f"=== analysis: '{step_name}' OK ===")
        except Exception as exc:
            ANALYSIS_LOGGER.error(f"=== analysis: '{step_name}' FAILED: {exc} ===\n{traceback.format_exc()}")
            summary[step_name] = {"error": str(exc)}
        finally:
            gc.collect()

    # --- retained, node-level suites ---
    _run("feature_importance", lambda: run_feature_importance_suite(nodes))
    _run("power_curves", lambda: run_power_curve_suite(nodes))
    _run("distributions", lambda: run_distribution_suite(nodes))
    _run("feature_distribution_gallery", lambda: run_feature_distribution_gallery(nodes))
    _run("clustering", lambda: run_clustering_suite(nodes))
    _run("clustering_pairwise", lambda: run_clustering_pairwise_suite("merged_router"))
    _run("metric_comparisons", lambda: run_metric_comparison_suite(nodes))
    _run("leaderboard_violin_suite", lambda: run_leaderboard_violin_suite(nodes))
    _run("leaderboard_violin_suite_panoramic", lambda: run_panoramic_violin_suite(nodes))
    _run("tree_gallery_per_node", lambda: run_best_tree_per_node_suite(nodes))
    _run("tree_gallery_family", lambda: run_tree_family_gallery("merged_router"))

    # --- validation-level suites ---
    _run("family_comparison", run_family_comparison_suite)
    _run("train_val_test", run_train_val_test_suite)
    _run("rare_class_shrinkage", run_rare_class_shrinkage_suite)
    _run("per_class_metrics", run_per_class_metrics_suite)
    _run("calibration", run_calibration_suite)
    _run("inference_summary", run_inference_summary_suite)
    _run("architecture_summary", run_architecture_summary_suite)
    _run("shap_explainability", run_shap_suite)

    return summary


# %%
ANALYSIS_SUMMARY = run_analysis_pipeline()

print("=" * 78)
print("fermi_analysis — pipeline summary")
print("=" * 78)
for _step, _result in ANALYSIS_SUMMARY.items():
    if isinstance(_result, dict) and "error" in _result and len(_result) == 1:
        _status = f"FAILED: {_result['error']}"
    elif isinstance(_result, dict) and _result.get("skipped"):
        _status = "SKIPPED"
    elif _result is None:
        _status = "NOT RUN"
    else:
        _status = "OK"
    print(f"  [{_status:>8}]  {_step}")
print("=" * 78)
print(f"Analysis outputs written under: {ANALYSIS_OUT_DIR}")
print("=" * 78)
