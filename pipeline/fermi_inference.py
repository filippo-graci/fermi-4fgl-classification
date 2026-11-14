# %% [markdown]
# # fermi_inference.py

# %% [markdown]
# ## Unassociated-source inference — Module 5
#
# Stage 9 of the validation pipeline (plus 15a/15b/15c), kept in its own
# file so that running inference against the unassociated-source catalog
# does not require re-running (or re-importing, which would re-run)
# `fermi_validation.py`'s whole validation driver.
#
# **This file does not `import fermi_validation`.** That module has no
# `if __name__ == "__main__":` guard (nothing in this codebase does — every
# module here is a jupytext-style `# %%`-cell script meant to be run
# top-to-bottom, not a library), so importing it would re-execute its
# entire Section 13 driver (Stage 1, 3, 4, 5, 7, 8, 9) on every import.
# Instead, exactly like every other file in this pipeline reads its
# predecessors' PERSISTED artifacts off disk rather than their in-memory
# objects, this file:
#
# 1. Re-derives `PROCESSING_ROOT`/`OUTPUT_ROOT`/`VALIDATION_*_DIR`/`SPLIT0`
#    with the identical formulas `fermi_validation.py`'s Section 0/13
#    use (same `VERSION_TAG`/`SELECTION_METRIC` bare names, same
#    `multi_split_manifest.json` discovery order) — so `SPLIT0` here is
#    guaranteed to be the same split whose pooled train+test rows Stage 7
#    actually deployed against, without needing that value passed in.
# 2. Loads Stage 7's deployed `{fine,merged,3level}_deployed.joblib`
#    artifacts straight from `VALIDATION_MODELS_DIR` (fixed, deterministic
#    filenames — see `fermi_validation.py`'s `run_stage7`).
# 3. Loads Stage 8's `corrected_winner_family` straight from the persisted
#    `cross_pipeline_validation_report.json`, instead of taking
#    `FINAL_REPORT` as an in-memory argument.
# 4. Rebuilds split-0's `(fine_ctx, merged_ctx, threelevel_ctx)` itself, via
#    the same sibling-module bare context builders
#    (`build_fine_context`/`build_merged_context`/`build_3level_context`)
#    `fermi_validation.py`'s own Section 3 uses — this file only ever
#    needs split 0 (Stage 7 pooled split 0's train+test; nothing here reads
#    any other validation split).
#
# **Still runs in the same shared kernel as everything else in this
# pipeline** (matching every other module's own docstring: `SEED`,
# `VERSION_TAG`, `SELECTION_METRIC`, `TAXONOMY`, `SUPERCLASS_ORDER`,
# `SUPER_LE`, `GATE_LE`, `BLAZAR_SUBGATE_LE`, `NONBLAZAR_ROUTER_LE`,
# `BLAZAR_SUPERCLASSES`, `NONBLAZAR_SUPERCLASS_ORDER`, `build_gate_labels`,
# `build_fine_context`, `build_merged_context`, `build_3level_context`,
# `get_section_logger` — bare names, in-scope, not imported cross-file).
# Run `fermi_processing.py`, `fermi_fine.py`, `fermi_merged.py`,
# `fermi_3levels.py`, and `fermi_validation.py` (through at least
# Stage 8) at least once first, so their persisted artifacts exist on disk;
# after that, this file can be re-run on its own without re-touching any of
# them — it only ever reads, never recomputes, anything upstream of Stage 7.

# %%
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

# %% [markdown]
# ## Section 0 — Config, output layout (mirrors `fermi_validation.py`'s own Section 0)
#
# Every path formula below is copied verbatim from `fermi_validation.py`
# so the two files always agree on where things live, without this file
# depending on that one having just run in this process.

# %%
PROCESSING_ROOT = _cfg("PROCESSING_ROOT", Path(f"./fermi_module1_processing_{VERSION_TAG}/"))
OUTPUT_ROOT = _cfg("OUTPUT_ROOT", Path(f"./outputs_{VERSION_TAG}_{SELECTION_METRIC}/"))
VALIDATION_DIR = OUTPUT_ROOT / "validation"
VALIDATION_LOGS_DIR = VALIDATION_DIR / "logs"
VALIDATION_RESULTS_DIR = VALIDATION_DIR / "results"
VALIDATION_MODELS_DIR = VALIDATION_DIR / "deployed_models"
VALIDATION_CHECKPOINTS_DIR = VALIDATION_DIR / "checkpoints"
VALIDATION_CSV_DIR = VALIDATION_DIR / "csv_reports"
for _d in (VALIDATION_LOGS_DIR, VALIDATION_RESULTS_DIR, VALIDATION_MODELS_DIR,
           VALIDATION_CHECKPOINTS_DIR, VALIDATION_CSV_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FINE_SETUP_KEY_VALIDATION = "B"  # matches fermi_validation.py — fixed choice, not re-derived per family
FAMILIES = list(_cfg("FAMILIES", ["fine", "merged", "3level"]))  # must match fermi_validation.py
FAMILY_LABELS = {"fine": "Fine-Grain (flat)", "merged": "Hierarchical Merged (2-level)",
                  "3level": "Hierarchical 3-Level"}

# --- Confidence-score summary thresholds — defined here since this file is their only reader.
CONFIDENCE_SUMMARY_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

FORCE_RECOMPUTE_INFERENCE = _cfg("FORCE_RECOMPUTE_INFERENCE", False)  # set True to ignore any cached checkpoint below and recompute

_module_logger = get_section_logger("_module_inference", VALIDATION_LOGS_DIR)
_module_logger.notice(f"\n{'=' * 78}\nMODULE 5 (Unassociated-Source Inference) — fermi_inference.py\n{'=' * 78}")
_module_logger.notice(f"  Selection metric        : {SELECTION_METRIC}")
_module_logger.notice(f"  Processing root          : {PROCESSING_ROOT.resolve()}")
_module_logger.notice(f"  Validation dir           : {VALIDATION_DIR.resolve()}")


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
SPLIT0 = ALL_DISCOVERED_SPLITS[0]  # same split Stage 7 pooled train+test and deployed against
_module_logger.notice(f"  Split 0                  : {SPLIT0}  (same split fermi_validation.py's "
                       "Stage 7 deployed against)")


# %% [markdown]
# ## Section 1 — `_m2_predict_proba_safe` (re-ported from `fermi_validation.py`'s Section 1)
#
# Needed by the merged/3level predictors below (`predict_proba` can come
# back pandas- or cuDF-flavored from a GPU/cuML builder) — verbatim copy,
# same reason this file's own Section 1 copy exists.

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


# %% [markdown]
# ## Section 2 — Rebuilding split 0's contexts (mirrors `fermi_validation.py`'s Section 3)

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


# %% [markdown]
# ## Section 3 — Unassociated-source catalog loading (was Section 15)

# %%
UNKNOWN_ID_TAG_COLUMN = "Source_Name"
UNKNOWN_SOURCES_FILENAME = "unknown_sources_full_processed.csv"
# Same DATA_PATH Module 1 reads (`./data/processed/without_optional_features/`).
UNKNOWN_CATALOG_DATA_PATH = _cfg("DATA_PATH", Path("./data/processed/without_optional_features/"))
UNKNOWN_SOURCES_CANDIDATE_PATHS = [
    UNKNOWN_CATALOG_DATA_PATH / UNKNOWN_SOURCES_FILENAME,                  # confirmed — matches fermi_processing.py's DATA_PATH
    PROCESSING_ROOT / UNKNOWN_SOURCES_FILENAME,                            # fallback: single, unsplit Module-1 export
    PROCESSING_ROOT / SPLIT0 / UNKNOWN_SOURCES_FILENAME,                   # fallback: per-split top level
    PROCESSING_ROOT / SPLIT0 / "datasets" / UNKNOWN_SOURCES_FILENAME,      # fallback: per-split, alongside tag CSVs
]


def _locate_unknown_catalog() -> Optional[Path]:
    for p in UNKNOWN_SOURCES_CANDIDATE_PATHS:
        if p.exists():
            return p
    return None


def _positional_columns(df_unk: pd.DataFrame) -> dict:
    """Carries common positional/identification columns into the prediction output
    when present (relevant given this pipeline's |b|>10° vs |b|<=10°
    Galactic-latitude stratification), without hard-failing on a naming
    mismatch. Returns {} if none of these exist."""
    candidates = ["RAJ2000", "DEJ2000", "GLON", "GLAT"]
    return {c: df_unk[c].to_numpy() for c in candidates if c in df_unk.columns}


def _unknown_id_column(df_unk: pd.DataFrame) -> np.ndarray:
    if UNKNOWN_ID_TAG_COLUMN in df_unk.columns:
        return df_unk[UNKNOWN_ID_TAG_COLUMN].to_numpy()
    print(f"  ⚠ '{UNKNOWN_ID_TAG_COLUMN}' not in the unknown-sources CSV — using "
          "positional fallback tags.")
    return np.array([f"row_{i}" for i in df_unk.index])


def _resolve_unknown_frame(ctx, df_unk: pd.DataFrame, tag: str, local_meta: Optional[dict]) -> pd.DataFrame:
    """Selects the column subset for `tag` out of the raw, unlabeled
    unassociated-source catalog `df_unk`, reading a global tag's column
    NAMES off split 0's already-loaded training context
    (`ctx.x_train_by_tag[tag].columns`)."""
    if tag in ctx.x_train_by_tag:
        cols = list(ctx.x_train_by_tag[tag].columns)
    elif local_meta and tag in local_meta.get("feature_sets", {}):
        cols = list(local_meta["feature_sets"][tag])
    else:
        raise KeyError(f"Tag '{tag}' is neither a global/router-local tag nor a known branch-local tag.")
    missing = [c for c in cols if c not in df_unk.columns]
    if missing:
        raise KeyError(f"Unassociated-source catalog is missing column(s) required by tag '{tag}': {missing}")
    return df_unk[cols]


# %% [markdown]
# ## Section 4 — Per-family unassociated-source predictors (was Section 15a)

# %%
def predict_unknown_sources_fine(df_unk: pd.DataFrame, deployed: dict, csv_dir: Optional[Path] = None) -> pd.DataFrame:
    """`csv_dir`: where the EM prior table goes (default VALIDATION_CSV_DIR)."""
    model, label_encoder, features = deployed["model"], deployed["label_encoder"], deployed["features"]
    missing = [c for c in features if c not in df_unk.columns]
    if missing:
        raise KeyError(f"[fine] Unassociated-source catalog is missing column(s) the deployed "
                        f"model needs: {missing}")
    x_unknown = df_unk[features]
    y_pred_enc = np.asarray(model.predict(x_unknown)).reshape(-1)
    classes = label_encoder.classes_
    pred_df = pd.DataFrame({
        "source_name": _unknown_id_column(df_unk), **_positional_columns(df_unk),
        "predicted_label": classes[y_pred_enc],
    })
    if not hasattr(model, "predict_proba"):
        pred_df["confidence"] = float("nan")
        return pred_df
    proba_raw = np.asarray(model.predict_proba(x_unknown))
    rows = np.arange(len(y_pred_enc))
    pred_df["confidence_raw"] = proba_raw[rows, y_pred_enc]
    calibrator = deployed.get("calibrator")
    proba = calibrator.transform(proba_raw) if calibrator is not None else proba_raw
    # `confidence` (what the threshold summaries and the v41 check cut on) is the CALIBRATED
    # probability of the predicted label when a calibrator was deployed; the label itself is the
    # weighted model's argmax either way.
    for i, cls in enumerate(classes):
        pred_df[f"proba_{cls}"] = proba[:, i]
    pred_df["confidence"] = proba[rows, y_pred_enc]
    if calibrator is not None and deployed.get("train_priors") is not None:
        # Sensitivity check, not a correction to trust blindly: EM re-estimation of the class priors
        # of the unassociated sources (Saerens et al. 2002), assuming label shift.
        proba_adj, priors_hat = em_prior_adjust(proba, deployed["train_priors"])
        pred_df["confidence_prior_adjusted"] = proba_adj[rows, y_pred_enc]
        pred_df["predicted_label_prior_adjusted"] = classes[proba_adj.argmax(axis=1)]
        priors_path = (csv_dir or VALIDATION_CSV_DIR) / "unassociated_class_priors_em_fine.csv"
        pd.DataFrame({"fine_class": classes, "training_prior": deployed["train_priors"],
                      "em_estimated_unassociated_prior": priors_hat}).to_csv(priors_path, index=False)
        _module_logger.notice(f"  [fine] calibrated confidences (T={calibrator.temperature_:.3f}); "
                              f"EM-estimated class priors of the unassociated sources -> {priors_path}")
    return pred_df


def predict_unknown_sources_merged(df_unk: pd.DataFrame, deployed: dict, merged_ctx) -> pd.DataFrame:
    router_model, router_cfg = deployed["router_model"], deployed["router_cfg"]
    branch_models, branch_cfgs = deployed["branch_models"], deployed["branch_cfgs"]
    fine_label_encoder = deployed["fine_label_encoder"]
    router_meta, branch_meta = deployed["router_meta"], deployed["branch_meta"]

    x_router = _resolve_unknown_frame(merged_ctx, df_unk, router_cfg["tag"], router_meta)
    all_superclasses = list(SUPER_LE.classes_)
    super_idx = {s: i for i, s in enumerate(all_superclasses)}
    p_super = _m2_predict_proba_safe(router_model, x_router, len(all_superclasses))
    y_pred_super_str = np.array(all_superclasses)[p_super.argmax(axis=1)]
    router_confidence = p_super.max(axis=1)

    # Soft routing, identical to `_fit_predict_merged_core` (the path the
    # reported test metrics come from): P(fine) = sum_sc P(sc) * P(fine | sc).
    # Hard top-1 routing here would deploy a different classifier than the
    # one that was evaluated.
    n = len(df_unk)
    all_fine_classes = list(fine_label_encoder.classes_)
    fine_idx = {c: i for i, c in enumerate(all_fine_classes)}
    p_fine = np.zeros((n, len(all_fine_classes)))

    for sc in SUPERCLASS_ORDER:
        p_sc = p_super[:, super_idx[sc]]
        bm = branch_models.get(sc)
        if bm is None:
            # Mirrors `_fit_predict_merged_core`'s own "no trained branch"
            # fallback (the code path Stage 3/5/7/16b's EVALUATION already uses) instead
            # of the previous bare 'UNKNOWN'. `_m2_build_branch_datasets`/
            # `fermi_merged.py`'s own `build_branch_datasets` skip training a branch
            # classifier whenever a superclass has < 2 distinct fine classes in training
            # data — which, after the ssrq/css->fsrq physical merge (Setup B), is
            # BLAZAR_HE (fsrq only) and BLAZAR_LE (bll only) EVERY time, by taxonomy
            # construction, not because training failed on this run. There is nothing to
            # classify there: the router's own superclass call already IS the fine-grain
            # answer. For any other superclass this ever fires on (a genuinely thin/empty
            # branch, not a taxonomically degenerate one), fall back to that superclass's
            # training-set mode fine class — identical to what Stage 5/7 already compute
            # internally, so Stage 9's deployment predictions stay consistent with the
            # validated evaluation path instead of silently discarding real sources
            # (previously 1067/2563 = 42% of unassociated sources for "merged").
            fine_tr_sc = merged_ctx.y_train_fine_str[merged_ctx.y_train_super_str == sc]
            if len(fine_tr_sc):
                fallback = pd.Series(fine_tr_sc).mode().iloc[0]
                p_fine[:, fine_idx[fallback]] += p_sc
            else:
                print(f"  ⚠ [merged|{sc}] no trained branch AND no training-set fallback available "
                      f"for this superclass — its probability mass is dropped.")
            continue
        model_b, le_branch = bm["model"], bm["label_encoder"]
        try:
            x_branch = _resolve_unknown_frame(merged_ctx, df_unk, branch_cfgs[sc]["tag"], branch_meta.get(sc))
        except KeyError as e:
            fine_tr_sc = merged_ctx.y_train_fine_str[merged_ctx.y_train_super_str == sc]
            fallback = pd.Series(fine_tr_sc).mode().iloc[0] if len(fine_tr_sc) else all_fine_classes[0]
            print(f"  ⚠ [merged|{sc}] {e} — branch mass routed to fallback '{fallback}'.")
            p_fine[:, fine_idx[fallback]] += p_sc
            continue
        p_branch = _m2_predict_proba_safe(model_b, x_branch, len(le_branch.classes_))
        for k, fc in enumerate(le_branch.classes_):
            p_fine[:, fine_idx[fc]] += p_sc * p_branch[:, k]

    return pd.DataFrame({
        "source_name": _unknown_id_column(df_unk), **_positional_columns(df_unk),
        "predicted_superclass": y_pred_super_str,
        "predicted_label": np.array(all_fine_classes)[p_fine.argmax(axis=1)],
        "router_confidence": router_confidence,
        "confidence": p_fine.max(axis=1),
        **{f"proba_{c}": p_fine[:, i] for i, c in enumerate(all_fine_classes)},
    })


def predict_unknown_sources_3level(df_unk: pd.DataFrame, deployed: dict, tl_ctx) -> pd.DataFrame:
    gate_model, gate_cfg, gate_meta = deployed["gate_model"], deployed["gate_cfg"], deployed["gate_meta"]
    subgate_model = deployed["subgate_model"]
    subgate_cfg, subgate_meta = deployed["subgate_cfg"], deployed["subgate_meta"]
    nonblazar_model = deployed["nonblazar_model"]
    nonblazar_cfg, nonblazar_meta = deployed["nonblazar_cfg"], deployed["nonblazar_meta"]
    branch_models, branch_cfgs, branch_meta = deployed["branch_models"], deployed["branch_cfgs"], deployed["branch_meta"]
    fine_label_encoder = deployed["fine_label_encoder"]

    # Soft routing, identical to `_fit_predict_3level_core` (the path the
    # reported test metrics come from): P(super) = P(gate) * P(super | gate),
    # P(fine) = sum_sc P(sc) * P(fine | sc).
    n = len(df_unk)
    x_gate = _resolve_unknown_frame(tl_ctx, df_unk, gate_cfg["tag"], gate_meta)
    gate_classes = list(GATE_LE.classes_)
    p_gate = _m2_predict_proba_safe(gate_model, x_gate, len(gate_classes))
    p_blazar, p_other = p_gate[:, gate_classes.index("BLAZAR")], p_gate[:, gate_classes.index("OTHER")]

    x_sub = _resolve_unknown_frame(tl_ctx, df_unk, subgate_cfg["tag"], subgate_meta)
    p_sub = _m2_predict_proba_safe(subgate_model, x_sub, len(BLAZAR_SUBGATE_LE.classes_))
    x_nb = _resolve_unknown_frame(tl_ctx, df_unk, nonblazar_cfg["tag"], nonblazar_meta)
    p_nb = _m2_predict_proba_safe(nonblazar_model, x_nb, len(NONBLAZAR_ROUTER_LE.classes_))

    all_superclasses = list(SUPER_LE.classes_)
    super_idx = {s: i for i, s in enumerate(all_superclasses)}
    p_super = np.zeros((n, len(all_superclasses)))
    for j, sc in enumerate(BLAZAR_SUBGATE_LE.classes_):
        p_super[:, super_idx[sc]] = p_blazar * p_sub[:, j]
    for j, sc in enumerate(NONBLAZAR_ROUTER_LE.classes_):
        p_super[:, super_idx[sc]] = p_other * p_nb[:, j]

    all_fine_classes = list(fine_label_encoder.classes_)
    fine_idx = {c: i for i, c in enumerate(all_fine_classes)}
    p_fine = np.zeros((n, len(all_fine_classes)))
    for sc in BLAZAR_SUPERCLASSES:
        fines = sorted(c for c, s in TAXONOMY.items() if s == sc and c in fine_label_encoder.classes_)
        p_fine[:, fine_idx[fines[0]]] += p_super[:, super_idx[sc]]

    for sc in NONBLAZAR_SUPERCLASS_ORDER:
        p_sc = p_super[:, super_idx[sc]]
        bm = branch_models.get(sc)
        if bm is None:
            fine_tr_sc = tl_ctx.y_train_fine_str[tl_ctx.y_train_super_str == sc]
            if len(fine_tr_sc):
                p_fine[:, fine_idx[pd.Series(fine_tr_sc).mode().iloc[0]]] += p_sc
            else:
                print(f"  ⚠ [3level|{sc}] no trained branch AND no training-set fallback available "
                      f"for this superclass — its probability mass is dropped.")
            continue
        model_b, le_branch = bm["model"], bm["label_encoder"]
        try:
            x_branch = _resolve_unknown_frame(tl_ctx, df_unk, branch_cfgs[sc]["tag"], branch_meta.get(sc))
        except KeyError as e:
            fine_tr_sc = tl_ctx.y_train_fine_str[tl_ctx.y_train_super_str == sc]
            fallback = pd.Series(fine_tr_sc).mode().iloc[0] if len(fine_tr_sc) else all_fine_classes[0]
            print(f"  ⚠ [3level|{sc}] {e} — branch mass routed to fallback '{fallback}'.")
            p_fine[:, fine_idx[fallback]] += p_sc
            continue
        p_branch = _m2_predict_proba_safe(model_b, x_branch, len(le_branch.classes_))
        for k, fc in enumerate(le_branch.classes_):
            p_fine[:, fine_idx[fc]] += p_sc * p_branch[:, k]

    return pd.DataFrame({
        "source_name": _unknown_id_column(df_unk), **_positional_columns(df_unk),
        "predicted_gate": np.array(gate_classes)[p_gate.argmax(axis=1)],
        "predicted_superclass": np.array(all_superclasses)[p_super.argmax(axis=1)],
        "predicted_label": np.array(all_fine_classes)[p_fine.argmax(axis=1)],
        "gate_confidence": p_gate.max(axis=1),
        "superclass_confidence": p_super.max(axis=1),
        "confidence": p_fine.max(axis=1),
        **{f"proba_{c}": p_fine[:, i] for i, c in enumerate(all_fine_classes)},
    })


# %% [markdown]
# ## Section 5 — Confidence-threshold summary + high-confidence export (was Section 15b)

# %%
def export_confidence_threshold_summary(pred_df: pd.DataFrame, fam: str) -> pd.DataFrame:
    """One row per `CONFIDENCE_SUMMARY_THRESHOLDS` value: how many/what
    fraction of this family's unassociated-source predictions clear that
    confidence bar — "sources with >=0.8 confidence" is this table's
    `threshold==0.8` row, rather than a fresh filter each time."""
    n_total = len(pred_df)
    rows = []
    for thr in CONFIDENCE_SUMMARY_THRESHOLDS:
        n_above = int((pred_df["confidence"] >= thr).sum())
        rows.append({"family": fam, "confidence_threshold": thr, "n_sources": n_above,
                     "fraction_of_total": (n_above / n_total) if n_total else float("nan"),
                     "n_total_predicted": n_total})
    return pd.DataFrame(rows)


HIGH_CONFIDENCE_EXPORT_THRESHOLD = 0.80  # matches requirement #5's own example (">0.8 confidence")


# %% [markdown]
# ## Section 6 — Driver (was Section 15c)
#
# Loads what Stage 7/8 left on disk (rather than taking them as in-memory
# arguments) and runs inference exactly as `fermi_validation.py`'s old
# `run_stage9_unknown_inference` did — same function, same outputs, same
# CSVs under `VALIDATION_RESULTS_DIR`/`VALIDATION_CSV_DIR`.

# %%
def _dedupe_console_handlers(logger) -> None:
    """[local workaround, same convention as this file's other disclosed
    upstream-inconsistency workarounds]. `get_section_logger`'s
    `_configure_fermi_root_logger(console_level)` call has no 'already
    configured' guard and runs unconditionally on EVERY call — including
    cache hits — so after N stages have each called `get_section_logger`
    once, the shared root/console handler ends up duplicated N times.
    Idempotent, safe to call every time Stage 9 runs."""
    node = logger
    while node is not None:
        seen, deduped = set(), []
        for h in node.handlers:
            key = (type(h), getattr(h, "stream", None))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(h)
        node.handlers = deduped
        node = node.parent


def run_stage9_unknown_inference(stage7_deployment: dict, split_contexts: dict, corrected_winner_family: str) -> dict:
    logger = get_section_logger("stage9_unknown_inference", VALIDATION_LOGS_DIR)
    _dedupe_console_handlers(logger)
    logger.notice(f"\n{'=' * 70}\nSTAGE 9 — unassociated-source inference [deployment]\n{'=' * 70}")

    catalog_path = _locate_unknown_catalog()
    if catalog_path is None:
        logger.warning("  ⚠ No unassociated-source catalog found at any of "
                        f"{[str(p) for p in UNKNOWN_SOURCES_CANDIDATE_PATHS]} — run "
                        "fermi_processing.py (Module 1) first, or add its real output path to "
                        "UNKNOWN_SOURCES_CANDIDATE_PATHS above. Skipping Stage 9 inference.")
        return {}
    df_unk = pd.read_csv(catalog_path)
    logger.notice(f"  Loaded unassociated-source catalog: {df_unk.shape[0]} rows, "
                  f"{df_unk.shape[1]} cols <- {catalog_path}")
    if "CLASS1" in df_unk.columns:
        n_labeled = int((df_unk["CLASS1"].astype(str).str.lower() != "unk").sum())
        if n_labeled:
            logger.warning(f"  ⚠ {n_labeled} row(s) in {catalog_path.name} carry a non-'unk' CLASS1 — "
                            "check fermi_processing.py's split logic before trusting these predictions.")

    fine_ctx, merged_ctx, tl_ctx = split_contexts[SPLIT0]
    out = {"predictions": {}, "confidence_summary_paths": {}, "high_confidence_paths": {},
           "agreement_path": None}
    for fam in FAMILIES:
        deployed_path = Path(stage7_deployment[fam]["model_path"])
        deployed = joblib.load(deployed_path)
        marker = "  <== CORRECTED WINNER (Stage 8)" if fam == corrected_winner_family else ""
        try:
            if fam == "fine":
                pred_df = predict_unknown_sources_fine(df_unk, deployed)
            elif fam == "merged":
                pred_df = predict_unknown_sources_merged(df_unk, deployed, merged_ctx)
            else:
                pred_df = predict_unknown_sources_3level(df_unk, deployed, tl_ctx)
        except Exception as e:
            logger.error(f"  ⚠ [{fam}] unassociated-source inference FAILED: {type(e).__name__}: {e}")
            out["predictions"][fam] = None
            continue
        out_path = VALIDATION_RESULTS_DIR / f"predictions_unknown_sources_{fam}.csv"
        pred_df.to_csv(out_path, index=False)
        counts = pred_df["predicted_label"].value_counts().to_dict()
        logger.notice(f"  [{fam}]{marker} predicted {len(pred_df)} unassociated sources -> {out_path}")
        logger.notice(f"    class distribution: {counts}")
        out["predictions"][fam] = pred_df

        # confidence-threshold summary + high-confidence subset
        summary_df = export_confidence_threshold_summary(pred_df, fam)
        summary_path = VALIDATION_CSV_DIR / f"confidence_threshold_summary_{fam}.csv"
        summary_df.to_csv(summary_path, index=False)
        out["confidence_summary_paths"][fam] = str(summary_path)
        thr_row = summary_df[summary_df["confidence_threshold"] == HIGH_CONFIDENCE_EXPORT_THRESHOLD]
        n_high = int(thr_row["n_sources"].iloc[0]) if len(thr_row) else 0
        logger.notice(f"    [{fam}] {n_high}/{len(pred_df)} source(s) at "
                       f">={HIGH_CONFIDENCE_EXPORT_THRESHOLD} confidence -> {summary_path}")

        high_conf_df = pred_df[pred_df["confidence"] >= HIGH_CONFIDENCE_EXPORT_THRESHOLD]
        high_conf_path = VALIDATION_CSV_DIR / f"high_confidence_predictions_{fam}.csv"
        high_conf_df.to_csv(high_conf_path, index=False)
        out["high_confidence_paths"][fam] = str(high_conf_path)

    # cross-family prediction-agreement table
    valid_preds = {fam: df for fam, df in out["predictions"].items() if df is not None}
    if len(valid_preds) >= 2:
        merged_on_name = None
        for fam, df in valid_preds.items():
            sub = df[["source_name", "predicted_label", "confidence"]].rename(
                columns={"predicted_label": f"predicted_label_{fam}", "confidence": f"confidence_{fam}"})
            merged_on_name = sub if merged_on_name is None else merged_on_name.merge(sub, on="source_name", how="outer")
        label_cols = [f"predicted_label_{fam}" for fam in valid_preds]
        merged_on_name["n_families_agreeing"] = merged_on_name[label_cols].apply(
            lambda r: int(r.value_counts().max()) if r.notna().any() else 0, axis=1)
        merged_on_name["all_families_agree"] = merged_on_name["n_families_agreeing"] == len(valid_preds)
        merged_on_name["any_disagreement"] = ~merged_on_name["all_families_agree"]
        agreement_path = VALIDATION_CSV_DIR / "cross_family_prediction_agreement.csv"
        merged_on_name.sort_values("n_families_agreeing").to_csv(agreement_path, index=False)
        out["agreement_path"] = str(agreement_path)
        n_full_agree = int(merged_on_name["all_families_agree"].sum())
        logger.notice(f"  Cross-family agreement: {n_full_agree}/{len(merged_on_name)} source(s) where "
                       f"all {len(valid_preds)} families agree -> {agreement_path}")
    else:
        logger.warning("  ⚠ Fewer than 2 families produced valid unassociated-source predictions — "
                        "skipping cross-family agreement export.")

    return out


def _load_stage7_deployment() -> dict:
    """Reconstructs the {fam: {"model_path": ...}} shape `run_stage7` (in
    fermi_validation.py) returns and this driver consumes, straight off
    the fixed, deterministic filenames that function writes to —
    `VALIDATION_MODELS_DIR/{fam}_deployed.joblib` — no in-memory
    STAGE7_DEPLOYMENT dict required."""
    out = {}
    for fam in FAMILIES:
        path = VALIDATION_MODELS_DIR / f"{fam}_deployed.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"[{fam}] no deployed model found at {path} — run fermi_validation.py (through "
                "Stage 7, at least) first.")
        out[fam] = {"model_path": str(path)}
    return out


def _load_corrected_winner_family() -> str:
    """Reads Stage 8's persisted JSON report instead of taking
    FINAL_REPORT as an in-memory argument."""
    report_path = VALIDATION_DIR / "cross_pipeline_validation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} not found — run fermi_validation.py (through Stage 8) first.")
    with open(report_path) as f:
        report = json.load(f)
    return report["corrected_winner_family"]


_INFERENCE_CHECKPOINT_PATH = VALIDATION_CHECKPOINTS_DIR / "unknown_source_inference.joblib"
# Deliberately its own tiny checkpoint file, separate from fermi_validation.py's
# CHECKPOINT_SCHEMA_VERSION-wrapped "stageN.joblib" family — this file has no downstream stages
# depending on it and no STAGE_DEPENDENCIES cascade to participate in, so a bare joblib dump/load
# is all the idempotency this needs.
if _INFERENCE_CHECKPOINT_PATH.exists() and not FORCE_RECOMPUTE_INFERENCE:
    _module_logger.notice(f"  -> loading unassociated-source inference from checkpoint -> "
                           f"{_INFERENCE_CHECKPOINT_PATH}")
    UNKNOWN_SOURCE_PREDICTIONS = joblib.load(_INFERENCE_CHECKPOINT_PATH)
else:
    _stage7_deployment = _load_stage7_deployment()
    _corrected_winner_family = _load_corrected_winner_family()
    _split_contexts = {SPLIT0: _val_build_contexts(SPLIT0)}
    UNKNOWN_SOURCE_PREDICTIONS = run_stage9_unknown_inference(
        _stage7_deployment, _split_contexts, _corrected_winner_family)
    joblib.dump(UNKNOWN_SOURCE_PREDICTIONS, _INFERENCE_CHECKPOINT_PATH)
    _module_logger.notice(f"  -> checkpoint saved -> {_INFERENCE_CHECKPOINT_PATH}")


# %%
print("=" * 78)
print("Module 5 (Unassociated-Source Inference) — COMPLETE.")
for _fam in FAMILIES:
    _pred_df = UNKNOWN_SOURCE_PREDICTIONS.get("predictions", {}).get(_fam)
    if _pred_df is not None:
        print(f"  [{_fam:7s}] {len(_pred_df)} unassociated source(s) predicted")
if UNKNOWN_SOURCE_PREDICTIONS.get("agreement_path"):
    print(f"  Cross-family agreement -> {UNKNOWN_SOURCE_PREDICTIONS['agreement_path']}")
print(f"  Predictions + confidence exports -> {VALIDATION_RESULTS_DIR} / {VALIDATION_CSV_DIR}")
print("=" * 78)

# %% [markdown]
# ## Section 7 — Prospective validation against v41 (FY16 precursor of 5FGL)
#
# Cross-matches Stage 9's unassociated-source predictions (4FGL-DR4 v35's
# still-unknown sources) against the newly-resolved `CLASS1` in v41 (the FY16
# precursor of the future 5FGL catalog), positionally — GLON/GLAT
# nearest-neighbor matching on the sphere, with Fermi-LAT's own 95%-confidence
# positional-error ellipse (`Conf_95_SemiMajor`) setting an adaptive per-source
# match radius. `Source_Name` strings are NOT used for the cross-match itself
# (they differ between catalog releases); they're only used for a cheap DR4-v35
# provenance sanity check below.
#
# Three scoring scenarios, all **Setup B** (this whole file already only ever
# runs Setup B — `FINE_SETUP_KEY_VALIDATION` in Section 0) and all **excluding
# 'bcu'** outright — no partial credit for a 'bcu' row landing anywhere in the
# blazar family. A source whose real v41 CLASS1 is 'bcu', or still 'unk', carries no
# usable ground truth here and is dropped from every accuracy number below:
#
#   [A] best model only (Stage 8's `corrected_winner_family`), swept over
#       `CONFIDENCE_SUMMARY_THRESHOLDS` (plus an unfiltered 0.0 baseline).
#   [B] all 3 families' `predicted_label` agree (`all_families_agree`, same
#       criterion as Section 6's cross-family agreement table), no additional
#       confidence gating.
#   [C] same "all 3 agree" pool as [B], additionally requiring EACH of the 3
#       families' own `confidence` to individually clear a threshold, swept
#       over the same threshold list.
#
# `cross_family_prediction_agreement.csv` (Section 6) carries no GLON/GLAT (its
# merge is name/label/confidence-only across `source_name`), so [B]/[C] rebuild
# that same table here with positions reattached from one family's own Stage-9
# predictions — all three families predict off the identical `df_unk`, so
# positions agree across families by construction, not by coincidence.

# %%
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree
from sklearn.metrics import balanced_accuracy_score

V41_FITS_PATH = _cfg("V41_FITS_PATH", Path("./data/catalogs/gll_psc_v41.fit"))
DR4_V35_FITS_PATH = _cfg("DR4_V35_FITS_PATH", Path("./data/catalogs/gll_psc_v35.fit"))

# Adaptive match radius: bright/well-localized
# v41 sources get a tight radius, faint/poorly-localized ones get a wide one;
# floor absorbs the genuine DR4<->v41 astrometric solution shift the 95% ellipse
# alone doesn't capture, ceiling caps degenerate/very-poorly-localized sources
# from matching almost anything.
USE_ADAPTIVE_MATCH_RADIUS = True
ADAPTIVE_MATCH_FACTOR = 3.0
ADAPTIVE_MATCH_FLOOR_ARCMIN = 1.0
ADAPTIVE_MATCH_CEIL_ARCMIN = 15.0
MAX_MATCH_SEP_ARCMIN = 5.0  # fallback global radius, only used if USE_ADAPTIVE_MATCH_RADIUS=False

# Excluded from every accuracy number in this section, no exceptions: still-'unk'
# (no ground truth at all) AND 'bcu' (ground truth exists but is deliberately
# excluded outright here, by choice — no blazar-family credit).
# 'spp' likewise: dropped from the taxonomy for the same reason as 'bcu'.
UNRESOLVED_V41_CLASSES = {"unk", "bcu", "spp"}

FORCE_RECOMPUTE_V41_VALIDATION = _cfg("FORCE_RECOMPUTE_V41_VALIDATION", False)  # set True to ignore any cached checkpoint below and recompute


# %% [markdown]
# ### Section 7.1 — v41 catalog loading + spherical cross-match (only what
# Setup-B/no-bcu label-level scoring below actually needs)

# %%
def load_v41_catalog(fits_path: Path) -> pd.DataFrame:
    """Extracts Source_Name/GLON/GLAT/CLASS1/Conf_95_SemiMajor from v41 and
    normalizes CLASS1 to lowercase 'real_class1' ('unk' for blank/missing)."""
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
    table = Table(data)["Source_Name", "GLON", "GLAT", "CLASS1", "Conf_95_SemiMajor"]
    df = table.to_pandas()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    df["CLASS1"] = df["CLASS1"].astype(str).str.strip().str.lower()
    df["CLASS1"] = df["CLASS1"].replace({"": "unk", "nan": "unk", "none": "unk"})
    df["Conf_95_SemiMajor"] = df["Conf_95_SemiMajor"].astype(np.float64) * 60.0  # deg -> arcmin
    df = df.rename(columns={
        "Source_Name": "v41_source_name", "GLON": "v41_glon", "GLAT": "v41_glat",
        "CLASS1": "real_class1", "Conf_95_SemiMajor": "v41_conf95_semimajor_arcmin",
    })
    _module_logger.notice(f"  v41 (FY16) catalog loaded: {len(df)} rows | "
                           f"unk={int((df['real_class1'] == 'unk').sum())} | "
                           f"associated={int((df['real_class1'] != 'unk').sum())}")
    return df


def _verify_dr4_v35_provenance(pred_df: pd.DataFrame, v35_fits_path: Path) -> None:
    """Sanity check only (not a hard gate) — confirms the sources Stage 9 just
    scored really are drawn from 4FGL-DR4 v35 before trusting any v41
    cross-match below. Cheap: loads only the Source_Name column from v35."""
    if not v35_fits_path.exists():
        _module_logger.warning(f"  ⚠ DR4 v35 FITS not found at {v35_fits_path} — skipping provenance check.")
        return
    if "source_name" not in pred_df.columns:
        _module_logger.warning("  ⚠ Predictions frame has no 'source_name' column — skipping provenance check.")
        return
    with fits.open(v35_fits_path) as hdul:
        v35_names = set(pd.Series(np.asarray(hdul[1].data["Source_Name"])).astype(str).str.strip())
    pred_names = set(pd.Series(pred_df["source_name"]).astype(str).str.strip())
    overlap = len(pred_names & v35_names) / max(len(pred_names), 1)
    if overlap < 0.9:
        _module_logger.warning(
            f"  ⚠ Only {overlap:.1%} of the scored predictions' source_name values are found in "
            f"{v35_fits_path.name} — this may not be the DR4 release Module 1 actually processed. "
            "Cross-match results below may be unreliable.")
    else:
        _module_logger.notice(f"  Provenance check OK: {overlap:.1%} of predictions' source_name values "
                               f"found in {v35_fits_path.name}.")


def _lonlat_to_unit_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Cartesian unit vectors avoid the 0/360 wraparound and pole degeneracy a
    naive Euclidean distance on raw (GLON, GLAT) would introduce."""
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _chord_to_arcsec(chord: np.ndarray) -> np.ndarray:
    """3D Euclidean (chord) distance between unit vectors -> angular separation:
    theta = 2*arcsin(chord/2)."""
    chord = np.clip(chord, 0.0, 2.0)
    return np.degrees(2.0 * np.arcsin(chord / 2.0)) * 3600.0


def crossmatch_to_v41(pred_df: pd.DataFrame, v41_df: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor spherical cross-match of `pred_df` (must carry GLON/GLAT)
    against v41; adds `is_matched`/`sep_arcsec`/`match_radius_arcmin` plus every
    v41_* column onto a fresh copy of `pred_df`."""
    if "GLON" not in pred_df.columns or "GLAT" not in pred_df.columns:
        raise KeyError("Predictions frame has no GLON/GLAT columns — cannot cross-match to v41 (these "
                        "come from this file's Section 4 `_positional_columns`; confirm the "
                        "unassociated-source catalog carries GLON/GLAT).")
    v41_xyz = _lonlat_to_unit_xyz(v41_df["v41_glon"].values, v41_df["v41_glat"].values)
    pred_xyz = _lonlat_to_unit_xyz(pred_df["GLON"].values, pred_df["GLAT"].values)
    chord_dist, nn_idx = cKDTree(v41_xyz).query(pred_xyz, k=1)
    sep_arcsec = _chord_to_arcsec(chord_dist)
    sep_arcmin = sep_arcsec / 60.0

    matched = v41_df.iloc[nn_idx].reset_index(drop=True)
    merged = pd.concat([pred_df.reset_index(drop=True), matched], axis=1)
    merged["sep_arcsec"] = sep_arcsec

    if USE_ADAPTIVE_MATCH_RADIUS:
        radius_arcmin = np.clip(ADAPTIVE_MATCH_FACTOR * merged["v41_conf95_semimajor_arcmin"].values,
                                 ADAPTIVE_MATCH_FLOOR_ARCMIN, ADAPTIVE_MATCH_CEIL_ARCMIN)
        merged["match_radius_arcmin"] = radius_arcmin
        merged["is_matched"] = sep_arcmin <= radius_arcmin
    else:
        merged["match_radius_arcmin"] = MAX_MATCH_SEP_ARCMIN
        merged["is_matched"] = sep_arcmin <= MAX_MATCH_SEP_ARCMIN

    n_unmatched = int((~merged["is_matched"]).sum())
    frac_tiny_sep = float((sep_arcsec < 1.0).mean())
    if frac_tiny_sep > 0.90 and n_unmatched == 0:
        _module_logger.warning("  ⚠ >90% of matches at sub-arcsecond separation and none unmatched — check "
                                "this isn't a self-match (predictions' positions already == v41's).")
    _module_logger.notice(f"    cross-match: {len(merged)} rows | unmatched: {n_unmatched} | "
                           f"sep_arcsec p50/p90/p99: {np.percentile(sep_arcsec, [50, 90, 99]).round(2).tolist()}")
    return merged


def _resolved_eval_mask(merged: pd.DataFrame) -> pd.Series:
    """Setup-B, no-bcu evaluable mask: spatially matched AND real v41 class is
    neither still-'unk' nor 'bcu' (excluded outright — see Section 7 docstring)."""
    return merged["is_matched"] & (~merged["real_class1"].isin(UNRESOLVED_V41_CLASSES))


def _accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy with its exact Clopper-Pearson 95% CI and balanced accuracy with the Brodersen et al.
    (2010) posterior 95% CI (`bootstrap_metric_ci`) — the v41 subsets are small at high thresholds."""
    n = len(y_true)
    nan = float("nan")
    if n == 0:
        return {"n": 0, "accuracy": nan, "accuracy_ci_low": nan, "accuracy_ci_high": nan,
                "balanced_accuracy": nan, "balanced_accuracy_ci_low": nan, "balanced_accuracy_ci_high": nan}
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    k = int(np.sum(y_true == y_pred))
    acc_lo, acc_hi = clopper_pearson(k, n)
    ba = bootstrap_metric_ci(y_true, y_pred, lambda t, p, w: {"bal_accuracy": balanced_accuracy_score(t, p)},
                             n_boot=200)["bal_accuracy"]
    return {"n": n, "accuracy": k / n, "accuracy_ci_low": acc_lo, "accuracy_ci_high": acc_hi,
            "balanced_accuracy": ba["value"], "balanced_accuracy_ci_low": ba["ci_low"],
            "balanced_accuracy_ci_high": ba["ci_high"]}


# %% [markdown]
# ### Section 7.2 — Scenario [A]: best model only, by confidence threshold

# %%
def evaluate_best_model_thresholds(best_pred_df: pd.DataFrame, v41_df: pd.DataFrame, fam: str) -> tuple:
    merged = crossmatch_to_v41(best_pred_df, v41_df)
    merged["real_class1_norm"] = merged["real_class1"].replace(MERGE_FINE_CLASSES)  # ssrq/css -> FSRQ-like, as in training
    merged["predicted_label_norm"] = merged["predicted_label"].astype(str).str.strip().str.lower()
    eval_frame = merged[_resolved_eval_mask(merged)]
    n_eval_total = len(eval_frame)
    n_total_predicted = len(merged)

    rows = []
    for thr in [0.0] + list(CONFIDENCE_SUMMARY_THRESHOLDS):
        sub = eval_frame[eval_frame["confidence"] >= thr]
        m = _accuracy_metrics(sub["real_class1_norm"].values, sub["predicted_label_norm"].values)
        m.update({"family": fam, "confidence_threshold": thr, "n_evaluable_total": n_eval_total,
                   "n_total_predicted": n_total_predicted,
                   "coverage_of_evaluable": (len(sub) / n_eval_total) if n_eval_total else float("nan")})
        rows.append(m)
    cols = ["family", "confidence_threshold", "n", "n_evaluable_total", "n_total_predicted",
            "coverage_of_evaluable", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
            "balanced_accuracy", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high"]
    return pd.DataFrame(rows)[cols], merged


def v41_per_class(merged: pd.DataFrame, threshold: float = HIGH_CONFIDENCE_EXPORT_THRESHOLD) -> pd.DataFrame:
    """Per v41 class: n scored, correct, n with confidence >= threshold, correct among those."""
    ev = merged[_resolved_eval_mask(merged)]
    ok = ev["real_class1_norm"] == ev["predicted_label_norm"]
    hi = ev["confidence"] >= threshold
    return (pd.DataFrame({"v41_class": ev["real_class1_norm"], "correct": ok, "high_conf": hi,
                          "correct_high_conf": ok & hi})
            .groupby("v41_class").sum().assign(n=ev["real_class1_norm"].value_counts())
            .sort_values("n", ascending=False))


# %% [markdown]
# ### Section 7.3 — Scenarios [B]/[C]: all-3-families-agree (+ per-model confidence gate)

# %%
def _build_agreement_frame_with_positions(predictions: dict) -> pd.DataFrame:
    """Rebuilds Section 6's `cross_family_prediction_agreement.csv` shape but
    keeps GLON/GLAT (that CSV drops them) by carrying them through from
    whichever family's own Stage-9 prediction frame has them first — all three
    families predict off the identical `df_unk`, so positions agree across
    families by construction, not by coincidence."""
    label_cols, merged_on_name, position_cols = [], None, None
    for fam in FAMILIES:
        df = predictions.get(fam)
        if df is None:
            continue
        pos_candidates = [c for c in ("GLON", "GLAT", "RAJ2000", "DEJ2000") if c in df.columns]
        sub = df[["source_name", "predicted_label", "confidence"] + pos_candidates].rename(
            columns={"predicted_label": f"predicted_label_{fam}", "confidence": f"confidence_{fam}"})
        label_cols.append(f"predicted_label_{fam}")
        if position_cols is None and pos_candidates:
            position_cols = pos_candidates
        else:
            sub = sub.drop(columns=pos_candidates)
        merged_on_name = sub if merged_on_name is None else merged_on_name.merge(sub, on="source_name", how="outer")

    if position_cols is None or not {"GLON", "GLAT"}.issubset(position_cols):
        raise KeyError("None of the per-family Stage-9 prediction frames carry both GLON and GLAT — "
                        "cannot cross-match the agreement table to v41.")

    merged_on_name["n_families_agreeing"] = merged_on_name[label_cols].apply(
        lambda r: int(r.value_counts().max()) if r.notna().any() else 0, axis=1)
    merged_on_name["all_families_agree"] = merged_on_name["n_families_agreeing"] == len(label_cols)
    merged_on_name["consensus_label"] = merged_on_name[label_cols[0]]  # identical across families whenever all_families_agree
    return merged_on_name


def evaluate_full_agreement(agreement_df: pd.DataFrame, v41_df: pd.DataFrame) -> tuple:
    """Scenario [B]: all 3 families agree, no extra confidence gate."""
    all_agree = agreement_df[agreement_df["all_families_agree"]].copy()
    merged = crossmatch_to_v41(all_agree.rename(columns={"consensus_label": "predicted_label"}), v41_df)
    merged["real_class1_norm"] = merged["real_class1"].replace(MERGE_FINE_CLASSES)  # ssrq/css -> FSRQ-like, as in training
    merged["predicted_label_norm"] = merged["predicted_label"].astype(str).str.strip().str.lower()
    eval_frame = merged[_resolved_eval_mask(merged)]
    metrics = _accuracy_metrics(eval_frame["real_class1_norm"].values, eval_frame["predicted_label_norm"].values)
    n_total, n_agree = len(agreement_df), len(all_agree)
    metrics.update({"scenario": "all_3_agree", "n_sources_all_agree": n_agree, "n_total_predicted": n_total,
                     "coverage_of_all_predicted": (n_agree / n_total) if n_total else float("nan")})
    return metrics, merged


def evaluate_full_agreement_by_confidence(agreement_df: pd.DataFrame, v41_df: pd.DataFrame, thresholds) -> pd.DataFrame:
    """Scenario [C]: same all-3-agree pool as [B], additionally requiring EACH
    family's own `confidence_<fam>` to individually clear `thr` (not just their
    average/min) — a stricter gate than Stage 9's own combined `confidence`
    column would give if applied post-hoc to the consensus row."""
    base = agreement_df[agreement_df["all_families_agree"]].copy()
    conf_cols = [f"confidence_{fam}" for fam in FAMILIES if f"confidence_{fam}" in base.columns]
    if not conf_cols:
        raise KeyError("Agreement frame has no per-family confidence_<fam> columns to gate on.")
    merged = crossmatch_to_v41(base.rename(columns={"consensus_label": "predicted_label"}), v41_df)
    merged["real_class1_norm"] = merged["real_class1"].replace(MERGE_FINE_CLASSES)  # ssrq/css -> FSRQ-like, as in training
    merged["predicted_label_norm"] = merged["predicted_label"].astype(str).str.strip().str.lower()
    eval_mask = _resolved_eval_mask(merged).values
    n_all_agree, n_total = len(base), len(agreement_df)

    rows = []
    for thr in [0.0] + list(thresholds):
        conf_ok = np.logical_and.reduce([merged[c].values >= thr for c in conf_cols])
        n_pass_conf_all = int(conf_ok.sum())
        sub = merged[eval_mask & conf_ok]
        m = _accuracy_metrics(sub["real_class1_norm"].values, sub["predicted_label_norm"].values)
        m.update({"confidence_threshold": thr, "n_pass_all3_confidence": n_pass_conf_all,
                   "n_all_agree": n_all_agree, "n_total_predicted": n_total,
                   "coverage_of_all_agree": (n_pass_conf_all / n_all_agree) if n_all_agree else float("nan"),
                   "coverage_of_all_predicted": (n_pass_conf_all / n_total) if n_total else float("nan")})
        rows.append(m)
    cols = ["confidence_threshold", "n", "n_pass_all3_confidence", "n_all_agree", "n_total_predicted",
            "coverage_of_all_agree", "coverage_of_all_predicted", "accuracy", "accuracy_ci_low",
            "accuracy_ci_high", "balanced_accuracy", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high"]
    return pd.DataFrame(rows)[cols]


# %% [markdown]
# ### Section 7.4 — Driver (Stage 10)

# %%
def run_stage10_v41_prospective_validation(unknown_source_predictions: dict, corrected_winner_family: str) -> dict:
    logger = get_section_logger("stage10_v41_prospective_validation", VALIDATION_LOGS_DIR)
    _dedupe_console_handlers(logger)
    logger.notice(f"\n{'=' * 70}\nSTAGE 10 — prospective validation vs v41 (FY16 precursor of 5FGL)\n{'=' * 70}")

    if not V41_FITS_PATH.exists():
        logger.warning(f"  ⚠ v41 FITS not found at {V41_FITS_PATH} — skipping Stage 10 entirely.")
        return {}

    valid_predictions = {fam: df for fam, df in unknown_source_predictions.get("predictions", {}).items()
                          if df is not None}
    if corrected_winner_family not in valid_predictions:
        logger.warning(f"  ⚠ Corrected winner family '{corrected_winner_family}' has no valid Stage-9 "
                        "predictions — skipping Stage 10 entirely.")
        return {}

    v41_df = load_v41_catalog(V41_FITS_PATH)
    best_pred_df = valid_predictions[corrected_winner_family]
    _verify_dr4_v35_provenance(best_pred_df, DR4_V35_FITS_PATH)

    out = {"best_model_family": corrected_winner_family}

    logger.notice(f"\n  --- [A] best model only ('{corrected_winner_family}') by confidence threshold ---")
    best_thr_df, best_merged = evaluate_best_model_thresholds(best_pred_df, v41_df, corrected_winner_family)
    best_thr_path = VALIDATION_CSV_DIR / f"v41_validation_best_model_{corrected_winner_family}_by_threshold.csv"
    best_thr_df.to_csv(best_thr_path, index=False)
    # Per source (which v41 source it matched, its v41 class, our label and confidence) and per class.
    per_source_path = VALIDATION_CSV_DIR / f"v41_crossmatch_per_source_{corrected_winner_family}.csv"
    best_merged.to_csv(per_source_path, index=False)
    per_class = v41_per_class(best_merged)
    per_class.to_csv(VALIDATION_CSV_DIR / f"v41_validation_per_class_{corrected_winner_family}.csv")
    logger.notice(f"  per-class (v41 class: n, correct, n with confidence >= {HIGH_CONFIDENCE_EXPORT_THRESHOLD}, "
                  f"correct among them):\n{per_class.to_string()}\n  per source -> {per_source_path}")
    logger.notice("\n" + best_thr_df.to_string(index=False))
    logger.notice(f"  -> {best_thr_path}")
    out["best_model_by_threshold"] = best_thr_df
    out["best_model_by_threshold_path"] = str(best_thr_path)

    if len(valid_predictions) < 3:
        logger.warning("  ⚠ Fewer than 3 families have valid predictions (or FAMILIES has fewer than 3) — "
                        "skipping [B]/[C] ('all 3 agree') scenarios.")
        return out

    agreement_df = _build_agreement_frame_with_positions(valid_predictions)

    logger.notice("\n  --- [B] all 3 models agree (no extra confidence gate) ---")
    agree_metrics, _ = evaluate_full_agreement(agreement_df, v41_df)
    agree_path = VALIDATION_CSV_DIR / "v41_validation_all3_agree_summary.csv"
    pd.DataFrame([agree_metrics]).to_csv(agree_path, index=False)
    logger.notice(f"  n_all_agree={agree_metrics['n_sources_all_agree']}/{agree_metrics['n_total_predicted']} "
                  f"({agree_metrics['coverage_of_all_predicted']:.1%}) | evaluable(matched,resolved,no-bcu) "
                  f"n={agree_metrics['n']} | accuracy={agree_metrics['accuracy']:.4f} | "
                  f"balanced_accuracy={agree_metrics['balanced_accuracy']:.4f}")
    logger.notice(f"  -> {agree_path}")
    out["all3_agree_summary"] = agree_metrics
    out["all3_agree_summary_path"] = str(agree_path)

    logger.notice("\n  --- [C] all 3 models agree AND each model's own confidence clears a threshold ---")
    agree_thr_df = evaluate_full_agreement_by_confidence(agreement_df, v41_df, CONFIDENCE_SUMMARY_THRESHOLDS)
    agree_thr_path = VALIDATION_CSV_DIR / "v41_validation_all3_agree_by_confidence.csv"
    agree_thr_df.to_csv(agree_thr_path, index=False)
    logger.notice("\n" + agree_thr_df.to_string(index=False))
    logger.notice(f"  -> {agree_thr_path}")
    out["all3_agree_by_threshold"] = agree_thr_df
    out["all3_agree_by_threshold_path"] = str(agree_thr_path)

    return out


_V41_VALIDATION_CHECKPOINT_PATH = VALIDATION_CHECKPOINTS_DIR / "v41_prospective_validation.joblib"
if _V41_VALIDATION_CHECKPOINT_PATH.exists() and not FORCE_RECOMPUTE_V41_VALIDATION:
    _module_logger.notice(f"  -> loading v41 prospective validation from checkpoint -> "
                           f"{_V41_VALIDATION_CHECKPOINT_PATH}")
    V41_PROSPECTIVE_VALIDATION = joblib.load(_V41_VALIDATION_CHECKPOINT_PATH)
else:
    _v41_corrected_winner_family = _load_corrected_winner_family()
    V41_PROSPECTIVE_VALIDATION = run_stage10_v41_prospective_validation(
        UNKNOWN_SOURCE_PREDICTIONS, _v41_corrected_winner_family)
    joblib.dump(V41_PROSPECTIVE_VALIDATION, _V41_VALIDATION_CHECKPOINT_PATH)
    _module_logger.notice(f"  -> checkpoint saved -> {_V41_VALIDATION_CHECKPOINT_PATH}")

# %%
print("=" * 78)
print("Stage 10 (v41 / FY16 prospective validation) — COMPLETE.")
if V41_PROSPECTIVE_VALIDATION:
    print(f"  Best-model family scored : {V41_PROSPECTIVE_VALIDATION.get('best_model_family')}")
    for _key, _label in [("best_model_by_threshold_path", "[A] best-model-by-threshold"),
                          ("all3_agree_summary_path", "[B] all-3-agree summary"),
                          ("all3_agree_by_threshold_path", "[C] all-3-agree-by-confidence")]:
        if _key in V41_PROSPECTIVE_VALIDATION:
            print(f"  {_label:32s} -> {V41_PROSPECTIVE_VALIDATION[_key]}")
else:
    print("  Skipped (see warnings above).")
print("=" * 78)

# %% [markdown]
# ## Section 8 — Threshold-accuracy curves: winner model vs. all-3-agree

# %%
# %% [markdown]
# ## Section 8 — Threshold-accuracy curves: winner model vs. all-3-agree (Stage 10b)
#
# Pure visualization of numbers Stage 10 already computed — no recomputation,
# no new statistics. Two figures (`accuracy`, `balanced_accuracy`), each with
# two lines read straight off Stage 10's own per-threshold tables:
#
#   - **winner model only**  <- `V41_PROSPECTIVE_VALIDATION["best_model_by_threshold"]`
#     (scenario [A], `evaluate_best_model_thresholds`, Section 7.2)
#   - **all 3 models agree** <- `V41_PROSPECTIVE_VALIDATION["all3_agree_by_threshold"]`
#     (scenario [C]'s threshold sweep over the scenario-[B] agreement pool,
#     `evaluate_full_agreement_by_confidence`, Section 7.3)
#
# Both tables already share the same `confidence_threshold` grid
# (`[0.0] + CONFIDENCE_SUMMARY_THRESHOLDS`) and the same Setup-B/no-'bcu'
# scoring (`_resolved_eval_mask`), so the two curves are directly comparable.

# %%
VALIDATION_PLOTS_DIR = VALIDATION_DIR / "plots"
VALIDATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _annotate_n_correct(ax, df: pd.DataFrame, metric_col: str, color: str,
                         y_offset_pts: float, va: str = "bottom") -> None:
    """Labels each marker with the integer count of sources classified
    correctly at that threshold (round(n * accuracy)) — the count is the same
    regardless of which metric ('accuracy' or 'balanced_accuracy') is on the
    y-axis, since balanced accuracy is a per-class-averaged rate, not an
    alternate correct-count. `y_offset_pts`/`va` control placement above
    (positive offset, va='bottom') or below (negative offset, va='top') the
    marker."""
    for _, row in df.iterrows():
        if pd.isna(row["n"]) or pd.isna(row["accuracy"]) or row["n"] == 0:
            continue
        n_correct = int(round(row["n"] * row["accuracy"]))
        ax.annotate(str(n_correct), xy=(row["confidence_threshold"], row[metric_col]),
                    xytext=(0, y_offset_pts), textcoords="offset points",
                    ha="center", va=va, fontsize=8, color=color, fontweight="bold")
        
        
        

def plot_threshold_accuracy_curve(best_thr_df: Optional[pd.DataFrame], agree_thr_df: Optional[pd.DataFrame],
                                   metric_col: str, winner_family: str, plots_dir: Path,
                                   filename: str, logger=None) -> Optional[Path]:
    """Line plot of `metric_col` ('accuracy' or 'balanced_accuracy') vs.
    `confidence_threshold`, one line for scenario [A] (winner family only) and
    one for scenario [C]'s sweep over the scenario-[B] all-3-agree pool. Skips
    (returns None) if neither table is usable — e.g. Stage 10 was skipped
    (no v41 FITS) or fewer than 3 families had valid Stage-9 predictions."""
    have_best = best_thr_df is not None and not best_thr_df.empty and metric_col in best_thr_df.columns
    have_agree = agree_thr_df is not None and not agree_thr_df.empty and metric_col in agree_thr_df.columns
    if not have_best and not have_agree:
        msg = f"  ⚠ No usable threshold data for '{metric_col}' — skipping plot."
        (logger.warning(msg) if logger is not None else print(msg))
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    if have_best:
        ax.plot(best_thr_df["confidence_threshold"], best_thr_df[metric_col], marker="o",
                label=f"winner model only ({winner_family})", color="indianred",
                linewidth=2, markersize=6)
        _annotate_n_correct(ax, best_thr_df, metric_col, color="indianred",
                             y_offset_pts=-10, va="top")
    if have_agree:
        ax.plot(agree_thr_df["confidence_threshold"], agree_thr_df[metric_col], marker="s",
                label="all 3 models agree", color="steelblue", linewidth=2, markersize=6)
        _annotate_n_correct(ax, agree_thr_df, metric_col, color="steelblue",
                             y_offset_pts=8, va="bottom")

    metric_label = metric_col.replace("_", " ").title()
    ax.set_xlabel("Confidence threshold", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"{metric_label} vs. confidence threshold\n"
                 "FY16 catalog source prediction\n"
                 "(labels: # sources correctly classified)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    plt.tight_layout()

    out_path = plots_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    msg = f"  Saved -> {out_path}"
    (logger.notice(msg) if logger is not None else print(msg))
    return out_path


def run_stage10b_threshold_accuracy_plots(v41_prospective_validation: dict) -> dict:
    logger = get_section_logger("stage10b_threshold_accuracy_plots", VALIDATION_LOGS_DIR)
    _dedupe_console_handlers(logger)
    logger.notice(f"\n{'=' * 70}\nSTAGE 10b — threshold-accuracy curves "
                  f"(winner-only vs. all-3-agree)\n{'=' * 70}")

    if not v41_prospective_validation:
        logger.warning("  ⚠ V41_PROSPECTIVE_VALIDATION is empty (Stage 10 was skipped, e.g. no v41 "
                        "FITS found) — skipping Stage 10b entirely.")
        return {}

    winner_family = v41_prospective_validation.get("best_model_family", "?")
    best_thr_df = v41_prospective_validation.get("best_model_by_threshold")
    agree_thr_df = v41_prospective_validation.get("all3_agree_by_threshold")

    if agree_thr_df is None:
        logger.warning("  ⚠ No 'all3_agree_by_threshold' table (fewer than 3 families had valid "
                        "Stage-9 predictions) — plots below will show the winner-only curve alone.")

    out = {}
    for metric_col, fname in [("accuracy", "v41_threshold_accuracy_curve.png"),
                               ("balanced_accuracy", "v41_threshold_balanced_accuracy_curve.png")]:
        path = plot_threshold_accuracy_curve(best_thr_df, agree_thr_df, metric_col, winner_family,
                                              VALIDATION_PLOTS_DIR, fname, logger=logger)
        if path is not None:
            out[f"{metric_col}_plot_path"] = str(path)
    return out


THRESHOLD_ACCURACY_PLOTS = run_stage10b_threshold_accuracy_plots(V41_PROSPECTIVE_VALIDATION)

# %%
print("=" * 78)
print("Stage 10b (threshold-accuracy curves) — COMPLETE.")
if THRESHOLD_ACCURACY_PLOTS:
    for _key, _label in [("accuracy_plot_path", "accuracy vs. threshold"),
                          ("balanced_accuracy_plot_path", "balanced accuracy vs. threshold")]:
        if _key in THRESHOLD_ACCURACY_PLOTS:
            print(f"  {_label:32s} -> {THRESHOLD_ACCURACY_PLOTS[_key]}")
else:
    print("  Skipped (see warnings above).")
print("=" * 78)
