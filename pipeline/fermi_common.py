# %% [markdown]
# # fermi_common.py

# %%
"""
# fermi_common.py
Shared infrastructure — **Section 1** of the pipeline. Executed BEFORE
`fermi_processing.py` (Module 1), `fermi_fine.py` (Module 2),
`fermi_merged.py` (Module 3, 2-level), `fermi_3levels.py` (Module 3-alt),
`fermi_validation.py`, `fermi_inference.py` and `fermi_analysis.py`. All
modules run in one shared namespace (see `run_fermi.py`) — nothing here is
imported via `from fermi_common import ...`; every name defined below is
simply in-scope for whatever runs after it.

### What this file provides

- **Model registry** (`MODEL_SPECS`, `get_builder`) — GPU/CPU builders for
  every model family, switched by `USE_GPU`.
- **Six custom early-stopping estimators** (Section 1.4), all of which
  resolve their internal plateau/pruning criterion from the current
  `SELECTION_METRIC` at fit time (optional `stopping_metric` constructor
  parameter, default `None`) and expose `stopping_metric_used_`, which
  `get_fitted_summary` (Section 1.2) surfaces in every ledger row.
- **Model-selection funnel**: `select_and_refit_val_only` /
  `cv_select_and_refit` (Stage 0 scaler → Stage 1 feature set/PCA →
  Stage 2 weighting, coarse-then-fine), `cv_score_estimator`,
  `select_top_n_end_to_end`. HistGBC is scored and refit as the real,
  leakage-safe `StratifiedEarlyStoppingHGBC` for its OOF pass and
  train-refit, never touching a test set (Section 1.3g).
- **`dispatch_fit`** (Section 1.3e) — the one weighting-dispatch helper used
  by `score_on_val`, `_run_model_core`, the final refit and `fit_component`.
- **Feature selection**: `compute_selection_engine` (stability selection),
  `select_by_cumulative_power`.
- **Logging**: a custom `NOTICE` level (25, between `INFO` and `WARNING`)
  and `get_section_logger(label_set, logs_dir)` — a cached, per-`label_set`
  logger writing everything to `logs_dir/{label_set}.log` and propagating
  `NOTICE`+ to one shared console handler. The console shows errors,
  warnings, save confirmations and milestones; the full blow-by-blow (every
  coarse candidate, every fine-stage CV score) goes only to the log file.
  Functions that print accept an optional `logger`/`logs_dir`; without it
  they fall back to plain `print`.
- **Results I/O**: `save_result` / `save_result_multi` (atomic writes to one
  or several ledgers); `io_for(output_dir, split_id)` — the per-split I/O
  bundle that resolves per-branch results files
  (`OUTPUT_DIR/{split_id}/results/{label_set}.json`) and log files
  (`OUTPUT_DIR/{split_id}/logs/{label_set}.log`) from the `label_set` each
  call passes, next to the full-ledger `results/results_log.json`;
  `bootstrap_module_io` (single flat ledger); `publish_split_summary`
  (copies a module's final artifacts into
  `OUTPUT_DIR/results_aggregated/{split_id}/`); `load_all_results`
  (concatenates every per-branch results file for a split, or all splits).
- **Hyperparameter capture**: `select_and_refit_val_only`'s final-report row
  and `run_histgbc` include `fitted_summary` / `hyperparameters` /
  `normalization`.
- **Candidate-level logging**: `_select_winners` / `_coarse_then_fine` log
  EVERY candidate evaluated at every stage (one INFO line per coarse
  candidate — config plus score or FAILED — and per surviving fine-CV
  candidate), and each of Stage 0/1/2 emits one NOTICE line per model with
  what it decided.
- **Progress bars**: `pbar()` around every long loop this file owns.

### Known limitations (deliberate)

- The shared-namespace design (rather than importable Python modules) is
  kept; converting it would touch every file at once.
- The `_failed()` shape-sniffing sentinel pattern is correct but fragile.
- Two rare-path `print()` calls inside estimator `fit()` methods (the
  singleton-class notice in `StratifiedEarlyStoppingHGBC`, the adaptive-LOO
  notice in `DTreeCVPrunedClassifier`) stay plain prints: threading a logger
  into estimator constructors would touch every builder and call site.
- The two local-label-re-encoding implementations
  (`GBCWarmStartClassifier`'s inline fix and `_LocalLabelReencodeClassifier`)
  are not consolidated.
- CatBoost citation: Coronado-Blázquez (2022), *MNRAS* 515(2):1807–1822,
  arXiv:2207.04725.
- The GPU/CPU `sample_weight` non-equivalence (Section 1.0) needs a
  corresponding sentence in the paper's methods text.
"""

# %%
import gc
import json
import logging
import shutil
import sys
import time
import warnings
import inspect
from pathlib import Path
from typing import Literal, Optional, Union, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

# %%
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, cohen_kappa_score, accuracy_score,
    classification_report, confusion_matrix,
    precision_score, recall_score,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder   # needed by GBCWarmStartClassifier's local re-encoding
from sklearn.impute import IterativeImputer, SimpleImputer

# %%
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
"""
## Section 1.0 — Global constants
`SEED` and `VERSION_TAG` anchor reproducibility and catalog identity
respectively — every downstream module imports both rather than
restating them, so a run can never silently drift between modules
(e.g. Module 2 training against a different `VERSION_TAG`'s datasets
than Module 1 exported). `USE_GPU` is the single flag that swaps every
GPU-eligible model builder (RF/GBC/SVM/KNN/MLP/CatBoost) between its
CPU and CUDA implementation via `get_builder()` below — flipping it
requires no other code change anywhere in the pipeline.

`SELECTION_METRIC` is the one switch that decides which metric every
model-selection decision in every module sorts/argmaxes on. It
intentionally does not include `"kappa"` as an option — kappa and
macro-F1 remain diagnostic-only outputs, never sort keys, anywhere in
the pipeline, by design. Physically, balanced accuracy is
preferred as the default because 4FGL source classes are severely
imbalanced (a handful of `hmb`/`nov`/`sey`-scale rare classes against
hundreds of `bll`/`fsrq`), so raw accuracy can be dominated by whichever
class has the most rows; `SELECTION_METRIC="accuracy"` is kept available
as an explicit alternative run for comparison, not because it's
physically preferred. `SELECTION_METRIC` is also read, dynamically, by
all six custom early-stopping estimators in Section 1.4 (each one's
`stopping_metric` constructor parameter defaults to `None` → "use
whatever `SELECTION_METRIC` currently is at fit time"), so it is the one
switch read at every selection point, including each model family's own
internal capacity/pruning decision. (`run_fermi.py` sets it to macro-F1:
balanced accuracy ignores false positives and rewarded sending large
classes into rare ones.)

## Note — GPU vs. CPU `sample_weight` semantics are NOT equivalent

`sample_weight` is not a generic sklearn-API guarantee — it is
implemented per-estimator, and GPU backends (cuML here) frequently lag
or diverge from their CPU counterparts on this parameter because a
weighted variant of the underlying kernel is harder to vectorize on GPU
(weighted SMO for SVM). One backend in `MODEL_SPECS` is affected:
**SVM (GPU/cuML)** uses native weighting when the installed cuML version
supports it (detected once at import time via `inspect.signature`, not
probed per-fit) and falls back to a weighted bootstrap resample
(`resample_by_weight`) otherwise — an approximation of *which rows
exist*, not *how much each row counts*. A weighted SVM row selected under
`USE_GPU=True` is therefore not strictly apples-to-apples with the
equivalent CPU row; say so in the paper's methods text if SVM is reported.
**RandomForest always runs on CPU** (`RFPlateauClassifier`, native
per-row sample weights) — there is no cuML RF path (CUDA
illegal-memory-access errors, and its resample fallback inflated the OOB
stopping score). Every other entry (HistGBC, GBC, DecisionTree,
CatBoost, LDA/LogReg/KNN/MLP) is unaffected.
"""

# %%
def _cfg(name, default):
    """Settings hook for run_fermi.py: returns the master script's value for
    `name` if it set one (the `_FERMI_CONFIG` dict it injects into the shared
    namespace), else `default`. Without the master script (plain notebook
    run) every setting keeps its default below. Every module uses the same
    hook, so a setting given once in the master can't be silently
    re-assigned by a later file."""
    return globals().get("_FERMI_CONFIG", {}).get(name, default)


SEED = _cfg("SEED", 42)
np.random.seed(SEED)

# %%
"""
# Single point of catalog identity. Part of every output/processing folder
# name (outputs_{VERSION_TAG}_..., fermi_module1_processing_{VERSION_TAG}),
# so runs on different catalogs never share a folder.
"""
VERSION_TAG = _cfg("VERSION_TAG", "4FGL_DR4")

# %%
"""
# GPU flag — default off; every module inherits GPU-aware MODEL_SPECS
# builders for free by importing MODEL_SPECS from here (no per-module
# change).
"""
USE_GPU = _cfg("USE_GPU", True)

# %%
"""
# Read UNIFORMLY by every selection point in every downstream module —
# fine Setup A/B, Level-2/3 branches, blazar_subgate, Level-1 router,
# gate, nonblazar_router, AND the
# internal early-stopping/pruning criterion of all six custom estimators
# in Section 1.4. Kappa is intentionally excluded as an option: it is
# dropped from selection entirely (still computed/logged/printed).
"""
SELECTION_METRIC: Literal["accuracy", "bal_accuracy", "macro_f1"] = _cfg("SELECTION_METRIC", "bal_accuracy")

# %%
"""
# Shared ablation axes.
"""
PCA_BASE_TAG = "ALL_FEATURES"
PCA_VARIANTS = _cfg("PCA_VARIANTS", {"PCA95": 0.95, "PCA90": 0.90, "PCA80": 0.80})
WEIGHTING_MODES = _cfg("WEIGHTING_MODES", ["none", "balanced", "dampened", "balanced_uncapped"])
# Full registry — always complete so any saved config (e.g. a winner fit with
# "Robust") can still be rebuilt. Which scalers the SEARCH tries is SEARCH_SCALERS.
SCALER_VARIANTS = {
    "Standard": StandardScaler,
    "Robust": RobustScaler,
    "MinMax": MinMaxScaler,
}
DEFAULT_SCALER_NAME = _cfg("DEFAULT_SCALER_NAME", "Standard")

"""
# Search profile — applied in ONE place, at the top of
# select_and_refit_val_only (every fine/merged/3-level node search goes
# through it). None = no restriction (the full search).
#   SEARCH_MODELS       : model names searched at every node.
#   SEARCH_MODELS_EXTRA : {substring of label_set: [extra models]} — e.g.
#                         {"BINARY_SYSTEMS": ["LDA"]} adds LDA only on that branch.
#   SEARCH_SCALERS      : scaler names tried in Stage 0.
#   SEARCH_TAGS         : feature-set tags tried in Stage 1 (exact names).
# Weighting modes and PCA variants are WEIGHTING_MODES / PCA_VARIANTS above.
"""
SEARCH_MODELS = _cfg("SEARCH_MODELS", None)
SEARCH_MODELS_EXTRA = _cfg("SEARCH_MODELS_EXTRA", {})
SEARCH_SCALERS = _cfg("SEARCH_SCALERS", None)
SEARCH_TAGS = _cfg("SEARCH_TAGS", None)
"""
# Per-node checkpoint of select_and_refit_val_only's result, so a crashed or
# interrupted run resumes at the first unfinished node instead of the start of
# the module. None = off (notebook default); run_fermi.py sets it. Bump
# NODE_CACHE_VERSION after any code change that should invalidate old nodes.
"""
NODE_CACHE_DIR = _cfg("NODE_CACHE_DIR", None)
NODE_CACHE_VERSION = _cfg("NODE_CACHE_VERSION", 1)

# %%
FEATURE_POWER_THRESHOLDS = {"99": 0.99, "90": 0.90, "80": 0.80}
PHYSICS_CORE_TOP_N = 15

# %%
"""
# select_by_cumulative_power selects by importance-MASS fraction only —
# it has no idea how many training rows exist. This governs the HARD
# ceiling on selected feature count as n_train // MIN_SAMPLES_PER_FEATURE,
# independent of the power threshold.
"""
MIN_SAMPLES_PER_FEATURE = 5

# %%
"""
# Adaptive LOO thresholds for DTreeCVPrunedClassifier (and any other
# small-sample CV site that opts in).
"""
LOO_MIN_CLASS_COUNT_THRESHOLD = 5
LOO_MIN_BRANCH_TRAIN_THRESHOLD = 50

# %%
"""
# Stability selection by complementary pairs (CPSS; Shah & Samworth 2013).
# Each of STABILITY_N_PAIRS draws splits the training rows into two
# disjoint, class-stratified halves A and B; a HistGBC fit on A is scored
# by held-out permutation importance on B, and vice versa. That gives
# 2*STABILITY_N_PAIRS importance estimates on subsamples of size n/2 —
# the design Meinshausen & Bühlmann's (2010) selection frequencies and
# error bound are defined for. (The old scheme, 10 reshuffled 5-fold CV
# passes over the FULL training set, gave near-identical repeats, so the
# frequencies sat at 0 or 1 and the threshold controlled nothing.)
# Nodes with fewer than CPSS_MIN_N rows keep a single LOO/K-fold pass.
"""
STABILITY_N_PAIRS = _cfg("STABILITY_N_PAIRS", 25)
CPSS_MIN_N = 40
"""
# A feature must be selected by the per-subsample cumulative-power cut in
# at least this fraction of the 2*STABILITY_N_PAIRS half-sample fits.
# Meinshausen & Bühlmann (2010) recommend thresholds in 0.6-0.9; their
# bound E[V] <= q^2/((2*thr-1)*p) is logged per feature set.
"""
STABILITY_SELECTION_FRACTION = 0.7
"""
# z of the ranking score max(0, mean - z*std) over the half-sample fits.
# 0 = rank on the mean: the std across n/2 subsamples is the variability
# of a HALF-sample estimate, so mean - std would be an over-conservative
# second stability filter on top of the selection frequency, which is
# the principled one.
"""
IMPORTANCE_LCB_Z = 0.0
"""
# Hierarchical-clustering decorrelation cutoff (on 1-|Spearman rho|
# distance) applied to each threshold's selected feature set.
"""
CORR_CLUSTER_THRESHOLD = 0.85

# %%
"""
# Default split identifier used
# by every per-split IO helper (io_for, get_split_dir, ...) when a caller
# doesn't (yet) loop over multiple splits — which, today, is every
# caller: no downstream module currently loops over splits at all. Using
# one fixed, named split id (rather than a flat/no-split layout) keeps
# the on-disk shape "one-or-more split folders" always, so nothing here
# needs revisiting once real multi-split looping is wired into Modules
# 2-4 later.
"""
SPLIT_ID_DEFAULT = "split_00_primary"

# %%
"""
# Restated once, shared by every Module-3-family file — no module needs
# its own copy/docstring justifying the restatement. Hierarchy is based
# on gamma-ray EMISSION MECHANISMS, not morphology: BLAZAR_HE (external-
# Compton, soft PL, highly variable) vs. BLAZAR_LE (SSC-dominated, hard
# PL, low/moderate variability); AGN_OTHER (misaligned jets); PSR_ISOLATED
# (magnetospheric curvature radiation, ~zero variability); BINARY_SYSTEMS
# (orbitally modulated shock/wind/thermonuclear emission); SNR_PWN
# (hadronic/leptonic shock acceleration, extended, non-variable);
# GALACTIC_EXT (diffuse galactic/starburst CR interactions).
"""
# `fsrq` is "FSRQ-like" in the paper: fsrq + ssrq + css (merged upstream via
# MERGE_FINE_CLASSES; gamma-ray emission indistinguishable from features alone,
# although 4LAC lists css/ssrq as non-blazar AGN — disclose both).
# `spp` is deliberately NOT mapped: like `bcu` it marks an ambiguous association
# (SNR/PWN coincidence), not a class, so Setup B and everything downstream drop it;
# Setup A still keeps it.
TAXONOMY = {
    "fsrq": "BLAZAR_HE", "ssrq": "BLAZAR_HE", "css": "BLAZAR_HE",
    "bll": "BLAZAR_LE",
    "rdg": "AGN_OTHER", "nlsy1": "AGN_OTHER", "agn": "AGN_OTHER", "sey": "AGN_OTHER",
    "psr": "PSR_ISOLATED", "msp": "PSR_ISOLATED", "glc": "PSR_ISOLATED",
    "hmb": "BINARY_SYSTEMS", "lmb": "BINARY_SYSTEMS", "bin": "BINARY_SYSTEMS", "nov": "BINARY_SYSTEMS",
    "snr": "SNR_PWN", "pwn": "SNR_PWN",
    "sfr": "GALACTIC_EXT", "gal": "GALACTIC_EXT", "sbg": "GALACTIC_EXT",
}
SUPERCLASS_ORDER = [
    "BLAZAR_HE", "BLAZAR_LE", "AGN_OTHER",
    "PSR_ISOLATED", "BINARY_SYSTEMS", "SNR_PWN", "GALACTIC_EXT",
]

# %%
MLP_ARCHITECTURES = {
    "MLP_1x64":              (64,),
    "MLP_1x256":             (256,),
    "MLP_2x128_64":          (128, 64),
    "MLP_3x256_128_64":      (256, 128, 64),
    "MLP_2x512_256":         (512, 256),
    "MLP_4x512_256_128_64":  (512, 256, 128, 64),
}

# %%
"""
## Section 1.0b — Logging infrastructure

Without filtering, every module's console would get an unfiltered stream
of `print()` calls, with a coarse-survivor count and an "every model failed"
alert printed identically. Logging uses Python's standard `logging`
module with one custom level, `NOTICE` (25, between
`INFO`/20 and `WARNING`/30), used for the four categories that should
still reach the console — milestones/progress narration, save/upload
confirmations, plus genuine `WARNING`/`ERROR` records at their own
levels — while everything else (`DEBUG`/`INFO`: every coarse candidate
score, every intermediate diagnostic) is written ONLY to that branch's
log file.

`get_section_logger(label_set, logs_dir)` returns one cached
`logging.Logger` per `(logs_dir, label_set)` pair: a `FileHandler` at
`DEBUG` (the file is the complete record) plus propagation up to one
shared `"fermi"` root logger carrying a single console `StreamHandler`
floored at `NOTICE` (so the console only ever shows `NOTICE`/`WARNING`/
`ERROR`/`CRITICAL`, from every branch, without each branch needing its
own console handler). Every function in this file that produced console
noise now accepts an **optional** `logger` (or `logs_dir`) parameter;
passing nothing falls back to plain `print`.
"""

# %%
NOTICE_LEVEL_NUM = 25
logging.addLevelName(NOTICE_LEVEL_NUM, "NOTICE")


# %%
def _notice(self, message, *args, **kwargs):
    if self.isEnabledFor(NOTICE_LEVEL_NUM):
        self._log(NOTICE_LEVEL_NUM, message, args, **kwargs)


# %%
logging.Logger.notice = _notice  # noqa: attach once, module import time

# %%
_FERMI_ROOT_LOGGER_CONFIGURED = False
_SECTION_LOGGER_CACHE: dict = {}


# %%
def _safe_filename(*parts) -> str:
    """Turns
    an arbitrary label_set/model_name/etc. into a filesystem-safe token."""
    return "_".join(str(p) for p in parts).replace("/", "-").replace(" ", "_")


# %%
def _configure_fermi_root_logger(console_level: int = NOTICE_LEVEL_NUM) -> logging.Logger:
    """One-time setup of the shared '"fermi"' root logger's console
    handler. Idempotent — safe to call from every get_section_logger()
    invocation without accumulating duplicate handlers."""
    global _FERMI_ROOT_LOGGER_CONFIGURED
    root = logging.getLogger("fermi")
    root.setLevel(logging.DEBUG)  # handlers, not the logger itself, decide what's shown where
    if not _FERMI_ROOT_LOGGER_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(console_level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        root.propagate = False
        _FERMI_ROOT_LOGGER_CONFIGURED = True
    return root


# %%
def get_section_logger(label_set: str, logs_dir: Path,
                        console_level: int = NOTICE_LEVEL_NUM) -> logging.Logger:
    """Cached per-(logs_dir, label_set) logger. File handler captures
    everything (DEBUG+) into logs_dir/{label_set}.log; NOTICE+ also
    propagates to the shared console handler. `label_set="_module"` (by
    convention) is used for module-wide banners/section headers that
    aren't tied to one branch — see io_for below."""
    _configure_fermi_root_logger(console_level)
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{_safe_filename(label_set)}.log"
    cache_key = str(log_path)
    if cache_key in _SECTION_LOGGER_CACHE:
        return _SECTION_LOGGER_CACHE[cache_key]

    logger = logging.getLogger(f"fermi.{cache_key}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)
    _SECTION_LOGGER_CACHE[cache_key] = logger
    return logger


# %%
"""
## Section 1.1 — Results ledger

`results_log.json` is the single append-only record every module's
`fit_and_log`/`run_histgbc`/`safe_run` writes into, keyed by a
deterministic `run_id` (`label_set__model_name__feature_set__variant`)
so re-running a cell overwrites its own prior entry rather than
duplicating it. Writes are atomic (`tmp` file + `os.replace`) so a
killed process never leaves a half-written, unparseable ledger behind.
`save_result` takes an optional `logger` parameter so
its confirmation line can be routed to a branch's log file (as well as
the console, via `NOTICE`) instead of only `print`ed; passing nothing
falls back to `print`. `save_result_multi` is a thin
multi-path wrapper used by the new per-split/per-branch layout (§1.3g,
§1.6, §1.10 below) to keep the full split aggregator AND a per-branch
file in sync from one call site, without `save_result` itself ever
needing to know about more than one path.
"""


# %%
def make_results_path(output_dir: Path) -> Path:
    return output_dir / "results_log.json"


# %%
def load_results(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            backup = path.with_suffix(f".json.corrupted_{int(time.time())}")
            path.rename(backup)
            print(f"  ⚠ {path} was corrupted ({e}) — backed up to {backup}, "
                  "starting a fresh ledger. Recover the backup manually if needed.")
            return {}
    return {}


# %%
def save_result(run_id: str, entry: dict, path: Path,
                 logger: Optional[logging.Logger] = None) -> None:
    """Atomic write: dump to a sibling temp file, then os.replace() onto
    the real path. json.dump'ing directly into the target file leaves a
    truncated/corrupted ledger if the process dies mid-write (kernel
    restart, OOM kill, Ctrl-C) — os.replace is a single atomic filesystem
    rename, so readers only ever see either the fully old or fully new
    file, never a half-written one.

    `logger`, if given, receives the confirmation line at NOTICE
    (console + that branch's log file) instead of a bare `print` — one
    of the console categories (errors, warnings, save confirmations, milestones).
    Omit it to just `print`."""
    import os
    ledger = load_results(path)
    ledger[run_id] = {**entry, "timestamp": datetime.now(timezone.utc).isoformat()}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(ledger, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    msg = f"  ✓ Logged '{run_id}' → {path}"
    if logger is not None:
        logger.notice(msg)
    else:
        print(msg)


# %%
def save_result_multi(run_id: str, entry: dict, paths: Sequence[Path],
                       logger: Optional[logging.Logger] = None) -> None:
    """Writes the same entry to every path in
    `paths` via the existing, unchanged `save_result` — one call site,
    several ledgers (typically: the split's full `results_log.json`
    aggregator/backup, plus that branch's own per-label_set file). A
    single-element `paths` list reproduces exactly one `save_result`
    call."""
    for path in paths:
        save_result(run_id, entry, path, logger=logger)


# %%
def make_run_id(label_set: str, model_name: str, feature_set: str, variant: str) -> str:
    return f"{label_set}__{model_name}__{feature_set}__{variant}"


# %%
def results_to_dataframe(path: Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Prints (or logs, at NOTICE) a message on an empty ledger rather
    than failing silently."""
    ledger = load_results(path)
    if not ledger:
        msg = "  (ledger is empty — run at least one model subsection first)"
        (logger.notice(msg) if logger is not None else print(msg))
        return pd.DataFrame()
    return pd.DataFrame.from_dict(ledger, orient="index").reset_index(drop=True)


# %%
def load_all_results(output_dir: Path, split_id: Optional[str] = None) -> pd.DataFrame:
    """Concatenates every per-branch results
    file for one split (or every split under `output_dir`, if `split_id`
    is omitted) into one DataFrame, tagging each row with the branch
    file it came from. Falls back to that split's `results_log.json`
    full aggregator when no per-branch files exist yet (e.g. output
    produced through the flat bootstrap_module_io path, or a
    split where io_for's per-branch files simply haven't been created
    because no model has finished training there yet) — so this always
    returns *something* sensible rather than an empty frame the moment a
    caller hasn't migrated to the new per-branch layout."""
    output_dir = Path(output_dir)
    split_dirs = ([output_dir / split_id] if split_id is not None
                   else sorted(d for d in output_dir.glob("split_*") if d.is_dir()))
    frames = []
    for split_dir in split_dirs:
        results_dir = split_dir / "results"
        branch_files = sorted(p for p in results_dir.glob("*.json")
                               if p.name != "results_log.json") if results_dir.exists() else []
        if branch_files:
            for bf in branch_files:
                df = results_to_dataframe(bf)
                if not df.empty:
                    df["_split_id"] = split_dir.name
                    df["_source_file"] = bf.name
                    frames.append(df)
        else:
            agg_path = results_dir / "results_log.json" if results_dir.exists() else split_dir / "results_log.json"
            df = results_to_dataframe(agg_path)
            if not df.empty:
                df["_split_id"] = split_dir.name
                df["_source_file"] = agg_path.name
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# %%
"""
## Section 1.2 — Fitted-hyperparameter capture

Several custom estimators below select a hyperparameter *during* `fit`
rather than accepting it at construction time — `DTreeCVPrunedClassifier`
picks `ccp_alpha` by internal CV, `StratifiedEarlyStoppingHGBC` picks
`n_iter_` by validation plateauing, `AdaptiveKNNClassifier` clamps `k`
to the actual training-set size, and so on. None of that is visible from
`model.get_params()`, so `get_fitted_summary()` reaches into the fitted
estimator's own post-fit attributes and surfaces them into the results
ledger / `model_metadata.txt`. Also now captures
`stopping_metric_used_` (§5.5 — which metric governed that estimator's
internal plateau/pruning decision), when present.

`_normalization_type`/`_hyperparams` are promoted to plain module-level
functions so both
`RunTelemetry.track()` and `select_and_refit_val_only`'s final-report
row (§1.3g, closing the hyperparameter-logging gap per design-doc §5)
share one implementation instead of two. `RunTelemetry` keeps its own
`@staticmethod` wrappers delegating to these, so any existing code
calling `RunTelemetry._normalization_type(...)`/`RunTelemetry._hyperparams(...)`
directly keeps working unchanged.
"""


# %%
def get_fitted_summary(model) -> dict:
    """Extracts POST-FIT (not just constructor) hyperparameters:
      - DTreeCVPrunedClassifier.best_alpha_
      - AdaptiveKNNClassifier's actual clamped k (self.model_.n_neighbors,
        not self.n_neighbors)
      - RFPlateauClassifier.n_estimators_
      - GBCWarmStartClassifier.n_estimators_
      - MLPLeakageSafeClassifier(.*Torch).n_epochs_ / best_val_f1_
      - StratifiedEarlyStoppingHGBC.n_iter_ / n_singleton_classes_
      - stopping_metric_used_ on any of the six custom estimators
    Works whether `model` is a bare estimator or the last step of a
    sklearn Pipeline. Returns {} for anything with none of these attrs
    (e.g. plain SVC/LogisticRegression/CatBoost)."""
    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    summary = {}

    if hasattr(est, "best_alpha_"):
        summary["ccp_alpha_selected"] = float(est.best_alpha_)
    if isinstance(est, AdaptiveKNNClassifier) and hasattr(est, "model_"):
        summary["n_neighbors_clamped"] = int(est.model_.n_neighbors)
    if hasattr(est, "n_estimators_"):
        summary["n_estimators_selected"] = int(est.n_estimators_)
    if hasattr(est, "n_epochs_"):
        summary["n_epochs_selected"] = int(est.n_epochs_)
    if hasattr(est, "best_val_f1_"):
        val = est.best_val_f1_
        summary["best_val_f1"] = float(val) if val is not None else None
    if hasattr(est, "n_iter_") and est.n_iter_ is not None:
        n_iter_val = np.asarray(est.n_iter_).ravel()
        if n_iter_val.size == 1:
            summary["n_iter_selected"] = int(n_iter_val[0])
        elif n_iter_val.size > 1:
            summary["n_iter_selected"] = int(n_iter_val.max())
    if hasattr(est, "n_singleton_classes_") and est.n_singleton_classes_ is not None:
        summary["n_singleton_classes"] = int(est.n_singleton_classes_)
    if hasattr(est, "stopping_metric_used_") and est.stopping_metric_used_ is not None:
        summary["stopping_metric_used"] = str(est.stopping_metric_used_)

    return summary
def _normalization_type(model) -> str:
    if isinstance(model, Pipeline):
        names = [type(step).__name__ for _, step in model.steps]
        imputers = [n for n in names if "Imputer" in n]
        scalers = [n for n in names if n in
                   ("StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer")]
        pca = ["PCA"] if "PCA" in names else []
        parts = ["+".join(imputers)] if imputers else []
        parts += ["+".join(scalers)] if scalers else []
        parts += pca
        return " -> ".join(parts) if parts else "none (raw pass-through)"
    return "none (native NaN-routing, no pipeline)"
def _hyperparams(model) -> dict:
    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    try:
        return est.get_params(deep=False)
    except Exception:
        return {}
class RunTelemetry:
    """Wraps fit_and_log() to enrich each results_log.json entry (and
    every other results file it was also written to — see
    results_paths below) with normalization/hyperparameter/feature
    telemetry AND the fitted-summary hook above, then regenerates
    model_metadata.txt from the PRIMARY (first) ledger."""

    def __init__(self, metadata_path: Path, results_path: Union[Path, Sequence[Path]] = None,
                 results_paths: Optional[Sequence[Path]] = None):
        """Accepts either a single `results_path` kwarg
        or a `results_paths`
        sequence — the first path in the resolved list is treated as the
        PRIMARY ledger (the one model_metadata.txt is generated from);
        every path in the list gets the same telemetry fields written."""
        self.metadata_path = metadata_path
        if results_paths is not None:
            self.results_paths = list(results_paths)
        elif isinstance(results_path, (list, tuple)):
            self.results_paths = list(results_path)
        elif results_path is not None:
            self.results_paths = [results_path]
        else:
            raise ValueError("RunTelemetry requires results_path or results_paths")

    @staticmethod
    def _normalization_type(model) -> str:
        return _normalization_type(model)

    @staticmethod
    def _hyperparams(model) -> dict:
        return _hyperparams(model)

    def _write_manifest(self, ledger: dict):
        lines = ["MODEL METADATA MANIFEST",
                 f"Generated: {datetime.now(timezone.utc).isoformat()}", "=" * 78, ""]
        for run_id, e in ledger.items():
            train_line = (
                f"  Train metrics : accuracy={e.get('train_accuracy', 'n/a')}, "
                f"bal_accuracy={e.get('train_bal_accuracy', 'n/a')}, "
                f"macro_f1={e.get('train_macro_f1', 'n/a')}, kappa={e.get('train_kappa', 'n/a')}"
            )
            test_line = (
                f"  Test metrics  : accuracy={e.get('test_accuracy', e.get('accuracy', 'n/a'))}, "
                f"bal_accuracy={e.get('test_bal_accuracy', e.get('bal_accuracy', 'n/a'))}, "
                f"macro_f1={e.get('test_macro_f1', e.get('macro_f1', 'n/a'))}, "
                f"kappa={e.get('test_kappa', e.get('kappa', 'n/a'))}"
            )
            lines += [
                f"[{run_id}]",
                f"  Normalization : {e.get('normalization', 'n/a')}",
                f"  Feature set   : {e.get('feature_set', 'n/a')} ({e.get('n_features', '?')} features)",
                f"  Hyperparams   : {json.dumps(e.get('hyperparameters', {}), default=str)}",
                f"  Fitted values : {json.dumps(e.get('fitted_summary', {}), default=str)}",
                train_line,
                test_line,
                f"  Fit time (s)  : {e.get('fit_seconds', 'n/a')}",
                f"  Stopping rule : {e.get('stopping_rule', 'n/a')}",
                "",
            ]
        with open(self.metadata_path, "w") as f:
            f.write("\n".join(lines))

    def track(self, fit_and_log_fn):
        def wrapped(*args, **kwargs):
            model, metrics = fit_and_log_fn(*args, **kwargs)
            bound = inspect.signature(fit_and_log_fn).bind_partial(*args, **kwargs).arguments
            x_tr = bound.get("x_tr")
            run_id = make_run_id(bound.get("label_set"), bound.get("model_name"),
                                  bound.get("feature_set"), bound.get("variant"))
            extra = {
                "normalization": _normalization_type(model),
                "hyperparameters": _hyperparams(model),
                "fitted_summary": get_fitted_summary(model),
                "active_features": list(x_tr.columns) if hasattr(x_tr, "columns") else None,
            }
            primary_ledger = None
            for i, rp in enumerate(self.results_paths):
                ledger = load_results(rp)
                if run_id in ledger:
                    ledger[run_id].update(extra)
                    with open(rp, "w") as f:
                        json.dump(ledger, f, indent=2, default=str)
                if i == 0:
                    primary_ledger = ledger
            if primary_ledger is not None:
                self._write_manifest(primary_ledger)
            return model, metrics
        return wrapped


# %%
"""
## Section 1.3 — Weighting

Class imbalance in the 4FGL fine-grain taxonomy is severe (hundreds of
`bll`/`fsrq` vs. single-digit `hmb`/`nov`/`sey` counts in some setups),
so every model sweep tries four weighting regimes: `none` (baseline),
`balanced` (inverse-frequency, capped at `max_ratio`× the majority
class's weight so a handful of extreme outlier classes don't dominate
the loss), `dampened` (a damped power of `balanced`, softer), and
`balanced_uncapped` (inverse-frequency with no cap, letting rare classes
fully compete). Models without a native `sample_weight` (LDA, KNN)
instead get a weighted bootstrap resample via `resample_by_weight` — a
different mechanism reaching for the same effective goal.
"""


# %%
def class_weight_map(y, mode, damping=0.5, max_ratio=10.0):
    """`max_ratio` clamps relative to the *smallest* raw weight (the
    majority class), not the largest — `capped = min(w, min_w *
    max_ratio)`. `balanced_uncapped` is available to every caller."""
    classes, counts = np.unique(y, return_counts=True)
    if mode == "none":
        return {c: 1.0 for c in classes}
    n_total, n_classes = len(y), len(classes)
    if mode == "balanced":
        raw = {c: n_total / (n_classes * n) for c, n in zip(classes, counts)}
    elif mode == "balanced_uncapped":
        return {c: n_total / (n_classes * n) for c, n in zip(classes, counts)}
    elif mode == "dampened":
        raw = {c: (n_total / (n_classes * n)) ** damping for c, n in zip(classes, counts)}
    else:
        raise ValueError(f"Unknown weighting mode: {mode}")
    min_w = min(raw.values())
    return {c: min(w, min_w * max_ratio) for c, w in raw.items()}
def sample_weight_array(y, wmap):
    return np.array([wmap[v] for v in y])
def resample_by_weight(X, y, wmap, random_state=SEED):
    """Weighted bootstrap resample — fallback weighting mechanism for
    estimators with no native sample_weight (LDA, KNN)."""
    rng = np.random.RandomState(random_state)
    w = sample_weight_array(y, wmap)
    p = w / w.sum()
    idx = rng.choice(len(y), size=len(y), replace=True, p=p)
    X_res = X.iloc[idx].reset_index(drop=True) if hasattr(X, "iloc") else X[idx]
    return X_res, y[idx]
def _leakage_safe_val_split(y_arr, val_fraction, seed):
    """Per-class split guaranteeing every class keeps >=1 sample in FIT."""
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_arr)
    fit_idx, val_idx = [], []
    for cls in np.unique(y_arr):
        cls_idx = np.where(y_arr == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_val = int(np.floor(n * val_fraction))
        if n < 2 or n_val < 1:
            fit_idx.extend(cls_idx)
        else:
            val_idx.extend(cls_idx[:n_val])
            fit_idx.extend(cls_idx[n_val:])
    return np.array(sorted(fit_idx)), np.array(sorted(val_idx))


# %%
"""
## Section 1.3b — Constants
"""
COARSE_VAL_FRACTION = _cfg("COARSE_VAL_FRACTION", 0.15)
COARSE_SURVIVAL_FRACTION = _cfg("COARSE_SURVIVAL_FRACTION", 0.25)
COARSE_MIN_SURVIVORS = _cfg("COARSE_MIN_SURVIVORS", 3)
"""
# Every downstream module independently capped LOO at this same row
# count (a large branch with one 2-sample class would otherwise trigger
# an O(n) LeaveOneOut sweep per CV call) — one constant here instead of
# three copy-pasted module-local ones.
"""
MAX_LOO_N = 150

# %%
"""
## Section 1.3c — CV scoring primitive

`cv_score_estimator` pools every out-of-fold prediction and scores it
ONCE (`_pooled_oof_metrics`) instead of averaging n_folds per-fold
metrics. This matters specifically under adaptive LeaveOneOut: a
per-fold `balanced_accuracy_score` on a single held-out row is a
degenerate 0/1 draw, and `mean(fold_scores)` over those silently
collapses toward class-size-weighted accuracy while its std is pure
Bernoulli noise, not a genuine generalization estimate. Pooling first is
the standard OOF-CV fix; uncertainty comes from bootstrapping the pooled
(true, pred) pairs, which stays meaningful for any n_folds, including
n_folds=n (LOO). The naive fold-averaged number is kept alongside as
`fold_diagnostic_*` so a degenerate LOO fold-mean stays auditable
instead of silently vanishing. The fold loop is now wrapped in
`pbar()` — every
candidate's CV pass now shows live fold-by-fold progress instead of
running silently; this is a pure iteration-wrapper change, the scoring
math is untouched.
"""


# %%
def _score_by_name(metric_name, y_true, y_pred):
    if metric_name in ("bal_accuracy", "balanced_accuracy"):
        return balanced_accuracy_score(y_true, y_pred)
    if metric_name == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric_name == "macro_f1":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    raise ValueError(f"Unsupported metric_name for CV selection: {metric_name}")
def _failed(result) -> bool:
    """Single guard for every `safe_run_fn(...)` call site in this file.
    `safe_run` returns the literal 2-tuple (None, None) on a caught
    exception, REGARDLESS of what the wrapped function normally returns.
    Fragile but correct: a more robust sentinel type would touch every
    call site for no behavioral gain."""
    return result is None or (isinstance(result, tuple) and len(result) == 2
                               and result[0] is None and result[1] is None)
def _loo_structural_zero_classes(y):
    """Classes with exactly 1 training row. Under LOO such a class is
    NEVER present in any training fold, so every model scores 0% recall
    on it deterministically — a fixed offset shared by every candidate
    (safe for *ranking*), but one that makes the pooled LOO metric a
    conservative floor, not a forecast of the final refit's performance
    on that class."""
    classes, counts = np.unique(np.asarray(y), return_counts=True)
    return classes[counts < 2]
def _pooled_oof_metrics(oof_true, oof_pred, seed=SEED, n_bootstrap=1000, min_n_for_ci=5):
    """Score every out-of-fold prediction ONCE, pooled — see the section
    markdown above. Bootstraps the pooled (true, pred) pairs for an
    uncertainty estimate that stays meaningful even when n_folds == n
    (LOO)."""
    oof_true = np.asarray(oof_true)
    oof_pred = np.asarray(oof_pred)
    n = len(oof_true)
    point = {
        "accuracy": float(accuracy_score(oof_true, oof_pred)),
        "bal_accuracy": float(balanced_accuracy_score(oof_true, oof_pred)),
        "macro_f1": float(f1_score(oof_true, oof_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(oof_true, oof_pred)) if len(np.unique(oof_true)) > 1 else float("nan"),
    }
    if n < min_n_for_ci:
        return point, {k: float("nan") for k in point}

    rng = np.random.default_rng(seed)
    boot = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bt, bp = oof_true[idx], oof_pred[idx]
        if len(np.unique(bt)) < 2:
            continue
        boot["accuracy"].append(accuracy_score(bt, bp))
        boot["bal_accuracy"].append(balanced_accuracy_score(bt, bp))
        boot["macro_f1"].append(f1_score(bt, bp, average="macro", zero_division=0))
        boot["kappa"].append(cohen_kappa_score(bt, bp))
    boot_std = {k: (float(np.std(v)) if len(v) >= 20 else float("nan")) for k, v in boot.items()}
    return point, boot_std


def bootstrap_metric_ci(y_true, y_pred, score_fn, weights=None, n_boot=1000, confidence=0.95, seed=SEED):
    """Uncertainty of metrics computed on ONE fixed set of predictions (a test split, a pooled OOF set,
    a training refit). `score_fn(y_true, y_pred, weights) -> {metric: value}`.

    Every metric gets a class-stratified bootstrap over rows: rows are resampled within each true class,
    so every class keeps its count (Efron & Tibshirani 1993). Balanced accuracy (unweighted only)
    instead gets the posterior of Brodersen et al. (2010): the mean over classes of
    Beta(k_c+1, n_c-k_c+1). It also covers classes with a single row, which a stratified bootstrap
    resamples to themselves (zero variance). With classes of 1-3 rows that posterior is pulled toward
    0.5 (a recall of 1/1 is barely determined), so the observed value can lie outside the interval;
    `posterior_mean` is reported next to it. Both describe the sampling variability of the evaluated
    rows only, not of the training set (Bengio & Grandvalet 2004).
    Returns {metric: {"value", "std", "ci_low", "ci_high", "method"[, "posterior_mean"]}}."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    w = None if weights is None else np.asarray(weights, dtype=float)
    point = score_fn(y_true, y_pred, w)
    a = (1.0 - confidence) / 2.0
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix)) for ix in by_class])
        for k, v in score_fn(y_true[idx], y_pred[idx], None if w is None else w[idx]).items():
            draws[k].append(v)
    out = {k: {"value": float(point[k]), "std": float(np.nanstd(d, ddof=1)),
               "ci_low": float(np.nanquantile(d, a)), "ci_high": float(np.nanquantile(d, 1 - a)),
               "method": "stratified_bootstrap"} for k, d in draws.items()}
    if w is None:
        post = np.mean([rng.beta(np.sum(y_pred[ix] == y_true[ix]) + 1, np.sum(y_pred[ix] != y_true[ix]) + 1, 4000)
                        for ix in by_class], axis=0)
        # Only balanced accuracy: sklearn's macro recall also averages over classes that appear only
        # in y_pred (recall 0), so it is a different quantity when the two label sets differ.
        for k in ("bal_accuracy",):
            if k in out:
                out[k].update(std=float(post.std(ddof=1)), posterior_mean=float(post.mean()),
                              ci_low=float(np.quantile(post, a)),
                              ci_high=float(np.quantile(post, 1 - a)), method="beta_posterior_brodersen2010")
    return out


def clopper_pearson(k: int, n: int, confidence=0.95) -> tuple:
    """Exact binomial interval for k successes in n trials (Clopper & Pearson 1934); (nan, nan) for n=0."""
    if n == 0:
        return float("nan"), float("nan")
    from scipy.stats import binomtest
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=confidence, method="exact")
    return float(ci.low), float(ci.high)
def cv_score_estimator(builder_fn, build_kwargs, x, y, weighting, weight_support,
                        metric_name, n_splits=5,
                        loo_min_class_count=LOO_MIN_CLASS_COUNT_THRESHOLD,
                        seed=SEED, damping=0.5, max_ratio=10.0,
                        collect_oof=False, show_progress=True):
    """(Rare-class-aware) K-fold, or bounded LeaveOneOut, CV — see the
    section markdown for the pooled-OOF scoring rationale. Returns the
    full accuracy/bal_accuracy/macro_f1/kappa quartet plus 'mean'/'std'
    aliased to `metric_name`, always genuinely held-out. Every class with >= 2
    rows is validated (each of its rows in a different fold); only singleton
    classes ride along in every training fold and are absent from the pooled
    OOF score. (Making every class with fewer than `n_splits` rows
    train-only would make the CV score optimistic relative to the test
    score.) `collect_oof=True`
    additionally returns 'oof_true'/'oof_pred'/'oof_row_idx' for a
    confusion matrix that never touches test. `show_progress`
    wraps the fold loop in `pbar()` — set False to suppress (e.g. inside
    a caller that's already wrapping many of these calls in its own
    outer bar, to avoid nested-bar clutter)."""
    y = np.asarray(y)
    n_total = len(y)
    classes, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min())
    use_loo = min_class_count <= loo_min_class_count and n_total <= MAX_LOO_N

    if use_loo:
        cv_mode = "LeaveOneOut"
        fold_iter = LeaveOneOut().split(x, y)
        total_folds = n_total
    else:
        # Only singleton classes are train-only (one row cannot be both trained on and held out).
        # A class with 2..n_splits-1 rows is spread over that many folds, one row each, so each of
        # its rows is validated once by a model that still saw n_c - 1 >= 1 examples of it. The
        # pooled-OOF scoring below makes the uneven per-fold counts harmless (Forman & Scholz 2010).
        rare_mask = np.isin(y, classes[counts < 2])
        strat_idx = np.where(~rare_mask)[0]
        rare_idx = np.where(rare_mask)[0]
        y_strat = y[strat_idx]
        strat_max = int(np.unique(y_strat, return_counts=True)[1].max()) if len(strat_idx) else 0
        n_eff = max(2, min(n_splits, strat_max))
        cv = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)
        cv_mode = (f"StratifiedKFold(n_splits={n_eff}, "
                   f"{len(rare_idx)} singleton-class row(s) train-only)")
        total_folds = n_eff

        def _bounded_fold_iter():
            x_strat = x.iloc[strat_idx] if hasattr(x, "iloc") else x[strat_idx]
            for tr_local, va_local in cv.split(x_strat, y_strat):
                yield np.concatenate([strat_idx[tr_local], rare_idx]), strat_idx[va_local]
        fold_iter = _bounded_fold_iter()

    if show_progress:
        fold_iter = pbar(fold_iter, total=total_folds, desc=f"CV folds ({cv_mode})", leave=False)

    fold_scores, oof_true_parts, oof_pred_parts, oof_idx_parts = [], [], [], []
    for tr_idx, va_idx in fold_iter:
        x_tr_f = x.iloc[tr_idx] if hasattr(x, "iloc") else x[tr_idx]
        x_va_f = x.iloc[va_idx] if hasattr(x, "iloc") else x[va_idx]
        y_tr_f, y_va_f = y[tr_idx], y[va_idx]
        _, fitted = score_on_val(builder_fn, build_kwargs, x_tr_f, y_tr_f, x_va_f, y_va_f,
                                  weighting, weight_support, metric_name, seed=seed,
                                  damping=damping, max_ratio=max_ratio)
        y_pred_f = fitted.predict(x_va_f)
        fold_scores.append(_score_by_name(metric_name, y_va_f, y_pred_f))
        oof_true_parts.append(y_va_f)
        oof_pred_parts.append(y_pred_f)
        oof_idx_parts.append(va_idx)

    oof_true_all = np.concatenate(oof_true_parts)
    oof_pred_all = np.concatenate(oof_pred_parts)
    oof_idx_all = np.concatenate(oof_idx_parts)

    point, boot_std = _pooled_oof_metrics(oof_true_all, oof_pred_all, seed=seed)
    _metric_key = "bal_accuracy" if metric_name in ("bal_accuracy", "balanced_accuracy") else metric_name
    zero_recall_classes = (_loo_structural_zero_classes(y) if cv_mode == "LeaveOneOut"
                            else np.array([]))
    fold_scores_arr = np.array(fold_scores)

    result = {
        "mean": point[_metric_key], "std": boot_std[_metric_key],
        "n_folds": len(fold_scores_arr), "cv_mode": cv_mode,
        "accuracy_mean": point["accuracy"], "accuracy_std": boot_std["accuracy"],
        "bal_accuracy_mean": point["bal_accuracy"], "bal_accuracy_std": boot_std["bal_accuracy"],
        "macro_f1_mean": point["macro_f1"], "macro_f1_std": boot_std["macro_f1"],
        "kappa_mean": point["kappa"], "kappa_std": boot_std["kappa"],
        "fold_diagnostic_mean": float(fold_scores_arr.mean()) if len(fold_scores_arr) else float("nan"),
        "fold_diagnostic_std": float(fold_scores_arr.std()) if len(fold_scores_arr) else float("nan"),
        "oof_n": int(len(oof_true_all)),
        "n_structural_zero_recall_classes": int(len(zero_recall_classes)),
        "structural_zero_recall_classes": zero_recall_classes.tolist(),
    }
    if collect_oof:
        result["oof_true"] = oof_true_all
        result["oof_pred"] = oof_pred_all
        result["oof_row_idx"] = oof_idx_all
    return result


# %%
"""
## Section 1.3d — Train_core/val split
"""
def carve_train_core_val_split(x_tr, y_tr, val_fraction=COARSE_VAL_FRACTION,
                                seed=SEED, label="label_set", logger: Optional[logging.Logger] = None):
    """Per-class, leakage-safe. n<2 classes routed entirely into
    train_core. Returns (train_core_mask, val_mask, report) — masks are
    boolean numpy arrays aligned to x_tr's row order; apply the SAME masks
    to every tag's x_train_by_tag[tag] for this label_set. Optional
    `logger`: when given, the split-report line goes to NOTICE (console +
    file) instead of a bare print — this is exactly the kind of one-time,
    per-branch headline that belongs on the console."""
    y_arr = np.asarray(y_tr)
    n = len(y_arr)
    train_core_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    rng = np.random.RandomState(seed)
    singleton_classes = []
    for cls in np.unique(y_arr):
        idx = np.where(y_arr == cls)[0]
        if len(idx) < 2:
            train_core_mask[idx] = True
            singleton_classes.append(cls)
            continue
        idx_shuffled = rng.permutation(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        n_val = min(n_val, len(idx) - 1)
        val_mask[idx_shuffled[:n_val]] = True
        train_core_mask[idx_shuffled[n_val:]] = True

    report = {
        "label": label, "n_total": int(n),
        "n_train_core": int(train_core_mask.sum()), "n_val": int(val_mask.sum()),
        "val_fraction_requested": val_fraction,
        "singleton_classes_routed_to_train_core": [str(c) for c in singleton_classes],
    }
    msg = (f"  [{label}] train_core/val split: n_train_core={report['n_train_core']}  "
           f"n_val={report['n_val']}  singleton_classes={singleton_classes}")
    (logger.notice(msg) if logger is not None else print(msg))
    return train_core_mask, val_mask, report


# %%
"""
## Section 1.3e — Consolidated weighting-dispatch

`score_on_val`, `_run_model_core`, the final-refit block inside
`select_and_refit_val_only`, and `fit_component` all need the same
three-way branch: `weighting=="none"`
→ fit unweighted; `weight_support=="resample"` → weight-map, resample,
fit; otherwise → weight-map, sample-weight array, fit with
`sample_weight=`, further branching on bare-estimator vs. `Pipeline`
(needing the `f"{last_step_name}__sample_weight"` prefix). Per the
review (§5.6), that logic is now factored into one function,
`dispatch_fit`, and all four call sites delegate to it — a pure
refactor, no statistical behavior changes; the four copies were not
byte-for-byte identical before, which was itself the risk this removes.
"""


# %%
def dispatch_fit(builder, build_kwargs: dict, x, y, weighting: str, weight_support: str,
                  seed=SEED, damping=0.5, max_ratio=10.0, record_history: bool = False):
    """Builds a fresh estimator via
    `builder(**build_kwargs)` and fits it according to `weighting`/
    `weight_support`. `record_history=True` flips on the opt-in
    train/val-per-step trace on the five staged estimators in Section 1.4
    (MLP CPU/GPU, StratifiedEarlyStoppingHGBC, RFPlateauClassifier,
    GBCWarmStartClassifier) — a plain `hasattr` check, so this is a no-op
    for every other model family. Left False by default so the Stage
    0/1/2 search sweeps (the overwhelming majority of `dispatch_fit`
    calls) pay zero extra predict cost; only `select_and_refit_val_only`'s
    final per-model refit opts in."""
    est = builder(**build_kwargs)
    if record_history:
        _target = est.named_steps[est.steps[-1][0]] if hasattr(est, "steps") else est
        if hasattr(_target, "record_history"):
            _target.record_history = True
    if weighting == "none":
        est.fit(x, y)
        return est
    if weight_support == "resample":
        wmap = class_weight_map(y, weighting, damping=damping, max_ratio=max_ratio)
        x_res, y_res = resample_by_weight(x, y, wmap, random_state=seed)
        est.fit(x_res, y_res)
        return est
    wmap = class_weight_map(y, weighting, damping=damping, max_ratio=max_ratio)
    sw = sample_weight_array(y, wmap)
    if hasattr(est, "steps"):
        last = est.steps[-1][0]
        est.fit(x, y, **{f"{last}__sample_weight": sw})
    else:
        est.fit(x, y, sample_weight=sw)
    return est



def score_on_val(builder_fn, build_kwargs, x_train_core, y_train_core, x_val, y_val,
                  weighting, weight_support, metric_name, seed=SEED, damping=0.5, max_ratio=10.0):
    """One fit on train_core, one predict on val, one score. Shared by
    the coarse-filter phase (Stage 0/1/2) AND by the router/gate top-N
    end-to-end comparison. Returns (score, fitted_estimator) — the fitted
    model is returned too so the router-comparison can reuse it for
    downstream routing without a second fit. Fitting itself now
    delegates to `dispatch_fit` (§5.6) instead of inlining the
    weighting-dispatch branch."""
    est = dispatch_fit(builder_fn, build_kwargs, x_train_core, y_train_core,
                        weighting, weight_support, seed=seed, damping=damping, max_ratio=max_ratio)
    score = _score_by_name(metric_name, y_val, est.predict(x_val))
    return score, est
"""
## Section 1.3f — Coarse filter
"""
def coarse_filter(scored_candidates, fraction=COARSE_SURVIVAL_FRACTION,
                   min_survivors=COARSE_MIN_SURVIVORS):
    """Drops candidates whose 'coarse_val_score' is None BEFORE sorting —
    `safe_run` returns the tuple (None, None) on a failed fit, which
    without this guard reaches this function's sort key and crashes.
    `fraction` defaults to the top-25% found most reliable in practice —
    a caller sweeping many cheap candidates can still pass a smaller
    fraction explicitly for a more aggressive coarse cut."""
    valid = [c for c in scored_candidates if c.get("coarse_val_score") is not None]
    if not valid:
        return []
    ranked = sorted(valid, key=lambda c: c["coarse_val_score"], reverse=True)
    if len(ranked) <= min_survivors:
        return ranked
    n_keep = max(min_survivors, int(np.ceil(len(ranked) * fraction)))
    return ranked[:n_keep]


# %%
"""
## Section 1.3g — Canonical Stage-0/1/2 engine + two final-report modes

`_select_winners` runs Stage 0 (scaler) -> Stage 1 (feature-set/PCA) ->
Stage 2 (weighting), each stage using the coarse-then-fine funnel, and
is the ONE place that logic lives — every downstream module reuses it
rather than reimplementing it. `cv_select_and_refit` spends the test set
exactly once per surviving model; `select_and_refit_val_only` never
touches it, reporting one more OOF CV pass + a train refit instead.
`feat_label` is pulled out to module level since both final-report modes
need it. All three functions below now accept an optional
`logs_dir: Path` parameter — when given, `_select_winners` resolves one
`logger = get_section_logger(label_set, logs_dir)` at the top of the
call and threads it through every print site in this section (stage
headers and per-model winner selections at `NOTICE`; per-candidate
coarse/fine detail at `INFO`, file-only; failures at `WARNING`); the
per-model Stage 0/1/2 loops and the coarse/fine candidate loops inside
`_coarse_then_fine` are also now wrapped in `pbar()`.
Omitting `logs_dir` falls back to plain console output.

Every candidate `_coarse_then_fine` evaluates in its
coarse phase (Stage 0/1/2 alike, since all three stages route through
this one helper) now emits one INFO-level (file-only) line — config plus
score, or FAILED — instead of being silently dropped after
`coarse_filter`. Stage 0 and Stage 1 each also now emit one NOTICE-level
(console + file) per-model decision line when that stage concludes for a
given model, mirroring the line Stage 2 already had via `win_msg` — so
the console shows, for every model, what each of the three stages
decided, and the log file carries every candidate behind that decision.
"""


# %%
def feat_label(tag, pca_variance, scaler_name, scale_sensitive, pca_variants=PCA_VARIANTS) -> str:
    pca_suffix = f"_{[k for k, v in pca_variants.items() if v == pca_variance][0]}" \
        if pca_variance is not None else ""
    scaler_suffix = f"_{scaler_name}" if (scaler_name and scale_sensitive) else ""
    return f"{tag}{pca_suffix}{scaler_suffix}"
def ledger_row(results_path: Path, label_set: str, model_name: str, feat_label_str: str, variant: str) -> dict:
    """Fetches the FULL logged ledger entry (train_* AND test-aliased
    metrics) for one (model, feature_set, variant, label_set) combo, so
    winner_info/leaderboards can carry train metrics without re-fitting
    anything."""
    run_id = make_run_id(label_set, model_name, feat_label_str, variant)
    return dict(load_results(results_path).get(run_id, {}))
def _select_winners(x_train_by_tag, y_train_enc, tags, model_names, safe_run_fn,
                     model_specs, get_builder_fn, metric_key, ctx_prefix, label_set, n_splits,
                     scaler_variants, default_scaler_name, pca_base_tag, pca_variants,
                     weighting_modes, coarse_val_fraction, coarse_survival_fraction,
                     coarse_min_survivors, val_split_report_path,
                     logs_dir: Optional[Path] = None):
    logger = get_section_logger(label_set, logs_dir) if logs_dir is not None else None
    all_names = list(model_names)

    any_tag = tags[0]
    train_core_mask, val_mask, split_report = carve_train_core_val_split(
        x_train_by_tag[any_tag], y_train_enc, val_fraction=coarse_val_fraction,
        seed=SEED, label=label_set, logger=logger)
    if val_split_report_path is not None:
        with open(val_split_report_path, "w") as f:
            json.dump(split_report, f, indent=2, default=str)

    def _core(tag):
        x = x_train_by_tag[tag]
        return x.iloc[train_core_mask] if hasattr(x, "iloc") else x[train_core_mask]

    def _val(tag):
        x = x_train_by_tag[tag]
        return x.iloc[val_mask] if hasattr(x, "iloc") else x[val_mask]

    y_train_core = y_train_enc[train_core_mask]
    y_val = y_train_enc[val_mask]

    def _coarse_then_fine(model_name, candidates, stage_label):
        spec = model_specs[model_name]
        scale_sensitive = spec.get("scale_sensitive", False)
        support = weight_support_for(model_name)
        builder = get_builder_fn(model_name)

        scored = []
        for cand in pbar(candidates, desc=f"[{label_set}|{model_name}|{stage_label}] coarse", leave=False):
            build_kwargs = {"pca_variance": cand.get("pca_variance")}
            if scale_sensitive:
                scaler_name = cand.get("scaler_name")
                build_kwargs["scaler_cls"] = scaler_variants[scaler_name] if scaler_name else None
            result = safe_run_fn(score_on_val, builder, build_kwargs, _core(cand["tag"]),
                                  y_train_core, _val(cand["tag"]), y_val, cand["weighting"],
                                  support, metric_key, seed=SEED,
                                  context=f"{ctx_prefix}-{stage_label}-coarse|{model_name}|{cand}")
            coarse_score, _fitted = (None, None) if _failed(result) else result
            row = dict(cand)
            row["coarse_val_score"] = coarse_score
            scored.append(row)
            # Every coarse candidate
            # this stage tries — config plus score/FAILED — not just the
            # eventual survivors/winner. INFO/file-only, matching the
            # logging policy (per-candidate detail off console, on in
            # the branch's log file).
            cand_msg = (f"    [{label_set} | {model_name} | {stage_label} | coarse] "
                        f"tag={cand['tag']} pca={cand.get('pca_variance')} "
                        f"scaler={cand.get('scaler_name')} weighting={cand['weighting']} "
                        f"-> {metric_key}="
                        f"{'FAILED' if coarse_score is None else f'{coarse_score:.4f}'}")
            (logger.info(cand_msg) if logger is not None else print(cand_msg))

        survivors = coarse_filter(scored, coarse_survival_fraction, coarse_min_survivors)
        coarse_msg = (f"  [{ctx_prefix} | {model_name} | {stage_label}] coarse: "
                      f"{len(scored)} candidates -> {len(survivors)} survivors "
                      f"(top-{coarse_survival_fraction:.0%})")
        (logger.info(coarse_msg) if logger is not None else print(coarse_msg))

        fine_results = []
        for row in pbar(survivors, desc=f"[{label_set}|{model_name}|{stage_label}] fine", leave=False):
            build_kwargs = {"pca_variance": row.get("pca_variance")}
            if scale_sensitive:
                scaler_name = row.get("scaler_name")
                build_kwargs["scaler_cls"] = scaler_variants[scaler_name] if scaler_name else None
            r = safe_run_fn(cv_score_estimator, builder, build_kwargs, x_train_by_tag[row["tag"]],
                             y_train_enc, row["weighting"], support, metric_key,
                             n_splits=n_splits, seed=SEED, show_progress=False,
                             context=f"{ctx_prefix}-{stage_label}-fine|{model_name}|{row}")
            if _failed(r):
                fail_msg = (f"    [{label_set} | {model_name} | {stage_label} | fine] "
                            f"tag={row['tag']} pca={row.get('pca_variance')} "
                            f"scaler={row.get('scaler_name')} weighting={row['weighting']} "
                            f"-> FAILED fine CV")
                (logger.info(fail_msg) if logger is not None else print(fail_msg))
                continue
            row = dict(row)
            row["score"], row["score_std"], row["cv_mode"] = r["mean"], r["std"], r["cv_mode"]
            row["n_coarse_candidates"], row["n_coarse_survivors"] = len(scored), len(survivors)
            row["feature_set"] = feat_label(row["tag"], row.get("pca_variance"),
                                             row.get("scaler_name"), scale_sensitive, pca_variants)
            row["model_name"] = model_name
            # cv_score_estimator already
            # computes the full accuracy/bal_accuracy/macro_f1/kappa quartet
            # regardless of `metric_key` (see _pooled_oof_metrics) — it was
            # being discarded here down to just the one selection-metric
            # mean/std. Carrying the whole quartet lets a caller persist
            # every candidate row (not only the Stage-2 winner) with the
            # same val_* columns the final leaderboard uses.
            row["val_accuracy"], row["val_accuracy_std"] = r["accuracy_mean"], r["accuracy_std"]
            row["val_bal_accuracy"], row["val_bal_accuracy_std"] = r["bal_accuracy_mean"], r["bal_accuracy_std"]
            row["val_macro_f1"], row["val_macro_f1_std"] = r["macro_f1_mean"], r["macro_f1_std"]
            row["val_kappa"], row["val_kappa_std"] = r["kappa_mean"], r["kappa_std"]
            row["n_folds"] = r["n_folds"]
            fine_results.append(row)
            fine_msg = (f"    [{model_name} | {row['feature_set']} | {row['weighting']}] "
                        f"coarse={row['coarse_val_score']:.4f}  "
                        f"CV-{metric_key}={row['score']:.4f}±{row['score_std']:.4f}")
            (logger.info(fine_msg) if logger is not None else print(fine_msg))
        return fine_results

    h0 = (f"\n[{label_set}] STAGE 0 — SCALER SELECTION (top-{coarse_survival_fraction:.0%} "
          f"coarse-val filter + K={n_splits} CV; {metric_key})")
    (logger.notice(h0) if logger is not None else print(h0))
    winning_scaler = {}
    for model_name in pbar(all_names, desc=f"[{label_set}] Stage 0 — scaler"):
        if model_name == "HistGBC" or not model_specs[model_name]["scale_sensitive"]:
            winning_scaler[model_name] = None
            # Log the skip too, so the
            # per-model Stage 0 record is complete even for models that
            # never entered the scaler sweep at all.
            skip_msg = f"  · [{label_set}] {model_name}: Stage 0 skipped (not scale-sensitive)"
            (logger.info(skip_msg) if logger is not None else print(skip_msg))
            continue
        candidates = [{"tag": pca_base_tag, "pca_variance": None, "scaler_name": s, "weighting": "none"}
                      for s in scaler_variants]
        results = _coarse_then_fine(model_name, candidates, "stage0")
        winning_scaler[model_name] = (max(results, key=lambda r: r["score"])["scaler_name"]
                                       if results else default_scaler_name)
        # Per-model Stage 0 decision —
        # mirrors the Stage 2 win_msg that already existed, so all three
        # stages now announce their per-model outcome at NOTICE.
        s0_fallback = " (no survivors — fell back to default)" if not results else ""
        s0_msg = (f"  → [{label_set}] {model_name}: Stage 0 winner — "
                  f"scaler={winning_scaler[model_name]}{s0_fallback}")
        (logger.notice(s0_msg) if logger is not None else print(s0_msg))

    h1 = (f"\n[{label_set}] STAGE 1 — FEATURE-SET/PCA SELECTION (top-{coarse_survival_fraction:.0%} "
          f"coarse-val filter + K={n_splits} CV; {metric_key})")
    (logger.notice(h1) if logger is not None else print(h1))
    winner_info = {}
    for model_name in pbar(all_names, desc=f"[{label_set}] Stage 1 — feature/PCA"):
        candidates = [{"tag": t, "pca_variance": None, "scaler_name": winning_scaler[model_name],
                       "weighting": "none"} for t in tags]
        if model_name != "HistGBC" and model_specs[model_name]["supports_pca"]:
            candidates += [{"tag": pca_base_tag, "pca_variance": v,
                            "scaler_name": winning_scaler[model_name], "weighting": "none"}
                           for v in pca_variants.values()]
        results = _coarse_then_fine(model_name, candidates, "stage1")
        if not results:
            warn = f"  ⚠ {label_set}/{model_name}: every feature-set/PCA config failed — excluded."
            (logger.warning(warn) if logger is not None else print(warn))
            continue
        winner_info[model_name] = max(results, key=lambda r: r["score"])
        # Per-model Stage 1 decision.
        win1 = winner_info[model_name]
        s1_msg = (f"  → [{label_set}] {model_name}: Stage 1 winner — "
                  f"feature_set={win1['feature_set']} score={win1['score']:.4f}±{win1['score_std']:.4f}")
        (logger.notice(s1_msg) if logger is not None else print(s1_msg))
    if not winner_info:
        return {}

    h2 = (f"\n[{label_set}] STAGE 2 — WEIGHTING SELECTION (top-{coarse_survival_fraction:.0%} "
          f"coarse-val filter + K={n_splits} CV; {metric_key})")
    (logger.notice(h2) if logger is not None else print(h2))
    for model_name in pbar(list(winner_info.keys()), desc=f"[{label_set}] Stage 2 — weighting"):
        win = winner_info[model_name]
        candidates = [{"tag": win["tag"], "pca_variance": win.get("pca_variance"),
                       "scaler_name": win.get("scaler_name"), "weighting": w}
                      for w in weighting_modes]
        results = _coarse_then_fine(model_name, candidates, "stage2")
        # Keep every candidate this stage
        # actually scored (typically 3-4 of len(weighting_modes), whatever
        # survives coarse_filter), not just the argmax — `select_and_refit_
        # val_only`/`cv_select_and_refit` persist these so "none" (and every
        # other losing mode) gets a real, comparable CV score on disk
        # instead of only ever surfacing when it happens to win.
        winner_info[model_name]["weighting_sweep"] = results
        if results:
            best_w = max(results, key=lambda r: r["score"])
            winner_info[model_name].update({"weighting": best_w["weighting"],
                                             "cv_score": best_w["score"],
                                             "cv_score_std": best_w["score_std"]})
            win_msg = (f"  → [{label_set}] {model_name}: weighting={best_w['weighting']} "
                       f"cv_score={best_w['score']:.4f}±{best_w['score_std']:.4f}")
            (logger.notice(win_msg) if logger is not None else print(win_msg))
        else:
            winner_info[model_name]["weighting"] = "none"
            winner_info[model_name]["cv_score"] = winner_info[model_name]["score"]
            winner_info[model_name]["cv_score_std"] = winner_info[model_name]["score_std"]
            no_survivors_msg = (f"  → [{label_set}] {model_name}: Stage 2 — every weighting mode "
                                 f"failed — kept Stage 1's weighting='none' "
                                 f"score={winner_info[model_name]['score']:.4f}")
            (logger.notice(no_survivors_msg) if logger is not None else print(no_survivors_msg))

    return winner_info
def _persist_weighting_sweep(winner_info: dict, label_set: str, results_path: Path,
                              extra_results_paths: Optional[Sequence[Path]] = None,
                              logger: Optional[logging.Logger] = None) -> int:
    """Writes every Stage-2 weighting
    candidate `_select_winners` scored (`winner_info[model]["weighting_sweep"]`
    — see there) to the results ledger(s), one row per (model, weighting
    mode) actually evaluated. Distinct from the final winner's own
    `final_*`/`final_val_*` row (still written separately by
    `cv_select_and_refit`/`select_and_refit_val_only`) via
    `variant=f"stage2_sweep_{weighting}"` and `stage="stage2_weighting_sweep"`
    — so a downstream reader can tell "this model's chosen config, refit
    and properly evaluated" apart from "one candidate weighting mode's
    Stage-2 CV score," while both still carry the same val_accuracy/
    val_bal_accuracy/val_macro_f1/val_kappa columns and pool together in
    a `weighting`-grouped comparison. Called once, right after
    `_select_winners` returns, before the final refit loop — a failed
    final refit for a model doesn't need to take its already-scored sweep
    rows down with it. Returns the number of rows written."""
    paths = [results_path] + list(extra_results_paths or [])
    n_written = 0
    for model_name, win in winner_info.items():
        for cand in (win.get("weighting_sweep") or []):
            sweep_row = {**cand, "label_set": label_set, "stage": "stage2_weighting_sweep",
                         "variant": f"stage2_sweep_{cand['weighting']}"}
            run_id = make_run_id(label_set, model_name, cand["feature_set"], sweep_row["variant"])
            save_result_multi(run_id, sweep_row, paths, logger=logger)
            n_written += 1
    msg = f"  ✓ [{label_set}] weighting sweep persisted: {n_written} candidate row(s) across {len(winner_info)} model(s)"
    (logger.notice(msg) if logger is not None else print(msg))
    return n_written


def cv_select_and_refit(x_train_by_tag, x_test_by_tag, y_train_enc, y_test_enc,
                         class_names, label_set, tags, model_names,
                         fit_and_log_fn, run_histgbc_fn, safe_run_fn, results_path,
                         model_specs=None, get_builder_fn=None, selection_metric=None,
                         n_splits=5, context_prefix=None, final_hook=None,
                         scaler_variants=None, default_scaler_name=None,
                         pca_base_tag=None, pca_variants=None,
                         weighting_modes=None,
                         coarse_val_fraction=None,
                         coarse_survival_fraction=None,
                         coarse_min_survivors=None,
                         val_split_report_path=None,
                         logs_dir: Optional[Path] = None):
    """Stage 0/1/2 (`_select_winners`) then exactly ONE refit-and-test-
    evaluate per surviving model. x_test_by_tag/y_test_enc are read ONLY
    in the final loop below — spend the test set on model/config
    selection anywhere upstream and every number this function reports
    stops being an honest held-out estimate.

    `safe_run_fn(fn, *args, context=..., **kwargs)` is each module's own
    wrapper that already binds its results_path/ledger — call it WITHOUT
    passing results_path again here (it's bound inside safe_run_fn).

    `final_hook(model, model_name, x_te, y_test_enc, class_names, label_set,
    win)` runs once per surviving model right after its final refit.

    Optional `logs_dir` resolves one logger for this function's own
    two headline prints and is threaded into `_select_winners`; omit for
    plain console output. Returns (winner_info, leaderboard_df).

    `scaler_variants`/`default_scaler_name`/
    `pca_base_tag`/`pca_variants`/`weighting_modes`/`coarse_val_fraction`/
    `coarse_survival_fraction`/`coarse_min_survivors` now default to `None`
    and are resolved against the CURRENT module globals below, mirroring
    `model_specs`/`get_builder_fn`/`selection_metric`/`context_prefix`'s
    existing pattern. Previously these defaulted directly to the module
    globals (`=PCA_BASE_TAG`, etc.), which Python binds ONCE at function-
    definition time — if this cell was ever (re-)executed in a kernel
    session while one of those bare globals had been reassigned elsewhere
    (e.g. a scratch/test cell, an out-of-order rerun), this function would
    silently keep the stale value forever, immune to any later reset of
    the global. Resolving inside the body instead reads the live global on
    every call.
    """
    model_specs = model_specs or MODEL_SPECS
    get_builder_fn = get_builder_fn or get_builder
    metric_key = selection_metric or SELECTION_METRIC
    ctx_prefix = context_prefix or label_set
    scaler_variants = scaler_variants if scaler_variants is not None else SCALER_VARIANTS
    default_scaler_name = default_scaler_name if default_scaler_name is not None else DEFAULT_SCALER_NAME
    pca_base_tag = pca_base_tag if pca_base_tag is not None else PCA_BASE_TAG
    pca_variants = pca_variants if pca_variants is not None else PCA_VARIANTS
    weighting_modes = weighting_modes if weighting_modes is not None else WEIGHTING_MODES
    coarse_val_fraction = coarse_val_fraction if coarse_val_fraction is not None else COARSE_VAL_FRACTION
    coarse_survival_fraction = (coarse_survival_fraction if coarse_survival_fraction is not None
                                 else COARSE_SURVIVAL_FRACTION)
    coarse_min_survivors = coarse_min_survivors if coarse_min_survivors is not None else COARSE_MIN_SURVIVORS
    logger = get_section_logger(label_set, logs_dir) if logs_dir is not None else None

    winner_info = _select_winners(
        x_train_by_tag, y_train_enc, tags, model_names, safe_run_fn,
        model_specs, get_builder_fn, metric_key, ctx_prefix, label_set, n_splits,
        scaler_variants, default_scaler_name, pca_base_tag, pca_variants,
        weighting_modes, coarse_val_fraction, coarse_survival_fraction,
        coarse_min_survivors, val_split_report_path, logs_dir=logs_dir)
    if not winner_info:
        return {}, pd.DataFrame()

    _persist_weighting_sweep(winner_info, label_set, results_path, logger=logger)

    header = f"\n[{label_set}] FINAL REPORT — one refit per model, one test-set eval each"
    (logger.notice(header) if logger is not None else print(header))
    final_rows = []
    for model_name, win in winner_info.items():
        x_tr, x_te = x_train_by_tag[win["tag"]], x_test_by_tag[win["tag"]]
        ctx_label = f"{ctx_prefix}|{model_name}|FINAL-TEST"
        if model_name == "HistGBC":
            model, metrics = safe_run_fn(run_histgbc_fn, x_tr, y_train_enc, x_te, y_test_enc,
                                          win["tag"], win["weighting"], class_names, label_set,
                                          variant_prefix="final", context=ctx_label)
            feat_lookup = win["tag"]
        else:
            model, metrics = safe_run_fn(_run_model_core, x_tr, y_train_enc, x_te, y_test_enc,
                                          model_name, win["tag"], win["weighting"], class_names,
                                          label_set, fit_and_log_fn, pca_variance=win.get("pca_variance"),
                                          scaler_name=win.get("scaler_name"), variant_prefix="final",
                                          context=ctx_label)
            feat_lookup = win["feature_set"]
        if metrics is None:
            continue
        row = ledger_row(results_path, label_set, model_name, feat_lookup, f"final_{win['weighting']}")
        row["cv_score"], row["cv_score_std"] = win.get("cv_score"), win.get("cv_score_std")
        final_rows.append(row)
        notice = f"  → [{label_set}] {model_name}: test_{metric_key}={row.get(metric_key, 'n/a')}"
        (logger.notice(notice) if logger is not None else print(notice))
        if final_hook is not None:
            final_hook(model, model_name, x_te, y_test_enc, class_names, label_set, win)

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        raise RuntimeError(f"[{label_set}]: every final refit failed.")
    leaderboard = final_df.sort_values(metric_key, ascending=False)
    return winner_info, leaderboard



def select_and_refit_val_only(x_train_by_tag, y_train_enc, class_names, label_set, tags,
                               model_names, safe_run_fn, results_path,
                               model_specs=None, get_builder_fn=None, selection_metric=None,
                               n_splits=5, context_prefix=None, final_hook=None,
                               scaler_variants=None, default_scaler_name=None,
                               pca_base_tag=None, pca_variants=None,
                               weighting_modes=None,
                               coarse_val_fraction=None,
                               coarse_survival_fraction=None,
                               coarse_min_survivors=None,
                               val_split_report_path=None,
                               logs_dir: Optional[Path] = None,
                               extra_results_paths: Optional[Sequence[Path]] = None,
                               run_histgbc_deployable: bool = True,
                               histgbc_max_iter_ceiling: int = 1000):
    """Identical Stage 0/1/2 funnel to `cv_select_and_refit` (shared via
    `_select_winners`) but the final report NEVER reads a test set: each
    surviving model instead gets one more OOF K-fold CV pass (val
    quartet + confusion-matrix-ready predictions) and one plain
    refit-on-full-x_tr evaluated on x_tr itself — an explicit refit/
    memorization diagnostic, not a validation score.

    Every other model
    in this loop is scored/refit through `MODEL_SPECS`'s registered
    builder, which for `"HistGBC"` is the cheap, fixed-config search
    proxy (`_build_histgbc_search_proxy` / `_HGBCWeightedEarlyStop`) —
    appropriate for the Stage 0/1/2 SEARCH, but not a claim about the
    deployable model. `cv_select_and_refit` already avoids this by
    special-casing `"HistGBC"` in ITS final loop and dispatching to the
    real, leakage-safe `StratifiedEarlyStoppingHGBC` via `run_histgbc_fn`
    — but that dispatch needs a test set, which this val-only function's
    entire contract forbids reading. The fix implemented here instead
    swaps in a small adapter builder that constructs the SAME production
    `StratifiedEarlyStoppingHGBC` (with `weighting` baked into its own
    constructor, matching how `run_histgbc` already builds it) for both
    the OOF CV pass and the train-refit diagnostic — so this function's
    HistGBC row now reflects the real, deployable, leakage-safe
    early-stopped estimator, and NO test data is read anywhere in this
    function, before or after this fix. Set `run_histgbc_deployable=False`
    to fall back to the search-proxy behavior for HistGBC here (kept
    as an escape hatch, not because it's recommended).

    Every row also carries `fitted_summary`/
    `hyperparameters`/`normalization`, via the same module-level helpers
    `RunTelemetry` uses (§1.2) — this was the single highest-volume
    under-logged code path in the whole pipeline (every gate/subgate/
    router/branch winner across both hierarchical modules, plus Module
    2's val-only leaderboards, all funnel through this one function).

    `logs_dir` resolves one logger for this function's own prints
    (and is threaded into `_select_winners`); `extra_results_paths`
    lets a caller write each final row to additional files (e.g. a
    per-branch results file) beyond the required `results_path`, via
    `save_result_multi` — `save_result` itself is still called exactly
    once per path.

    Same fix as `cv_select_and_refit`:
    `scaler_variants`/`default_scaler_name`/`pca_base_tag`/`pca_variants`/
    `weighting_modes`/`coarse_val_fraction`/`coarse_survival_fraction`/
    `coarse_min_survivors` now default to `None` and resolve against the
    live module globals inside the body, instead of being frozen at
    function-definition time — a stale kernel-namespace reassignment of
    any of these globals (e.g. `PCA_BASE_TAG`) can no longer get baked
    into this function forever.
    """
    model_specs = model_specs or MODEL_SPECS
    get_builder_fn = get_builder_fn or get_builder
    metric_key = selection_metric or SELECTION_METRIC
    ctx_prefix = context_prefix or label_set
    scaler_variants = scaler_variants if scaler_variants is not None else SCALER_VARIANTS
    default_scaler_name = default_scaler_name if default_scaler_name is not None else DEFAULT_SCALER_NAME
    pca_base_tag = pca_base_tag if pca_base_tag is not None else PCA_BASE_TAG
    pca_variants = pca_variants if pca_variants is not None else PCA_VARIANTS
    weighting_modes = weighting_modes if weighting_modes is not None else WEIGHTING_MODES
    coarse_val_fraction = coarse_val_fraction if coarse_val_fraction is not None else COARSE_VAL_FRACTION
    coarse_survival_fraction = (coarse_survival_fraction if coarse_survival_fraction is not None
                                 else COARSE_SURVIVAL_FRACTION)
    coarse_min_survivors = coarse_min_survivors if coarse_min_survivors is not None else COARSE_MIN_SURVIVORS
    logger = get_section_logger(label_set, logs_dir) if logs_dir is not None else None
    _say = lambda m: logger.notice(m) if logger is not None else print(m)

    # Search profile (see SEARCH_* in the config section).
    if SEARCH_MODELS is not None:
        allowed = set(SEARCH_MODELS).union(*[v for k, v in SEARCH_MODELS_EXTRA.items() if k in label_set])
        model_names = [m for m in model_names if m in allowed]
    if SEARCH_SCALERS is not None:
        scaler_variants = {k: v for k, v in scaler_variants.items() if k in SEARCH_SCALERS}
        if default_scaler_name not in scaler_variants:
            default_scaler_name = next(iter(scaler_variants))
    if SEARCH_TAGS is not None:
        tags = [t for t in tags if t in SEARCH_TAGS]
    _say(f"[{label_set}] search space: models={list(model_names)}  tags={list(tags)}  "
         f"scalers={list(scaler_variants)}  pca={list(pca_variants)}  weighting={list(weighting_modes)}")

    # Per-node checkpoint: keyed on everything that determines the result
    # (data, search space, settings); code changes are covered by NODE_CACHE_VERSION.
    cache_path = None
    if NODE_CACHE_DIR is not None:
        key = joblib.hash((
            NODE_CACHE_VERSION, label_set, ctx_prefix, list(model_names), list(tags), list(scaler_variants),
            default_scaler_name, pca_base_tag, dict(pca_variants), list(weighting_modes), coarse_val_fraction,
            coarse_survival_fraction, coarse_min_survivors, metric_key, n_splits, SEED, USE_GPU,
            run_histgbc_deployable, histgbc_max_iter_ceiling, MLP_ARCHITECTURES,
            CATBOOST_MAX_ITERATIONS, CATBOOST_EARLY_STOPPING_ROUNDS, CATBOOST_LEARNING_RATE,
            np.asarray(y_train_enc),
            {t: x_train_by_tag[t] for t in sorted(set(tags) | {pca_base_tag}) if t in x_train_by_tag}))
        cache_path = Path(NODE_CACHE_DIR) / f"{_safe_filename(label_set)}__{key[:16]}.joblib"
        if cache_path.exists():
            winner_info, leaderboard, val_report_text = joblib.load(cache_path)
            if val_split_report_path is not None and val_report_text is not None:
                Path(val_split_report_path).write_text(val_report_text)
            # Re-write this node's rows into the current run's result ledgers.
            _persist_weighting_sweep(winner_info, label_set, results_path,
                                      extra_results_paths=extra_results_paths, logger=logger)
            paths = [results_path] + list(extra_results_paths or [])
            for row in leaderboard.to_dict("records"):
                save_result_multi(make_run_id(label_set, row["model_name"], row["feature_set"], row["variant"]),
                                  row, paths, logger=logger)
            _say(f"  ⏭ [{label_set}] resumed from node checkpoint {cache_path.name} — search skipped "
                 "(per-model plots from final_hook are not regenerated)")
            return winner_info, leaderboard

    winner_info = _select_winners(
        x_train_by_tag, y_train_enc, tags, model_names, safe_run_fn,
        model_specs, get_builder_fn, metric_key, ctx_prefix, label_set, n_splits,
        scaler_variants, default_scaler_name, pca_base_tag, pca_variants,
        weighting_modes, coarse_val_fraction, coarse_survival_fraction,
        coarse_min_survivors, val_split_report_path, logs_dir=logs_dir)
    if not winner_info:
        return {}, pd.DataFrame()

    _persist_weighting_sweep(winner_info, label_set, results_path,
                              extra_results_paths=extra_results_paths, logger=logger)

    header = (f"\n[{label_set}] FINAL REPORT — one OOF K={n_splits} CV pass + one train "
              f"refit per surviving model. NO TEST SET TOUCHED.")
    (logger.notice(header) if logger is not None else print(header))
    final_rows = []
    for model_name, win in pbar(list(winner_info.items()), desc=f"[{label_set}] Final val-only refit"):
        spec = model_specs[model_name]
        scale_sensitive = spec.get("scale_sensitive", False)
        support = weight_support_for(model_name)
        builder = get_builder_fn(model_name)
        build_kwargs = {"pca_variance": win.get("pca_variance")}
        if scale_sensitive:
            scaler_name = win.get("scaler_name")
            build_kwargs["scaler_cls"] = scaler_variants[scaler_name] if scaler_name else None
        weighting_for_dispatch = win["weighting"]

        is_histgbc_deployable = (model_name == "HistGBC" and run_histgbc_deployable)
        if is_histgbc_deployable:
            _weighting_baked = win["weighting"]

            def builder(pca_variance=None, scaler_cls=None, _w=_weighting_baked):
                return StratifiedEarlyStoppingHGBC(
                    weighting=_w, damping=0.5, max_ratio=10.0, random_state=SEED,
                    max_iter_ceiling=histgbc_max_iter_ceiling)
            build_kwargs = {}
            # weighting is baked into the constructor above, exactly as
            # run_histgbc() builds it — tell dispatch_fit/cv_score_estimator
            # to just plain-fit, not apply a second, redundant weighting pass.
            weighting_for_dispatch = "none"
            support = "sample_weight"

        x_tr = x_train_by_tag[win["tag"]]
        ctx_label = f"{ctx_prefix}|{model_name}|FINAL-VAL"

        cv_result = safe_run_fn(cv_score_estimator, builder, build_kwargs, x_tr, y_train_enc,
                                 weighting_for_dispatch, support, metric_key,
                                 n_splits=n_splits, seed=SEED, collect_oof=True,
                                 context=f"{ctx_label}-oof")
        if _failed(cv_result):
            warn = f"  ⚠ {label_set}/{model_name}: final OOF CV pass failed — excluded."
            (logger.warning(warn) if logger is not None else print(warn))
            continue

        est = dispatch_fit(builder, build_kwargs, x_tr, y_train_enc,
                            weighting_for_dispatch, support, seed=SEED, damping=0.5, max_ratio=10.0,
                            record_history=True)  # [NEW] only this one final-refit-per-family call opts in

        y_pred_train = est.predict(x_tr)
        train_metrics = evaluate_model(
            f"{model_name} [{label_set}] [TRAIN — refit metric, NOT held-out]",
            y_train_enc, y_pred_train, class_names, label_set=label_set,
            split="train", verbose=False)
        _, train_std = _pooled_oof_metrics(y_train_enc, y_pred_train, seed=SEED)  # same bootstrap as val_*_std

        feat_label_str = feat_label(win["tag"], win.get("pca_variance"), win.get("scaler_name"),
                                     scale_sensitive, pca_variants)
        row = {
            "model_name": model_name, "feature_set": feat_label_str, "tag": win["tag"],
            "pca_variance": win.get("pca_variance"), "scaler_name": win.get("scaler_name"),
            "weighting": win["weighting"], "variant": f"final_val_{win['weighting']}",
            "label_set": label_set, "n_features": x_tr.shape[1],
            "cv_mode": cv_result["cv_mode"], "n_folds": cv_result["n_folds"],
            "val_accuracy": cv_result["accuracy_mean"], "val_accuracy_std": cv_result["accuracy_std"],
            "val_bal_accuracy": cv_result["bal_accuracy_mean"], "val_bal_accuracy_std": cv_result["bal_accuracy_std"],
            "val_macro_f1": cv_result["macro_f1_mean"], "val_macro_f1_std": cv_result["macro_f1_std"],
            "val_kappa": cv_result["kappa_mean"], "val_kappa_std": cv_result["kappa_std"],
            "train_accuracy": train_metrics["accuracy"], "train_bal_accuracy": train_metrics["bal_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"], "train_kappa": train_metrics["kappa"],
            **{f"train_{m}_std": train_std[m] for m in ("accuracy", "bal_accuracy", "macro_f1", "kappa")},
            # hyperparameter-logging gap closed — same fields
            # RunTelemetry attaches to fit_and_log's rows.
            "fitted_summary": get_fitted_summary(est),
            "hyperparameters": _hyperparams(est),
            "normalization": _normalization_type(est),
            "histgbc_deployable_used": bool(is_histgbc_deployable),
        }
        row[f"val_{metric_key}"] = cv_result[f"{metric_key}_mean"]
        final_rows.append(row)

        run_id = make_run_id(label_set, model_name, feat_label_str, row["variant"])
        paths = [results_path] + list(extra_results_paths or [])
        save_result_multi(run_id, dict(row), paths, logger=logger)
        notice = f"  → [{label_set}] {model_name}: val_{metric_key}={row[f'val_{metric_key}']:.4f}"
        (logger.notice(notice) if logger is not None else print(notice))

        if final_hook is not None:
            final_hook(est, model_name, cv_result["oof_true"], cv_result["oof_pred"],
                       class_names, label_set, win)

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        raise RuntimeError(f"[{label_set}]: every final validation pass failed.")
    sort_col = f"val_{metric_key}"
    leaderboard = final_df.sort_values(sort_col, ascending=False)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        report = Path(val_split_report_path) if val_split_report_path is not None else None
        tmp = cache_path.with_suffix(".tmp")
        joblib.dump((winner_info, leaderboard, report.read_text() if report and report.exists() else None), tmp)
        tmp.replace(cache_path)  # atomic: a crash mid-write never leaves a half checkpoint
    return winner_info, leaderboard


# %%
"""
## Section 1.3i — Small per-module conveniences

Two helpers every hierarchical module previously copy-pasted verbatim: a
tqdm-optional progress bar, and a config-dict -> fitted-estimator step
used both for a one-time deployment refit and for per-fold refits inside
chain-CV.
"""


# %%
def pbar(iterable, total=None, desc="", leave=True):
    """Progress-bar wrapper. Uses tqdm.auto if installed; otherwise
    materializes the iterable and prints one summary line (no live bar).
    Always returns something iterable with the same items/order — safe
    to drop into any `for x in ...:` site with zero other change.
    Extended in this file into `_select_winners`'s
    per-model Stage 0/1/2 loops, `_coarse_then_fine`'s coarse/fine
    candidate loops, `cv_score_estimator`'s fold loop,
    `compute_selection_engine`'s repeat loop, and
    `select_top_n_end_to_end`'s candidate loop, and the hierarchical
    modules' branch loops."""
    items = list(iterable)
    try:
        from tqdm.auto import tqdm as _tqdm
        return _tqdm(items, total=total or len(items), desc=desc, leave=leave)
    except Exception:
        print(f"  [progress] {desc}: {len(items)} item(s) queued "
              f"(install tqdm for a live bar: pip install tqdm --break-system-packages)")
        return items
def fit_component(cfg: dict, model_specs, get_builder_fn, x_tr, y_tr, seed=SEED):
    """Fits one component (router/gate/branch) from an already-selected
    config dict — {'model_name','tag','pca_variance','scaler_name',
    'weighting'}. Used for the one-time deployment refit AND for every
    per-fold refit inside a chain-CV engine — single source of truth.
    Delegates to `dispatch_fit` instead of inlining the
    weighting-dispatch branch (fourth of the four consolidated call
    sites)."""
    spec = model_specs[cfg["model_name"]]
    builder = get_builder_fn(cfg["model_name"])
    build_kwargs = {"pca_variance": cfg.get("pca_variance")}
    if spec.get("scale_sensitive", False):
        scaler_name = cfg.get("scaler_name")
        build_kwargs["scaler_cls"] = SCALER_VARIANTS[scaler_name] if scaler_name else None
    support = weight_support_for(cfg["model_name"])
    return dispatch_fit(builder, build_kwargs, x_tr, y_tr, cfg["weighting"], support, seed=seed)


# %%
"""
## Section 1.3j — Probability calibration

The models are trained with class weights, so their probabilities are tilted
toward the rare classes and are not posteriors under the catalogue's class mix.
`ProbabilityCalibrator` is fit on OUT-OF-FOLD predictions of the training rows
(`oof_predict_proba`), never on test rows. The predicted class is left as the
weighted model's argmax (that is what balanced accuracy rewards); only the
probabilities attached to it change.
"""


def model_classes(m) -> np.ndarray:
    """Encoded classes a fitted estimator (or the last step of a Pipeline / a wrapper's inner model)
    knows, i.e. the column order of its predict_proba."""
    for obj in (m, getattr(m, "model_", None), getattr(m, "estimator_", None)):
        c = getattr(obj, "classes_", None) if obj is not None else None
        if c is not None:
            return np.asarray(c).astype(int)
    return model_classes(m.steps[-1][1])


def oof_predict_proba(cfg: dict, model_specs, get_builder_fn, x, y, n_splits=5, seed=SEED) -> np.ndarray:
    """Out-of-fold class probabilities on training rows: every row is predicted by a model fit on the
    other folds with the same config. Columns are classes 0..K-1; a class a fold never saw (a
    singleton held out) gets probability 0 there."""
    y = np.asarray(y)
    proba = np.zeros((len(y), int(y.max()) + 1))
    for tr, va in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x, y):
        m = fit_component(cfg, model_specs, get_builder_fn, x.iloc[tr], y[tr], seed)
        proba[np.ix_(va, model_classes(m))] = np.asarray(m.predict_proba(x.iloc[va]))
    return proba


class ProbabilityCalibrator:
    """Two-step post-hoc calibration, fit on out-of-fold training predictions:
      1. undo the class-weighting tilt. A model trained with class weights w_c estimates
         p_w(c|x) proportional to w_c p(c|x), so p(c|x) is proportional to p_w(c|x) / w_c
         (Elkan 2001, IJCAI; Saerens, Latinne & Decaestecker 2002, Neural Comput. 14:21);
      2. temperature scaling of the log-probabilities, T chosen to minimise the out-of-fold negative
         log-likelihood (Guo et al. 2017, ICML). One parameter, so it cannot overfit rare classes;
         per-class methods such as isotonic regression need far more rows per class
         (Niculescu-Mizil & Caruana 2005). `class_weights[c]` = the weight class c had in training
         (all 1 for weighting='none')."""

    def __init__(self, class_weights):
        self.class_weights = np.asarray(class_weights, dtype=float)

    def _untilt(self, proba):
        p = np.clip(np.asarray(proba, dtype=float), 1e-12, None) / self.class_weights
        return p / p.sum(axis=1, keepdims=True)

    @staticmethod
    def _softmax_t(logp, t):
        z = logp / t
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, proba_oof, y_oof):
        from scipy.optimize import minimize_scalar
        proba_oof, y = np.asarray(proba_oof, dtype=float), np.asarray(y_oof)
        ok = np.isfinite(proba_oof).all(axis=1)
        if not ok.all():  # a NaN row would turn the whole NLL into NaN and the fit into noise
            print(f"  ⚠ ProbabilityCalibrator: {int((~ok).sum())} out-of-fold row(s) with non-finite "
                  "probabilities ignored — check the model's predict_proba.")
            proba_oof, y = proba_oof[ok], y[ok]
        logp = np.log(self._untilt(proba_oof))
        nll = lambda t: -np.mean(np.log(np.clip(self._softmax_t(logp, t)[np.arange(len(y)), y], 1e-12, None)))
        self.temperature_ = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
        return self

    def transform(self, proba):
        return self._softmax_t(np.log(self._untilt(proba)), self.temperature_)


def fit_calibrator(cfg: dict, model_specs, get_builder_fn, x, y, seed=SEED) -> ProbabilityCalibrator:
    """`ProbabilityCalibrator` for one flat-model config, fit on that config's OOF probabilities over
    (x, y) — the training rows of whatever model it will be applied to."""
    y = np.asarray(y)
    wmap = class_weight_map(y, cfg.get("weighting", "none"))
    weights = [wmap.get(c, 1.0) for c in range(int(y.max()) + 1)]
    return ProbabilityCalibrator(weights).fit(oof_predict_proba(cfg, model_specs, get_builder_fn, x, y, seed=seed), y)


def em_prior_adjust(proba, train_priors, n_iter=200, tol=1e-6):
    """Saerens, Latinne & Decaestecker (2002) EM re-estimation of the class priors of an UNLABELLED
    set: returns (posteriors under the estimated priors, estimated priors). `proba` must be
    calibrated posteriors under the training priors `train_priors`. It assumes label shift
    (same p(x|class) in both sets); for unassociated Fermi sources that is only approximate
    (covariate shift, sources outside the taxonomy — Malyshev 2024, arXiv:2412.04675), so report it
    as a sensitivity check next to the unadjusted values."""
    pi0 = np.clip(np.asarray(train_priors, dtype=float), 1e-12, None)
    pi = pi0.copy()
    for _ in range(n_iter):
        p = proba * (pi / pi0)
        p /= p.sum(axis=1, keepdims=True)
        new = p.mean(axis=0)
        done = np.abs(new - pi).max() < tol
        pi = new
        if done:
            break
    p = proba * (pi / pi0)
    return p / p.sum(axis=1, keepdims=True), pi


# %%
"""
## Section 1.3h — Top-N router/gate/nonblazar_router val comparison
"""
def select_top_n_end_to_end(candidates, fit_router_fn, route_and_score_fn,
                             x_train_core, y_train_core, x_val, y_val,
                             safe_run_fn, metric_name=None, context_prefix="",
                             logger: Optional[logging.Logger] = None):
    """Generic val-scored comparison across N already-CV-selected router/
    gate/nonblazar_router candidates.

    fit_router_fn(candidate, x_tr, y_tr) -> fitted_router
    route_and_score_fn(fitted_router, x_eval, y_eval_true) -> (score, extra)

    Neither this function nor its callback contract ever reads test data —
    the caller is responsible for the one-time deployment refit + test
    report AFTER this function has already picked the winner.

    `logger`, if given, receives the per-candidate lines at INFO
    (file-only) and the final winner line at NOTICE (console + file) —
    the winner line is a milestone, so it goes to the console. The candidate loop is now
    wrapped in `pbar()`. Returns (winner_candidate,
    winner_end_to_end_val_score, all_scored)."""
    metric_name = metric_name or SELECTION_METRIC
    scored = []
    for i, cand in enumerate(pbar(candidates, desc=f"[{context_prefix}] top-N candidates")):
        ctx = f"{context_prefix}-topN-val|{i}"

        def _run(cand=cand):
            router = fit_router_fn(cand, x_train_core, y_train_core)
            return route_and_score_fn(router, x_val, y_val)

        result = safe_run_fn(_run, context=ctx)
        if result is None:
            continue
        end_to_end_val, extra = result
        scored.append({"candidate": cand, "end_to_end_val": end_to_end_val, "extra": extra})
        cand_msg = f"  [{context_prefix} | candidate {i}] end_to_end_val ({metric_name}) = {end_to_end_val:.4f}"
        (logger.info(cand_msg) if logger is not None else print(cand_msg))

    if not scored:
        raise RuntimeError(f"[{context_prefix}]: every top-N candidate failed val scoring.")
    winner = max(scored, key=lambda s: s["end_to_end_val"])
    win_msg = (f"  -> [{context_prefix}] winner: end_to_end_val={winner['end_to_end_val']:.4f} "
               f"(of {len(scored)} compared)")
    (logger.notice(win_msg) if logger is not None else print(win_msg))
    return winner["candidate"], winner["end_to_end_val"], scored


# %%
"""
## Section 1.4 — Custom convergence-based estimators

Several models below are hand-rolled wrappers around a stock sklearn
estimator, adding a leakage-safe, class-stratified early-stopping loop
on top: growing the model in chunks (more boosting rounds / more trees
/ more epochs) and stopping once a held-out validation metric plateaus
for `patience` chunks. This replaces relying on the underlying library's
own early-stopping (which is typically NOT stratified by class and can
silently starve or drop rare classes from its internal validation
split). `DTreeCVPrunedClassifier` additionally implements adaptive
Leave-One-Out cross-validation.

All six classes below now accept an
optional `stopping_metric: Optional[str] = None` constructor parameter.
Left unset (the default), each estimator resolves `metric_name =
self.stopping_metric or SELECTION_METRIC` **at `fit()` time** — reading
whatever `SELECTION_METRIC` currently is, exactly matching how
`cv_select_and_refit`/`select_and_refit_val_only` already resolve
`selection_metric or SELECTION_METRIC` dynamically — and uses that
metric (via `_score_by_name`, Section 1.3c) as its internal plateau/
pruning criterion instead of a hardcoded per-family metric. Every
estimator also now stores the resolved value on `self.stopping_metric_used_`
after fitting, which `get_fitted_summary()` (Section 1.2) surfaces into
every ledger row automatically — so a leaderboard row now shows not just
what a model's chosen capacity was (`n_iter_`, `n_estimators_`,
`ccp_alpha_selected`, ...) but what metric that choice was made against.
This closes the exact "every one of these estimators is privately tuning
its own internal capacity for a different objective than the outer
funnel" inconsistency. Two
per-family notes on how the swap was implemented, since "just call
`_score_by_name` instead" isn't equally mechanical for all six:

- **`RFPlateauClassifier`** has no explicit held-out validation split at
  all — it plateaus on scikit-learn's out-of-bag (OOB) estimate, which
  is already "free" (computed from the trees that didn't see each row,
  no data withheld from training). Converting this to an explicit
  leakage-safe validation split would give up that efficiency advantage
  for no real benefit, so instead this estimator now derives a
  **metric-aware OOB score** directly from `RandomForestClassifier`'s
  own `oob_decision_function_` (the per-row, per-class OOB vote
  fractions scikit-learn already computes when `oob_score=True`) —
  `argmax` gives each row's OOB-predicted class, and `_score_by_name`
  scores that against the true labels using whatever `SELECTION_METRIC`
  is, rather than the plain OOB accuracy `1 - rf.oob_score_`.
  This costs nothing extra computationally (the decision function is
  already being computed) and keeps the "every row's information is
  used, none withheld for validation" property that is the entire
  reason to prefer OOB over an explicit split for a forest.
- **`GBCWarmStartClassifier`** already had a leakage-safe validation
  split (`X_val`/`y_val`, via `_leakage_safe_val_split`) and monitored
  validation log-loss/deviance on it. The swap replaces the deviance
  computation with `_score_by_name(metric_name, y_val, gbc.predict(X_val))`
  and flips the comparison direction accordingly (deviance is minimized;
  `_score_by_name`'s metrics are maximized) — mechanical, since the
  split and the per-chunk fitted model were already there.
"""


# %%
class StratifiedEarlyStoppingHGBC(BaseEstimator, ClassifierMixin):
    """Per-class stratified fit/val split (n<2 routed entirely into fit)
    — replaces HistGBC's native early_stopping=True (unstratified,
    silently drops rare classes). `max_iter_ceiling` has no hardcoded
    module-specific default here — every call site passes its own value
    explicitly. `stopping_metric` — see Section 1.4 markdown."""

    def __init__(self, val_fraction=0.15, chunk=25, patience=4, tol=1e-4,
                 max_iter_ceiling=1000, learning_rate=0.05, max_depth=6,
                 max_leaf_nodes=31, min_samples_leaf=20, random_state=SEED,
                 weighting="balanced", damping=0.5, max_ratio=10.0,
                 stopping_metric=None, record_history=False):
        self.val_fraction = val_fraction
        self.chunk = chunk
        self.patience = patience
        self.tol = tol
        self.max_iter_ceiling = max_iter_ceiling
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.weighting = weighting
        self.damping = damping
        self.max_ratio = max_ratio
        self.stopping_metric = stopping_metric
        self.n_iter_ = None
        self.best_val_f1_ = None
        self.model_ = None
        self.n_singleton_classes_ = None
        self.stopping_metric_used_ = None
        self.record_history = record_history  # [NEW]

    @staticmethod
    def _stratified_split_allow_singletons(y, val_fraction, random_state):
        rng = np.random.RandomState(random_state)
        fit_idx, val_idx, singleton_classes = [], [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            n = len(idx)
            if n < 2:
                fit_idx.extend(idx)
                singleton_classes.append(cls)
                continue
            idx = rng.permutation(idx)
            n_val = max(1, int(round(n * val_fraction)))
            n_val = min(n_val, n - 1)
            val_idx.extend(idx[:n_val])
            fit_idx.extend(idx[n_val:])
        return np.array(sorted(fit_idx)), np.array(sorted(val_idx)), singleton_classes

    def _sample_weight(self, y):
        if self.weighting == "none":
            return np.ones(len(y))
        return sample_weight_array(y, class_weight_map(y, self.weighting, self.damping, self.max_ratio))

    def fit(self, X, y, sample_weight=None):
        # sample_weight param kept for interface symmetry with the other
        # estimators — weighting is driven by self.weighting via the
        # constructor, exactly as intended.
        metric_name = self.stopping_metric or SELECTION_METRIC
        self.stopping_metric_used_ = metric_name
        self.history_, self.history_metric_ = [], metric_name  # [NEW]
        y_arr = np.asarray(y)
        fit_idx, val_idx, singleton_classes = self._stratified_split_allow_singletons(
            y_arr, self.val_fraction, self.random_state)
        self.n_singleton_classes_ = len(singleton_classes)
        if singleton_classes:
            print(f"    [StratifiedEarlyStoppingHGBC] {len(singleton_classes)} class(es) "
                  f"with n<2 excluded from validation split: {singleton_classes}")

        X_fit = X.iloc[fit_idx] if hasattr(X, "iloc") else X[fit_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_fit, y_val = y_arr[fit_idx], y_arr[val_idx]

        if len(val_idx) == 0:
            n_iter = self.chunk * self.patience
            sw_full = self._sample_weight(y_arr)
            self.n_iter_ = n_iter
            self.model_ = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate, max_depth=self.max_depth,
                max_iter=n_iter, max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                early_stopping=False, random_state=self.random_state)
            self.model_.fit(X, y_arr, sample_weight=sw_full)
            return self

        best_score, best_n, stall, trace = -np.inf, self.chunk, 0, []
        n_iter = self.chunk
        while n_iter <= self.max_iter_ceiling:
            m = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate, max_depth=self.max_depth,
                max_iter=n_iter, max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                early_stopping=False, random_state=self.random_state)
            sw_fit = self._sample_weight(y_fit)
            m.fit(X_fit, y_fit, sample_weight=sw_fit)
            val_score = _score_by_name(metric_name, y_val, m.predict(X_val))
            trace.append((n_iter, val_score))
            if self.record_history:
                train_score = _score_by_name(metric_name, y_fit, m.predict(X_fit))  # [NEW]
                self.history_.append({"step": n_iter, "train_score": train_score, "val_score": val_score})
            if val_score > best_score + self.tol:
                best_score, best_n, stall = val_score, n_iter, 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
            n_iter += self.chunk

        self.n_iter_, self.best_val_f1_, self.trace_ = best_n, best_score, trace
        sw_full = self._sample_weight(y_arr)
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate, max_depth=self.max_depth,
            max_iter=best_n, max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            early_stopping=False, random_state=self.random_state)
        self.model_.fit(X, y_arr, sample_weight=sw_full)
        return self

        best_score, best_n, stall, trace = -np.inf, self.chunk, 0, []
        n_iter = self.chunk
        while n_iter <= self.max_iter_ceiling:
            m = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate, max_depth=self.max_depth,
                max_iter=n_iter, max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                early_stopping=False, random_state=self.random_state)
            sw_fit = self._sample_weight(y_fit)
            m.fit(X_fit, y_fit, sample_weight=sw_fit)
            val_score = _score_by_name(metric_name, y_val, m.predict(X_val))
            trace.append((n_iter, val_score))
            if val_score > best_score + self.tol:
                best_score, best_n, stall = val_score, n_iter, 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
            n_iter += self.chunk

        self.n_iter_, self.best_val_f1_, self.trace_ = best_n, best_score, trace
        sw_full = self._sample_weight(y_arr)
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate, max_depth=self.max_depth,
            max_iter=best_n, max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            early_stopping=False, random_state=self.random_state)
        self.model_.fit(X, y_arr, sample_weight=sw_full)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
class _HGBCWeightedEarlyStop(BaseEstimator, ClassifierMixin):
    """[CONSOLIDATED] Same chunked, leakage-safe stratified early-stopping
    pattern as `StratifiedEarlyStoppingHGBC` above, but takes
    `sample_weight` EXTERNALLY at `.fit()` time — this is what lets it
    plug straight into `_select_winners`/`dispatch_fit`'s generic
    sample_weight/resample dispatch, as the HistGBC search proxy.
    `stopping_metric` — see Section 1.4 markdown."""

    def __init__(self, val_fraction=0.15, chunk=25, patience=4, tol=1e-4,
                 max_iter_ceiling=500, learning_rate=0.05, max_depth=6,
                 random_state=SEED, stopping_metric=None):
        self.val_fraction = val_fraction
        self.chunk = chunk
        self.patience = patience
        self.tol = tol
        self.max_iter_ceiling = max_iter_ceiling
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.stopping_metric = stopping_metric
        self.stopping_metric_used_ = None

    def fit(self, X, y, sample_weight=None):
        metric_name = self.stopping_metric or SELECTION_METRIC
        self.stopping_metric_used_ = metric_name
        y_arr = np.asarray(y)
        fit_idx, val_idx = _leakage_safe_val_split(y_arr, self.val_fraction, self.random_state)
        if len(val_idx) == 0:
            n_iter = self.chunk * self.patience
            self.n_iter_ = n_iter
            self.model_ = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate, max_depth=self.max_depth,
                max_iter=n_iter, random_state=self.random_state)
            self.model_.fit(X, y_arr, sample_weight=sample_weight)
            return self

        X_fit = X.iloc[fit_idx] if hasattr(X, "iloc") else X[fit_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_fit, y_val = y_arr[fit_idx], y_arr[val_idx]
        sw_fit = sample_weight[fit_idx] if sample_weight is not None else None

        best_score, best_n, stall, n_iter = -np.inf, self.chunk, 0, self.chunk
        while n_iter <= self.max_iter_ceiling:
            m = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate, max_depth=self.max_depth,
                max_iter=n_iter, random_state=self.random_state)
            m.fit(X_fit, y_fit, sample_weight=sw_fit)
            val_score = _score_by_name(metric_name, y_val, m.predict(X_val))
            if val_score > best_score + self.tol:
                best_score, best_n, stall = val_score, n_iter, 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
            n_iter += self.chunk

        self.n_iter_ = best_n
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate, max_depth=self.max_depth,
            max_iter=best_n, random_state=self.random_state)
        self.model_.fit(X, y_arr, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
class RFPlateauClassifier(BaseEstimator, ClassifierMixin):
    """RandomForest grown in chunks to a metric-aware OOB plateau. Both
    `predict` and `predict_proba` always exposed. `stopping_metric`
    — see Section 1.4 markdown for how the OOB-decision-function-based
    scoring works."""

    def __init__(self, chunk_size=50, max_estimators=2000, tol=1e-4, patience=3,
                 max_features="sqrt", random_state=SEED, stopping_metric=None,
                 record_history=False):
        self.chunk_size = chunk_size
        self.max_estimators = max_estimators
        self.tol = tol
        self.patience = patience
        self.max_features = max_features
        self.random_state = random_state
        self.stopping_metric = stopping_metric
        self.stopping_metric_used_ = None
        self.record_history = record_history  # [NEW] opt-in (X, y)_step train/val trace, gated off by default

    @staticmethod
    def _oob_score(rf, y_arr, metric_name):
        oob_dec = getattr(rf, "oob_decision_function_", None)
        if oob_dec is None:
            return float(rf.oob_score_)
        valid = ~np.isnan(oob_dec).any(axis=1)
        if valid.sum() < 2 or len(np.unique(y_arr[valid])) < 2:
            return float(rf.oob_score_)
        oob_pred = rf.classes_[np.argmax(oob_dec[valid], axis=1)]
        return _score_by_name(metric_name, y_arr[valid], oob_pred)

    def fit(self, X, y, sample_weight=None):
        metric_name = self.stopping_metric or SELECTION_METRIC
        self.stopping_metric_used_ = metric_name
        self.history_, self.history_metric_ = [], metric_name  # [NEW]
        y_arr = np.asarray(y)
        rf = RandomForestClassifier(
            n_estimators=self.chunk_size, warm_start=True, oob_score=True,
            max_features=self.max_features, random_state=self.random_state, n_jobs=-1)
        n_est, stall, prev_score = self.chunk_size, 0, None
        while n_est <= self.max_estimators:
            rf.set_params(n_estimators=n_est)
            rf.fit(X, y_arr, sample_weight=sample_weight)
            score = self._oob_score(rf, y_arr, metric_name)
            if self.record_history:
                # [NEW] train side = in-bag vote (rf.predict), val side = OOB
                # (already free — no separate held-out split exists for a forest).
                train_score = _score_by_name(metric_name, y_arr, rf.predict(X))
                self.history_.append({"step": n_est, "train_score": train_score, "val_score": score})
            if prev_score is not None and (score - prev_score) < self.tol:
                stall += 1
            else:
                stall = 0
            if stall >= self.patience:
                break
            prev_score = score
            n_est += self.chunk_size
        self.model_, self.n_estimators_ = rf, n_est
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
class GBCWarmStartClassifier(BaseEstimator, ClassifierMixin):
    """GradientBoostingClassifier grown in chunks with manual, leakage-
    safe validation early stopping. `stopping_metric` — see
    Section 1.4 markdown (the same `_score_by_name` criterion every other
    estimator uses, resolved from SELECTION_METRIC by default, rather than
    validation log-loss).

    sklearn's GradientBoostingClassifier
    (multinomial-deviance loss) requires y's unique values to be an
    EXACT contiguous range 0..n_classes-1. A CV/LOO fold that drops the
    sole remaining example of some global class label leaves a GAP in
    that fold's y — sklearn raises "Invalid classes inferred from unique
    values of y" in that case. Fixed by locally re-encoding y to
    whatever classes ARE present in THIS fit() call, fitting/predicting
    in that local contiguous space, then mapping predict() back to the
    original (global) label values."""

    def __init__(self, chunk=25, patience=4, tol=1e-4, max_estimators=2000,
                 learning_rate=0.05, max_depth=4, subsample=0.8,
                 val_fraction=0.15, random_state=SEED, stopping_metric=None,
                 record_history=False):
        self.chunk = chunk
        self.patience = patience
        self.tol = tol
        self.max_estimators = max_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.stopping_metric = stopping_metric
        self.stopping_metric_used_ = None
        self.record_history = record_history  # [NEW]

    def fit(self, X, y, sample_weight=None):
        metric_name = self.stopping_metric or SELECTION_METRIC
        self.stopping_metric_used_ = metric_name
        self.history_, self.history_metric_ = [], metric_name  # [NEW]
        self.imputer_ = IterativeImputer(random_state=self.random_state, max_iter=10)
        X_imp = self.imputer_.fit_transform(X)
        if hasattr(X, "columns"):
            X_imp = pd.DataFrame(X_imp, columns=X.columns, index=X.index)

        y_arr = np.asarray(y)
        # [FIX] local contiguous re-encoding — see class docstring.
        self.local_le_ = LabelEncoder().fit(y_arr)
        y_local = self.local_le_.transform(y_arr)

        tr_idx, val_idx = _leakage_safe_val_split(y_local, self.val_fraction, self.random_state)

        if len(val_idx) == 0:
            gbc = GradientBoostingClassifier(
                n_estimators=self.chunk * self.patience, learning_rate=self.learning_rate,
                max_depth=self.max_depth, subsample=self.subsample, random_state=self.random_state)
            gbc.fit(X_imp, y_local, sample_weight=sample_weight)
            self.model_, self.n_estimators_ = gbc, gbc.n_estimators_
            return self

        X_fit = X_imp.iloc[tr_idx] if hasattr(X_imp, "iloc") else X_imp[tr_idx]
        X_val = X_imp.iloc[val_idx] if hasattr(X_imp, "iloc") else X_imp[val_idx]
        y_fit, y_val = y_local[tr_idx], y_local[val_idx]
        sw_fit = sample_weight[tr_idx] if sample_weight is not None else None

        gbc = GradientBoostingClassifier(
            n_estimators=self.chunk, learning_rate=self.learning_rate,
            max_depth=self.max_depth, subsample=self.subsample,
            warm_start=True, random_state=self.random_state)
        best_score, best_n, stall, n_est = -np.inf, self.chunk, 0, self.chunk
        while n_est <= self.max_estimators:
            gbc.set_params(n_estimators=n_est)
            gbc.fit(X_fit, y_fit, sample_weight=sw_fit)
            val_score = _score_by_name(metric_name, y_val, gbc.predict(X_val))
            if self.record_history:
                train_score = _score_by_name(metric_name, y_fit, gbc.predict(X_fit))  # [NEW]
                self.history_.append({"step": n_est, "train_score": train_score, "val_score": val_score})
            if val_score > best_score + self.tol:
                best_score, best_n, stall = val_score, n_est, 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
            n_est += self.chunk

        final = GradientBoostingClassifier(
            n_estimators=best_n, learning_rate=self.learning_rate,
            max_depth=self.max_depth, subsample=self.subsample, random_state=self.random_state)
        final.fit(X_imp, y_local, sample_weight=sample_weight)
        self.model_, self.n_estimators_ = final, best_n
        return self

        X_fit = X_imp.iloc[tr_idx] if hasattr(X_imp, "iloc") else X_imp[tr_idx]
        X_val = X_imp.iloc[val_idx] if hasattr(X_imp, "iloc") else X_imp[val_idx]
        y_fit, y_val = y_local[tr_idx], y_local[val_idx]
        sw_fit = sample_weight[tr_idx] if sample_weight is not None else None

        gbc = GradientBoostingClassifier(
            n_estimators=self.chunk, learning_rate=self.learning_rate,
            max_depth=self.max_depth, subsample=self.subsample,
            warm_start=True, random_state=self.random_state)
        best_score, best_n, stall, n_est = -np.inf, self.chunk, 0, self.chunk
        while n_est <= self.max_estimators:
            gbc.set_params(n_estimators=n_est)
            gbc.fit(X_fit, y_fit, sample_weight=sw_fit)
            val_score = _score_by_name(metric_name, y_val, gbc.predict(X_val))
            if val_score > best_score + self.tol:
                best_score, best_n, stall = val_score, n_est, 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
            n_est += self.chunk

        final = GradientBoostingClassifier(
            n_estimators=best_n, learning_rate=self.learning_rate,
            max_depth=self.max_depth, subsample=self.subsample, random_state=self.random_state)
        final.fit(X_imp, y_local, sample_weight=sample_weight)
        self.model_, self.n_estimators_ = final, best_n
        return self

    def predict(self, X):
        X_imp = self.imputer_.transform(X)
        if hasattr(X, "columns"):
            X_imp = pd.DataFrame(X_imp, columns=X.columns, index=X.index)
        pred_local = self.model_.predict(X_imp)
        # [FIX] map back to original (global) label values.
        return self.local_le_.inverse_transform(pred_local)

    def predict_proba(self, X):
        """Columns are now in LOCAL-fit class order (self.local_le_.classes_),
        which only diverges from full-global width during a CV/LOO fold
        that legitimately never saw some rare class. The final branch
        refit always sees the FULL branch data, so no behavior change
        there."""
        X_imp = self.imputer_.transform(X)
        if hasattr(X, "columns"):
            X_imp = pd.DataFrame(X_imp, columns=X.columns, index=X.index)
        return self.model_.predict_proba(X_imp)
class DTreeCVPrunedClassifier(BaseEstimator, ClassifierMixin):
    """DecisionTree with ccp_alpha selected by CV over its OWN
    cost-complexity pruning path, re-derived every fit.

    Adaptive LOO: when the fold's minority class is very rare or the
    branch is tiny, Stratified-K-Fold degrades to genuine `LeaveOneOut`
    instead of clamping toward K=2, bounded by `MAX_LOO_N`.

    `MAX_ALPHAS` [CONSOLIDATED] subsamples `cost_complexity_pruning_path`'s
    candidate alphas to at most this many evenly-spaced values before the
    CV loop.

    `stopping_metric` — when given (or resolved from
    SELECTION_METRIC by default), maps to the corresponding
    `cross_val_score(scoring=...)` string (`"balanced_accuracy"`,
    `"accuracy"`, or `"f1_macro"`) instead of a hardcoded
    `scoring="f1_macro"`. The `scoring` constructor argument is kept,
    unchanged in meaning, as the fallback used only when `stopping_metric`
    doesn't map to a recognized value."""

    MAX_ALPHAS = 30
    _METRIC_TO_SKLEARN_SCORING = {
        "bal_accuracy": "balanced_accuracy", "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy", "macro_f1": "f1_macro",
    }

    def __init__(self, cv_splits=5, scoring="f1_macro", random_state=SEED,
                 loo_min_class_count=LOO_MIN_CLASS_COUNT_THRESHOLD,
                 loo_min_branch_train=LOO_MIN_BRANCH_TRAIN_THRESHOLD,
                 stopping_metric=None):
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.random_state = random_state
        self.loo_min_class_count = loo_min_class_count
        self.loo_min_branch_train = loo_min_branch_train
        self.stopping_metric = stopping_metric
        self.stopping_metric_used_ = None

    def fit(self, X, y, sample_weight=None):
        metric_name = self.stopping_metric or SELECTION_METRIC
        sklearn_scoring = self._METRIC_TO_SKLEARN_SCORING.get(metric_name, self.scoring)
        self.stopping_metric_used_ = metric_name

        y_arr = np.asarray(y)
        path = DecisionTreeClassifier(random_state=self.random_state).cost_complexity_pruning_path(X, y_arr)
        ccp_alphas = np.unique(path.ccp_alphas)
        self.n_alphas_full_ = int(len(ccp_alphas))
        if len(ccp_alphas) > self.MAX_ALPHAS:
            idx = np.linspace(0, len(ccp_alphas) - 1, self.MAX_ALPHAS).round().astype(int)
            ccp_alphas = ccp_alphas[np.unique(idx)]
        self.n_alphas_used_ = int(len(ccp_alphas))

        min_class_count = int(np.unique(y_arr, return_counts=True)[1].min())
        use_loo = ((min_class_count <= self.loo_min_class_count
                    or len(y_arr) <= self.loo_min_branch_train)
                   and len(y_arr) <= MAX_LOO_N)

        if use_loo:
            cv = LeaveOneOut()
            self.cv_mode_ = "LeaveOneOut"
            print(f"    [DTreeCVPrunedClassifier] adaptive LOO triggered "
                  f"(min_class_count={min_class_count}, n={len(y_arr)}) — using LeaveOneOut CV.")
        else:
            n_splits = max(2, min(self.cv_splits, min_class_count))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            self.cv_mode_ = f"StratifiedKFold(n_splits={n_splits})"

        best_alpha, best_score = 0.0, -np.inf
        for alpha in ccp_alphas:
            t = DecisionTreeClassifier(random_state=self.random_state, ccp_alpha=alpha)
            try:
                score = cross_val_score(t, X, y_arr, cv=cv, scoring=sklearn_scoring, n_jobs=1).mean()
            except ValueError:
                continue
            if score > best_score:
                best_score, best_alpha = score, alpha

        self.best_alpha_ = float(best_alpha)
        self.model_ = DecisionTreeClassifier(random_state=self.random_state, ccp_alpha=best_alpha)
        self.model_.fit(X, y_arr, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(X)
KNN_PER_CLASS_CLAMP = False # see AdaptiveKNNClassifier docstring.
# Left False here so fermi_fine.py (flat classification) is unaffected.
# fermi_merged.py and fermi_3levels.py set this to True near the
# top of their branch-level search, before the shared funnel runs — this
# file's globals are simply in-scope in those notebooks' kernel, not
# imported cross-file, so a plain reassignment there is sufficient.
class AdaptiveKNNClassifier(BaseEstimator, ClassifierMixin):
    """KNeighborsClassifier with n_neighbors clamped to the actual
    training-set size — degrades gracefully instead of raising.

    When KNN_PER_CLASS_CLAMP is True, k
    is ALSO clamped to the smallest class's count minus one — a branch's
    smallest class can be tiny even when the branch's total row count is
    not. Left off for fine.py's flat classification (unset by default)."""

    def __init__(self, n_neighbors=7):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        k = max(1, min(self.n_neighbors, len(y) - 1))
        if KNN_PER_CLASS_CLAMP:
            _, counts = np.unique(y, return_counts=True)
            k = max(1, min(k, int(counts.min()) - 1))
        self.model_ = KNeighborsClassifier(n_neighbors=k)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
class MLPLeakageSafeClassifier(BaseEstimator, ClassifierMixin):
    """CPU MLPClassifier wrapped with a leakage-safe, per-class
    stratified validation split for early stopping. `stopping_metric`
    — see Section 1.4 markdown."""

    def __init__(self, hidden_layer_sizes=(256, 128, 64), chunk_epochs=10,
                 patience=4, tol=1e-4, max_epochs=1000, val_fraction=0.15,
                 activation="relu", solver="adam", alpha=1e-4,
                 learning_rate_init=1e-3, random_state=SEED, stopping_metric=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.chunk_epochs = chunk_epochs
        self.patience = patience
        self.tol = tol
        self.max_epochs = max_epochs
        self.val_fraction = val_fraction
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self.stopping_metric = stopping_metric
        self.stopping_metric_used_ = None

    def _new_net(self, max_iter=1):
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
            solver=self.solver, alpha=self.alpha, learning_rate_init=self.learning_rate_init,
            max_iter=max_iter, warm_start=True, early_stopping=False,
            random_state=self.random_state)

    def fit(self, X, y, sample_weight=None):
        metric_name = self.stopping_metric or SELECTION_METRIC
        self.stopping_metric_used_ = metric_name
        y_arr = np.asarray(y)
        classes = self.classes_ = np.unique(y_arr)
        fit_idx, val_idx = _leakage_safe_val_split(y_arr, self.val_fraction, self.random_state)
        X_fit = X.iloc[fit_idx] if hasattr(X, "iloc") else X[fit_idx]
        y_fit = y_arr[fit_idx]
        sw_fit = None if sample_weight is None else np.asarray(sample_weight)[fit_idx]
        has_val = len(val_idx) > 0

        net = self._new_net(max_iter=1)

        if not has_val:
            net.set_params(max_iter=self.max_epochs, warm_start=False)
            net.fit(X_fit, y_fit, sample_weight=sw_fit)
            self.model_, self.n_epochs_ = net, self.max_epochs
            self.best_val_f1_ = None
            return self

        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_val = y_arr[val_idx]

        best_score, best_epoch, stall, best_params = -np.inf, self.chunk_epochs, 0, None
        epoch = 0
        while epoch < self.max_epochs:
            for _ in range(self.chunk_epochs):
                net.partial_fit(X_fit, y_fit, classes=classes, sample_weight=sw_fit)
                epoch += 1
                if epoch >= self.max_epochs:
                    break
            val_score = _score_by_name(metric_name, y_val, net.predict(X_val))
            if val_score > best_score + self.tol:
                best_score, best_epoch, stall = val_score, epoch, 0
                best_params = ([c.copy() for c in net.coefs_], [b.copy() for b in net.intercepts_])
            else:
                stall += 1
                if stall >= self.patience:
                    break

        if best_params is not None:
            net.coefs_, net.intercepts_ = best_params

        self.model_, self.n_epochs_, self.best_val_f1_ = net, best_epoch, best_score
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# %%
"""
### GPU/Torch MLP variant

Identical chunked-epoch, leakage-safe-validation, early-stopping
structure to `MLPLeakageSafeClassifier` above, but runs under
`torch.amp.autocast('cuda')` with `pin_memory=True` DataLoaders and
`.to(device, non_blocking=True)` transfers. VRAM is explicitly reclaimed
(`del`, `gc.collect()`, `torch.cuda.empty_cache()`) at the end of every
`fit()` call. `MLP_ARCHITECTURES` is unchanged; this is purely a backend
swap selected by `USE_GPU` in `MODEL_SPECS` below. `stopping_metric`
— same treatment as the CPU variant.
"""

# %%
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = torch.cuda.is_available()
except Exception:
    _TORCH_AVAILABLE = False
class MLPLeakageSafeClassifierTorch(BaseEstimator, ClassifierMixin):
    """PyTorch reimplementation of MLPLeakageSafeClassifier."""

    def __init__(self, hidden_layer_sizes=(256, 128, 64), chunk_epochs=10,
                 patience=4, tol=1e-4, max_epochs=1000, val_fraction=0.15,
                 lr=1e-3, weight_decay=1e-4, batch_size=256, random_state=SEED,
                 stopping_metric=None, record_history=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.chunk_epochs = chunk_epochs
        self.patience = patience
        self.tol = tol
        self.max_epochs = max_epochs
        self.val_fraction = val_fraction
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state
        self.stopping_metric = stopping_metric
        self.stopping_metric_used_ = None
        self.record_history = record_history  # [NEW]

    def _build_net(self, n_in, n_out):
        layers, prev = [], n_in
        for h in self.hidden_layer_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        return nn.Sequential(*layers)

    def fit(self, X, y, sample_weight=None):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("MLPLeakageSafeClassifierTorch requires a CUDA-capable "
                                "torch install; USE_GPU should not select this builder "
                                "on a machine without CUDA.")
        metric_name = self.stopping_metric or SELECTION_METRIC
        self.stopping_metric_used_ = metric_name
        self.history_, self.history_metric_ = [], metric_name  # [NEW]
        torch.manual_seed(self.random_state)
        device = torch.device("cuda")

        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        self.classes_ = classes
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[v] for v in y_arr])

        X_arr = X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0)  # torch net has no native NaN routing

        fit_idx, val_idx = _leakage_safe_val_split(y_idx, self.val_fraction, self.random_state)
        has_val = len(val_idx) > 0

        net = self._build_net(X_arr.shape[1], len(classes)).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # The net's last layer is linear (logits); CrossEntropyLoss applies
        # log-softmax itself. reduction="none" so per-row class weights can
        # be applied exactly instead of via a duplicating bootstrap resample.
        criterion = nn.CrossEntropyLoss(reduction="none")
        scaler_amp = torch.amp.GradScaler("cuda")

        sw = np.ones(len(y_idx), dtype=np.float32) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float32)
        X_fit_t = torch.tensor(X_arr[fit_idx])
        y_fit_t = torch.tensor(y_idx[fit_idx], dtype=torch.long)
        w_fit_t = torch.tensor(sw[fit_idx])
        loader = DataLoader(TensorDataset(X_fit_t, y_fit_t, w_fit_t), batch_size=self.batch_size,
                             shuffle=True, pin_memory=True)

        def _run_epoch():
            net.train()
            for xb, yb, wb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                wb = wb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    out = net(xb)
                    loss = (criterion(out.float(), yb) * wb).sum() / wb.sum()
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()

        def _predict_idx(X_np):
            net.eval()
            with torch.no_grad():  # full precision: fp16 logits overflow on extreme inputs
                xb = torch.tensor(X_np, dtype=torch.float32).to(device, non_blocking=True)
                logits = net(xb)
                return logits.argmax(dim=1).cpu().numpy()

        try:
            if not has_val:
                for _ in range(self.max_epochs):
                    _run_epoch()
                self.n_epochs_, self.best_val_f1_ = self.max_epochs, None
            else:
                X_fit_np = X_arr[fit_idx]  # [NEW] kept as numpy for the record_history train-side predict
                X_val_np = X_arr[val_idx]
                y_val_idx = y_idx[val_idx]
                best_score, best_epoch, stall, best_state = -np.inf, self.chunk_epochs, 0, None
                epoch = 0
                while epoch < self.max_epochs:
                    for _ in range(self.chunk_epochs):
                        _run_epoch()
                        epoch += 1
                        if epoch >= self.max_epochs:
                            break
                    val_pred_idx = _predict_idx(X_val_np)
                    val_score = _score_by_name(metric_name, y_val_idx, val_pred_idx)
                    if self.record_history:
                        train_pred_idx = _predict_idx(X_fit_np)  # [NEW]
                        train_score = _score_by_name(metric_name, y_idx[fit_idx], train_pred_idx)
                        self.history_.append({"step": epoch, "train_score": train_score, "val_score": val_score})
                    if val_score > best_score + self.tol:
                        best_score, best_epoch, stall = val_score, epoch, 0
                        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    else:
                        stall += 1
                        if stall >= self.patience:
                            break
                if best_state is not None:
                    net.load_state_dict(best_state)
                self.n_epochs_, self.best_val_f1_ = best_epoch, best_score

            # Persist inference-only weights (CPU numpy) so the estimator
            # survives after we free the CUDA graph below.
            self._state_dict_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            self._n_in, self._n_out = X_arr.shape[1], len(classes)
        finally:
            # VRAM cleanup — mandatory per branch/topology loop.
            del net, optimizer, scaler_amp
            gc.collect()
            torch.cuda.empty_cache()
        return self

    def _rebuild_and_load(self):
        device = torch.device("cuda")
        net = self._build_net(self._n_in, self._n_out).to(device)
        net.load_state_dict(self._state_dict_cpu)
        net.eval()
        return net, device

    def predict_proba(self, X):
        """Softmax over the logits — the probabilities CrossEntropyLoss was
        trained against. Without this, every caller fell back to a one-hot
        matrix from predict(): fake 1.0 confidences and hard routing in
        the hierarchical chains."""
        X_arr = X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        net, device = self._rebuild_and_load()
        try:
            # Full precision (no autocast): in fp16, large logits from extreme standardized inputs
            # overflow to inf and the softmax returns NaN probabilities. Inference is cheap anyway.
            with torch.no_grad():
                xb = torch.tensor(X_arr, dtype=torch.float32).to(device, non_blocking=True)
                logits = net(xb)
            return torch.softmax(logits.float(), dim=1).cpu().numpy()
        finally:
            del net
            gc.collect()
            torch.cuda.empty_cache()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# %%
"""
## Section 1.5 — Evaluation harness

`evaluate_model` computes the full metric quartet — accuracy, balanced
accuracy, macro-F1, Cohen's kappa — on every call, regardless of which
one `SELECTION_METRIC` currently points at. `split="train"/"test"` only
changes the printed header/label and the returned dict's semantics,
never the computation itself. `labels=range(len(class_names))` is passed
explicitly to `classification_report` because sklearn otherwise infers
the label set from whichever encoded values actually appear in a given
split/resample. All four functions in this section now accept
an optional `logger` parameter. `evaluate_model`'s verbose block (which
can be large — a full per-class `classification_report`) is routed to
`logger.info` (file-only) rather than `logger.notice`, when a logger is
given — detail like this stays OFF the console; the console-worthy summary line for a finished model is instead
emitted by whichever caller invokes `evaluate_model` (`fit_and_log`,
`select_and_refit_val_only`, ...) at `NOTICE`. `plot_confusion_matrix`/
`plot_comparison_table`/`save_per_sample_predictions`'s "Saved -> ..."
confirmations route to `NOTICE` (console + file) when a logger is given
— one of the console categories (errors, warnings, save confirmations, milestones). Omitting `logger` everywhere in this section falls back to
plain `print`.
"""


# %%
def evaluate_model(name, y_true, y_pred, class_names, label_set="fine",
                    split: str = "test", verbose: bool = True,
                    logger: Optional[logging.Logger] = None):
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    if verbose:
        lines = [
            f"\n{'━' * 65}",
            f"  {name}  [{label_set} / {split}]",
            f"{'━' * 65}",
            f"  Accuracy         : {acc:.4f}",
            f"  Balanced Accuracy: {bal_acc:.4f}",
            f"  Precision (macro): {precision_macro:.4f}",
            f"  Recall (macro)   : {recall_macro:.4f}",
            f"  Macro-F1         : {macro_f1:.4f}",
            f"  Cohen's Kappa    : {kappa:.4f}",
        ]
        if split == "test":
            lines.append(f"\n{classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0)}")
        block = "\n".join(lines)
        if logger is not None:
            logger.info(block)
        else:
            print(block)

    return {"name": name, "label_set": label_set, "split": split,
            "accuracy": acc, "bal_accuracy": bal_acc, "macro_f1": macro_f1, "kappa": kappa,
            "precision_macro": precision_macro, "recall_macro": recall_macro}
def plot_confusion_matrix(name, y_true, y_pred, class_names, label_set, plots_dir: Path,
                           logger: Optional[logging.Logger] = None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cmn = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    n = len(class_names)
    fig_h = max(6, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_h + 2, fig_h))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax, linewidths=0.3)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{name}  |  Normalised Confusion Matrix [{label_set}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = plots_dir / f"confusion_matrix_{_safe_filename(label_set, name)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    msg = f"  Saved -> {out_path}"
    (logger.notice(msg) if logger is not None else print(msg))
    return cm
def plot_comparison_table(results_list, plots_dir: Path, metric="bal_accuracy", title="",
                           filename=None, logger: Optional[logging.Logger] = None):
    """`metric` should be called with `SELECTION_METRIC` by every
    downstream caller (rather than a hardcoded "bal_accuracy") — this
    function itself stays metric-agnostic, it just plots whichever
    single column it's told to."""
    df_res = pd.DataFrame(results_list).sort_values(metric, ascending=True)
    if "name" not in df_res.columns:
        df_res["name"] = (
            df_res.get("model_name", pd.Series(["?"] * len(df_res), index=df_res.index)).astype(str)
            + " | " + df_res.get("feature_set", pd.Series(["?"] * len(df_res), index=df_res.index)).astype(str)
            + " | " + df_res.get("variant", pd.Series(["?"] * len(df_res), index=df_res.index)).astype(str)
        )
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(df_res)))

    fig, ax = plt.subplots(figsize=(10, max(5, len(df_res) * 0.38)))
    bars = ax.barh(df_res["name"], df_res[metric], color=colors, edgecolor="k", linewidth=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"Model Comparison — {metric}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(df_res[metric].max(), color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = plots_dir / (filename or f"comparison_{_safe_filename(title or metric)}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    msg = f"  Saved -> {out_path}"
    (logger.notice(msg) if logger is not None else print(msg))
def save_per_sample_predictions(model, x, y_true_enc, class_names, output_path,
                                  index=None, extra_cols: dict = None,
                                  logger: Optional[logging.Logger] = None):
    """Persists one row per test sample: true/pred label, correct flag,
    and — if the model supports predict_proba — confidence (top-1 proba),
    margin (top1-top2), and the probability mass the model put on the
    TRUE class. Sorted so the most useful rows for error analysis come
    first. `extra_cols` lets a caller attach routing context as extra
    columns."""
    y_pred_enc = model.predict(x)
    idx = np.asarray(index) if index is not None else np.arange(len(y_true_enc))

    df = pd.DataFrame({
        "row_index": idx,
        "true_label": np.asarray(class_names)[y_true_enc],
        "pred_label": np.asarray(class_names)[y_pred_enc],
        "correct": np.asarray(y_true_enc) == np.asarray(y_pred_enc),
    })

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x)
            if hasattr(proba, "to_numpy"):
                proba = proba.to_numpy()
            elif hasattr(proba, "values") and not isinstance(proba, np.ndarray):
                proba = proba.values
            proba = np.asarray(proba)
        except Exception as e:
            warn = f"  ⚠ predict_proba unavailable ({type(e).__name__}: {e}) — confidence columns will be NaN."
            (logger.warning(warn) if logger is not None else print(warn))

    if proba is not None:
        sorted_proba = np.sort(proba, axis=1)
        df["confidence"] = sorted_proba[:, -1]
        df["margin_top1_top2"] = (sorted_proba[:, -1] - sorted_proba[:, -2]
                                   if proba.shape[1] > 1 else sorted_proba[:, -1])
        df["true_class_proba"] = proba[np.arange(len(y_true_enc)), y_true_enc]
    else:
        df["confidence"] = np.nan
        df["margin_top1_top2"] = np.nan
        df["true_class_proba"] = np.nan

    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v

    errors = df[~df["correct"]].sort_values("confidence", ascending=False)
    correct = df[df["correct"]].sort_values("confidence", ascending=True)
    df_ranked = pd.concat([errors, correct], ignore_index=True)
    df_ranked.insert(0, "error_rank", np.arange(1, len(df_ranked) + 1))

    df_ranked.to_csv(output_path, index=False)
    n_err = int((~df_ranked["correct"]).sum())
    msg = (f"  Per-sample predictions saved -> {output_path}  "
           f"({n_err}/{len(df_ranked)} errors, {n_err/len(df_ranked):.2%})")
    (logger.notice(msg) if logger is not None else print(msg))
    return df_ranked


# %%
"""
## Section 1.6 — Generic fit/log drivers

`fit_and_log` and `run_histgbc` are the two universal entry points every
module's `run_model` wrapper eventually calls into. Both compute and
persist the full train + test metric quartet in every ledger entry.
`safe_run` wraps every individual (model, feature-set, weighting) fit
attempt in a try/except so one bad combination never aborts an entire
multi-hour sweep.

`_make_fit_and_log`/`make_run_histgbc` are now factored around
two small callbacks instead of a single bound `Path`:
`results_paths_fn(label_set) -> list[Path]` and, optionally,
`logger_fn(label_set) -> Optional[logging.Logger]`. This is what lets
ONE factory instance serve every branch/label_set trained during a
module's run while still writing each branch's rows to its own file and
its own log — the resolution happens **inside** the returned
`fit_and_log`/`run_histgbc` closures, using the `label_set` argument
those functions already receive per call. `bootstrap_module_io` (below) binds
a trivial `results_paths_fn = lambda label_set: [results_path]` and no
`logger_fn` — a flat, single-ledger, print-only layout. `io_for`
(Section 1.10) binds
`results_paths_fn = lambda label_set: [split_results_path, branch_path(label_set)]`
and `logger_fn = lambda label_set: get_section_logger(label_set, logs_dir)`
— the per-split/per-branch, logger-aware layout. Every caller of
`safe_run` may pass `results_path` as a single `Path` or a list of
`Path`s (multi-ledger), plus an optional `logger`.
"""


# %%
def _make_fit_and_log(results_paths_fn: Callable[[str], Sequence[Path]], plots_dir: Path,
                       logger_fn: Optional[Callable[[str], Optional[logging.Logger]]] = None):
    """Factory so every module gets its own fit_and_log bound to its own
    results-path resolution / plots_dir / logger resolution, while
    sharing the identical implementation — including the train+test
    double-evaluation."""

    def fit_and_log(model, x_tr, y_tr, x_te, y_te, label_set, model_name, feature_set,
                     variant, class_names, stopping_rule, plot=False, sample_weight=None):
        logger = logger_fn(label_set) if logger_fn is not None else None
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if sample_weight is not None:
                if hasattr(model, "steps"):
                    last_step_name = model.steps[-1][0]
                    model.fit(x_tr, y_tr, **{f"{last_step_name}__sample_weight": sample_weight})
                else:
                    model.fit(x_tr, y_tr, sample_weight=sample_weight)
            else:
                model.fit(x_tr, y_tr)
        fit_sec = time.perf_counter() - t0

        conv_warns = [w for w in caught if "convergence" in str(w.message).lower()]
        if conv_warns:
            warn = f"  ⚠  ConvergenceWarning from {model_name}: solver hit max_iter before tol."
            (logger.warning(warn) if logger is not None else print(warn))
            stopping_rule += " [DID NOT CONVERGE — hit max_iter]"

        # Train AND test evaluation. Diagnostic only — winner-picks
        # everywhere continue to select on SELECTION_METRIC-on-test;
        # train metrics are never compared/used to break ties.
        y_pred_train = model.predict(x_tr)
        train_metrics = evaluate_model(
            name=f"{model_name} [{variant}] [TRAIN — refit metric, NOT a held-out "
                 f"k-fold/LOO validation score]",
            y_true=y_tr, y_pred=y_pred_train, class_names=class_names,
            label_set=label_set, split="train", verbose=True, logger=logger)
        y_pred_test = model.predict(x_te)
        test_metrics = evaluate_model(name=f"{model_name} [{variant}] [TEST — held-out]",
                                       y_true=y_te, y_pred=y_pred_test, class_names=class_names,
                                       label_set=label_set, split="test", verbose=True, logger=logger)
        if plot:
            plot_confusion_matrix(name=f"{model_name}_{variant}", y_true=y_te, y_pred=y_pred_test,
                                   class_names=class_names, label_set=label_set, plots_dir=plots_dir,
                                   logger=logger)

        run_id = make_run_id(label_set, model_name, feature_set, variant)
        entry = {
            "model_name": model_name, "variant": variant, "feature_set": feature_set,
            "label_set": label_set,
            "n_features": x_tr.shape[1], "stopping_rule": stopping_rule,
            "fit_seconds": round(fit_sec, 2),
            "test_accuracy": test_metrics["accuracy"], "test_bal_accuracy": test_metrics["bal_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"], "test_kappa": test_metrics["kappa"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "train_accuracy": train_metrics["accuracy"], "train_bal_accuracy": train_metrics["bal_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"], "train_kappa": train_metrics["kappa"],
            "train_precision_macro": train_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "accuracy": test_metrics["accuracy"], "bal_accuracy": test_metrics["bal_accuracy"],
            "macro_f1": test_metrics["macro_f1"], "kappa": test_metrics["kappa"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
        }
        save_result_multi(run_id, entry, results_paths_fn(label_set), logger=logger)
        return model, test_metrics

    return fit_and_log
def make_run_histgbc(results_paths_fn: Callable[[str], Sequence[Path]],
                      logger_fn: Optional[Callable[[str], Optional[logging.Logger]]] = None):
    """Shared HistGBC driver (bypasses fit_and_log — no Pipeline wrapper)
    — same train+test double-eval and schema. Now also persists
    the full constructor `hyperparameters` dict (§5 of the redesign —
    previously only `fitted_summary` was captured for HistGBC, unlike
    every Pipeline-wrapped model)."""

    def run_histgbc(x_tr, y_tr, x_te, y_te, tag, weighting, class_names, label_set,
                     variant_prefix="featset", max_iter_ceiling=1000):
        logger = logger_fn(label_set) if logger_fn is not None else None
        hgbc = StratifiedEarlyStoppingHGBC(
            val_fraction=0.15, chunk=25, patience=4, tol=1e-4, max_iter_ceiling=max_iter_ceiling,
            learning_rate=0.05, max_depth=6, random_state=SEED,
            weighting=weighting, damping=0.5, max_ratio=10.0)
        hgbc.fit(x_tr, y_tr)

        y_pred_train = hgbc.predict(x_tr)
        train_metrics = evaluate_model(
            f"HistGBC [{label_set}/{tag}/{weighting}] [TRAIN — refit metric, NOT the "
            f"internal val_fraction=0.15 early-stopping score]",
            y_tr, y_pred_train, class_names, label_set,
            split="train", verbose=True, logger=logger)
        y_pred_test = hgbc.predict(x_te)
        test_metrics = evaluate_model(f"HistGBC [{label_set}/{tag}/{weighting}] [TEST — held-out]",
                                       y_te, y_pred_test, class_names, label_set, split="test",
                                       logger=logger)

        variant = f"{variant_prefix}_{weighting}"
        run_id = make_run_id(label_set, "HistGBC", tag, variant)
        save_result_multi(run_id, {
            "model_name": "HistGBC", "variant": variant, "feature_set": tag,
            "label_set": label_set,
            "n_features": x_tr.shape[1],
            "stopping_rule": (f"StratifiedEarlyStoppingHGBC; label_set={label_set}; "
                              f"feature_set={tag}; weighting={weighting}; "
                              f"n_iter={hgbc.n_iter_}; n_singleton_classes={hgbc.n_singleton_classes_}"),
            "fit_seconds": None,
            "fitted_summary": get_fitted_summary(hgbc),
            "hyperparameters": _hyperparams(hgbc), # closes the run_histgbc hyperparameter gap
            "normalization": _normalization_type(hgbc),
            "test_accuracy": test_metrics["accuracy"], "test_bal_accuracy": test_metrics["bal_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"], "test_kappa": test_metrics["kappa"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "train_accuracy": train_metrics["accuracy"], "train_bal_accuracy": train_metrics["bal_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"], "train_kappa": train_metrics["kappa"],
            "train_precision_macro": train_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "accuracy": test_metrics["accuracy"], "bal_accuracy": test_metrics["bal_accuracy"],
            "macro_f1": test_metrics["macro_f1"], "kappa": test_metrics["kappa"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
        }, results_paths_fn(label_set), logger=logger)
        return hgbc, test_metrics

    return run_histgbc
def safe_run(run_fn, *args, results_path: Union[Path, Sequence[Path]], context="",
             logger: Optional[logging.Logger] = None, **kwargs):
    """Catches any exception from run_model/run_histgbc so ONE bad
    (model, tag/PCA, weighting) combo never aborts the whole sweep.
    `results_path` accepts either a single `Path` or
    a sequence of `Path`s (written via `save_result_multi`); `logger`, if
    given, receives the failure line at `ERROR` (console + file) instead
    of a bare `print`."""
    try:
        return run_fn(*args, **kwargs)
    except Exception as e:
        msg = f"  ⚠ SKIPPED [{context}] — {type(e).__name__}: {e}"
        (logger.error(msg) if logger is not None else print(msg))
        run_id = f"{context}__FAILED__{int(time.time() * 1000)}"
        paths = list(results_path) if isinstance(results_path, (list, tuple)) else [results_path]
        save_result_multi(run_id, {
            "status": "failed", "error": f"{type(e).__name__}: {e}",
            "context": context, "test_bal_accuracy": None, "test_macro_f1": None,
            "test_kappa": None, "test_accuracy": None,
            "test_precision_macro": None, "test_recall_macro": None,
            "train_precision_macro": None, "train_recall_macro": None,
            "bal_accuracy": None, "macro_f1": None, "kappa": None, "accuracy": None,
            "precision_macro": None, "recall_macro": None,
        }, paths, logger=logger)
        return None, None
"""
# Stable, never-shadowed alias — every downstream module (fine/merged/
# 3-levels) must capture ITS "_safe_run_common" from THIS name, not from
# the bare `safe_run` global. In single-kernel jupytext execution, `safe_run`
# gets progressively re-wrapped by each module that runs earlier in the same
# kernel, so `_safe_run_common = safe_run` silently grabs the PREVIOUS
# module's wrapper instead of this base implementation, double-injecting
# results_path on every subsequent wrap.
"""
_BASE_SAFE_RUN = safe_run

# %%
"""
## Section 1.7 — Pipeline steps + MODEL_SPECS (CPU + GPU builders)

Every scale-sensitive model (LDA, LogReg, KNN, SVM, MLP) is built as an
`IterativeImputer -> [scaler] -> [PCA] -> classifier` `Pipeline`,
constructed fresh per fold/config so imputation and scaling statistics
are always fit on that fold's training rows only (leakage-safe). Tree
ensembles (DecisionTree/RandomForest/GBC/HistGBC) and CatBoost are
scale-invariant and consume raw NaN-containing features directly — no
pipeline wrapper. `get_builder()` is the single place `USE_GPU` is
consulted.
"""


# %%
def _pipeline_steps(pca_variance=None, scaler_cls=None):
    """scaler_cls=None means 'genuinely scale-invariant' (tree models) —
    NOT 'skip scaling unless PCA requested'. Every scale-sensitive model
    passes an explicit scaler_cls even when pca_variance is None."""
    steps = [("imputer", IterativeImputer(random_state=SEED, max_iter=10))]
    if scaler_cls is not None:
        steps.append(("scaler", scaler_cls()))
    if pca_variance is not None:
        if scaler_cls is None:
            raise ValueError(
                "pca_variance was requested but scaler_cls is None — "
                "PCA-eligible models must always pass an explicit scaler.")
        from sklearn.decomposition import PCA
        steps.append(("pca", PCA(n_components=pca_variance, random_state=SEED)))
    return steps
def build_lda(pca_variance=None, scaler_cls=None):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    scaler_cls = scaler_cls or SCALER_VARIANTS[DEFAULT_SCALER_NAME]
    return Pipeline(_pipeline_steps(pca_variance, scaler_cls) + [
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
def build_logreg(pca_variance=None, scaler_cls=None):
    from sklearn.linear_model import LogisticRegression
    scaler_cls = scaler_cls or SCALER_VARIANTS[DEFAULT_SCALER_NAME]
    return Pipeline(_pipeline_steps(pca_variance, scaler_cls) + [
        ("clf", LogisticRegression(solver="saga", C=1.0, tol=1e-4,
                                    max_iter=2000, random_state=SEED, n_jobs=-1)),
    ])
def build_knn(pca_variance=None, scaler_cls=None):
    scaler_cls = scaler_cls or SCALER_VARIANTS[DEFAULT_SCALER_NAME]
    return Pipeline(_pipeline_steps(pca_variance, scaler_cls) + [
        ("clf", AdaptiveKNNClassifier(n_neighbors=7)),
    ])
"""
# probability=True (CPU and GPU) — the hierarchical chains soft-route on
# predict_proba; without it an SVM stage fell back to a one-hot matrix
# (fake 1.0 confidences, hard routing). The probabilities are Platt-scaled
# via an internal 5-fold CV, so their argmax can disagree with the raw SVM
# decision; predict() is therefore redefined as argmax(predict_proba) so
# model selection (scored on predict) and deployment (argmax of the
# chained probabilities) use the same decision. Caveat: Platt is noisy
# for classes with fewer than ~5 training rows.
"""
from sklearn.svm import SVC


class ProbaSVC(SVC):
    """SVC whose predict() is argmax(predict_proba) — see note above."""

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_svm(pca_variance=None, scaler_cls=None):
    scaler_cls = scaler_cls or SCALER_VARIANTS[DEFAULT_SCALER_NAME]
    return Pipeline(_pipeline_steps(pca_variance, scaler_cls) + [
        ("clf", ProbaSVC(C=10.0, kernel="rbf", gamma="scale", probability=True,
                          tol=1e-4, max_iter=5000, cache_size=1000, random_state=SEED)),
    ])
def build_dtree(pca_variance=None, scaler_cls=None):
    return DTreeCVPrunedClassifier(random_state=SEED)
def build_rf(pca_variance=None, scaler_cls=None):
    return RFPlateauClassifier(random_state=SEED)
def build_gbc(pca_variance=None, scaler_cls=None):
    return GBCWarmStartClassifier(random_state=SEED)
def build_mlp_variant(hidden_layer_sizes):
    def _build(pca_variance=None, scaler_cls=None):
        scaler_cls = scaler_cls or SCALER_VARIANTS[DEFAULT_SCALER_NAME]
        return Pipeline(_pipeline_steps(pca_variance, scaler_cls) + [
            ("clf", MLPLeakageSafeClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=SEED)),
        ])
    return _build
def build_mlp_variant_gpu(hidden_layer_sizes):
    """GPU/Torch MLP builder — same topology dict, different backend.
    Still wrapped in a Pipeline so imputation/scaling/PCA stay identical
    to the CPU arm; only the final estimator changes."""
    def _build(pca_variance=None, scaler_cls=None):
        scaler_cls = scaler_cls or SCALER_VARIANTS[DEFAULT_SCALER_NAME]
        return Pipeline(_pipeline_steps(pca_variance, scaler_cls) + [
            ("clf", MLPLeakageSafeClassifierTorch(hidden_layer_sizes=hidden_layer_sizes, random_state=SEED)),
        ])
    return _build


# %%
"""
### GPU tree/SVM/KNN/boosting baselines

Best-effort optional imports for XGBoost, LightGBM, and cuML — if a
given GPU library isn't installed in the current environment, the
corresponding `build_*_gpu` function prints a warning and transparently
falls back to the CPU implementation rather than raising.
"""


# %%
def _try_import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except Exception:
        return None
def _try_import_lgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except Exception:
        return None
def _try_import_cuml():
    try:
        from cuml.ensemble import RandomForestClassifier as cuRF
        from cuml.svm import SVC as cuSVC
        from cuml.neighbors import KNeighborsClassifier as cuKNN
        return cuRF, cuSVC, cuKNN
    except Exception:
        return None, None, None
_XGB_CLS = _try_import_xgb()
_LGBM_CLS = _try_import_lgbm()
_CUML_RF, _CUML_SVC, _CUML_KNN = _try_import_cuml()
class _LocalLabelReencodeClassifier(BaseEstimator, ClassifierMixin):
    """Wraps ANY classifier whose .fit() enforces contiguous 0..n-1
    integer labels (XGBoost, LightGBM, GradientBoostingClassifier) so a
    CV/LOO fold that legitimately drops the sole remaining row of some
    global class label doesn't crash with "Invalid classes inferred from
    unique values of y". Re-encodes y to a LOCAL contiguous label space
    with a FRESH LabelEncoder on every fit() call, fits the wrapped
    estimator in that space, and maps predict()/predict_proba() back to
    original global label values/columns. estimator_cls/estimator_kwargs
    (not a pre-built instance) so a fresh, unfitted inner estimator is
    constructed per fit() call, and keeps this class picklable."""

    def __init__(self, estimator_cls, estimator_kwargs=None,
                 early_stopping_rounds=None, val_fraction=0.15, random_state=SEED):
        self.estimator_cls = estimator_cls
        self.estimator_kwargs = estimator_kwargs or {}
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        y_arr = np.asarray(y)
        self.local_le_ = LabelEncoder().fit(y_arr)
        y_local = self.local_le_.transform(y_arr)
        kwargs = dict(self.estimator_kwargs)

        if self.early_stopping_rounds is not None:
            fit_idx, val_idx = _leakage_safe_val_split(
                y_local, self.val_fraction, self.random_state)
            if len(val_idx) > 0:
                X_fit = X.iloc[fit_idx] if hasattr(X, "iloc") else X[fit_idx]
                X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
                y_fit, y_val = y_local[fit_idx], y_local[val_idx]
                sw_fit = sample_weight[fit_idx] if sample_weight is not None else None
                kwargs["early_stopping_rounds"] = self.early_stopping_rounds
                self.estimator_ = self.estimator_cls(**kwargs)
                # NOTE XGBoost >=2.0 takes early_stopping_rounds as a
                # constructor kwarg (as above); older XGBoost and LightGBM's
                # sklearn API instead expect it passed to .fit() (or via
                # callbacks=[lgb.early_stopping(n)] on recent LightGBM) —
                # verify against installed versions before relying on this.
                self.estimator_.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                                     sample_weight=sw_fit, verbose=False)
                self.classes_ = self.local_le_.classes_
                return self
            # too few rows/classes for a val split — fall through to a plain fit

        self.estimator_ = self.estimator_cls(**kwargs)
        if sample_weight is not None:
            self.estimator_.fit(X, y_local, sample_weight=sample_weight)
        else:
            self.estimator_.fit(X, y_local)
        self.classes_ = self.local_le_.classes_
        return self

    def predict(self, X):
        pred_local = self.estimator_.predict(X)
        return self.local_le_.inverse_transform(pred_local)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
def build_gbc_gpu(pca_variance=None, scaler_cls=None):
    if _XGB_CLS is not None:
        return _LocalLabelReencodeClassifier(
            _XGB_CLS, dict(
                tree_method="hist", device="cuda", random_state=SEED, n_estimators=2000,
                learning_rate=0.05, max_depth=4, subsample=0.8),
            early_stopping_rounds=25, val_fraction=0.15, random_state=SEED)
    if _LGBM_CLS is not None:
        return _LocalLabelReencodeClassifier(
            _LGBM_CLS, dict(
                device="gpu", random_state=SEED, n_estimators=2000,
                learning_rate=0.05, max_depth=4, subsample=0.8),
            early_stopping_rounds=25, val_fraction=0.15, random_state=SEED)
    print("  ⚠ No GPU boosting backend (XGBoost/LightGBM) available — falling back to CPU GBCWarmStartClassifier.")
    return build_gbc(pca_variance, scaler_cls)
def build_histgbc_gpu_hint():
    """HistGBC has no first-party GPU backend in sklearn; XGBoost/LightGBM
    with device='cuda'/'gpu' are the drop-in GPU alternatives used above
    for RF/GBC. HistGBC's own StratifiedEarlyStoppingHGBC wrapper stays
    CPU-only (run_histgbc is dispatched separately, unaffected by
    MODEL_SPECS' build_gpu key)."""
    return None
def _cuml_svc_accepts_sample_weight():
    """Defined once;
    kept as one definition."""
    if _CUML_SVC is None:
        return False
    try:
        return "sample_weight" in inspect.signature(_CUML_SVC.fit).parameters
    except Exception:
        return False
_CUML_SVC_SUPPORTS_SAMPLE_WEIGHT = _cuml_svc_accepts_sample_weight()
class _CumlSVCSampleWeightAdapter(BaseEstimator, ClassifierMixin):
    """cuML SVC may or may not accept sample_weight depending on the
    installed cuML version — detected ONCE at import time
    (_CUML_SVC_SUPPORTS_SAMPLE_WEIGHT). Native sample_weight is used when
    available; falls back to the same weighted-bootstrap mechanism as
    LDA/KNN otherwise.

    estimator_cls/estimator_kwargs (NOT a lambda/closure factory) — a
    closure over _CUML_SVC can't be pickled by joblib.dump at deployment
    time, since every fitted branch/router model in this pipeline
    eventually gets persisted via persist_hierarchical_models."""

    def __init__(self, estimator_cls, estimator_kwargs=None, random_state=SEED):
        self.estimator_cls = estimator_cls
        self.estimator_kwargs = estimator_kwargs or {}
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.estimator = self.estimator_cls(**self.estimator_kwargs)
        if sample_weight is not None and not np.allclose(sample_weight, sample_weight[0]):
            if _CUML_SVC_SUPPORTS_SAMPLE_WEIGHT:
                self.estimator.fit(X, y, sample_weight=sample_weight)
            else:
                wmap = {}
                for cls in np.unique(y):
                    mask = y == cls
                    wmap[cls] = float(sample_weight[mask][0]) if mask.any() else 1.0
                X_res, y_res = resample_by_weight(X, y, wmap, random_state=self.random_state)
                self.estimator.fit(X_res, y_res)
        else:
            self.estimator.fit(X, y)
        self.classes_ = np.asarray(getattr(self.estimator, "classes_", np.unique(y)))
        return self

    def predict(self, X):
        # argmax(predict_proba) when Platt probabilities exist — see ProbaSVC.
        if self.estimator_kwargs.get("probability"):
            return self.classes_[np.asarray(self.predict_proba(X)).argmax(axis=1)]
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
def _cuml_svc_supported_kwargs(kwargs):
    """Filters kwargs down to what this installed
    cuML SVC build actually accepts — mirrors the existing defensive
    pattern used by _cuml_svc_accepts_sample_weight, since probability/
    tol/max_iter support varies across cuML versions and an unsupported
    kwarg should degrade gracefully, not crash the whole sweep."""
    try:
        accepted = set(inspect.signature(_CUML_SVC.__init__).parameters)
    except Exception:
        return {}
    return {k: v for k, v in kwargs.items() if k in accepted}
def build_svm_gpu(pca_variance=None, scaler_cls=None):
    scaler_cls = scaler_cls or SCALER_VARIANTS[DEFAULT_SCALER_NAME]
    if _CUML_SVC is not None:
        full_kwargs = {"C": 10.0, "kernel": "rbf", "tol": 1e-4,
                        "max_iter": 5000, "probability": True}
        return Pipeline(_pipeline_steps(pca_variance, scaler_cls) + [
            ("clf", _CumlSVCSampleWeightAdapter(
                estimator_cls=_CUML_SVC,
                estimator_kwargs=_cuml_svc_supported_kwargs(full_kwargs),
                random_state=SEED)),
        ])
    print("  ⚠ No GPU SVM backend (cuML) available — falling back to CPU SVC.")
    return build_svm(pca_variance, scaler_cls)
class AdaptiveKNNClassifierGPU(BaseEstimator, ClassifierMixin):
    """cuML KNeighborsClassifier with n_neighbors clamped to the actual
    training-set size — GPU counterpart to AdaptiveKNNClassifier. The CPU class clamps to len(y)-1 (total training
    rows), NOT a per-class count — mirrored exactly here, including the
    same KNN_PER_CLASS_CLAMP switch for merged/3-levels branches."""

    def __init__(self, n_neighbors=7):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        k = max(1, min(self.n_neighbors, len(y) - 1))
        if KNN_PER_CLASS_CLAMP:
            _, counts = np.unique(y, return_counts=True)
            k = max(1, min(k, int(counts.min()) - 1))
        self.model_ = _CUML_KNN(n_neighbors=k)
        self.model_.fit(X, y)
        self.classes_ = getattr(self.model_, "classes_", np.unique(y))
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
def build_knn_gpu(pca_variance=None, scaler_cls=None):
    scaler_cls = scaler_cls or SCALER_VARIANTS[DEFAULT_SCALER_NAME]
    if _CUML_KNN is not None:
        return Pipeline(_pipeline_steps(pca_variance, scaler_cls) + [
            ("clf", AdaptiveKNNClassifierGPU(n_neighbors=7)),
        ])
    print("  ⚠ No GPU KNN backend (cuML) available — falling back to CPU AdaptiveKNNClassifier.")
    return build_knn(pca_variance, scaler_cls)


# %%
"""
### CatBoost

Added for direct comparison against **Coronado-Blázquez (2022), *MNRAS*
515(2):1807–1822, arXiv:2207.04725**, which used CatBoost as a 4FGL
blazar/pulsar-classification baseline — including it here lets this
pipeline's leaderboards be read against that published benchmark.
The same citation is used in the RuntimeError messages below and the
MODEL_SPECS comment.
No pipeline/imputer wrapper is used: CatBoost has native missing-value
handling and is treated as `scale_sensitive=False`. `weight_support=
"sample_weight"` routes through the existing `sample_weight_array`/
`_run_model_core` path unchanged.
"""


# %%
def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier, Pool
        return CatBoostClassifier, Pool
    except Exception:
        return None, None
_CATBOOST_CLS, _CATBOOST_POOL = _try_import_catboost()
CATBOOST_MAX_ITERATIONS = _cfg("CATBOOST_MAX_ITERATIONS", 5000)
CATBOOST_EARLY_STOPPING_ROUNDS = _cfg("CATBOOST_EARLY_STOPPING_ROUNDS", 100)
CATBOOST_LEARNING_RATE = _cfg("CATBOOST_LEARNING_RATE", 0.05)
CATBOOST_FALLBACK_ITERATIONS = 500  # used only when a branch is too small for a val split


class CatBoostEarlyStopClassifier(BaseEstimator, ClassifierMixin):
    """CatBoost with early stopping instead of a fixed 500 trees.

    Same pattern as GBCWarmStartClassifier: a leakage-safe per-class val
    split (`_leakage_safe_val_split`) picks the number of trees (up to
    `max_iterations`, stopping after `early_stopping_rounds` without
    improvement), then the model is refit on ALL rows with that many trees,
    so small branches don't lose 15% of their data. Stopping uses
    CatBoost's own (class-weighted) MultiClass log-loss on the val split —
    smoother than balanced accuracy on a small split, and it directly tunes
    the probabilities the hierarchical soft routing multiplies together.
    The fitted CatBoost model is `estimator_`, so analysis/SHAP unwrap it
    like any other wrapper."""

    def __init__(self, max_iterations=5000, early_stopping_rounds=100, learning_rate=0.05,
                 depth=6, val_fraction=0.15, task_type="CPU", random_state=SEED):
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate = learning_rate
        self.depth = depth
        self.val_fraction = val_fraction
        self.task_type = task_type
        self.random_state = random_state

    def _new(self, iterations):
        kw = dict(iterations=iterations, learning_rate=self.learning_rate, depth=self.depth,
                  loss_function="MultiClass", random_seed=self.random_state, verbose=False,
                  task_type=self.task_type)
        if self.task_type == "GPU":
            kw["devices"] = "0"
        return _CATBOOST_CLS(**kw)

    def fit(self, X, y, sample_weight=None):
        y_arr = np.asarray(y)
        sw = None if sample_weight is None else np.asarray(sample_weight)
        fit_idx, val_idx = _leakage_safe_val_split(y_arr, self.val_fraction, self.random_state)
        _rows = (lambda idx: X.iloc[idx]) if hasattr(X, "iloc") else (lambda idx: X[idx])
        n_iter = CATBOOST_FALLBACK_ITERATIONS
        if len(val_idx) > 0:
            probe = self._new(self.max_iterations)
            probe.fit(_rows(fit_idx), y_arr[fit_idx], sample_weight=None if sw is None else sw[fit_idx],
                      eval_set=_CATBOOST_POOL(_rows(val_idx), y_arr[val_idx],
                                              weight=None if sw is None else sw[val_idx]),
                      early_stopping_rounds=self.early_stopping_rounds, use_best_model=True)
            n_iter = max(1, int(probe.get_best_iteration()) + 1)
        self.estimator_ = self._new(n_iter).fit(X, y_arr, sample_weight=sw)
        self.n_estimators_ = n_iter  # recorded by get_fitted_summary as n_estimators_selected
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).reshape(-1).astype(self.classes_.dtype)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def build_catboost(pca_variance=None, scaler_cls=None, task_type="CPU"):
    if _CATBOOST_CLS is None:
        raise RuntimeError(
            "catboost is not installed — `pip install catboost` to use the "
            "CatBoost baseline (added for comparison against Coronado-Blázquez "
            "2022, MNRAS 515(2):1807-1822, arXiv:2207.04725)."
        )
    return CatBoostEarlyStopClassifier(
        max_iterations=CATBOOST_MAX_ITERATIONS, early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS,
        learning_rate=CATBOOST_LEARNING_RATE, depth=6, task_type=task_type, random_state=SEED)
def build_catboost_gpu(pca_variance=None, scaler_cls=None):
    return build_catboost(pca_variance, scaler_cls, task_type="GPU")


# %%
"""
### MODEL_SPECS registry

The single source of truth every module sweeps over. `"HistGBC"` uses
`_HGBCWeightedEarlyStop` (leakage-safe chunked early-stopping, real
external sample_weight) as its Stage-0/1/2 scoring proxy — the FINAL
winning HistGBC config is still refit with the real, production
`StratifiedEarlyStoppingHGBC`: `cv_select_and_refit` special-cases
`model_name == "HistGBC"` to dispatch through `run_histgbc_fn`, and
`select_and_refit_val_only` does the val-only
equivalent internally — see that function's docstring.
"""


# %%
def _build_histgbc_search_proxy(pca_variance=None, scaler_cls=None):
    return _HGBCWeightedEarlyStop(random_state=SEED)
MODEL_SPECS = {
    "LDA":          dict(build=build_lda,      build_gpu=build_lda,        weight_support="resample",      supports_pca=True,  scale_sensitive=True),
    "LogReg":       dict(build=build_logreg,   build_gpu=build_logreg,     weight_support="sample_weight", supports_pca=True,  scale_sensitive=True),
    "KNN":          dict(build=build_knn,      build_gpu=build_knn_gpu,    weight_support="resample",      supports_pca=True,  scale_sensitive=True),
    "SVM":          dict(build=build_svm,      build_gpu=build_svm_gpu,    weight_support="sample_weight", supports_pca=True,  scale_sensitive=True),
    "DecisionTree": dict(build=build_dtree,    build_gpu=build_dtree,      weight_support="sample_weight", supports_pca=False, scale_sensitive=False),
    "RandomForest": dict(build=build_rf,       build_gpu=build_rf,         weight_support="sample_weight", supports_pca=False, scale_sensitive=False),
    "GBC":          dict(build=build_gbc,      build_gpu=build_gbc_gpu,    weight_support="sample_weight", supports_pca=False, scale_sensitive=False),
    # CatBoost baseline — Coronado-Blázquez (2022), arXiv:2207.04725.
    "CatBoost":     dict(build=build_catboost, build_gpu=build_catboost_gpu, weight_support="sample_weight", supports_pca=False, scale_sensitive=False),
    # search-only proxy — see markdown above. Real winner is refit via
    # run_histgbc_fn (cv_select_and_refit) or the internal deployable
    # adapter (select_and_refit_val_only), never via this entry's
    # build/build_gpu.
    "HistGBC":      dict(build=_build_histgbc_search_proxy, build_gpu=_build_histgbc_search_proxy,
                          weight_support="sample_weight", supports_pca=False, scale_sensitive=False),
}
for _mlp_name, _hidden_sizes in MLP_ARCHITECTURES.items():
    MODEL_SPECS[_mlp_name] = dict(
        build=build_mlp_variant(_hidden_sizes),
        build_gpu=build_mlp_variant_gpu(_hidden_sizes),
        # sample_weight, not resample: a bootstrap resample put duplicate rows
        # on both sides of the MLP's internal early-stopping split.
        weight_support="sample_weight", supports_pca=True, scale_sensitive=True,
    )
ALL_MODEL_NAMES = list(MODEL_SPECS.keys())


def weight_support_for(model_name: str) -> str:
    """SVM declares weight_support='sample_weight' for its CPU builder, but
    cuML's SVC ignores sample_weight for multiclass fits under USE_GPU=True,
    so it falls back to 'resample' there. RandomForest always runs on CPU
    (cuML RF hit CUDA illegal-memory-access errors) and keeps native
    sample weights: a resample would put copies of a row both in-bag and
    out-of-bag and inflate the OOB score RFPlateauClassifier stops on."""
    base = MODEL_SPECS[model_name]["weight_support"]
    if USE_GPU and model_name == "SVM":
        return "resample"
    return base


globals().setdefault("_BASE_MODEL_SPECS", MODEL_SPECS)

def get_builder(model_name: str, model_specs=None):
    """Single selection point every module's run_model uses: returns the
    GPU builder if USE_GPU else the CPU builder. HistGBC is dispatched
    separately (run_histgbc) and is unaffected by USE_GPU.
    `model_specs` lets a caller (e.g. fine.py's own
    FINE_MODEL_SPECS) resolve against its own spec dict without
    reassigning the bare `MODEL_SPECS` global — defaults to the
    snapshot taken above, at this module's own load time, so a
    downstream module's local override can never contaminate this
    default for callers that don't opt in."""
    specs = model_specs if model_specs is not None else _BASE_MODEL_SPECS
    spec = specs[model_name]
    return spec["build_gpu"] if USE_GPU else spec["build"]


# %%
"""
## Section 1.8 — Shared feature-selection engine

A cumulative-power cutoff is used instead of a fixed top-N because a
fixed N has no principled connection to how much of a model's actual
discriminative signal is concentrated in the head of the importance
ranking. `compute_selection_engine` runs `HistGradientBoostingClassifier
-> permutation_importance(scoring="balanced_accuracy") -> cumulative-
power cuts` — the same three-step engine used by Module 1's global
99/90/80 feature sets AND every downstream local selection.
`balanced_accuracy` is used as the shuffle-scoring metric specifically
because every 4FGL setup here is class-imbalanced; this stays hardcoded
regardless of `SELECTION_METRIC` — feature-ranking stability under class
imbalance is a different concern from which model wins the leaderboard
(deliberately not tied to `SELECTION_METRIC`, unlike the six custom
estimators in Section 1.4). Accepts an optional `logs_dir` (resolves one logger for
this call's `label`); the repeat loop (`for r in range(effective_n_repeats)`)
is now wrapped in `pbar()`.
"""


# %%
def spearman_clusters(x, threshold=CORR_CLUSTER_THRESHOLD):
    """Groups features whose |Spearman rho| >= `threshold`: average-linkage hierarchical clustering on
    1 - |rho| (pairwise-complete correlations), cut at 1 - threshold — the collinearity clustering
    reviewed by Dormann et al. (2013, Ecography 36:27). Returns {representative: [members]} in
    column order; the representative is the member with the fewest missing values, then the highest
    mean |rho| to the other members (the cluster medoid), then the name."""
    cols = list(x.columns)
    if len(cols) < 2:
        return {c: [c] for c in cols}
    corr = x.corr(method="spearman").abs().fillna(0.0).to_numpy()
    np.fill_diagonal(corr, 1.0)
    dist = np.clip(1.0 - corr, 0.0, None)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    ids = fcluster(linkage(squareform(dist, checks=False), method="average"),
                   t=1.0 - threshold, criterion="distance")
    n_nan = x.isna().sum().to_numpy()
    out = {}
    for cid in dict.fromkeys(ids):
        idx = [i for i, c in enumerate(ids) if c == cid]
        rep = min(idx, key=lambda i: (n_nan[i], -corr[i, idx].mean(), cols[i]))
        out[cols[rep]] = [cols[i] for i in idx]
    return out


def grouped_permutation_importance(model, x, y, groups, n_repeats=10, seed=SEED):
    """Permutation importance of feature GROUPS (Gregorutti, Michel & Saint-Pierre 2017, Stat.
    Comput. 27:659; Au et al. 2022, Data Min. Knowl. Disc. 36:1401): all columns of a group are
    permuted with the SAME row permutation, so a correlated twin cannot stand in for a permuted
    feature and make both look unimportant (Strobl et al. 2008; Hooker, Mentch & Zhou 2021).
    Score: drop in balanced accuracy. Returns one value per group."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = balanced_accuracy_score(y, model.predict(x))
    xp = x.copy()
    out = np.empty(len(groups))
    for g, cols in enumerate(groups):
        orig = x[cols].to_numpy()
        drops = []
        for _ in range(n_repeats):
            xp[cols] = orig[rng.permutation(len(x))]
            drops.append(base - balanced_accuracy_score(y, model.predict(xp)))
        xp[cols] = orig
        out[g] = float(np.mean(drops))
    return out


def expand_clusters(units, clusters):
    """Selected cluster representatives -> all member features, in order."""
    return [f for u in units for f in clusters[u]]


def per_feature_importance(unit_df, clusters):
    """Cluster-level importance table -> one row per feature (each feature carries its cluster's
    importance, plus a `cluster` column naming the representative)."""
    return pd.DataFrame([{**row, "feature": f, "cluster": row["feature"]}
                         for row in unit_df.to_dict("records") for f in clusters[row["feature"]]])


def cpss_importance(x, y, n_pairs=STABILITY_N_PAIRS, seed=SEED, n_perm_repeats=10, groups=None):
    """Complementary-pairs subsampling (Shah & Samworth 2013) for stability selection. Each draw
    splits the rows into two disjoint halves, class by class (an odd-sized class gives its extra row
    to a random half, a singleton class lands in one half only). A HistGBC fit on one half is scored
    on the other by grouped permutation importance with balanced accuracy, and vice versa, so every
    estimate is held-out and every subsample has size n/2. `groups`: list of column lists (default:
    one column each). Returns (importance matrix [2*n_pairs, n_groups], held-out balanced
    accuracies, held-out macro-F1s), one row/entry per half-sample fit."""
    groups = groups or [[c] for c in x.columns]
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    imps, accs, f1s = [], [], []
    for _ in pbar(range(n_pairs), desc="CPSS complementary pairs", leave=False):
        halves = ([], [])
        for c in np.unique(y):
            ix = rng.permutation(np.flatnonzero(y == c))
            cut = len(ix) // 2 + (len(ix) % 2) * int(rng.integers(2))
            halves[0].append(ix[:cut])
            halves[1].append(ix[cut:])
        a, b = np.concatenate(halves[0]), np.concatenate(halves[1])
        for fit_idx, eval_idx in ((a, b), (b, a)):
            if len(np.unique(y[fit_idx])) < 2 or len(eval_idx) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=seed)
            m.fit(x.iloc[fit_idx], y[fit_idx])
            y_hat = m.predict(x.iloc[eval_idx])
            accs.append(balanced_accuracy_score(y[eval_idx], y_hat))
            f1s.append(f1_score(y[eval_idx], y_hat, average="macro", zero_division=0))
            imps.append(grouped_permutation_importance(m, x.iloc[eval_idx], y[eval_idx], groups,
                                                       n_repeats=n_perm_repeats, seed=seed + len(imps)))
    return np.vstack(imps), np.asarray(accs), np.asarray(f1s)


def stability_error_bound(mean_n_selected, n_features, threshold=STABILITY_SELECTION_FRACTION):
    """Meinshausen & Bühlmann (2010, Thm 1) bound on the expected number of falsely selected
    features, q^2 / ((2*threshold - 1) * p), with q the mean number of features selected per
    subsample. It assumes exchangeable noise features; Shah & Samworth (2013, Thm 1) give an
    assumption-free version for CPSS in terms of low-selection-probability features. Reported, not
    enforced: with cumulative-power cuts q is large and the bound is loose."""
    return float(mean_n_selected ** 2 / ((2 * threshold - 1) * n_features))


def select_by_cumulative_power(importance_df, target_fraction, importance_col="importance",
                                max_features=None):
    """Smallest top-ranked feature set whose importances cumulatively
    cover `target_fraction` of total POSITIVE permutation-importance
    mass. `max_features`, when given, is a hard ceiling applied AFTER
    the power-fraction cut."""
    pos = importance_df[importance_df[importance_col] > 0].copy()
    if pos.empty:
        # Nothing beats noise (common on tiny branches): keep the highest-ranked features, but never
        # more than the max_features cap, so the samples-per-feature limit still holds on the
        # small branches where it matters most.
        ranked = importance_df.sort_values(importance_col, ascending=False)["feature"].tolist()
        return ranked[:max_features] if max_features is not None else ranked
    pos = pos.sort_values(importance_col, ascending=False).reset_index(drop=True)
    cum_frac = pos[importance_col].cumsum() / pos[importance_col].sum()
    n_keep = min(int(np.searchsorted(cum_frac.values, target_fraction) + 1), len(pos))
    if max_features is not None:
        n_keep = min(n_keep, max_features)
    return pos["feature"].iloc[:n_keep].tolist()
def compute_selection_engine(x_tr, x_te, y_tr_enc, y_te_enc, label: str,
                              tag_prefix: str, seed=SEED,
                              thresholds=FEATURE_POWER_THRESHOLDS,
                              top_n=PHYSICS_CORE_TOP_N,
                              include_physics_core: bool = False,
                              n_splits: int = 5,
                              loo_min_class_count=LOO_MIN_CLASS_COUNT_THRESHOLD,
                              n_pairs: int = STABILITY_N_PAIRS,
                              stability_fraction: float = STABILITY_SELECTION_FRACTION,
                              lcb_z: float = IMPORTANCE_LCB_Z,
                              corr_threshold: float = CORR_CLUSTER_THRESHOLD,
                              logs_dir: Optional[Path] = None):
    """[Stability-selection upgrade] Same call signature and 4-tuple
    return as before. x_te/y_te_enc remain unread, kept only for
    signature compatibility.

    Importances come from complementary-pairs subsampling (`cpss_importance`,
    Shah & Samworth 2013) when the node has >= CPSS_MIN_N rows; smaller
    nodes get one LOO/K-fold pass instead (no stability filter possible).
    Ranking/selection uses `max(0, mean - lcb_z*std)` (lcb_z = 0: the
    mean). A selection-FREQUENCY filter (Meinshausen & Bühlmann 2010) is
    layered on top: a feature must be selected by the per-subsample
    cumulative-power cut in >= `stability_fraction` of the half-sample
    fits to survive; if that would empty a threshold's set, the ranked
    selection is kept unfiltered instead.

    Importance, ranking, the stability filter and the cumulative-power
    cut all work on CLUSTERS of correlated features (`spearman_clusters`,
    |rho| >= corr_threshold), each permuted as one unit
    (`grouped_permutation_importance`). A tag's feature set is every member
    of its selected clusters; `{tag_prefix}_{tag}_DECORR` keeps one
    representative per selected cluster. The `MIN_SAMPLES_PER_FEATURE` cap
    counts clusters. The returned importance table has one row per feature
    (its cluster's importance, plus a `cluster` column)."""
    logger = get_section_logger(label, logs_dir) if logs_dir is not None else None
    clusters = spearman_clusters(x_tr, corr_threshold)
    groups = list(clusters.values())
    min_class_count = int(np.unique(y_tr_enc, return_counts=True)[1].min())
    use_cpss = len(x_tr) >= CPSS_MIN_N
    use_loo = (not use_cpss) and min_class_count <= loo_min_class_count
    cv_mode = (f"CPSS({n_pairs} complementary pairs)" if use_cpss else "LeaveOneOut" if use_loo else
               f"StratifiedKFold(n_splits={max(2, min(n_splits, min_class_count))})")

    def _run_one_pass(repeat_seed):
        cv = LeaveOneOut() if use_loo else StratifiedKFold(
            n_splits=max(2, min(n_splits, min_class_count)), shuffle=True,
            random_state=repeat_seed)
        fold_importances, fold_bal_accs, fold_f1s = [], [], []
        for tr_idx, va_idx in cv.split(x_tr, y_tr_enc):
            x_f_tr, x_f_va = x_tr.iloc[tr_idx], x_tr.iloc[va_idx]
            y_f_tr, y_f_va = y_tr_enc[tr_idx], y_tr_enc[va_idx]
            hgbc_fi = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=6, random_state=seed)
            hgbc_fi.fit(x_f_tr, y_f_tr)
            y_pred = hgbc_fi.predict(x_f_va)
            fold_bal_accs.append(balanced_accuracy_score(y_f_va, y_pred))
            fold_f1s.append(f1_score(y_f_va, y_pred, average="macro", zero_division=0))
            if len(x_f_va) < 2:
                continue
            n_reps_perm = 20 if len(x_f_va) >= 30 else 10
            fold_importances.append(grouped_permutation_importance(
                hgbc_fi, x_f_va, y_f_va, groups, n_repeats=n_reps_perm, seed=seed))

        if not fold_importances:
            if len(x_tr) < 2:
                raise RuntimeError(f"[{label}] branch has <2 rows total — "
                                    "cannot compute feature importance at all.")
            note = (f"    [{label}] every CV fold's validation split was <2 rows "
                    f"(cv_mode={cv_mode}) — falling back to IN-SAMPLE permutation "
                    f"importance on the full branch. NOT held-out: treat as a "
                    f"noisier/optimistic ranking, adequate for feature-set "
                    f"selection on a branch too small for genuine held-out "
                    f"importance, not for reporting as a validated metric.")
            (logger.warning(note) if logger is not None else print(note))
            hgbc_fi = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=6, random_state=seed)
            hgbc_fi.fit(x_tr, y_tr_enc)
            n_reps_perm = 20 if len(x_tr) >= 30 else 10
            fold_importances.append(grouped_permutation_importance(
                hgbc_fi, x_tr, y_tr_enc, groups, n_repeats=n_reps_perm, seed=seed))

        imp_arr = np.vstack(fold_importances)
        return imp_arr.mean(axis=0), float(np.mean(fold_bal_accs)), float(np.mean(fold_f1s))

    if use_cpss:
        imp_cpss, acc_cpss, f1_cpss = cpss_importance(x_tr, y_tr_enc, n_pairs=n_pairs, seed=seed, groups=groups)
        repeat_importances, repeat_bal_accs, repeat_f1s = list(imp_cpss), list(acc_cpss), list(f1_cpss)
    else:
        imp_r, bal_acc_r, f1_r = _run_one_pass(seed)
        repeat_importances, repeat_bal_accs, repeat_f1s = [imp_r], [bal_acc_r], [f1_r]
    effective_n_repeats = len(repeat_importances)

    imp_matrix = np.vstack(repeat_importances)
    mean_imp, std_imp = imp_matrix.mean(axis=0), imp_matrix.std(axis=0)
    lcb_imp = np.clip(mean_imp - lcb_z * std_imp, 0.0, None)

    feature_names = np.array(list(clusters))  # one "unit" per cluster, named by its representative
    importance_df = pd.DataFrame({
        "feature": feature_names, "importance": lcb_imp,
        "importance_mean": mean_imp, "std": std_imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    bal_acc_fi, macro_f1_fi = float(np.mean(repeat_bal_accs)), float(np.mean(repeat_f1s))
    baseline_msg = (f"    [{label}] {x_tr.shape[1]} features in {len(clusters)} Spearman cluster(s) "
                    f"(|rho| >= {corr_threshold}); CV baseline HistGBC ({cv_mode}, {effective_n_repeats} fit(s)) — "
                     f"Bal-Acc={bal_acc_fi:.4f}±{np.std(repeat_bal_accs):.4f}  Macro-F1={macro_f1_fi:.4f}")
    (logger.notice(baseline_msg) if logger is not None else print(baseline_msg))

    n_train_local = len(x_tr)
    max_feats_for_n = max(2, n_train_local // MIN_SAMPLES_PER_FEATURE)
    n_pos_features = int((importance_df["importance"] > 0).sum())
    total_pos = importance_df.loc[importance_df["importance"] > 0, "importance"].sum()

    per_repeat_dfs = [pd.DataFrame({"feature": feature_names, "importance": imp_r})
                       for imp_r in repeat_importances]
    freq_by_tag = {}
    for tag, frac in thresholds.items():
        counts = pd.Series(0.0, index=feature_names)
        for df_r in per_repeat_dfs:
            sel_r = select_by_cumulative_power(df_r, frac, max_features=max_feats_for_n)
            counts.loc[sel_r] += 1.0
        freq_by_tag[tag] = counts / effective_n_repeats

    feats, achieved_power = {}, {}
    for tag, frac in thresholds.items():
        key = f"{tag_prefix}_{tag}" if tag_prefix else tag
        sel_lcb = select_by_cumulative_power(importance_df, frac, max_features=max_feats_for_n)

        if effective_n_repeats > 1:
            freq = freq_by_tag[tag]
            sel_stable = [f for f in sel_lcb if freq.get(f, 0.0) >= stability_fraction]
            if sel_stable:
                sel = sel_stable
            else:
                empty_warn = (f"    [{label}] {key}: stability filter (>={stability_fraction:.0%} "
                               f"of {effective_n_repeats} repeats) would empty this feature set — "
                               "keeping the LCB-ranked selection unfiltered instead.")
                (logger.warning(empty_warn) if logger is not None else print(empty_warn))
                sel = sel_lcb
        else:
            sel = sel_lcb

        feats[key] = expand_clusters(sel, clusters)
        achieved = importance_df.set_index("feature").loc[sel, "importance"].clip(lower=0).sum()
        achieved_power[tag] = float(achieved / total_pos) if total_pos > 0 else 0.0
        if len(sel_lcb) == max_feats_for_n and max_feats_for_n < n_pos_features:
            cap_warn = (f"    [{label}] {key}: capped at {max_feats_for_n} clusters "
                        f"(n_train={n_train_local}, {MIN_SAMPLES_PER_FEATURE} samples/feature floor) "
                        f"— only reached {achieved_power[tag]:.0%} of the {frac:.0%} power target; "
                        "this branch is dimensionality-limited, not just sample-limited.")
            (logger.warning(cap_warn) if logger is not None else print(cap_warn))

        feats[f"{key}_DECORR"] = list(sel)  # one representative per selected cluster

    if include_physics_core:
        key = f"{tag_prefix}_PHYSICS_CORE" if tag_prefix else "PHYSICS_CORE"
        feats[key] = expand_clusters(importance_df.head(top_n)["feature"].tolist(), clusters)

    # ALLPOS — every feature with positive (LCB-clipped) importance, no
    # cumulative-power cut, no stability frequency filter. Discarded
    # entirely for this train path (NOT truncated) if that count exceeds
    # the MIN_SAMPLES_PER_FEATURE floor — a silently-shortened "ALLPOS"
    # would no longer be "all features minus negative importance."
    key_allpos = f"{tag_prefix}_ALLPOS" if tag_prefix else "ALLPOS"
    sel_allpos = importance_df.loc[importance_df["importance"] > 0, "feature"].tolist()
    if len(sel_allpos) > max_feats_for_n:
        cap_warn = (f"    [{label}] {key_allpos}: discarded — {len(sel_allpos)} positive-importance "
                    f"features exceeds the {max_feats_for_n}-feature floor "
                    f"(n_train={n_train_local}, {MIN_SAMPLES_PER_FEATURE} samples/feature).")
        (logger.warning(cap_warn) if logger is not None else print(cap_warn))
    elif len(sel_allpos) == 0:
        empty_warn = f"    [{label}] {key_allpos}: no feature has positive importance — set is empty; skipped."
        (logger.warning(empty_warn) if logger is not None else print(empty_warn))
    else:
        feats[key_allpos] = expand_clusters(sel_allpos, clusters)
        achieved_power["ALLPOS"] = 1.0
        feats[f"{key_allpos}_DECORR"] = list(sel_allpos)

    summary1 = f"    [{label}] local feature sets: " + ", ".join(f"{k}={len(v)}" for k, v in feats.items())
    summary2 = (f"    [{label}] achieved power — "
                + ", ".join(f"{k}%={achieved_power[k]:.3f}" for k in thresholds)
                + (f", ALLPOS%={achieved_power['ALLPOS']:.3f}" if "ALLPOS" in achieved_power else ""))

    (logger.notice(summary1) if logger is not None else print(summary1))
    (logger.notice(summary2) if logger is not None else print(summary2))

    baseline = {"bal_acc": bal_acc_fi, "macro_f1": macro_f1_fi, "cv_mode": cv_mode}
    if effective_n_repeats > 1:
        baseline["stability_error_bound"] = {
            tag: stability_error_bound(float(freq_by_tag[tag].sum()), len(feature_names), stability_fraction)
            for tag in thresholds}
        bound_msg = (f"    [{label}] Meinshausen-Bühlmann bound on E[false selections] — "
                     + ", ".join(f"{t}%={v:.1f}" for t, v in baseline["stability_error_bound"].items()))
        (logger.notice(bound_msg) if logger is not None else print(bound_msg))
    baseline["n_clusters"] = len(clusters)
    return feats, per_feature_importance(importance_df, clusters), achieved_power, baseline


# %%
"""
## Section 1.9 — Shared model-fit core (`_run_model_core`)

The behavior behind every module's own `run_model` wrapper. This
function owns weighting-mode dispatch (resample vs. sample_weight) and
PCA/scaler-suffix feature-set labeling. Unlike
`score_on_val`/`fit_component`/`select_and_refit_val_only`'s final
refit — which all fit immediately, and now share `dispatch_fit` — this
function hands an **unfitted** estimator plus prepared inputs to
`fit_and_log_fn`, which does the actual `.fit()` call itself (so it can
time it, catch convergence warnings, and log the result in one place).
That deferred-fit contract means it can't literally call `dispatch_fit`
(which fits immediately), so the weighting-branch logic is instead
shared via `_prepare_fit_inputs` — the same primitive `dispatch_fit`
itself is built on — rather than being reimplemented a fourth time.
"""

# %%


def _prepare_fit_inputs(builder, build_kwargs: dict, x, y, weighting: str, weight_support: str,
                         seed=SEED, damping=0.5, max_ratio=10.0):
    """Builds a
    fresh estimator via `builder(**build_kwargs)` and resolves
    `(x_use, y_use, sample_weight_or_None)` according to `weighting`/
    `weight_support`, WITHOUT calling `.fit()`. `dispatch_fit` (§1.3e)
    is this same logic plus an immediate fit; `_run_model_core` below
    uses this directly because its caller needs an unfitted model."""
    est = builder(**build_kwargs)
    if weighting == "none":
        return est, x, y, None
    if weight_support == "resample":
        wmap = class_weight_map(y, weighting, damping=damping, max_ratio=max_ratio)
        x_res, y_res = resample_by_weight(x, y, wmap, random_state=seed)
        return est, x_res, y_res, None
    wmap = class_weight_map(y, weighting, damping=damping, max_ratio=max_ratio)
    sw = sample_weight_array(y, wmap)
    return est, x, y, sw

def _run_model_core(x_tr_raw, y_tr, x_te, y_te, model_name, tag, weighting,
                     class_names, label_set, fit_and_log_fn, pca_variance=None,
                     scaler_name=None, variant_prefix="featset",
                     model_specs=None, get_builder_fn=None):
    specs = model_specs if model_specs is not None else _BASE_MODEL_SPECS
    spec = specs[model_name]
    support = weight_support_for(model_name)
    scale_sensitive = spec.get("scale_sensitive", False)
    scaler_cls = SCALER_VARIANTS[scaler_name] if scaler_name else None
    builder = (get_builder_fn or get_builder)(model_name, specs)

    pca_suffix = f"_{[k for k, v in PCA_VARIANTS.items() if v == pca_variance][0]}" \
        if pca_variance is not None else ""
    scaler_suffix = f"_{scaler_name}" if (scaler_name and scale_sensitive) else ""
    feat_label_str = f"{tag}{pca_suffix}{scaler_suffix}"

    build_kwargs = {"pca_variance": pca_variance}
    if scale_sensitive:
        build_kwargs["scaler_cls"] = scaler_cls

    est, x_use, y_use, sw = _prepare_fit_inputs(builder, build_kwargs, x_tr_raw, y_tr,
                                                 weighting, support, seed=SEED,
                                                 damping=0.5, max_ratio=10.0)

    variant = f"{variant_prefix}_{weighting}"
    model, metrics = fit_and_log_fn(
        model=est, x_tr=x_use, y_tr=y_use, x_te=x_te, y_te=y_te,
        label_set=label_set, model_name=model_name, feature_set=feat_label_str,
        variant=variant, class_names=class_names,
        stopping_rule=f"feature_set={feat_label_str}; weighting={weighting}; "
                      f"support={support}; scaler={scaler_name or 'n/a'}; "
                      f"backend={'GPU' if USE_GPU else 'CPU'}",
        sample_weight=sw, plot=False)
    return model, metrics


# %%
"""
## Section 1.10 — Module bootstrap helpers

`bootstrap_module_io` sets up a flat, single-ledger layout (one
`results_log.json`). `io_for` is the per-split layout every module uses: one call per split (not per branch/label_set)
that sets up `OUTPUT_DIR/{split_id}/{logs,results,plots}/`, and returns
`fit_and_log`/`run_histgbc` closures that resolve their per-branch
results file AND per-branch logger dynamically from whatever `label_set`
each call passes in — see Section 1.6's markdown for the mechanism.
`publish_split_summary` copies a module's already-produced final
artifacts into the cross-split `results_aggregated/{split_id}/` folder
(a plain file copy, no new computation).
"""


# %%
def bootstrap_module_io(output_dir: Path, plots_dir: Path):
    """Common per-module setup: ensure dirs exist,
    return bound (fit_and_log, run_histgbc, results_path) triple, flat,
    single-ledger, print-only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_path = make_results_path(output_dir)
    metadata_manifest_path = output_dir / "model_metadata.txt"

    telemetry = RunTelemetry(metadata_path=metadata_manifest_path, results_paths=[results_path])
    fit_and_log = telemetry.track(_make_fit_and_log(lambda label_set: [results_path], plots_dir))
    run_histgbc = make_run_histgbc(lambda label_set: [results_path])

    return {
        "results_path": results_path,
        "metadata_manifest_path": metadata_manifest_path,
        "fit_and_log": fit_and_log,
        "run_histgbc": run_histgbc,
    }
def get_split_dir(output_dir: Path, split_id: str = SPLIT_ID_DEFAULT) -> dict:
    """Ensures OUTPUT_DIR/{split_id}/{logs,results,plots}/
    exist and returns their paths, without binding any IO callbacks —
    the lower-level building block `io_for` (below) is built on."""
    split_dir = output_dir / split_id
    logs_dir, results_dir, plots_dir = split_dir / "logs", split_dir / "results", split_dir / "plots"
    for d in (logs_dir, results_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {"split_dir": split_dir, "logs_dir": logs_dir,
            "results_dir": results_dir, "plots_dir": plots_dir}
def io_for(output_dir: Path, split_id: str = SPLIT_ID_DEFAULT) -> dict:
    """The recommended per-split IO bundle.
    Call ONCE per split per module (NOT once per branch/label_set) —
    every branch trained during that split reuses the returned
    `fit_and_log`/`run_histgbc`, which resolve their per-branch results
    file (`results/{label_set}.json`) and per-branch logger
    (`logs/{label_set}.log`) dynamically from whatever `label_set` each
    call passes, exactly mirroring how `label_set` already flows through
    `fit_and_log`/`run_histgbc`.

    Every row also lands in `results/results_log.json` (the full-ledger
    aggregator/backup).

    Returns a dict: split_dir, logs_dir, results_dir, plots_dir,
    split_results_path, metadata_manifest_path, fit_and_log,
    run_histgbc, split_id."""
    dirs = get_split_dir(output_dir, split_id)
    logs_dir, results_dir, plots_dir = dirs["logs_dir"], dirs["results_dir"], dirs["plots_dir"]
    split_results_path = results_dir / "results_log.json"
    metadata_manifest_path = results_dir / "model_metadata.txt"

    def _results_paths_fn(label_set):
        return [split_results_path, results_dir / f"{_safe_filename(label_set)}.json"]

    def _logger_fn(label_set):
        return get_section_logger(label_set, logs_dir)

    telemetry = RunTelemetry(metadata_path=metadata_manifest_path, results_paths=[split_results_path])
    fit_and_log = telemetry.track(_make_fit_and_log(_results_paths_fn, plots_dir, logger_fn=_logger_fn))
    run_histgbc = make_run_histgbc(_results_paths_fn, logger_fn=_logger_fn)

    return {
        "split_dir": dirs["split_dir"], "logs_dir": logs_dir,
        "results_dir": results_dir, "plots_dir": plots_dir,
        "split_results_path": split_results_path,
        "metadata_manifest_path": metadata_manifest_path,
        "fit_and_log": fit_and_log, "run_histgbc": run_histgbc,
        "split_id": split_id,
        "results_paths_fn": _results_paths_fn, "logger_fn": _logger_fn,
    }
def publish_split_summary(split_id: str, output_dir: Path, summary_files: Sequence[Path]) -> Path:
    """Copies a module's already-produced
    final artifacts (leaderboard CSVs, winner_info*.json,
    hierarchical_metrics.json / merged_classification_metrics.json,
    model_metadata.txt, ...) into `OUTPUT_DIR/results_aggregated/{split_id}/`.
    A plain file copy, no new computation — call once at the end of a
    module's run, after the source files already exist. Missing source
    files are skipped with a printed warning rather than raising, since
    a module might call this after only some of its optional artifacts
    were produced (e.g. a run that skipped plotting)."""
    dest_dir = Path(output_dir) / "results_aggregated" / split_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in summary_files:
        src = Path(src)
        if not src.exists():
            print(f"  ⚠ publish_split_summary: {src} does not exist yet — skipped.")
            continue
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        copied.append(dest)
    print(f"  ✓ publish_split_summary [{split_id}]: {len(copied)}/{len(summary_files)} "
          f"artifact(s) copied -> {dest_dir}")
    return dest_dir


# %%
"""
## Config banner

Printed once when this file's cells finish executing, so every
downstream module's own config banner has a known-good baseline to
compare against in a saved notebook output.
"""

# %%
print(f"fermi_common ready — SEED={SEED}  VERSION_TAG={VERSION_TAG}  "
      f"USE_GPU={USE_GPU}  SELECTION_METRIC={SELECTION_METRIC}")
print(f"  MODEL_SPECS: {list(MODEL_SPECS.keys())}")
print(f"  CatBoost available: {_CATBOOST_CLS is not None}")
print(f"  Default split id: {SPLIT_ID_DEFAULT}  (per-split layout: OUTPUT_DIR/{{split_id}}/{{logs,results,plots}}/)")
