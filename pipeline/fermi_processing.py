# %% [markdown]
# # fermi_processing.py

# %% [markdown]
# ## Fermi 4FGL processing — Module 1
#
# Runs against the shared `fermi_common.py` infrastructure (already
# executed earlier in this kernel — every name below, e.g. `SEED`,
# `VERSION_TAG`, `TAXONOMY`, `SUPERCLASS_ORDER`, `FEATURE_POWER_THRESHOLDS`,
# `LOO_MIN_CLASS_COUNT_THRESHOLD`, `LOO_MIN_BRANCH_TRAIN_THRESHOLD`,
# `select_by_cumulative_power`, `compute_selection_engine`, `USE_GPU`,
# `SELECTION_METRIC`, is simply in-scope, not imported cross-file).
#
# ## Overview
#
# Each point is expanded in its own markdown cell below, where it is most
# relevant:
#
# 1. **Multi-split validation** — `ENABLE_MULTI_SPLIT_VALIDATION` flag; runs
#    the whole split → feature-selection → export pipeline once, or
#    `N_VALIDATION_SPLITS` (default 5) times on independent splits.
# 2. **Nested CV explicitly NOT implemented** — a disclosed, deliberate
#    simplification, not an oversight (see "Design decision (a)" below).
# 3. **Feature-set tags**: `99`/`90`/`80` cumulative-power cuts, their
#    `_DECORR`/`_SHIFT_ROBUST` companions, and `ALL_FEATURES`.
# 4. **Explicit on-disk accounting of every dropped or merged class**
#    (fsrq/ssrq/css merge, `bcu`) — a `label_accounting.csv` per split.
# 5. **The engine's tunable constants** are each annotated below as either
#    grounded in a citable convention, or explicitly flagged "arbitrary but
#    legitimate": a pragmatic engineering choice, not tuned on this dataset.
# 6. **`GLAT`/`GLON` as model features** — justified explicitly below.
# 7. **Cache staleness** — accepted by design: delete a split's `cache/`
#    folder (or change `VERSION_TAG`) for a clean re-run; no hashing
#    machinery, not worth the complexity for a single-operator pipeline.
# 8. **Output folder layout** — see "Folder layout" below.
# 9. Setup A/B are driven from one `{"A": {...}, "B": {...}}` loop
#    everywhere (splitting, feature selection, plotting, export, metadata),
#    and the whole per-split pipeline is one function (`process_split`)
#    called once per split.
# 10. Results-section support, without touching any fitted model: (a) a
#     merged/superclass class-distribution plot (Section 4/end), (b)
#     per-node importance (`imp_df`) exported from
#     `run_local_feature_selection`/`run_branch_feature_selection`
#     (Section 3/5), feeding an engineered-feature "node-rescue" heatmap
#     (Section 4), (c) a cross-split multi-model-agreement aggregator —
#     Spearman + Jaccard-AUC + RBO(p=0.90) (Section 4/6c), and (d) the
#     optional-features comparison (Section 6b), pickle-cached like every
#     other expensive step in this file.

# %% [markdown]
# ## Design decision — (a) why feature selection is NOT nested inside model selection
#
# The statistically "purest" design would re-run feature selection inside
# *every single fold* of *every* downstream model's hyperparameter search,
# for all four pipelines (flat Setup A/B, 2-level merged, 3-level), the
# hierarchical ones having up to ~15 selection nodes each (router, gate,
# sub-gate, non-blazar router, one branch per superclass). At
# `STABILITY_N_PAIRS=25` complementary pairs (50 half-sample fits) each, that
# is a multiplier of (#hyperparameter configs x #folds) on top of an
# already-expensive procedure — not affordable on a single local
# workstation, and arguably not worth it: the point of stability selection
# is precisely to make the *chosen feature set* insensitive to which
# particular subsample produced it, which already substantially reduces the
# marginal benefit of re-deriving it inside every model-selection fold too.
#
# We instead run feature selection ONCE per split (this module), and hand
# the resulting feature set to Modules 2/3 as a fixed input for their own,
# separately-cross-validated model/hyperparameter search. This means the
# cross-validated *model-selection* scores reported by Modules 2/3 carry a
# small, well-known optimism (feature selection and model selection are
# not statistically independent of each other, because both draw on
# overlapping training folds) — see Cawley & Talbot (2010, *Journal of
# Machine Learning Research* 11:2079-2107, "On Over-fitting in Model
# Selection and Subsequent Selection Bias in Performance Evaluation") for
# the general phenomenon. Critically, this optimism affects only the
# internal CV scores used to *pick* a model — it does NOT affect the final
# held-out *test*-set metric, since test data is never touched by feature
# selection (this module) or by model selection (Modules 2/3) at any point
# (true per source under the `single_split` and `per_split` designs of
# run_fermi.py; not under the legacy `shared` design). Ambroise & McLachlan
# (2002, PNAS 99:6562) show the size of this bias for selection done outside
# CV. This is a disclosed simplification: state it plainly in the paper
# rather than presenting internal CV scores as if they had no selection bias.

# %% [markdown]
# ## Design decision — (c) why feature selection stays HistGBC-specific
#
# Every importance ranking in this module (global Setup A/B and every
# hierarchical node) is computed from a single model family,
# `HistGradientBoostingClassifier`. This is a deliberate, justified choice,
# not an oversight: HistGBC is nonlinear (captures interactions a linear
# ranking would miss), handles missing values and mixed feature scales
# natively (no imputation/scaling artifacts can bias the ranking), and is
# fast enough to run under the repeated-CV budget this pipeline needs
# (10 repeats x 5 folds, at up to ~15 nodes, x up to 5 splits).
#
# We do NOT re-derive a separate feature set per downstream model family
# (e.g. one ranking for tree ensembles, another for KNN/LDA/SVM). The
# literature (Kohavi & John's classic result that feature relevance is
# relative to the induction algorithm; see also the "ensembles for feature
# selection" literature) supports that model-specific selection can help,
# but it multiplies the number of feature-selection runs and reintroduces
# exactly the multiple-comparisons/researcher-degrees-of-freedom risk this
# pipeline is trying to keep in check. Instead we keep ONE relevance
# ranking (HistGBC-derived, stability-selected) and let each downstream
# model family consume a *transformation* of the SAME ranking tailored to
# its own inductive bias — this module already computes both the
# undecorrelated tag (`90`, best for tree ensembles, which tolerate
# correlated inputs) and the decorrelated companion (`90_DECORR`, best for
# distance-/covariance-based models like KNN/LDA/SVM, which do not). Which
# tag each model family should pull is a Module 2/3 decision; this module's
# job is only to make both available. The ranking's cross-model validity is
# checked, not enforced: `compute_feature_sets` cross-checks the HistGBC
# ranking against SHAP, Random Forest, and Logistic-Regression importance
# (Spearman correlation + top-k Jaccard overlap), logged and plotted per
# setup — if a paper's best-performing model ends up being linear or
# distance-based, cite that agreement-summary number as evidence the
# HistGBC-derived selection did not handicap it.

# %% [markdown]
# ## Design decision — the fsrq / ssrq / css merge
#
# `ssrq` (steep-spectrum radio quasar, 2 sources) and `css` (compact
# steep-spectrum source, 6 sources) are merged into `fsrq` before any split
# or selection, and the class is reported as **"FSRQ-like"**. This is a
# disclosed CONVENTION, not a physical identity:
#
# - For: all three are radio-loud, broad-line quasars with a relativistic
#   jet, and 8 sources cannot be learned or validated as separate classes.
# - Against (state it in the paper): 4LAC lists CSS and SSRQ among the
#   NON-blazar ("misaligned") AGN, with radio galaxies (Ajello et al. 2020,
#   ApJ 892:105), and in AGN unification steep-spectrum quasars are the
#   misaligned parent population of FSRQs (Urry & Padovani 1995, PASP
#   107:803) — so mapping them to AGN_OTHER would follow the catalogue.
# - A preliminary classifier gave them roughly uniform (1/3, 1/3, 1/3)
#   posteriors; with 8 sources (and class weighting) that is what any
#   classifier does, so it is not evidence of indistinguishability.
#
# They are 0.3% of Setup B; a merge-vs-drop sensitivity check settles
# whether the choice matters.

# %% [markdown]
# ## Design decision — why `bcu` is dropped from the primary pipeline (Setup B)
#
# `bcu` ("blazar candidate of uncertain type") is not a physical class: it
# is an *observational* flag for a source that is confidently a blazar
# (via radio/X-ray/optical counterpart and broadband SED) but whose optical
# spectrum was not good enough to confidently classify as FSRQ (broad
# emission lines) or BL Lac (weak/no emission lines). Asking a classifier
# to predict "is this source's true subclass unknown" is not a meaningful
# physical target — it would be learning something about which sources
# happened to get a good enough follow-up spectrum, not about blazar
# physics. Setup A retains `bcu` anyway, purely to report a full-taxonomy
# baseline comparable to "every label present in the catalog"; Setup B
# (the primary pipeline) excludes it. This mirrors the precedent set by
# "Classification of Fermi-LAT unidentified gamma-ray sources using
# CatBoost gradient boosting decision trees," the paper this project's
# methodology follows most closely, which drops `bcu` for the same reason.

# %% [markdown]
# ## Design decision — (f) why `GLAT`/`GLON` are kept as model features
#
# Galactic latitude/longitude are retained as ordinary input features, not
# used only for the spatial cross-check. This is a deliberate choice: the
# different source populations in this taxonomy really are non-uniformly
# distributed on the sky for genuine astrophysical reasons (pulsars and
# SNR/PWN cluster near the Galactic plane; blazars are extragalactic and
# roughly isotropic), so position carries real physical signal, not just a
# survey artifact. The risk this creates — a model learning sky position as
# a shortcut (counterpart surveys are shallower in the plane, so |b| also
# predicts "was this source associated") — is NOT tested by the spatial
# cross-check in `compute_spatial_agreement`: it compares in-sample
# importance RANKINGS above and below |b| = 10°, which differ anyway because
# the class mix differs, and a latitude shortcut can rank high on both sides.
# Behavioural evidence is what counts: test metrics stratified by |b|
# (validation Section 16a) and an ablation without GLAT/GLON. GLON is fed as
# a raw angle (0° and 359° look far apart to linear/MLP models); sin/cos
# encoding would fix that.

# %% [markdown]
# ## Design decision — (g) cache staleness
#
# Every expensive step below is pickle-cached to its split's `cache/`
# folder and silently reused on a later run. No hash-based invalidation is
# implemented: if you change a constant (e.g. `STABILITY_N_PAIRS`) you
# must manually delete the affected split's `cache/` folder before
# re-running, or the run will silently reuse stale results. This is a
# known, accepted trade-off for a single-operator research pipeline —
# adding a constants-hash into every cache filename would remove the
# hazard but adds real complexity for a failure mode that is easy to avoid
# by convention ("delete the cache before a clean run"). Not implemented.
#
# The optional-features comparison (Section 6b) now follows the
# same convention — it previously had NO cache guard at all and refit
# Setup A/B from scratch on every re-run regardless of whether anything
# had changed; it now writes/reads
# `optional_features_comparison/cache/optional_features_results.pkl`, same
# "delete the cache folder for a clean re-run" rule as everywhere else.

# %% [markdown]
# ## Parameter justification
#
# Every tunable constant below is annotated as either grounded in a
# citable external convention, or flagged plainly as an arbitrary-but-
# legitimate pragmatic choice (i.e., not tuned on this dataset, not claimed
# optimal — a defensible default, nothing more).
#
# | Constant | Value | Status |
# |---|---|---|
# | `STABILITY_SELECTION_FRACTION` | 0.70 | **Literature-grounded.** Meinshausen & Bühlmann (2010, *JRSS-B* 72(4):417-473, "Stability Selection") show results are insensitive to the selection-frequency threshold across 0.6-0.9 when frequencies are computed over subsamples of size n/2 — the design used here (complementary pairs, Shah & Samworth 2013, *JRSS-B* 75:55-80). Their bound on expected false selections, q^2/((2*thr-1)p), is logged per feature set. |
# | `CORR_CLUSTER_THRESHOLD` | 0.85 | **Literature-grounded convention.** \|Spearman rho\| thresholds in the 0.7-0.9 range are the standard rule-of-thumb band in applied-statistics treatments of multicollinearity; 0.85 sits within it. Not derived from one canonical source — a convention, not a law. |
# | `MIN_SAMPLES_PER_FEATURE` | 5 | **Literature-grounded.** Small-sample pattern-recognition literature (e.g. Raudys & Jain, 1991, *IEEE TPAMI*, "Small Sample Size Effects in Statistical Pattern Recognition") commonly recommends on the order of 5-10 training rows per feature to avoid the curse of dimensionality; 5 is the permissive end of that band. |
# | `PSI_FLAG_THRESHOLD` | 0.25 | **Literature-grounded.** The standard "severe population shift" cutoff in the Population Stability Index literature used in credit-risk/model-monitoring practice (<0.10 negligible, 0.10-0.25 moderate, >=0.25 severe); see e.g. Yurdakul (2018), *Statistical Properties of the Population Stability Index*, Western Michigan University dissertation. |
# | `GLAT_STRATIFICATION_THRESHOLD` | 10.0 deg | **Literature-grounded.** Matches the \|b\| > 10 deg "clean sample" convention used by the Fermi-LAT Fourth AGN Catalog (4LAC) team to separate a low-contamination extragalactic sample from the Galactic-plane-confused region. |
# | `IMPORTANCE_LCB_Z` | 0.0 | Features are ranked on the mean importance over the half-sample fits. Their std is the variability of a HALF-sample estimate, so mean - z*std would be an extra, over-conservative stability filter on top of the selection frequency. |
# | `STABILITY_N_PAIRS` | 25 | **Budget choice.** 25 complementary pairs = 50 half-sample fits, the same number of model fits as a 10x5-fold scheme; Shah & Samworth (2013) use 50 pairs. |
# | `COVARIATE_SHIFT_ALPHA` | 0.01 | **Arbitrary but legitimate.** No multiple-comparisons correction is applied across the ~71 KS tests; for this data it would change nothing (every flagged feature has p < 1e-100, and PSI >= 0.25 is the binding criterion). Rabanser et al. (2019, NeurIPS) use Bonferroni for the same detector. |
# | `LOO_MIN_CLASS_COUNT_THRESHOLD` / `LOO_MIN_BRANCH_TRAIN_THRESHOLD` / `BRANCH_MIN_TRAIN_FOR_SELECTION_MERGED` / `BRANCH_MIN_TRAIN_FOR_SELECTION_3LEVEL` / `MAX_LOO_N` | 5 / 50 / 40 / 8 / 150 | **Arbitrary but legitimate.** Pragmatic thresholds sized to this dataset's specific class-count distribution, not derived from a formula. State them as engineering judgment calls in the paper, not as validated statistical cutoffs. |

# %% [markdown]
# ## Folder layout
#
# All outputs live under one clearly-named root,
# `fermi_module1_processing_{VERSION_TAG}/`, with one subfolder per split:
# `split_00_primary/` (the split used for the full Module-2/3 model search)
# and, only when `ENABLE_MULTI_SPLIT_VALIDATION=True`, `split_01/` ..
# `split_0{N-1}/` (independent re-confirmation splits — see the decision
# log doc). Each split folder has the same four subfolders:
#
# - `datasets/`   — per-tag, per-setup train/test CSVs ready for Module 2/3.
# - `encoders/`   — fitted `LabelEncoder`s for Setup A and Setup B.
# - `diagnostics/`— every CSV report (importance, SHAP, agreement, spatial,
#                   missingness, covariate shift, class counts,
# `label_accounting.csv`, per-node
#                   `node_importance_*.csv`) plus `feature_metadata.json`.
# - `cache/`      — pickled intermediate results (feature-selection engine
#                   outputs) so a re-run of this split is fast.
#
# `plots/` exists only under `split_00_primary/` — diagnostic plots are a
# sanity check on the primary split; regenerating them 5x would be pure
# clutter with zero scientific benefit, so confirmation splits (1..N-1)
# are numeric-only. A `multi_split_manifest.json` at the root records which
# splits were built and where, for Modules 2/3 to discover them.
#
# `optional_features_comparison/` (only when `ENABLE_OPTIONAL_FEATURES_
# COMPARISON=True`) sits directly under the root, alongside the split
# folders, not inside one — it is a one-off diagnostic against split 0,
# not another split. now also has its own `cache/` and `plots/`
# subfolders.
#
# Two more catalog-level (not per-split) folders:
# `plots_catalog/` (merged/superclass class-distribution plot) and, at
# `BASE_OUTPUT_DIR` root, `multi_model_agreement_{aggregate,per_split}_B.csv`
# / `multi_model_agreement_setup_A_split0.csv` / the leadership-flip CSV —
# derived post-hoc from every split's already-exported diagnostics, so
# they live at the root next to `multi_split_manifest.json`, not inside
# any one split folder.

# %%
import json
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp, chi2_contingency
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score

import shap
import joblib

# %% [markdown]
# ## Section 0 — Configuration, folder layout, multi-split switch

# %%
APPLY_PHYSICAL_MERGE = True
MERGE_FINE_CLASSES = {"ssrq": "fsrq", "css": "fsrq"}   # label `fsrq` = "FSRQ-like"; see design-decision cell above
CLASSES_TO_DROP = ["gc", "unk"]                         # not coherent target classes
ID_COLUMNS_TO_DROP = ["Source_Name", "CLASS1", "Was_Imputed"]
# RA/Dec repeat GLON/GLAT; raw GLON is replaced by GLON_sin/GLON_cos (preprocessing: an angle,
# 359 deg next to 1 deg). GLON stays in the CSVs for cross-matching, just not as a feature.
REDUNDANT_SPATIAL_COLUMNS = ["RAJ2000", "DEJ2000", "GLON"]
ID_TAG_COLUMN = "Source_Name"

DATA_PATH = _cfg("DATA_PATH", Path("./data/processed/without_optional_features/"))
BASE_OUTPUT_DIR = _cfg("PROCESSING_ROOT", Path(f"./fermi_module1_processing_{VERSION_TAG}/"))

# --- Optional-features comparison (physics-driven vs. ML-driven selection) --
# `DATA_PATH` and `DATA_PATH_WITH_OPTIONAL` are two on-disk cuts of the SAME
# underlying catalog: `DATA_PATH` has already had redundant/double-counted
# columns manually purged on physical grounds; `DATA_PATH_WITH_OPTIONAL`
# restores them. This lets the fine (Setup A/B) selection engine be run
# against both and the two selections diffed — a direct empirical check on
# whether the physics-motivated manual purge agrees with what a purely
# data-driven importance ranking would have kept anyway (see §3.11 of the
# review doc). Setup A/B ONLY (not the hierarchical merged/3-level trees —
# this is a targeted robustness check on the curation, not a second full
# pipeline) and against a single split (`SEED`) — this is a sanity check on
# feature curation, not a headline result needing multi-split confirmation.
DATA_PATH_WITH_OPTIONAL = _cfg("DATA_PATH_WITH_OPTIONAL", Path("./data/processed/with_optional_features/"))

# --- Processing parts: each can run on its own and resumes from its caches ----
#   core     — splits, Setup B (+ A) global feature selection, dataset exports. Everything needs it.
#   hier     — hierarchical local selections (router/gate/subgate/non-blazar router/branches);
#              only the merged and 3-level modules need it.
#   diag     — SHAP / RF+LogReg / latitude / missingness diagnostics; never change the selection.
#   optional — the with-optional-features comparison (a one-off check of the manual purge).
# A part not listed is not computed, but its finished caches are still loaded and exported.
PROCESSING_PARTS = set(_cfg("PROCESSING_PARTS", ["core", "hier", "diag", "optional"]))
PROCESS_SETUP_A = _cfg("RUN_FINE_SETUP_A", True)  # Setup A (bcu kept) only feeds fine's A-vs-B comparison

# --- THE multi-split switch --------------------------
ENABLE_MULTI_SPLIT_VALIDATION = True   # True -> run N_VALIDATION_SPLITS independent splits
N_VALIDATION_SPLITS = _cfg("N_VALIDATION_SPLITS", 5)  # 1 = a single train/test split

# --- Global (Setup A/B) stability-selection engine constants --------------
STABILITY_N_PAIRS = _cfg("STABILITY_N_PAIRS", 25)   # complementary pairs (CPSS) -> 50 half-sample fits
STABILITY_SELECTION_FRACTION = 0.7
IMPORTANCE_LCB_Z = 0.0   # rank on the mean importance — see fermi_common.py

# --- Decorrelation ----------------------------------------------------------
CORR_CLUSTER_THRESHOLD = 0.85

# --- Covariate-shift (domain_rules: Covariate Shift) -----------------------
COVARIATE_SHIFT_ALPHA = 0.01
PSI_FLAG_THRESHOLD = 0.25

# --- Spatial-bias (domain_rules: Spatial Bias) ------------------------------
GLAT_STRATIFICATION_THRESHOLD = 10.0

# --- Hierarchical local-selection branch-size floors ------------------------
# Two different floors, one per pipeline plan, preserved as-is: the 3-level
# pipeline's branches can be smaller because compute_selection_engine (in
# fermi_common.py) has its own in-sample-importance fallback for
# branches this small — 8 is the point below which even one held-out-safe
# split is impossible, not a claim of validated quality at n=8.
BRANCH_MIN_TRAIN_FOR_SELECTION_MERGED = 40
BRANCH_MIN_TRAIN_FOR_SELECTION_3LEVEL = 8

ROUTER_TAG_PREFIX, BRANCH_TAG_PREFIX = "router", "branch"
GATE_TAG_PREFIX, SUBGATE_TAG_PREFIX, NONBLAZAR_TAG_PREFIX = "gate", "subgate", "nonblazar"

# Superclass taxonomy partition used by the 3-level pipeline.
BLAZAR_SUPERCLASSES = {"BLAZAR_HE", "BLAZAR_LE"}
NONBLAZAR_SUPERCLASS_ORDER = [sc for sc in SUPERCLASS_ORDER if sc not in BLAZAR_SUPERCLASSES]
SUPER_LE = LabelEncoder().fit(SUPERCLASS_ORDER)
GATE_LE = LabelEncoder().fit(["BLAZAR", "OTHER"])
BLAZAR_SUBGATE_LE = LabelEncoder().fit(sorted(BLAZAR_SUPERCLASSES))
NONBLAZAR_ROUTER_LE = LabelEncoder().fit(NONBLAZAR_SUPERCLASS_ORDER)

print(f"Config ready — SEED={SEED}  VERSION_TAG={VERSION_TAG}  USE_GPU={USE_GPU}  "
      f"SELECTION_METRIC={SELECTION_METRIC}")
print(f"  Multi-split validation: {'ON — ' + str(N_VALIDATION_SPLITS) + ' splits' if ENABLE_MULTI_SPLIT_VALIDATION else 'OFF — single split'}")
print(f"  Stability engine: CPSS n_pairs={STABILITY_N_PAIRS}  stability_frac>={STABILITY_SELECTION_FRACTION}  lcb_z={IMPORTANCE_LCB_Z}")
print(f"  Output root -> {BASE_OUTPUT_DIR.resolve()}")

# %% [markdown]
# ## Section 1 — Data loading & split-independent preprocessing
#
# Everything in this section runs **once**, regardless of how many
# validation splits are requested: loading, class merging, taxonomy
# mapping, the feature matrix, the covariate-shift diagnostic (associated
# vs. unassociated catalog — a property of the catalog, not of any one
# train/test partition), and the plain per-feature NaN report. Only the
# things that genuinely depend on *which rows land in train vs. test*
# (Section 2 onward) are repeated inside the per-split loop. This is a
# deliberate compute optimisation: it means enabling 5-split validation
# costs roughly 4x a narrow re-run, not 5x the full pipeline.
#
# Nothing in this section fits a model — it is pure I/O, merging and two
# non-parametric statistical tests (KS + chi-square), so re-running it
# after a kernel restart is always cheap regardless of caching.

# %%
df = pd.read_csv(DATA_PATH / "full_processed_dataset_with_string_labels.csv")
df_unknown = pd.read_csv(DATA_PATH / "unknown_sources_full_processed.csv")
print(f"Labeled dataset : {df.shape[0]:>5} rows x {df.shape[1]} cols")
print(f"Unknown dataset : {df_unknown.shape[0]:>5} rows x {df_unknown.shape[1]} cols")

x_raw = df.drop(columns=ID_COLUMNS_TO_DROP + REDUNDANT_SPATIAL_COLUMNS, errors="ignore").copy()
y_raw = df["CLASS1"].copy()
tag_raw = (df[ID_TAG_COLUMN].copy() if ID_TAG_COLUMN in df.columns
           else pd.Series([f"row_{i}" for i in df.index], index=df.index, name=ID_TAG_COLUMN))

# Physical merge (fsrq/ssrq/css) — applied before any split/selection; see
# the design-decision cell above for the astrophysical + empirical
# justification. `merge_accounting` is carried into label_accounting.csv
# per split so the merge is auditable, not just asserted in a docstring.
merge_accounting = []
if APPLY_PHYSICAL_MERGE:
    for src, dst in MERGE_FINE_CLASSES.items():
        n = int((y_raw == src).sum())
        merge_accounting.append({"action": "merged", "class": src, "into": dst, "n_rows": n})
    y_raw = y_raw.replace(MERGE_FINE_CLASSES)
    print(f"Physical merge applied: {[m['n_rows'] for m in merge_accounting]} sources "
          f"{list(MERGE_FINE_CLASSES.items())} -> fsrq, before any split/selection.")

mask_keep = ~y_raw.isin(CLASSES_TO_DROP)
for cls in CLASSES_TO_DROP:
    merge_accounting.append({"action": "excluded_not_a_target_class", "class": cls,
                              "into": None, "n_rows": int((y_raw == cls).sum())})
x_raw, y_raw, tag_raw = x_raw[mask_keep].copy(), y_raw[mask_keep].copy(), tag_raw[mask_keep].copy()
print(f"Dropped {int((~mask_keep).sum())} sources with classes: {CLASSES_TO_DROP}")

# Superclass taxonomy mapping (fine -> physical superclass, from fermi_common).
y_super = y_raw.map(TAXONOMY)
unmapped_idx = y_super[y_super.isna()].index
if len(unmapped_idx):
    unmapped_classes = sorted(y_raw.loc[unmapped_idx].unique().tolist())
    print(f"{len(unmapped_idx)} sources unmapped (dropped): {unmapped_classes}")
    for cls in unmapped_classes:
        merge_accounting.append({"action": "excluded_unmapped_in_taxonomy", "class": cls,
                                  "into": None, "n_rows": int((y_raw.loc[unmapped_idx] == cls).sum())})
else:
    print("All fine classes successfully mapped to superclasses.")

# Feature matrix: x_all keeps every remaining native column; rows unmapped
# by TAXONOMY are dropped (no standalone superclass-only model is built).
valid = y_super.notna()
x_all, y_all, y_all_s, tag_all = x_raw[valid].copy(), y_raw[valid].copy(), y_super[valid].copy(), tag_raw[valid].copy()
print(f"Final feature matrix : {x_all.shape[0]} sources x {x_all.shape[1]} features  "
      f"({int(x_all.isna().sum().sum())} NaN cells)")

assert {"GLAT", "GLON_sin", "GLON_cos"} <= set(x_all.columns), \
    "Galactic coordinates (GLAT, GLON_sin/cos) must be retained — run preprocessing.py"
assert not {"RAJ2000", "DEJ2000", "GLON"} & set(x_all.columns), "RA/Dec and raw GLON must not be features"
assert ID_TAG_COLUMN not in x_all.columns, f"{ID_TAG_COLUMN} is a tag, not a feature"

# These three are properties of the CATALOG (not of any one split), so they
# are written once, directly under BASE_OUTPUT_DIR rather than per-split.
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
y_raw.value_counts().rename("count").to_csv(BASE_OUTPUT_DIR / "fine_class_counts_post_filter.csv")
y_super.value_counts().reindex(SUPERCLASS_ORDER).rename("count").to_csv(BASE_OUTPUT_DIR / "super_class_counts.csv")



# %% [markdown]
# ### Covariate-shift diagnostic (associated vs. unassociated), once
#
# Trains exclusively on `x_all` (resolved `CLASS1`) but the finished model
# is applied to `df_unknown` (unassociated sources). Per-feature two-sample
# KS test + Population Stability Index (binned on the associated
# distribution's own deciles); a feature is flagged only when BOTH the KS
# test is significant AND PSI clears the severe-shift line (0.25) — pairing
# effect-size with significance avoids over-flagging on thousands of rows.
# Diagnostic only: see the design-decision on item (g)/shift-robust sets —
# the `_SHIFT_ROBUST` feature-set companion is what actually acts on this.

# %%
def compute_covariate_shift(x_assoc, df_unassoc, feature_names,
                             alpha=COVARIATE_SHIFT_ALPHA, psi_thr=PSI_FLAG_THRESHOLD, n_bins=10):
    rows = []
    unassoc = df_unassoc.reindex(columns=feature_names)
    for feat in feature_names:
        a, b = x_assoc[feat].dropna().to_numpy(), unassoc[feat].dropna().to_numpy()
        if len(a) < 10 or len(b) < 10:
            rows.append({"feature": feat, "ks_stat": np.nan, "ks_pvalue": np.nan, "psi": np.nan,
                         "n_assoc": len(a), "n_unassoc": len(b), "shift_flag": False,
                         "reason": "insufficient_non_nan_rows"})
            continue
        ks_stat, ks_p = ks_2samp(a, b)
        edges = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            psi = np.nan
        else:
            a_frac = np.clip(np.histogram(a, bins=edges)[0] / max(len(a), 1), 1e-6, None)
            b_frac = np.clip(np.histogram(b, bins=edges)[0] / max(len(b), 1), 1e-6, None)
            psi = float(np.sum((b_frac - a_frac) * np.log(b_frac / a_frac)))
        flagged = bool((ks_p < alpha) and not np.isnan(psi) and (psi >= psi_thr))
        rows.append({"feature": feat, "ks_stat": float(ks_stat), "ks_pvalue": float(ks_p), "psi": psi,
                     "n_assoc": len(a), "n_unassoc": len(b), "shift_flag": flagged,
                     "reason": "ks_and_psi_exceed_threshold" if flagged else ""})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


COVARIATE_SHIFT_DF = compute_covariate_shift(x_all, df_unknown, x_all.columns.tolist())
SHIFT_FLAGGED_FEATURES = COVARIATE_SHIFT_DF.loc[COVARIATE_SHIFT_DF["shift_flag"], "feature"].tolist()
COVARIATE_SHIFT_DF.to_csv(BASE_OUTPUT_DIR / "covariate_shift_diagnostic.csv", index=False)
print(f"Covariate-shift: {len(SHIFT_FLAGGED_FEATURES)}/{len(x_all.columns)} feature(s) flagged "
      f"(KS p<{COVARIATE_SHIFT_ALPHA} AND PSI>={PSI_FLAG_THRESHOLD}): {SHIFT_FLAGGED_FEATURES}")

# Plain per-feature NaN report — descriptive only, computed once on x_all
# (the per-SETUP missingness-*informativeness* test against class labels
# is necessarily split-dependent and lives inside process_split below).
NAN_REPORT = pd.DataFrame({
    "n_nan_cells": x_all.isna().sum(),
    "pct_nan": (x_all.isna().sum() / len(x_all) * 100).round(3),
}).sort_values("n_nan_cells", ascending=False)
NAN_REPORT.to_csv(BASE_OUTPUT_DIR / "nan_report_by_feature.csv")


# %% [markdown]
# ## Section 2 — Feature-selection engine (shared by every split)
#
# Everything in this section is a pure function of whatever `(x_tr, y_tr)`
# it is called with — it holds no split-specific state, so it is defined
# once and reused by every split in the loop. This is the statistical core
# of the module.
# Brief recap of the five-step recipe `compute_feature_sets` runs:
# (0) cluster correlated features, |Spearman rho| >= `CORR_CLUSTER_THRESHOLD`
# (`spearman_clusters`) -> (1) HistGBC held-out GROUPED permutation importance
# (each cluster permuted as one unit; Gregorutti et al. 2017) on
# `STABILITY_N_PAIRS` complementary pairs of class-stratified half-samples
# (CPSS, Shah & Samworth 2013) -> (2) rank clusters by mean importance ->
# (3) cumulative-power cutoff (smallest cluster subset covering 99%/90%/80%
# of positive importance mass) -> (4) stability filter (clusters selected in
# >= `STABILITY_SELECTION_FRACTION` of the half-sample fits; Meinshausen &
# Bühlmann 2010) -> (5) expand: a tag keeps every member of its selected
# clusters, `_DECORR` one representative per cluster, `_SHIFT_ROBUST` drops
# covariate-shift-flagged features.
# There is no fixed "top-15 by importance" set: the cumulative-power cuts
# above already adapt the feature count to each node.

# %%
def determine_cv_eligibility(y_enc, loo_min_class_count=LOO_MIN_CLASS_COUNT_THRESHOLD,
                              loo_total_n_threshold=LOO_MIN_BRANCH_TRAIN_THRESHOLD):
    """Which rows can ever land in a HELD-OUT CV fold, decided once per
    setup. Rare classes (<= loo_min_class_count members) ride along in
    every fold's training rows but are never held out; a genuinely small
    problem (n <= loo_total_n_threshold) uses true LeaveOneOut instead."""
    y_arr = np.asarray(y_enc)
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(y_arr) <= loo_total_n_threshold:
        return np.ones(len(y_arr), dtype=bool), "LeaveOneOut", int(counts.min()), [], True
    rare = classes[counts <= loo_min_class_count].tolist()
    eligible = ~np.isin(y_arr, rare)
    if eligible.sum() == 0 or len(np.unique(y_arr[eligible])) < 2:
        return np.ones(len(y_arr), dtype=bool), "LeaveOneOut", int(counts.min()), rare, True
    mode = "StratifiedKFold" + (f" ({len(rare)} rare class(es) train-only)" if rare else "")
    return eligible, mode, int(counts.min()), rare, False


def make_cv_splitter(y_enc, n_splits=5, seed=SEED,
                      loo_min_class_count=LOO_MIN_CLASS_COUNT_THRESHOLD,
                      loo_total_n_threshold=LOO_MIN_BRANCH_TRAIN_THRESHOLD):
    min_count = int(np.unique(y_enc, return_counts=True)[1].min())
    if len(y_enc) <= loo_total_n_threshold or min_count <= loo_min_class_count:
        return LeaveOneOut(), "LeaveOneOut", min_count
    n_eff = max(2, min(n_splits, min_count))
    return StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed), f"StratifiedKFold(n_splits={n_eff})", min_count


def stability_select(imp_matrix, feature_names, thresholds, lcb_z=IMPORTANCE_LCB_Z,
                      stability_frac=STABILITY_SELECTION_FRACTION):
    """LCB-adjusted point estimate fed to select_by_cumulative_power, plus
    a per-repeat selection-frequency filter (Meinshausen & Buhlmann)."""
    n_repeats = imp_matrix.shape[0]
    mean_imp, std_imp = imp_matrix.mean(axis=0), imp_matrix.std(axis=0)
    lcb_df = pd.DataFrame({"feature": feature_names,
                            "importance": np.clip(mean_imp - lcb_z * std_imp, 0.0, None),
                            "importance_mean": mean_imp, "importance_std": std_imp})

    freq = {tag: np.zeros(len(feature_names)) for tag in thresholds}
    for r in range(n_repeats):
        rep_df = pd.DataFrame({"feature": feature_names, "importance": imp_matrix[r]})
        for tag, frac in thresholds.items():
            sel = set(select_by_cumulative_power(rep_df, frac))
            freq[tag] += np.array([f in sel for f in feature_names], dtype=float)
    for tag in freq:
        freq[tag] /= n_repeats
    stability_freq = pd.DataFrame({"feature": feature_names,
                                    **{f"stability_freq_{t}": freq[t] for t in thresholds}})

    feats = {}
    for tag, frac in thresholds.items():
        sel_raw = select_by_cumulative_power(lcb_df, frac, importance_col="importance")
        sel_stable = [f for f in sel_raw if freq[tag][feature_names.index(f)] >= stability_frac]
        feats[tag] = sel_stable if sel_stable else sel_raw
        if not sel_stable and sel_raw:
            print(f"  ⚠ stability filter emptied {tag}% — falling back to LCB-only set ({len(sel_raw)} feats).")
    return feats, stability_freq, lcb_df


def apply_shift_robust(feats, shift_flagged_features):
    kept = [f for f in feats if f not in shift_flagged_features]
    return kept if kept else list(feats)


def _shap_mean_abs(model, x_bg, x_ev, feature_names, seed=SEED, max_bg=30, max_ev=50):
    """TreeExplainer with a re-subsampled model-agnostic fallback."""
    try:
        raw = shap.TreeExplainer(model, data=x_bg, feature_perturbation="interventional") \
                  .shap_values(x_ev, check_additivity=False)
    except Exception as e:
        print(f"    ⚠ TreeExplainer failed ({type(e).__name__}) — falling back to a slower, resubsampled Explainer.")
        rng = np.random.RandomState(seed)
        bg_idx = rng.choice(len(x_bg), size=min(max_bg, len(x_bg)), replace=False)
        ev_idx = rng.choice(len(x_ev), size=min(max_ev, len(x_ev)), replace=False)
        raw = shap.Explainer(model.predict_proba, x_bg.iloc[bg_idx])(x_ev.iloc[ev_idx]).values
    if isinstance(raw, list):
        return np.mean([np.abs(c).mean(axis=0) for c in raw], axis=0)
    arr = np.asarray(raw)
    if arr.ndim == 3:
        axis = 2 if arr.shape[1] == len(feature_names) else 1
        return np.abs(arr).mean(axis=tuple(i for i in range(arr.ndim) if i != (1 if axis == 2 else 2)))
    return np.abs(arr).mean(axis=0)


def compute_shap_cross_check(x_tr, y_tr_enc, feature_names, cv, cv_mode, max_bg=100, max_ev=300, seed=SEED):
    """CV-fold-averaged mean|SHAP|, cross-checked against permutation
    importance — a different attribution method should broadly agree."""
    rng, fold_vals = np.random.RandomState(seed), []
    if cv_mode != "LeaveOneOut":
        for tr_idx, va_idx in cv.split(x_tr, y_tr_enc):
            if len(va_idx) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=seed)
            m.fit(x_tr.iloc[tr_idx], y_tr_enc[tr_idx])
            bg = rng.choice(len(tr_idx), size=min(max_bg, len(tr_idx)), replace=False)
            ev = rng.choice(len(va_idx), size=min(max_ev, len(va_idx)), replace=False)
            fold_vals.append(_shap_mean_abs(m, x_tr.iloc[tr_idx].iloc[bg], x_tr.iloc[va_idx].iloc[ev], feature_names, seed))
    if not fold_vals:
        print(f"    SHAP cross-check: LOO/tiny folds — in-sample fallback (NOT held-out).")
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=seed)
        m.fit(x_tr, y_tr_enc)
        bg = rng.choice(len(x_tr), size=min(max_bg, len(x_tr)), replace=False)
        ev = rng.choice(len(x_tr), size=min(max_ev, len(x_tr)), replace=False)
        fold_vals.append(_shap_mean_abs(m, x_tr.iloc[bg], x_tr.iloc[ev], feature_names, seed))
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.vstack(fold_vals).mean(axis=0)}) \
             .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def compute_multi_model_agreement(x_tr, y_tr_enc, cv, cv_mode, feature_names, perm_df, seed=SEED, top_k=15, n_reps=10):
    """CV-fold-averaged RF/LogReg permutation importance vs. HistGBC —
    a ranking only one model family likes is weaker evidence than one
    two unrelated model families (tree + linear) both pick up."""


    def _fit_score(x_f_tr, y_f_tr, x_f_va, y_f_va):
        rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=seed, n_jobs=-1)
        rf.fit(x_f_tr, y_f_tr)
        rf_imp = permutation_importance(rf, x_f_va, y_f_va, n_repeats=n_reps, scoring="balanced_accuracy",
                                         random_state=seed, n_jobs=-1).importances_mean
        x_f_tr_g, x_f_va_g = _guard_allnan_fold_columns(x_f_tr, x_f_va)
        lr = _fit_lr_robust(x_f_tr_g, y_f_tr, seed)
        lr_imp = permutation_importance(lr, x_f_va_g, y_f_va, n_repeats=n_reps, scoring="balanced_accuracy",
                                         random_state=seed, n_jobs=-1).importances_mean
        return rf_imp, lr_imp

    # --- everything below is UNCHANGED original code — must still be present ---
    fold_rf, fold_lr = [], []
    if cv_mode != "LeaveOneOut":
        for tr_idx, va_idx in cv.split(x_tr, y_tr_enc):
            if len(va_idx) < 2:
                continue
            rf_i, lr_i = _fit_score(x_tr.iloc[tr_idx], y_tr_enc[tr_idx], x_tr.iloc[va_idx], y_tr_enc[va_idx])
            fold_rf.append(rf_i); fold_lr.append(lr_i)
    if not fold_rf:
        print(f"    Multi-model agreement: LOO/tiny folds — in-sample fallback (NOT held-out).")
        rf_i, lr_i = _fit_score(x_tr, y_tr_enc, x_tr, y_tr_enc)
        fold_rf, fold_lr = [rf_i], [lr_i]

    agree = pd.DataFrame({"feature": feature_names})
    agree["imp_histgbc"] = perm_df.set_index("feature").loc[feature_names, "importance"].values
    agree["imp_rf"], agree["imp_logreg"] = np.vstack(fold_rf).mean(axis=0), np.vstack(fold_lr).mean(axis=0)
    rho_rf, _ = spearmanr(agree["imp_histgbc"], agree["imp_rf"])
    rho_lr, _ = spearmanr(agree["imp_histgbc"], agree["imp_logreg"])
    top = lambda col: set(agree.sort_values(col, ascending=False)["feature"].head(top_k))
    jacc = lambda a, b: len(a & b) / len(a | b)
    print(f"    Spearman(HistGBC,RF)={rho_rf:.3f}  Spearman(HistGBC,LogReg)={rho_lr:.3f}  "
          f"Jaccard(RF)={jacc(top('imp_histgbc'), top('imp_rf')):.3f}  Jaccard(LogReg)={jacc(top('imp_histgbc'), top('imp_logreg')):.3f}")
    return agree, {"spearman_rf": float(rho_rf), "spearman_logreg": float(rho_lr),
                    "jaccard_rf": jacc(top("imp_histgbc"), top("imp_rf")),
                    "jaccard_logreg": jacc(top("imp_histgbc"), top("imp_logreg"))}



def _guard_allnan_fold_columns(x_f_tr, x_f_va, fill_value=0.0):
    """IterativeImputer's BayesianRidge sub-estimators require finite
    input. A column that is 100% NaN within THIS fold (common for
    optional/sparse catalog columns restricted to a subset of classes)
    makes SimpleImputer's own per-column mean NaN, which then leaks
    into every other feature it's used as a predictor for. Const-fill
    only fully-NaN columns; anything with partial coverage is left
    untouched for IterativeImputer to handle normally."""
    x_f_tr, x_f_va = x_f_tr.copy(), x_f_va.copy()
    allnan = x_f_tr.columns[x_f_tr.isna().all(axis=0)]
    if len(allnan):
        x_f_tr[allnan] = x_f_tr[allnan].fillna(fill_value)
        x_f_va[allnan] = x_f_va[allnan].fillna(fill_value)
    return x_f_tr, x_f_va


def _fit_lr_robust(x_f_tr, y_f_tr, seed):
    """IterativeImputer's chained BayesianRidge sub-fits can emit NaN/Inf
    coefficients when predictor columns are near-constant or near-duplicate
    (the 'optional' catalog reintroduces exactly these — they were purged
    from the primary feature set for this reason). A NaN prediction for one
    feature poisons every later feature in the same imputation pass, so a
    column-level pre-check can't reliably catch it. Try the IterativeImputer
    pipeline; on any numerical failure, fall back to a plain per-column
    SimpleImputer(median) pipeline for this fold only."""
    try:
        lr = Pipeline([("imputer", IterativeImputer(random_state=seed, max_iter=10)), ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(solver="saga", C=1.0, tol=1e-4, max_iter=2000,
                                                    random_state=seed, n_jobs=-1))])
        lr.fit(x_f_tr, y_f_tr)
        return lr
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    [warn] IterativeImputer unstable ({e.__class__.__name__}) on this fold "
              f"(likely near-collinear optional-feature columns) — falling back to SimpleImputer(median).")
        lr = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(solver="saga", C=1.0, tol=1e-4, max_iter=2000,
                                                    random_state=seed, n_jobs=-1))])
        lr.fit(x_f_tr, y_f_tr)
        return lr




def compute_spatial_agreement(x_tr, y_tr_enc, feature_names, glat_thr=GLAT_STRATIFICATION_THRESHOLD, seed=SEED, setup_label="setup"):
    """Independent importance ranking on |GLAT|>thr vs. |GLAT|<=thr — a
    feature important only because of plane-specific contamination would
    show up as a big disagreement between the two hemispheres."""
    if "GLAT" not in x_tr.columns:
        return None, {"error": "GLAT not in feature matrix"}
    glat = x_tr["GLAT"].abs()
    mask_hi = (glat > glat_thr).to_numpy()

    def _quick(mask, label):
        x_sub, y_sub = x_tr.iloc[mask], y_tr_enc[mask]
        counts = np.unique(y_sub, return_counts=True)[1] if len(y_sub) else np.array([])
        if len(x_sub) < 20 or len(counts) < 2 or counts.min() < 2:
            print(f"    [{setup_label}/{label}] too few rows/classes — spatial check skipped.")
            return None
        hgbc = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=seed)
        hgbc.fit(x_sub, y_sub)
        perm = permutation_importance(hgbc, x_sub, y_sub, n_repeats=20 if len(x_sub) >= 30 else 10,
                                       scoring="balanced_accuracy", random_state=seed, n_jobs=-1)
        return pd.Series(perm.importances_mean, index=feature_names)

    imp_hi, imp_lo = _quick(mask_hi, f"|b|>{glat_thr:g}"), _quick(~mask_hi, f"|b|<={glat_thr:g}")
    if imp_hi is None or imp_lo is None:
        return None, {"error": "one hemisphere too small", "n_hi": int(mask_hi.sum()), "n_lo": int((~mask_hi).sum())}
    rho, _ = spearmanr(imp_hi.values, imp_lo.values)
    top_k = min(15, len(feature_names))
    top_hi, top_lo = set(imp_hi.nlargest(top_k).index), set(imp_lo.nlargest(top_k).index)
    jaccard = len(top_hi & top_lo) / len(top_hi | top_lo)
    print(f"    [{setup_label}] spatial cross-check — Spearman={rho:.3f}  Jaccard(top{top_k})={jaccard:.3f}")
    return (pd.DataFrame({"feature": feature_names, "importance_hi_lat": imp_hi.values, "importance_lo_lat": imp_lo.values}),
            {"spearman_hi_lo": float(rho), "jaccard_top15_hi_lo": float(jaccard),
             "n_hi_lat": int(mask_hi.sum()), "n_lo_lat": int((~mask_hi).sum())})


def compute_missingness_informativeness(x_tr, y_tr_enc, setup_label="setup"):
    """Cramer's V between (feature is NaN) and true class — tests whether
    the missingness PATTERN itself carries class information (distinct
    from the plain per-feature NaN rate). The V statistic is diagnostic only,
    but the missingness itself IS used by the models: HistGBC/CatBoost route
    NaN natively and the `*_is_null` columns are features — legitimate for
    prediction (Josse et al. 2019, arXiv:1902.06931). Chi-square with tiny
    expected counts for rare classes is unreliable (Cochran 1954)."""
    rows, y_arr = [], np.asarray(y_tr_enc)
    for feat in x_tr.columns:
        is_nan = x_tr[feat].isna().to_numpy()
        if is_nan.sum() in (0, len(is_nan)):
            rows.append({"feature": feat, "pct_nan": float(is_nan.mean() * 100), "cramers_v": 0.0})
            continue
        contingency = pd.crosstab(is_nan, y_arr)
        chi2, _, _, _ = chi2_contingency(contingency, correction=False)
        n, (r, k) = contingency.to_numpy().sum(), contingency.shape
        rows.append({"feature": feat, "pct_nan": float(is_nan.mean() * 100),
                     "cramers_v": float(np.sqrt((chi2 / n) / (min(r - 1, k - 1) or 1)))})
    out = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    flagged = out[out["cramers_v"] >= 0.1]
    if len(flagged):
        print(f"  [{setup_label}] missingness informativeness — {len(flagged)} feature(s) with Cramer's V>=0.10: " +
              ", ".join(f"{r.feature}({r.cramers_v:.2f})" for r in flagged.itertuples()))
    return out


def compute_feature_sets(x_tr, y_tr_enc, seed=SEED, thresholds=FEATURE_POWER_THRESHOLDS,
                          setup_label="setup", n_splits=5, n_pairs=STABILITY_N_PAIRS,
                          shift_flagged_features=SHIFT_FLAGGED_FEATURES):
    """The five-step engine described in the Section 2 markdown cell above.
    Entirely test-set-blind: only ever called with training data."""
    all_feats = x_tr.columns.tolist()
    # Correlated features are clustered first (|Spearman rho| >= CORR_CLUSTER_THRESHOLD) and every
    # step below works on clusters, each permuted as one unit — see `grouped_permutation_importance`.
    clusters = spearman_clusters(x_tr, CORR_CLUSTER_THRESHOLD)
    units = list(clusters)
    imp_matrix, accs, _ = cpss_importance(x_tr, y_tr_enc, n_pairs=n_pairs, seed=seed,
                                          groups=[clusters[u] for u in units])
    cv_mode, bal_acc, bal_acc_std = f"CPSS({n_pairs} complementary pairs)", float(accs.mean()), float(accs.std())
    print(f"[{setup_label}] {len(all_feats)} features in {len(units)} Spearman cluster(s); CPSS importance: "
          f"{len(imp_matrix)} half-sample fits, held-out Bal-Acc={bal_acc:.4f}±{bal_acc_std:.4f}")

    unit_df = pd.DataFrame({"feature": units, "importance": imp_matrix.mean(axis=0),
                            "std": imp_matrix.std(axis=0)}).sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df = per_feature_importance(unit_df, clusters)
    feats_u, stability_freq, lcb_df = stability_select(imp_matrix, units, thresholds)
    stability_freq["cluster_members"] = [";".join(clusters[u]) for u in stability_freq["feature"]]
    feats = {tag: expand_clusters(us, clusters) for tag, us in feats_u.items()}
    feats["ALL_FEATURES"] = all_feats

    total_pos = lcb_df.loc[lcb_df["importance"] > 0, "importance"].sum()
    achieved_power = {tag: (float(lcb_df.set_index("feature").loc[feats_u[tag], "importance"].clip(lower=0).sum() / total_pos)
                             if feats_u[tag] and total_pos > 0 else 0.0) for tag in thresholds}
    for tag in list(thresholds):
        feats[f"{tag}_DECORR"] = list(feats_u[tag])  # one representative per selected cluster
        feats[f"{tag}_SHIFT_ROBUST"] = apply_shift_robust(feats[tag], shift_flagged_features)

    # ALLPOS — mirrors fermi_common.compute_selection_engine's ALLPOS at
    # the GLOBAL/setup level: every feature with positive (LCB-clipped)
    # importance, no cumulative-power cut. Discarded entirely (not
    # truncated) if it exceeds the MIN_SAMPLES_PER_FEATURE floor for this
    # setup's n_train.
    max_feats_for_n = max(2, len(x_tr) // MIN_SAMPLES_PER_FEATURE)
    sel_allpos_units = lcb_df.loc[lcb_df["importance"] > 0, "feature"].tolist()
    sel_allpos = expand_clusters(sel_allpos_units, clusters)
    if len(sel_allpos) > max_feats_for_n:
        print(f"  ⚠ [{setup_label}] ALLPOS: discarded — {len(sel_allpos)} positive-importance features "
              f"exceeds the {max_feats_for_n}-feature floor (n_train={len(x_tr)}, "
              f"{MIN_SAMPLES_PER_FEATURE} samples/feature).")
    elif len(sel_allpos) == 0:
        print(f"  ⚠ [{setup_label}] ALLPOS: no feature has positive importance — set is empty; skipped.")
    else:
        feats["ALLPOS"] = sel_allpos
        achieved_power["ALLPOS"] = 1.0
        feats["ALLPOS_DECORR"] = list(sel_allpos_units)
        feats["ALLPOS_SHIFT_ROBUST"] = apply_shift_robust(sel_allpos, shift_flagged_features)

    print(f"[{setup_label}] feature sets: " + ", ".join(f"{k}={len(v)}" for k, v in feats.items()))
    print(f"[{setup_label}]   achieved power (LCB-adjusted) — "
          + ", ".join(f"{k}%={achieved_power[k]:.3f}" for k in thresholds)
          + (f", ALLPOS%={achieved_power['ALLPOS']:.3f}" if "ALLPOS" in achieved_power else ""))

    # Sum of a tag's selection frequencies = mean number of features selected per half-sample (q).
    error_bound = {tag: stability_error_bound(float(stability_freq[f"stability_freq_{tag}"].sum()), len(units))
                   for tag in thresholds}
    print(f"[{setup_label}]   Meinshausen-Bühlmann bound on E[false selections] — "
          + ", ".join(f"{t}%={v:.1f}" for t, v in error_bound.items()))
    baseline = {"bal_acc": bal_acc, "bal_acc_std": bal_acc_std, "cv_mode": cv_mode,
                "stability_error_bound": error_bound, "n_clusters": len(units)}
    return dict(feats=feats, importance_df=importance_df, achieved_power=achieved_power, baseline=baseline,
                stability_freq=stability_freq)


def compute_feature_diagnostics(x_tr, y_tr_enc, importance_df, seed=SEED, setup_label="setup", n_splits=5):
    """Diagnostics only — none of these change which features are selected
    (the 'diag' processing part; see PROCESSING_PARTS): SHAP and RF/LogReg
    cross-checks of the HistGBC ranking, the Galactic-latitude check, and
    missingness informativeness."""
    all_feats = x_tr.columns.tolist()
    eligible, _, _, _, is_loo = determine_cv_eligibility(y_tr_enc)
    x_cv, y_cv = (x_tr.iloc[eligible], y_tr_enc[eligible]) if not is_loo else (x_tr, y_tr_enc)
    cv, cv_mode_cc, _ = make_cv_splitter(y_cv, n_splits=n_splits, seed=seed)

    print(f"[{setup_label}] SHAP cross-check ...")
    shap_df = compute_shap_cross_check(x_cv, y_cv, all_feats, cv, cv_mode_cc, seed=seed)
    rho_shap, _ = spearmanr(importance_df.set_index("feature").loc[all_feats, "importance"],
                             shap_df.set_index("feature").loc[all_feats, "mean_abs_shap"])
    print(f"    Spearman(permutation importance, mean|SHAP|) = {rho_shap:.3f}")

    print(f"[{setup_label}] Multi-model importance agreement ...")
    agree_df, agree_summary = compute_multi_model_agreement(x_cv, y_cv, cv, cv_mode_cc, all_feats, importance_df, seed=seed)
    agree_summary["spearman_shap"] = float(rho_shap)

    print(f"[{setup_label}] Spatial (Galactic latitude) cross-check ...")
    spatial_per_feature, spatial_summary = compute_spatial_agreement(x_tr, y_tr_enc, all_feats, seed=seed, setup_label=setup_label)
    missingness_df = compute_missingness_informativeness(x_tr, y_tr_enc, setup_label)
    return dict(shap_df=shap_df, agree_df=agree_df, agree_summary=agree_summary,
                spatial_per_feature=spatial_per_feature, spatial_summary=spatial_summary,
                missingness_df=missingness_df)


# %% [markdown]
# ## Section 3 — Hierarchical local-selection helpers (shared by every split)
#
# Thin wrappers around `compute_selection_engine` (fermi_common.py,
# already carries its own repeated-CV/LCB/decorrelation machinery) used by
# every node of the merged (2-level) and 3-level pipelines. Also
# split-independent — takes whatever `(x_tr_all, y_tr_enc)` it is given.

# %%
def _cached(path, compute_fn, run, label=""):
    """Load `path` if it exists. Otherwise compute and save it when `run` is
    True, else return None. Writes go through a temp file, so a run stopped
    mid-way never leaves a half-written cache: a re-run resumes at the first
    unit (setup, node, branch) whose cache is missing."""
    if path.exists():
        with open(path, "rb") as f:
            out = pickle.load(f)
        print(f"Loaded {label or path.stem} <- {path}")
        return out
    if not run:
        return None
    out = compute_fn()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    tmp.replace(path)
    print(f"Saved {label or path.stem} -> {path}")
    return out


def run_local_feature_selection(x_tr_all, y_tr_enc, label, tag_prefix, cache_path, min_train=0, compute=True):
    """One hierarchical local-selection unit (router/gate/subgate/
    nonblazar_router), cached at `cache_path`. Returns `(meta, imp_df)`, or
    `(None, None)` below `min_train` (the caller then falls back to the global
    feature sets) or when the cache is missing and `compute` is False."""
    n_train = len(y_tr_enc)
    if n_train < min_train:
        print(f"  ⚠ [{label}] train={n_train} < {min_train} — too small; falling back to global tags.")
        return None, None
    res = _cached(cache_path, lambda: compute_selection_engine(x_tr_all, None, y_tr_enc, None, label=label,
                                                               tag_prefix=tag_prefix),
                  compute, label=f"[{label}] local feature selection")
    if res is None:
        return None, None
    feats, imp_df, power, baseline = res
    meta = {"n_train": n_train, "achieved_power": power, "baseline_histgbc": baseline, "feature_sets": dict(feats)}
    return meta, imp_df


def run_branch_feature_selection(x_all_features, y_fine_str, route_super_str, superclass_list, min_train,
                                 cache_dir, compute=True):
    """One compute_selection_engine call per superclass with >=2 fine
    classes and >=min_train rows, each cached on its own
    (`branch_<sc>_cache.pkl`). Computed ONCE per superclass and shared by the
    merged (Level-2) and 3-level (Level-3) pipelines, which each filter it."""
    raw_cache, meta = {}, {}
    for sc in superclass_list:
        tr_mask = route_super_str == sc
        fine_tr, n_train_sc = y_fine_str[tr_mask], int(tr_mask.sum())
        if len(set(fine_tr)) < 2:
            print(f"  [{sc}] single fine class — passthrough, no branch selection needed.")
            continue
        if n_train_sc < min_train:
            print(f"  [{sc}] train={n_train_sc} < {min_train} — too small; falling back to global tags.")
            continue
        le_branch = LabelEncoder().fit(fine_tr)
        res = _cached(cache_dir / f"branch_{sc}_cache.pkl",
                      lambda: compute_selection_engine(x_all_features[tr_mask].reset_index(drop=True), None,
                                                       le_branch.transform(fine_tr), None,
                                                       label=sc, tag_prefix=BRANCH_TAG_PREFIX),
                      compute, label=f"[{sc}] branch feature selection ({n_train_sc} rows)")
        if res is None:
            continue
        feats, imp_df, power, baseline = res
        raw_cache[sc] = res
        meta[sc] = {"n_train": n_train_sc, "achieved_power": power, "baseline_histgbc": baseline,
                    "feature_sets": dict(feats)}
    return raw_cache, meta


# %% [markdown]
# ## Section 4 — Diagnostic plots (primary split only — see folder layout)
#
# Extended with: a merged/superclass class-distribution plot
# (catalog-level, called once at the end of this section); a per-node
# engineered-feature percentile heatmap; a Jaccard-AUC/RBO cross-split
# multi-model-agreement aggregator (metrics + plots); and an
# optional-features overlap plot. All of these read data this file already
# computes/exports — none of them fits a new model.

# %%
def plot_feature_importance(imp_df, label, plots_dir, top_n=30):
    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(top))))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="steelblue")
    ax.set_xlabel("Repeated-CV permutation importance (balanced accuracy, mean±std)")
    ax.set_title(f"Top {top_n} feature importances — setup {label}")
    plt.tight_layout(); fig.savefig(plots_dir / f"feature_importance_{label}.png", dpi=150); plt.close(fig)


def plot_cumulative_power(imp_df, label, plots_dir, thresholds=FEATURE_POWER_THRESHOLDS):
    pos = imp_df[imp_df["importance"] > 0].sort_values("importance", ascending=False).reset_index(drop=True)
    cum_frac = pos["importance"].cumsum() / pos["importance"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(cum_frac) + 1), cum_frac.values, color="darkorange")
    for tag, frac in thresholds.items():
        ax.axhline(frac, linestyle="--", color="gray", linewidth=0.8)
        ax.text(len(cum_frac) * 0.98, frac, f"{tag}%", va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("Number of features (ranked by importance)"); ax.set_ylabel("Cumulative positive importance fraction")
    ax.set_title(f"Cumulative predictive-power retention — setup {label}"); ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout(); fig.savefig(plots_dir / f"cumulative_power_{label}.png", dpi=150); plt.close(fig)


def plot_class_distribution(y_series, label, plots_dir):
    counts = y_series.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color="teal")
    ax.set_ylabel("Count"); ax.set_title(f"Class distribution — setup {label}"); ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); fig.savefig(plots_dir / f"class_distribution_{label}.png", dpi=150); plt.close(fig)


def plot_correlation_heatmap(x_df, feature_list, label, plots_dir):
    corr = x_df[feature_list].corr()
    fig, ax = plt.subplots(figsize=(0.4 * len(feature_list) + 2, 0.4 * len(feature_list) + 2))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax, cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
    ax.set_title(f"Correlation heatmap — setup {label} (80% power set)")
    plt.tight_layout(); fig.savefig(plots_dir / f"correlation_heatmap_{label}_80.png", dpi=150); plt.close(fig)


def plot_stability_frequency(stability_freq_df, label, plots_dir, tag="90", top_n=30):
    col = f"stability_freq_{tag}"
    if col not in stability_freq_df.columns:
        return
    top = stability_freq_df.sort_values(col, ascending=False).head(top_n).iloc[::-1]
    colors = ["seagreen" if v >= STABILITY_SELECTION_FRACTION else "lightcoral" for v in top[col]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(top))))
    ax.barh(top["feature"], top[col], color=colors); ax.axvline(STABILITY_SELECTION_FRACTION, linestyle="--", color="gray")
    ax.set_xlabel(f"Fraction of {2 * STABILITY_N_PAIRS} half-sample fits selecting this feature at {tag}% power")
    ax.set_title(f"Stability-selection frequency — setup {label} ({tag}% tag)")
    plt.tight_layout(); fig.savefig(plots_dir / f"stability_frequency_{label}_{tag}.png", dpi=150); plt.close(fig)


def plot_spatial_agreement(spatial_df, label, plots_dir, top_n=25):
    if spatial_df is None:
        return
    top = spatial_df.reindex(spatial_df[["importance_hi_lat", "importance_lo_lat"]].mean(axis=1).sort_values(ascending=False).head(top_n).index)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(top["importance_hi_lat"], top["importance_lo_lat"], color="purple", alpha=0.7)
    lims = [0, max(top["importance_hi_lat"].max(), top["importance_lo_lat"].max()) * 1.1 + 1e-6]
    ax.plot(lims, lims, linestyle="--", color="gray")
    ax.set_xlabel(f"Importance — |GLAT|>{GLAT_STRATIFICATION_THRESHOLD:g}°"); ax.set_ylabel(f"Importance — |GLAT|<={GLAT_STRATIFICATION_THRESHOLD:g}°")
    ax.set_title(f"Spatial-latitude importance agreement — setup {label}")
    plt.tight_layout(); fig.savefig(plots_dir / f"spatial_agreement_{label}.png", dpi=150); plt.close(fig)


def plot_missingness(missingness_df, label, plots_dir, top_n=20):
    top = missingness_df.sort_values("cramers_v", ascending=False).head(top_n).iloc[::-1]
    top = top[top["cramers_v"] > 0]
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(top))))
    ax.barh(top["feature"], top["cramers_v"], color="darkorange")
    ax.set_xlabel("Cramer's V — (feature is NaN) vs. true class"); ax.set_title(f"Missingness informativeness — setup {label}")
    plt.tight_layout(); fig.savefig(plots_dir / f"missingness_informativeness_{label}.png", dpi=150); plt.close(fig)


def plot_covariate_shift(shift_df, plots_dir, top_n=20):
    top = shift_df.dropna(subset=["psi"]).sort_values("psi", ascending=False).head(top_n).iloc[::-1]
    if top.empty:
        return
    colors = ["crimson" if f else "steelblue" for f in top["shift_flag"]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(top))))
    ax.barh(top["feature"], top["psi"], color=colors)
    ax.axvline(PSI_FLAG_THRESHOLD, linestyle="--", color="gray", label=f"flag threshold ({PSI_FLAG_THRESHOLD})")
    ax.set_xlabel("Population Stability Index (associated vs. unassociated)"); ax.set_title(f"Top {top_n} covariate-shift features")
    ax.legend(); plt.tight_layout(); fig.savefig(plots_dir / "covariate_shift_top_features.png", dpi=150); plt.close(fig)


# %% [markdown]
# ### New plots — merged class distribution, per-node engineered-
# ### feature heatmap, cross-split multi-model agreement, optional-features overlap

# %%
def plot_super_class_distribution(y_super_series, plots_dir, order=SUPERCLASS_ORDER):
    """Catalog-level (not split-dependent) — the '7 merged physical
    classes' slide's missing companion plot."""
    counts = y_super_series.value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="darkslateblue")
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(f"Merged superclass distribution (n={int(counts.sum())} sources, {len(order)} classes)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(plots_dir / "super_class_distribution.png", dpi=150)
    plt.close(fig)


def plot_engineered_feature_node_heatmap(global_imp_df, node_importance, plots_dir,
                                          name_patterns=("HR_", "Sqrt_TS_Band", "nuFnu_Band",
                                                          "Flux_History", "Sqrt_TS_History")):
    """Percentile rank of every engineered feature (matched by name
    prefix) within EACH node's own importance ranking — reproduces the
    'globally weak, node-rescued' vs. 'dead weight' classification from
    raw per-node numbers instead of ad hoc cross-referencing. Consumes
    already-computed importance DataFrames only; fits nothing."""
    def _percentile(imp_df):
        col = "importance" if "importance" in imp_df.columns else "mean_abs_shap"
        return imp_df.set_index("feature")[col].rank(pct=True)

    nodes = {"global_B": _percentile(global_imp_df),
             **{k: _percentile(v) for k, v in node_importance.items() if v is not None}}
    eng_feats = sorted({f for f in global_imp_df["feature"] if any(f.startswith(p) for p in name_patterns)})
    if not eng_feats:
        return None
    mat = pd.DataFrame({node: s.reindex(eng_feats) for node, s in nodes.items()}).T  # nodes x features

    fig, ax = plt.subplots(figsize=(0.55 * len(eng_feats) + 3, 0.4 * len(mat.index) + 2))
    sns.heatmap(mat, cmap="RdYlGn", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "percentile rank within node (1.0 = top feature)"})
    ax.set_title("Engineered-feature percentile rank, per node")
    plt.tight_layout()
    fig.savefig(plots_dir / "engineered_feature_node_heatmap.png", dpi=150)
    plt.close(fig)

    global_row = mat.loc["global_B"]
    other_max = mat.drop(index="global_B").max(axis=0) if len(mat.index) > 1 else global_row
    dead_weight = mat.columns[(mat < 0.60).all(axis=0)].tolist()
    rescued = mat.columns[(global_row < other_max - 0.20) & (other_max >= 0.85)].tolist()
    print(f"Engineered features — never clears 0.60 percentile in any node: {dead_weight}")
    print(f"Engineered features — node-rescued (global<node-0.20 AND node>=0.85): {rescued}")
    return mat


def jaccard_auc(order_a, order_b):
    """Mean Jaccard(top_k(A), top_k(B)) over EVERY k=1..n — area under
    the curve, replacing one arbitrary Jaccard@15 cutoff."""
    a, b, n = list(order_a), list(order_b), len(order_a)
    sa, sb, vals = set(), set(), []
    for d in range(1, n + 1):
        sa.add(a[d - 1]); sb.add(b[d - 1])
        vals.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(vals))


def rank_biased_overlap(order_a, order_b, p=0.90):
    """RBO for two COMPLETE, equal-membership rankings (same feature set,
    different order — the case for every ranker here), so the simple
    closed form applies with no Webber-et-al. extrapolation machinery
    needed: RBO(p) = (1-p) * sum_{d=1}^{n} p^(d-1) * |top_d(A) ∩ top_d(B)| / d."""
    a, b = list(order_a), list(order_b)
    assert set(a) == set(b), "rank_biased_overlap assumes identical feature membership"
    n = len(a)
    running_a, running_b, acc = set(), set(), 0.0
    for d in range(1, n + 1):
        running_a.add(a[d - 1]); running_b.add(b[d - 1])
        acc += (p ** (d - 1)) * (len(running_a & running_b) / d)
    return float((1 - p) * acc)


def compute_agreement_metrics_from_split(diag_dir, setup_key="B", top_k_report=15):
    """Reads this split's already-exported per-feature importance CSVs
    (no refit) and derives Spearman + Jaccard-AUC + RBO(p=0.90) for
    RF/LogReg/SHAP vs. HistGBC — none of which compute_multi_model_
    agreement/compute_shap_cross_check persist today (only the raw
    per-feature importance columns, plus a single Jaccard@15 that was
    printed to console and never saved)."""
    agree_df = pd.read_csv(diag_dir / f"multi_model_agreement_{setup_key}.csv")
    shap_df = pd.read_csv(diag_dir / f"shap_importance_{setup_key}.csv")
    merged = agree_df.merge(shap_df, on="feature", how="inner")
    orders = {
        "HistGBC": merged.sort_values("imp_histgbc", ascending=False)["feature"].tolist(),
        "RF":      merged.sort_values("imp_rf", ascending=False)["feature"].tolist(),
        "LogReg":  merged.sort_values("imp_logreg", ascending=False)["feature"].tolist(),
        "SHAP":    merged.sort_values("mean_abs_shap", ascending=False)["feature"].tolist(),
    }
    ref = orders["HistGBC"]
    competitor_col = {"RF": "imp_rf", "LogReg": "imp_logreg", "SHAP": "mean_abs_shap"}
    rows = []
    for comp, col in competitor_col.items():
        rho, _ = spearmanr(merged["imp_histgbc"], merged[col])
        top_a, top_b = set(ref[:top_k_report]), set(orders[comp][:top_k_report])
        rows.append({"vs_histgbc": comp, "spearman": float(rho),
                      "jaccard_auc": jaccard_auc(ref, orders[comp]),
                      "jaccard_at_k": len(top_a & top_b) / len(top_a | top_b),
                      "rbo_p90": rank_biased_overlap(ref, orders[comp], p=0.90)})
    return pd.DataFrame(rows), orders


def plot_multi_model_agreement(agg_df, plots_dir, setup_key="B"):
    metrics = ["spearman", "jaccard_auc", "rbo_p90"]
    labels = {"spearman": "Spearman ρ", "jaccard_auc": "Jaccard-AUC", "rbo_p90": "RBO (p=0.90)"}
    competitors = agg_df["vs_histgbc"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / len(competitors)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, comp in enumerate(competitors):
        row = agg_df[agg_df["vs_histgbc"] == comp].iloc[0]
        means = [row[f"{m}_mean"] for m in metrics]
        stds = [row[f"{m}_std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=comp)
    ax.set_xticks(x + width * (len(competitors) - 1) / 2)
    ax.set_xticklabels([labels[m] for m in metrics])
    ax.set_ylabel("Agreement with HistGBC ranking")
    ax.set_title(f"Multi-model agreement — Setup {setup_key}, mean±std across splits")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / f"multi_model_agreement_{setup_key}.png", dpi=150)
    plt.close(fig)


def plot_jaccard_curve(orders_by_split, plots_dir, setup_key="B"):
    """Jaccard(top_k HistGBC, top_k competitor) vs. k — the picture
    behind 'Jaccard@15 Alone Is the Wrong Number': bold=split 0,
    faint=splits 1..N-1, showing the curve spans ~0->1 within one split."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"RF": "tab:blue", "LogReg": "tab:orange", "SHAP": "tab:green"}
    for split_idx, orders in orders_by_split.items():
        ref, n = orders["HistGBC"], len(orders["HistGBC"])
        for comp in ("RF", "LogReg", "SHAP"):
            sa, sb, vals = set(), set(), []
            for d in range(1, n + 1):
                sa.add(ref[d - 1]); sb.add(orders[comp][d - 1])
                vals.append(len(sa & sb) / len(sa | sb))
            ax.plot(range(1, n + 1), vals, color=colors[comp],
                     alpha=1.0 if split_idx == 0 else 0.25,
                     linewidth=2.0 if split_idx == 0 else 0.8,
                     label=comp if split_idx == 0 else None)
    ax.set_xlabel("k (top-k cutoff)"); ax.set_ylabel("Jaccard(top-k HistGBC, top-k competitor)")
    ax.set_title(f"Jaccard-@k curve — Setup {setup_key} (bold=split 0, faint=confirmation splits)")
    ax.legend(); ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / f"jaccard_vs_k_{setup_key}.png", dpi=150)
    plt.close(fig)


def aggregate_multi_model_agreement(split_summaries, base_output_dir, plots_dir, setup_key="B",
                                     top_k_report=15, rbo_p=0.90):
    """Post-hoc ONLY — reads each split's already-exported diagnostics
    CSVs (written unconditionally by process_split, for every split, not
    just make_plots=True ones). Fits nothing; safe to call any number of
    times, on any subset of splits that already ran."""
    per_split_rows, orders_by_split = [], {}
    for s in split_summaries:
        diag_dir = Path(s["split_dir"]) / "diagnostics"
        f_path = diag_dir / f"multi_model_agreement_{setup_key}.csv"
        if not f_path.exists():
            continue  # e.g. Setup A, which is only exported for split 0
        metrics_df, orders = compute_agreement_metrics_from_split(diag_dir, setup_key=setup_key, top_k_report=top_k_report)
        metrics_df.insert(0, "split_idx", s["split_idx"])
        per_split_rows.append(metrics_df)
        orders_by_split[s["split_idx"]] = orders
    if not per_split_rows:
        print(f"  ⚠ aggregate_multi_model_agreement: no per-split diagnostics found for Setup {setup_key}.")
        return None, None
    all_df = pd.concat(per_split_rows, ignore_index=True)
    all_df.to_csv(base_output_dir / f"multi_model_agreement_per_split_{setup_key}.csv", index=False)

    agg = all_df.groupby("vs_histgbc")[["spearman", "jaccard_auc", "rbo_p90"]].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(base_output_dir / f"multi_model_agreement_aggregate_{setup_key}.csv", index=False)

    print(f"\nMulti-model agreement — Setup {setup_key}, mean±std across {all_df['split_idx'].nunique()} split(s)")
    for r in agg.itertuples():
        print(f"  vs.HistGBC={r.vs_histgbc:<7} Spearman={r.spearman_mean:.3f}±{r.spearman_std:.3f}  "
              f"Jaccard-AUC={r.jaccard_auc_mean:.3f}±{r.jaccard_auc_std:.3f}  "
              f"RBO(p={rbo_p:g})={r.rbo_p90_mean:.3f}±{r.rbo_p90_std:.3f}")

    if all_df["split_idx"].nunique() > 1:
        rf_lr = all_df[all_df["vs_histgbc"].isin(["RF", "LogReg"])].pivot(index="split_idx", columns="vs_histgbc", values="spearman")
        rf_lr["leader"] = np.where(rf_lr["RF"] > rf_lr["LogReg"], "RF", "LogReg")
        n_flips = int((rf_lr["leader"] != rf_lr["leader"].shift()).sum() - 1)
        print(f"  RF-vs-LogReg Spearman leadership flips across splits: {n_flips}/{len(rf_lr) - 1} transitions")
        rf_lr.to_csv(base_output_dir / f"multi_model_agreement_leadership_flips_{setup_key}.csv")

    plot_multi_model_agreement(agg, plots_dir, setup_key=setup_key)
    plot_jaccard_curve(orders_by_split, plots_dir, setup_key=setup_key)
    return all_df, agg


def plot_optional_features_overlap(overlap_summary_df, plots_dir):
    tags = [c.replace("jaccard_", "") for c in overlap_summary_df.columns if c.startswith("jaccard_")]
    x = np.arange(len(tags))
    width = 0.8 / max(len(overlap_summary_df), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, row in overlap_summary_df.reset_index(drop=True).iterrows():
        vals = [row[f"jaccard_{t}"] for t in tags]
        ax.bar(x + i * width, vals, width, label=f"Setup {row['setup']}")
    ax.set_xticks(x + width * (len(overlap_summary_df) - 1) / 2)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylabel("Jaccard(no-optional selection, with-optional selection)")
    ax.set_title("Physics-purged vs. ML-driven feature selection agreement")
    ax.axhline(1.0, linestyle=":", color="gray", linewidth=0.7)
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "optional_features_overlap.png", dpi=150)
    plt.close(fig)


# --- Merged (superclass) class distribution — catalog-level, once ---------
# Placed here (end of Section 4) rather than in Section 1, purely so the
# function definition and its single call sit together; it only needs
# y_all_s, which Section 1 already computed and which is still in scope.
CATALOG_PLOTS_DIR = BASE_OUTPUT_DIR / "plots_catalog"
CATALOG_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
plot_super_class_distribution(y_all_s, CATALOG_PLOTS_DIR)
print(f"Merged superclass distribution -> {CATALOG_PLOTS_DIR / 'super_class_distribution.png'}")


# %% [markdown]
# ## Section 5 — `process_split`: everything that depends on ONE train/test draw
#
# This is the function the multi-split loop (Section 6) calls once per
# split. It builds Setup A/B's stratified splits, derives the hierarchical
# (merged/3-level) row targets from Setup B, runs the full feature-
# selection engine (global + every hierarchical node), writes every
# diagnostic/dataset/cache file into `split_dir`, and — only when
# `make_plots=True` (the primary split) — regenerates the diagnostic plots.

# %%
def stratified_split_dropping_rare(x, y, seed, test_size=0.30, label_name="label"):
    """Stratified split that drops only classes unstratifiable for THIS
    target (count < 2). Returns the split PLUS the dropped-class
    accounting."""
    counts = y.value_counts()
    rare = counts[counts < 2].index.tolist()
    accounting = [{"class": c, "n_total": int(counts[c]), "reason": "fewer_than_2_samples_cannot_stratify"} for c in rare]
    if rare:
        print(f"  [{label_name}] classes with < 2 samples excluded: {rare}")
        keep = ~y.isin(rare)
        x, y = x[keep], y[keep]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)
    return x_tr, x_te, y_tr, y_te, accounting


def safe_encode(le, y_series, label_name="label"):
    known = set(le.classes_)
    mask = y_series.isin(known)
    accounting = []
    if not mask.all():
        unseen = y_series[~mask].unique().tolist()
        print(f"  [{label_name}] dropping {int((~mask).sum())} test rows with unseen labels: {unseen}")
        accounting = [{"class": c, "n_total": int((y_series == c).sum()), "reason": "unseen_in_train_split"} for c in unseen]
    return le.transform(y_series[mask]), mask, accounting


def process_split(split_idx, split_seed, split_dir, make_plots):
    """Runs Setup A/B splitting + global feature selection + hierarchical
    (merged/3-level) local feature selection for ONE split, and exports
    every dataset/diagnostic/cache file under `split_dir`. Returns a small
    summary dict for the top-level multi_split_manifest.json.

    Every expensive step below (compute_feature_sets, run_local_feature_
    selection, run_branch_feature_selection) is pickle-cached under
    `split_dir/cache/`: re-calling process_split for a split that already
    ran, with the same split_dir, hits those caches and does NOT refit
    anything — only cheap CSV/plot regeneration happens."""
    datasets_dir, encoders_dir, diag_dir, cache_dir = (split_dir / d for d in ("datasets", "encoders", "diagnostics", "cache"))
    plots_dir = split_dir / "plots"
    for d in (datasets_dir, encoders_dir, diag_dir, cache_dir, *((plots_dir,) if make_plots else ())):
        d.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\nSPLIT {split_idx} (seed={split_seed})\n{'=' * 70}")

    # --- Setup A (fine, +bcu) and Setup B (fine, -bcu) stratified splits ---
    label_accounting = list(merge_accounting)  # carry the global merge/drop log into every split's report
    splits = {}
    for key, (x_src, y_src) in {"A": (x_raw, y_raw), "B": (x_all, y_all)}.items():
        x_tr, x_te, y_tr, y_te, acc = stratified_split_dropping_rare(x_src, y_src, split_seed, label_name=f"setup_{key}")
        for a in acc:
            label_accounting.append({**a, "setup": key})
        le = LabelEncoder().fit(y_tr)
        y_tr_enc = le.transform(y_tr)
        y_te_enc, mask_te, acc2 = safe_encode(le, y_te, f"setup_{key}")
        for a in acc2:
            label_accounting.append({**a, "setup": key})
        # x_te and y_te are always filtered together (mask_te is virtually
        # always all-True — a class surviving the pre-split rare-class filter
        # is, by construction of a stratified split, present in both train
        # and test — but filtering only one would mismatch lengths).
        x_te, y_te = x_te[mask_te], y_te[mask_te]
        if key == "A":
            assert "bcu" in le.classes_, "Setup A must retain 'bcu'"
        joblib.dump(le, encoders_dir / f"label_encoder_fine_{key}.joblib")
        splits[key] = dict(x_tr=x_tr, x_te=x_te, y_tr=y_tr, y_te=y_te, y_tr_enc=y_tr_enc, y_te_enc=y_te_enc, le=le)
        print(f"Setup {key} split — train: {x_tr.shape[0]:>5}  test: {x_te.shape[0]:>5}  classes: {len(le.classes_)}")

    pd.DataFrame(label_accounting).to_csv(diag_dir / "label_accounting.csv", index=False)

    # Setup A only feeds the primary split's A-vs-B comparison, and only when fine runs Setup A.
    SETUP_KEYS_THIS_SPLIT = ("A", "B") if (split_idx == 0 and PROCESS_SETUP_A) else ("B",)

    # --- Hierarchical (merged/3-level) row targets, derived from Setup B ---
    x_b_all = splits["B"]["x_tr"].reset_index(drop=True)
    y_fine_str = splits["B"]["y_tr"].to_numpy()
    y_super_str = np.array([TAXONOMY[c] for c in y_fine_str])
    y_gate_str = np.where(np.isin(y_super_str, list(BLAZAR_SUPERCLASSES)), "BLAZAR", "OTHER")

    # --- [part: core] Global feature selection, one cache per setup -------
    global_results = {}
    for k in SETUP_KEYS_THIS_SPLIT:
        res = _cached(cache_dir / f"feature_selection_{k}.pkl",
                      lambda k=k: compute_feature_sets(splits[k]["x_tr"], splits[k]["y_tr_enc"], seed=split_seed,
                                                       setup_label=f"{k}_split{split_idx}"),
                      "core" in PROCESSING_PARTS, label=f"[{k}] global feature selection")
        if res is not None:
            global_results[k] = res
    if "B" not in global_results:
        raise RuntimeError(f"Split {split_idx}: no Setup B feature selection in {cache_dir} — run the 'core' "
                           "processing part (step 'processing') first.")
    for k, res in global_results.items():
        res["importance_df"].to_csv(diag_dir / f"feature_importance_{k}.csv", index=False)
        res["stability_freq"].to_csv(diag_dir / f"stability_frequency_{k}.csv", index=False)

    # --- [part: diag] Diagnostics (never change the selected features) ----
    diagnostics = {}
    for k in global_results:
        d = _cached(cache_dir / f"diagnostics_{k}.pkl",
                    lambda k=k: compute_feature_diagnostics(splits[k]["x_tr"], splits[k]["y_tr_enc"],
                                                            global_results[k]["importance_df"], seed=split_seed,
                                                            setup_label=f"{k}_split{split_idx}"),
                    "diag" in PROCESSING_PARTS, label=f"[{k}] diagnostics")
        if d is None:
            continue
        diagnostics[k] = d
        for name, df_ in (("shap_importance", d["shap_df"]), ("multi_model_agreement", d["agree_df"]),
                          ("spatial_agreement", d["spatial_per_feature"]),
                          ("missingness_informativeness", d["missingness_df"])):
            if df_ is not None:
                df_.to_csv(diag_dir / f"{name}_{k}.csv", index=False)

    # --- [part: hier] Hierarchical local feature selection, one cache per node ---
    # Written into feature_metadata.json only once EVERY node has finished (marker file):
    # a partial set would read downstream as "branch too small -> fall back to global tags".
    hier_done_flag = cache_dir / "hierarchical_selection.done"
    hier_compute = "hier" in PROCESSING_PARTS
    hier, node_importance = None, {}
    if hier_compute or hier_done_flag.exists():
        router_meta, router_imp = run_local_feature_selection(
            x_b_all, SUPER_LE.transform(y_super_str), "L1-router", ROUTER_TAG_PREFIX,
            cache_dir / "level1_router_cache.pkl", compute=hier_compute)
        branch_raw, branch_meta_all = run_branch_feature_selection(
            x_b_all, y_fine_str, y_super_str, SUPERCLASS_ORDER, BRANCH_MIN_TRAIN_FOR_SELECTION_3LEVEL, cache_dir,
            compute=hier_compute)
        gate_meta, gate_imp = run_local_feature_selection(
            x_b_all, GATE_LE.transform(y_gate_str), "L1-gate", GATE_TAG_PREFIX,
            cache_dir / "level1_gate_cache.pkl", compute=hier_compute)
        blazar_mask = np.isin(y_super_str, list(BLAZAR_SUPERCLASSES))
        subgate_meta, subgate_imp = run_local_feature_selection(
            x_b_all[blazar_mask].reset_index(drop=True), BLAZAR_SUBGATE_LE.transform(y_super_str[blazar_mask]),
            "L2-blazar_subgate", SUBGATE_TAG_PREFIX, cache_dir / "level2_subgate_cache.pkl",
            min_train=BRANCH_MIN_TRAIN_FOR_SELECTION_3LEVEL, compute=hier_compute)
        nonblazar_router_meta, nonblazar_imp = run_local_feature_selection(
            x_b_all[~blazar_mask].reset_index(drop=True), NONBLAZAR_ROUTER_LE.transform(y_super_str[~blazar_mask]),
            "L2-nonblazar_router", NONBLAZAR_TAG_PREFIX, cache_dir / "level2_nonblazar_router_cache.pkl",
            min_train=BRANCH_MIN_TRAIN_FOR_SELECTION_3LEVEL, compute=hier_compute)
        hier_done_flag.touch()

        merged_branch_meta = {sc: m for sc, m in branch_meta_all.items()
                              if m["n_train"] >= BRANCH_MIN_TRAIN_FOR_SELECTION_MERGED}
        threelevel_branch_meta = {sc: m for sc, m in branch_meta_all.items() if sc in NONBLAZAR_SUPERCLASS_ORDER}
        hier = {
            "merged_2level": {"level1_router": router_meta, "level2_branches": merged_branch_meta,
                              "min_train_for_branch_selection": BRANCH_MIN_TRAIN_FOR_SELECTION_MERGED},
            "3level": {"level1_gate": gate_meta, "level2_blazar_subgate": subgate_meta,
                       "level2_nonblazar_router": nonblazar_router_meta, "level3_branches": threelevel_branch_meta,
                       "min_train_for_local_selection": BRANCH_MIN_TRAIN_FOR_SELECTION_3LEVEL},
        }
        # Node-level importance export (feeds the per-node rescue heatmap) — cached data, no refit.
        node_importance = {"router": router_imp, "gate": gate_imp,
                           "blazar_subgate": subgate_imp, "nonblazar_router": nonblazar_imp,
                           **{f"branch_{sc}": branch_raw[sc][1] for sc in threelevel_branch_meta}}
        for node_name, node_imp_df in node_importance.items():
            if node_imp_df is not None:
                node_imp_df.to_csv(diag_dir / f"node_importance_{node_name}.csv", index=False)

    # --- Plots (primary split only) ---------------------------------------
    if make_plots:
        plot_covariate_shift(COVARIATE_SHIFT_DF, plots_dir)
        for k, res in global_results.items():
            plot_feature_importance(res["importance_df"], k, plots_dir)
            plot_cumulative_power(res["importance_df"], k, plots_dir)
            plot_class_distribution(splits[k]["y_tr"], k, plots_dir)
            plot_correlation_heatmap(splits[k]["x_tr"], res["feats"]["80"], k, plots_dir)
            plot_stability_frequency(res["stability_freq"], k, plots_dir)
            if k in diagnostics:
                plot_spatial_agreement(diagnostics[k]["spatial_per_feature"], k, plots_dir)
                plot_missingness(diagnostics[k]["missingness_df"], k, plots_dir)
        if hier is not None:
            plot_engineered_feature_node_heatmap(global_results["B"]["importance_df"], node_importance, plots_dir)

        print("\n--- Feature-selection results digest (Setup B, primary split) ---")
        res_b = global_results["B"]
        print(f"  CV balanced accuracy (HistGBC, ALL_FEATURES): {res_b['baseline']['bal_acc']:.4f} ± "
              f"{res_b['baseline']['bal_acc_std']:.4f}  (cv_mode={res_b['baseline']['cv_mode']})")
        for tag in FEATURE_POWER_THRESHOLDS:
            print(f"  {tag}% power tag: {len(res_b['feats'][tag])} features "
                  f"(decorr: {len(res_b['feats'][f'{tag}_DECORR'])}, "
                  f"shift-robust: {len(res_b['feats'][f'{tag}_SHIFT_ROBUST'])}) "
                  f"— achieved power {res_b['achieved_power'][tag]:.3f}")
        print(f"  ALLPOS: {len(res_b['feats']['ALLPOS'])} features" if "ALLPOS" in res_b["feats"]
              else "  ALLPOS: discarded (exceeds samples-per-feature floor)")
        print(f"  Covariate shift: {len(SHIFT_FLAGGED_FEATURES)}/{len(x_all.columns)} features flagged")

    # --- Scaling parameters (diagnostic only — models use native/pipeline scaling downstream) ---
    def _scaling_params(x_train, feature_list):
        sub = x_train[feature_list]
        q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
        return {"mean": sub.mean(skipna=True).to_dict(), "std": sub.std(skipna=True).replace(0, 1.0).to_dict(),
                "median": sub.median(skipna=True).to_dict(), "iqr": (q3 - q1).replace(0, 1.0).to_dict()}

    scaling_params = {k: {tag: _scaling_params(splits[k]["x_tr"], feats) for tag, feats in global_results[k]["feats"].items()}
                       for k in global_results}
    with open(diag_dir / "scaling_parameters.json", "w") as f:
        json.dump(scaling_params, f, indent=2)

    # --- Export per-tag train/test datasets (Setup A & B only — the
    # hierarchical selections are row-subsets of Setup B, consumed by
    # Modules 2/3 as feature-column lists against Setup B's own export) ---
    for k in global_results:
        s, feats = splits[k], global_results[k]["feats"]
        tagset = tag_raw if k == "A" else tag_all
        for tag, flist in feats.items():
            for split_name, x_df, y_series, y_enc in ((("train", s["x_tr"], s["y_tr"], s["y_tr_enc"])),
                                                        (("test", s["x_te"], s["y_te"], s["y_te_enc"]))):
                out = x_df[flist].copy()
                out.insert(0, "source_name", tagset.loc[x_df.index].values)
                out["label"], out["label_encoded"] = y_series.values, y_enc
                out.to_csv(datasets_dir / f"{k}_{split_name}_{tag}.csv", index=False)
        print(f"[{k}] datasets written for feature sets: {list(feats)}")

    # --- feature_metadata.json — single source of truth for this split ---
    metadata = {
        "split_idx": split_idx, "split_seed": split_seed, "version_tag": VERSION_TAG,
        "stability_selection": {"method": "CPSS (Shah & Samworth 2013)", "n_complementary_pairs": STABILITY_N_PAIRS,
                                 "stability_frac_threshold": STABILITY_SELECTION_FRACTION,
                                 "importance_lcb_z": IMPORTANCE_LCB_Z, "corr_cluster_threshold": CORR_CLUSTER_THRESHOLD},
        "covariate_shift": {"alpha": COVARIATE_SHIFT_ALPHA, "psi_flag_threshold": PSI_FLAG_THRESHOLD,
                             "n_flagged": len(SHIFT_FLAGGED_FEATURES), "flagged_features": SHIFT_FLAGGED_FEATURES},
        "spatial_bias": {"glat_threshold_deg": GLAT_STRATIFICATION_THRESHOLD},
        "physical_merge": {"applied": APPLY_PHYSICAL_MERGE, "merge_map": MERGE_FINE_CLASSES},
        "setups": {k: {"n_classes": len(splits[k]["le"].classes_), "classes": splits[k]["le"].classes_.tolist(),
                        "n_train": int(splits[k]["x_tr"].shape[0]), "n_test": int(splits[k]["x_te"].shape[0]),
                        "baseline_histgbc": global_results[k]["baseline"],
                        "feature_sets": {tag: {"n_features": len(v), "achieved_power": global_results[k]["achieved_power"].get(tag)}
                                          for tag, v in global_results[k]["feats"].items()}}
                   for k in global_results},
        # None until the 'hier' processing part has finished every node (merged/3-level need it).
        "hierarchical_feature_selection": hier,
        "processing_parts_available": {"core": True, "diag": sorted(diagnostics), "hier": hier is not None},
    }
    with open(diag_dir / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Split {split_idx} complete. Diagnostics -> {diag_dir}   Datasets -> {datasets_dir}")
    return {"split_idx": split_idx, "split_seed": split_seed, "split_dir": str(split_dir),
            "n_train_A": int(splits["A"]["x_tr"].shape[0]), "n_train_B": int(splits["B"]["x_tr"].shape[0]),
            "n_test_A": int(splits["A"]["x_te"].shape[0]), "n_test_B": int(splits["B"]["x_te"].shape[0])}


# %% [markdown]
# ## Section 6 — Driver: run one split, or `N_VALIDATION_SPLITS` of them
#
# Split 0 (seed = `SEED`, folder `split_00_primary`) is always built and is
# the split Modules 2/3 should run their full model/hyperparameter search
# against. When `ENABLE_MULTI_SPLIT_VALIDATION=True`, splits 1..N-1 (seeds
# `SEED+1` .. `SEED+N-1`) are additionally built for Modules 2/3 to
# re-confirm their split-0-selected shortlist on. The splits are
# independent draws with feature selection re-run fresh on each, not just
# a model refit on frozen features.
#
# **Re-running this loop after a prior successful run does NOT refit
# anything**: every split's `process_split` call hits that split's own
# `cache/*.pkl` files (feature_selection_cache.pkl, level1/2/3 caches) and
# loads instead of recomputing — see the "(g) cache staleness" design-
# decision cell. If you only need split 0's outputs refreshed (e.g. to
# pick up the new plots/exports below), you can replace this loop with a
# single call:
# `process_split(split_idx=0, split_seed=SEED, split_dir=BASE_OUTPUT_DIR / "split_00_primary", make_plots=True)`
# — still cache-backed, still no refit, just skips iterating splits 1..N-1.

# %%
n_splits_to_run = N_VALIDATION_SPLITS if ENABLE_MULTI_SPLIT_VALIDATION else 1
split_summaries = []
for i in range(n_splits_to_run):
    split_tag = "split_00_primary" if i == 0 else f"split_{i:02d}"
    summary = process_split(split_idx=i, split_seed=SEED + i, split_dir=BASE_OUTPUT_DIR / split_tag, make_plots=(i == 0))
    split_summaries.append(summary)

manifest = {
    "version_tag": VERSION_TAG, "multi_split_validation": ENABLE_MULTI_SPLIT_VALIDATION,
    "n_splits": n_splits_to_run, "primary_split": "split_00_primary",
    "confirmation_splits": [s["split_dir"] for s in split_summaries[1:]], "splits": split_summaries,
}
with open(BASE_OUTPUT_DIR / "multi_split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)


# %% [markdown]
# ## Section 6c — Multi-model agreement, aggregated across splits
#
# Purely post-hoc: reads each split's already-exported `multi_model_
# agreement_{B}.csv` / `shap_importance_{B}.csv` (every split wrote these
# unconditionally in Section 5, not only the primary one) and derives
# Spearman/Jaccard-AUC/RBO(p=0.90) — nothing here re-fits HistGBC, RF,
# LogReg, or re-runs SHAP. Safe to re-run on its own, any number of times,
# as long as `split_summaries` (or `multi_split_manifest.json`) and each
# split's `diagnostics/` folder are on disk. If your kernel restarted and
# `split_summaries` isn't in memory, rebuild it first:
# ```python
# with open(BASE_OUTPUT_DIR / "multi_split_manifest.json") as f:
#     split_summaries = json.load(f)["splits"]
# ```

# %%
PRIMARY_PLOTS_DIR = BASE_OUTPUT_DIR / "split_00_primary" / "plots"
aggregate_multi_model_agreement(split_summaries, BASE_OUTPUT_DIR, PRIMARY_PLOTS_DIR, setup_key="B")

_split0_diag = Path(split_summaries[0]["split_dir"]) / "diagnostics"
if (_split0_diag / "multi_model_agreement_A.csv").exists():
    metrics_a, _ = compute_agreement_metrics_from_split(_split0_diag, setup_key="A")
    metrics_a.to_csv(BASE_OUTPUT_DIR / "multi_model_agreement_setup_A_split0.csv", index=False)
    print("\nMulti-model agreement — Setup A (single split, split 0):")
    for r in metrics_a.itertuples():
        print(f"  vs.HistGBC={r.vs_histgbc:<7} Spearman={r.spearman:.3f}  "
              f"Jaccard-AUC={r.jaccard_auc:.3f}  RBO(p=0.90)={r.rbo_p90:.3f}")


# %% [markdown]
# ## Section 6b — Optional-features comparison (Setup A/B fine selection only)
#
# A scientific check, not an engineering one: does the manual, physics-
# motivated column purge behind `DATA_PATH` agree with what a purely
# data-driven importance ranking would select if the purged (optional /
# redundant) columns were put back? Loads the `with_optional_features`
# catalog through the same merge/taxonomy/filter logic as Section 1
# (factored out into `load_catalog` below so it isn't duplicated), reruns
# ONLY the flat Setup A/B fine feature-selection engine against it on the
# SAME split seed as the primary split, and diffs the resulting feature
# sets against split 0's already-cached selection — per-threshold Jaccard
# overlap plus the explicit "kept only because of the purge" /
# "kept only once optional columns are restored" set differences.
#
# This block previously had NO cache guard and refit Setup A/B
# from scratch (`compute_feature_sets`, the full repeated-CV/SHAP/multi-
# model engine) on every single re-run, unlike every other expensive step
# in this file. It now follows the same pickle-cache pattern as
# `process_split`: `cache/optional_features_results.pkl`. Delete that one
# file (not the whole `optional_features_comparison/` folder — the CSVs
# and plot are regenerated from the cache either way) to force a clean
# re-run of just this block.

# %%
def load_catalog(data_path):
    """Section 1's loading + physical-merge + taxonomy-mapping logic,
    factored out so it can run against either catalog variant. Returns
    (x_raw, y_raw) — Setup A's base (bcu retained, taxonomy-unfiltered) —
    and (x_all, y_all) — Setup B's base (taxonomy-valid rows only, which
    is what naturally excludes bcu: TAXONOMY has no 'bcu' entry). Does NOT
    recompute the covariate-shift/NaN diagnostics — not needed for this
    comparison's purpose, and those are catalog-column-set-specific."""
    df_ = pd.read_csv(data_path / "full_processed_dataset_with_string_labels.csv")
    x_raw_ =df_.drop(columns=ID_COLUMNS_TO_DROP + REDUNDANT_SPATIAL_COLUMNS, errors="ignore").copy()
    y_raw_ = df_["CLASS1"].copy()
    if APPLY_PHYSICAL_MERGE:
        y_raw_ = y_raw_.replace(MERGE_FINE_CLASSES)
    mask_keep_ = ~y_raw_.isin(CLASSES_TO_DROP)
    x_raw_, y_raw_ = x_raw_[mask_keep_].copy(), y_raw_[mask_keep_].copy()
    valid_ = y_raw_.map(TAXONOMY).notna()
    return x_raw_, y_raw_, x_raw_[valid_].copy(), y_raw_[valid_].copy()


def summarize_feature_set_overlap(feats_primary, feats_variant, thresholds=FEATURE_POWER_THRESHOLDS):
    """Per-threshold Jaccard overlap + explicit set difference between two
    compute_feature_sets() outputs — the actual comparison artifact."""
    rows = []
    for tag in list(thresholds) + [f"{t}_DECORR" for t in thresholds]:
        a, b = set(feats_primary.get(tag, [])), set(feats_variant.get(tag, []))
        rows.append({"tag": tag, "n_primary_no_optional": len(a), "n_with_optional": len(b),
                      "jaccard": (len(a & b) / len(a | b)) if (a or b) else float("nan"),
                      "only_in_no_optional_purge": sorted(a - b), "only_in_with_optional": sorted(b - a)})
    return pd.DataFrame(rows)


if "optional" in PROCESSING_PARTS:
    print(f"\n{'=' * 70}\nOPTIONAL-FEATURES COMPARISON (Setup A/B, single split)\n{'=' * 70}")
    comp_dir = BASE_OUTPUT_DIR / "optional_features_comparison"
    comp_cache_dir = comp_dir / "cache"
    comp_plots_dir = comp_dir / "plots"
    for d in (comp_dir, comp_cache_dir, comp_plots_dir):
        d.mkdir(parents=True, exist_ok=True)
    comp_cache_path = comp_cache_dir / "optional_features_results.pkl"

    # Split 0's cached Setup A/B selection is the "no-optional" (primary,
    # physics-purged) side of the comparison — already computed above,
    # reloaded here rather than recomputed.
    global_results_primary = {}
    for key in ("A", "B"):
        _p = BASE_OUTPUT_DIR / "split_00_primary" / "cache" / f"feature_selection_{key}.pkl"
        if _p.exists():
            with open(_p, "rb") as f:
                global_results_primary[key] = pickle.load(f)

    if comp_cache_path.exists():
        with open(comp_cache_path, "rb") as f:
            overlap_summary, per_key_importance = pickle.load(f)
        print(f"Loaded optional-features comparison cache <- {comp_cache_path}")
    else:
        x_raw_opt, y_raw_opt, x_all_opt, y_all_opt = load_catalog(DATA_PATH_WITH_OPTIONAL)
        overlap_summary, per_key_importance = [], {}
        for key, (x_src, y_src) in {"A": (x_raw_opt, y_raw_opt), "B": (x_all_opt, y_all_opt)}.items():
            if key not in global_results_primary:
                continue  # Setup A not processed (RUN_FINE_SETUP_A off)
            x_tr, _, y_tr, _, _ = stratified_split_dropping_rare(x_src, y_src, SEED, label_name=f"optional_{key}")
            le = LabelEncoder().fit(y_tr)
            res_opt = compute_feature_sets(x_tr, le.transform(y_tr), seed=SEED, setup_label=f"{key}_with_optional")
            per_key_importance[key] = res_opt["importance_df"]

            overlap_df = summarize_feature_set_overlap(global_results_primary[key]["feats"], res_opt["feats"])
            overlap_df.to_csv(comp_dir / f"feature_set_overlap_{key}.csv", index=False)
            overlap_summary.append({"setup": key, **{f"jaccard_{r.tag}": r.jaccard for r in overlap_df.itertuples()}})
            print(f"[{key}] optional-features overlap -> {comp_dir / f'feature_set_overlap_{key}.csv'}")
        with open(comp_cache_path, "wb") as f:
            pickle.dump((overlap_summary, per_key_importance), f)
        print(f"Saved optional-features comparison cache -> {comp_cache_path}")

    for key, imp_df in per_key_importance.items():
        imp_df.to_csv(comp_dir / f"feature_importance_{key}_with_optional.csv", index=False)

    overlap_summary_df = pd.DataFrame(overlap_summary)
    overlap_summary_df.to_csv(comp_dir / "overlap_summary.csv", index=False)
    plot_optional_features_overlap(overlap_summary_df, comp_plots_dir)
    print(f"Optional-features comparison complete -> {comp_dir}")

# %% [markdown]
# ## Section 7 — Summary

# %%
print("=" * 70)
print(f"Module 1 complete — {n_splits_to_run} split(s) processed.")
for s in split_summaries:
    print(f"  split {s['split_idx']} (seed={s['split_seed']}): "
          f"A train/test={s['n_train_A']}/{s['n_test_A']}  B train/test={s['n_train_B']}/{s['n_test_B']}  "
          f"-> {s['split_dir']}")
print(f"Manifest -> {BASE_OUTPUT_DIR / 'multi_split_manifest.json'}")
print("=" * 70)
