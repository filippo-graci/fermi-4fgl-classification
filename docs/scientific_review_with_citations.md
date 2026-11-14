# Scientific review of the Fermi 4FGL classification pipeline, with the justification and literature for each step

Review date: 2026-10-30. Code reviewed: everything in `pipeline/` as it stands today
(`run_fermi.py`, `fermi_common.py`, `fermi_processing.py`, `fermi_fine.py`,
`fermi_merged.py`, `fermi_3levels.py`, `fermi_validation.py`, `fermi_inference.py`,
`fermi_analysis.py`) plus the upstream `preprocessing.py`.

How it was read: `run_fermi.py`, processing, the shared engine in common (weighting, CV, staged search,
early-stopping wrappers, feature-selection engine, model registry), fine, validation (all stages and
reporting sections) and inference were read in full. Merged and 3-level were read for their branch
logic, chain-CV and routing; the rest of those two files is plumbing shared with fine. Analysis was
skimmed: it only makes descriptive plots and tables.

This document does three things:

1. **Part A** justifies each step of the pipeline, in execution order, with the paper that supports it,
   and says where the implementation departs from what that paper describes. Standard models
   (random forest, gradient boosting, CatBoost, logistic regression, MLP, LDA) and standard tools
   (one-hot encoding, standard scaling) are not justified here, as requested.
2. **Part B** is the scientific review: what is still wrong or unsupported, ranked by how much it
   affects the reported numbers. It includes issues found in this pass and the status of every issue
   from an earlier review made the same day (not included in this repository).
3. **Part C** is the reference list.

## Status after the follow-up changes (same day)

| Finding | Status |
|---|---|
| B1.4 RandomForest resampled under `USE_GPU` | **Fixed**: RF always CPU with native sample weights; the dead cuML RF path was removed (`weight_support_for`, `fermi_common.py`) |
| B1.1 no uncertainty on the single test score | **Fixed**: every Stage-3 validation, Stage-5 test and train evaluation, on every split, gets a row-level 95% CI (`bootstrap_metric_ci`: class-stratified bootstrap; Brodersen et al. 2010 Beta posterior for balanced accuracy). Across splits: Nadeau–Bengio t; with one split, that split's own CI. Also: per-class recall with Clopper–Pearson per split, train std in Module-2 leaderboards, CIs on the shift strata, the Brier score and the v41 check. New CSVs: `winner_per_split_metrics_ci.csv`, `per_class_recall_ci_<fam>.csv`, `calibration_brier_per_split_ci.csv` |
| B2.3 stability selection | **Fixed**: complementary-pairs subsampling (CPSS, Shah & Samworth 2013), 25 pairs = 50 held-out half-sample fits, threshold 0.7, ranking on the mean importance; the Meinshausen–Bühlmann bound is logged per feature set |
| B3.3 two OOF definitions in `_val_score_fine_candidate` | **Fixed**: one pooled OOF (`cv_score_estimator(collect_oof=True)`) |
| B3.9 stale text, Demšar citation | **Fixed** (Borda kept by decision; citation replaced by Dwork et al. 2001 for Borda rank aggregation; overfit gap stated as a preference) |
| B3.12 provenance | **Fixed**: `pipeline/preprocessing.py` (CLI + run log) reproduces the DR4 input CSVs from `gll_psc_v35.fit` to float32 rounding |
| Found by the smoke run | **Fixed**: (1) fine leaderboard tags such as `90` were read back from CSV as integers, so Stage 1/3 crashed for any winner on a plain 99/90/80 set; (2) a stage in which every candidate failed was still checkpointed and silently reused by `--resume`; (3) Stage 5 called `_metric_ci_across_splits` before Section 14 defined it |
| B1.2 excess variance, B3.8 flags, GLON | **Fixed in the data** (`preprocessing.py` → `data/processed/`, `VERSION_TAG` 4FGL_DR4): excess variance and peak-to-median removed (corrected excess variance has Spearman ρ = 0.98 with the catalogue's `Frac_Variability`; peak-to-median is noise for faint sources), `Conf_95_*` removed (exactly 1.62146 × `Conf_68_*`), never-set flags dropped (DR4: 8, 12), `GLON_sin`/`GLON_cos` replace raw GLON as features |
| Selection metric | **Changed to macro-F1** (all four weighting modes searched). Balanced accuracy = macro recall ignores false positives; with 18 classes and uncapped weights it selected a model that sent 72 test `bll` to `sbg` and 65 `fsrq` to `nlsy1` (accuracy 0.59). Macro-F1 run (fine, single split): LogReg / 80% set / `dampened`; test accuracy 0.832 [0.810, 0.854], macro-F1 0.319 [0.287, 0.354]; calibrated ECE 0.128 → 0.043; v41 newly associated sources at calibrated confidence ≥ 0.8: 59/60 correct |
| MLP half-precision inference | **Fixed**: fp16 logits overflowed to NaN probabilities on extreme inputs (broke calibration and corrupted MLP search scores); inference now runs in fp32 |
| B2.2 correlated / duplicated features | **Fixed**: features are clustered by \|Spearman ρ\| ≥ 0.85 *before* importance; each cluster is permuted as one unit (grouped permutation importance, Gregorutti et al. 2017; Au et al. 2022) inside CPSS; selection is per cluster; plain tags keep all members of the selected clusters, `_DECORR` one representative (fewest NaN, then medoid) |
| B2.4 rare classes never validated in CV | **Fixed**: in `cv_score_estimator` only singleton classes stay training-only; a class with 2–4 rows has each row validated in a different fold (pooled OOF, Forman & Scholz 2010). CV and test now score the same class set |
| B2.8 / B1.3 uncalibrated probabilities, priors | **Fixed for the flat family**: `ProbabilityCalibrator` (undo the class-weight tilt, Elkan 2001 / Saerens et al. 2002; then temperature scaling, Guo et al. 2017), fit on out-of-fold training predictions only. Section 16b reports Brier, NLL, ECE and reliability for raw vs calibrated on the test rows; the deployed model carries its calibrator, so `confidence` in the unassociated predictions is calibrated; EM prior re-estimation (Saerens et al. 2002) is exported as a sensitivity check (`confidence_prior_adjusted`, `unassociated_class_priors_em_fine.csv`). The predicted label stays the weighted argmax. Merged/3-level probabilities are not yet calibrated |

---

## 0. What the code ran at review time

This section describes the configuration on the review date. The final run differs: it searches all
three families, selects on macro-F1 and runs every step through analysis (see `RESULTS.md`).

From `_RF_CONFIG` in `run_fermi.py` at review time:

| Setting | Value | Consequence |
|---|---|---|
| `VERSION_TAG` | `4FGL_DR4` | fresh processing folder, because dropping `spp` changed Setup B |
| `PROFILE` | `reduced` | 7 model families (+LDA on two branches), Standard scaler only, no PCA, weighting ∈ {`balanced_uncapped`, `none`}, all feature-set tags |
| `VALIDATION_DESIGN` | `single_split` | one stratified 70/30 split; search and selection see only its training part; one test score |
| `FAMILIES` | `["fine"]` | only the flat classifier is searched, validated and deployed; merged/3-level are skipped |
| `STEPS` | processing → fine → validation → inference | diagnostics, hierarchical selection, optional-features comparison and analysis are not run |
| `SELECTION_METRIC` | `bal_accuracy` | |

So the deliverable of the current configuration is **one flat 18-class classifier**, chosen on the
training part of one split, scored once on its test part, refit on all labelled data, and applied to
the 2,563 unassociated sources.

Setup B after dropping `spp` from the taxonomy (A2.2): 2,876 labelled sources in 18 classes (bll 1490, fsrq 828
incl. 8 ssrq/css, msp 179, psr 141, rdg 53, snr 44, glc 41, pwn 22, hmb 11, bin 10, lmb 9, and
nlsy1, agn, sbg, nov 8 each, gal 6, sfr 6, sey 4). With a 30% stratified test split, about 861 test
rows, and **10 of the 18 classes have 1–3 test rows**.

---

# Part A — Step-by-step justification

Legend for the verdict column:
- ✅ justified and citable as implemented
- ⚠️ justified in principle, but the implementation departs from the cited method (see Part B)
- 🟡 an engineering judgment with no specific paper; acceptable if stated as such
- ❌ not justified as implemented

## A1. Catalogue ingestion and feature engineering (`preprocessing.py`)

### A1.1 Removing association-process columns (leakage)

**What:** `CLASS2`, `ASSOC*`, `ASSOC_PROB_BAY/LR`, `ASSOC_TEV`/`TEVCAT_FLAG`, the counterpart coordinates,
and bookkeeping columns are dropped before anything else.

**Why:** these columns are produced by the same procedure that assigns `CLASS1`. A feature that
encodes how the label was made will look highly predictive and carry no information about
unassociated sources, for which the association process failed by definition. This is the textbook
definition of target leakage (Kaufman et al. 2012). Kapoor & Narayanan (2023) document it as the
most common cause of irreproducible ML results in science. Feature selection cannot catch it, because
leaky features rank at the top of every importance measure, which is why it has to happen upstream.

**Verdict:** ✅

### A1.2 Flags bitmask → binary indicators, bits 13 and 14 excluded

**What:** `FLAG_BITS_USED = 1..12 minus {7, 11}` (`preprocessing.py:273`), one 0/1 column per bit.

**Why:** the flags describe analysis caveats (confusion, background dominance, localization
problems), which really do differ between populations. Decoding the bitmask avoids treating an
integer code as an ordinal number.

The exclusion of bit 14 is correct, but for a reason the code does not give. In 4FGL-DR3, **Flag 14
marks "Soft Galactic Unassociated" sources, and it is set only for sources classed SGU, SPP or UNK**
(Abdollahi et al. 2022, Table 4 and §6.2). It depends on association status, so it is leakage under
A1.1. Flag 13 means "TS < 25 at 12 yr" (same table), which is a legitimate property of the source and
could be kept. Bit 8 is unused in DR3 but kept, so it is a constant column.

**Verdict:** ✅ for the result; fix the comment (see B3.8).

### A1.3 log10 of flux-like quantities, with a clip floor

**What:** fluxes, energy flux, `Npred`, `Variability_Index`, `Pivot_Energy`, the SED peak and the
variability statistics are log-transformed.

**Why:** these quantities span several orders of magnitude, and catalogue classifiers since 2FGL have
used them in log space (Ackermann et al. 2012; Saz Parkinson et al. 2016). The transform matters for
the scale-sensitive models (logistic regression, MLP). Trees are invariant to it.

**Verdict:** ✅

### A1.4 Hardness ratios between adjacent energy bands

**What:** `HR_i_{i+1} = (F_{i+1} − F_i)/(F_{i+1} + F_i)` from `Flux_Band`.

**Why:** this is the hardness-ratio definition of the 2FGL catalogue (Nolan et al. 2012). It was
used as a primary feature in the first LAT source-class classifiers (Ackermann et al. 2012) and in
later ones (Saz Parkinson et al. 2016). It captures spectral curvature without assuming a spectral
model, which is what separates pulsars (curved, cut-off spectra; Abdo et al. 2013) from blazars.

**Caveat:** in 4FGL, a band with TS < 1 reports an upper-limit-derived uncertainty with a NULL lower
error (Abdollahi et al. 2022, Table 6 note c). The ratio then mixes measured fluxes and limit-like
values, which matters most for faint sources (see B1.2).

**Verdict:** ✅ (with the caveat)

### A1.5 SED peak (`nuFnu_Band_peak`, `nuFnu_Band_peak_band`)

**Why:** the peak location of the high-energy SED hump is the basis of the LSP/ISP/HSP blazar
sequence and separates FSRQs (mostly LSP) from BL Lacs (often ISP/HSP) (Abdo et al. 2010b; Ajello et
al. 2020, 4LAC). A band-index argmax is a coarse proxy for that, computed from LAT data only.

**Verdict:** ✅

### A1.6 Variability: normalized excess variance and peak-to-median

**What:** `_normalized_excess_variance` (`preprocessing.py:318`) computes
`(S² − ⟨σ²_err⟩)/⟨x⟩²`, floored at 0, on the yearly (and, for DR3, 2-month) light curves.

**Why:** the normalized excess variance is the standard noise-corrected variability amplitude
(Vaughan et al. 2003). It removes the measurement variance, which naive std/mean does not, so faint
steady sources do not look variable. Variability is the main discriminator between blazars and
pulsars in LAT data (Abdollahi et al. 2020; Ackermann et al. 2012). Peak-to-median is a heuristic
flare indicator. It complements the catalogue's own `Variability_Index` and `Frac_Variability`.

**Implementation problem:** see B1.2. Bins with TS < 1 have a NULL lower error, so `mean_unc` is NaN
there and `np.nanmean` drops their measurement variance while `np.nanvar` keeps their flux. This
inflates the excess variance of faint sources, which are mostly the unassociated ones.

**Verdict:** ⚠️

### A1.7 Infinities and missing values

**What:** `±inf` sentinels are mapped to 0 (with a `*_is_null` indicator) or to NaN. No global
statistic is imputed before the split; imputation happens later, inside each pipeline.

**Why:** imputing with pre-split statistics leaks test information into training. Fitting the
imputer inside the CV loop is the standard remedy (Hastie et al. 2009, §7.10.2). Keeping NaN for the
trees lets them route missing values natively ("missing incorporated in attributes", Twala et al.
2008), which Josse et al. (2019) show is consistent for prediction when missingness is informative.

**Verdict:** ✅ (but see B3.7: the claim that missingness is never used as a signal is false for the
trees, and that is fine)

### A1.8 Label normalization: lowercase = uppercase

**What:** `CLASS1` is lowercased, so identified (uppercase) and associated (lowercase) sources form one
class.

**Why:** this is how 4FGL defines the classes: the case only encodes the strength of the evidence for
the association (Abdollahi et al. 2020). Pooling is standard practice in the classifiers above.
Identified sources could get a higher weight, as optional label-noise handling.

**Verdict:** ✅

### A1.9 Galactic coordinates kept as features, equatorial dropped

**Why:** source classes really are distributed differently on the sky: pulsars, SNRs and binaries are
concentrated near the plane, while AGN are isotropic. Saz Parkinson et al. (2016) and Bhat & Malyshev
(2022) use latitude-dependent features. RA/Dec carry the same information in another frame.

**Implementation caveats:** `GLON` is fed as a raw angle (0° and 359° look far apart to the linear
and MLP models), and signed `GLAT` is used even though the physics is symmetric about the plane. Sky
position also correlates with counterpart-survey depth, which is weaker in the plane. Latitude can
therefore act as a proxy for "was this source associated" (see B2.6).

**Verdict:** ⚠️

### A1.10 Provenance

At review time, the committed preprocessing script read the DR3 file (`gll_psc_v31.fit`) and wrote to
`DR3_data`. The DR4 CSVs the pipeline used (`*/full_processed_dataset_with_string_labels.csv`) contained
`HR_6_7`, no `Flux2_History_*` columns, and keep `Conf_*` and `*_Flux_Density`. So they were produced by
a different script, which is not in the repository (B3.12).

---

## A2. Labels and taxonomy (`fermi_common.py:462`, `fermi_processing.py:317`)

### A2.1 Dropping `bcu` (Setup B)

**Why:** `bcu` means "blazar of uncertain type": the source is a blazar, but its optical spectrum did
not allow FSRQ vs BL Lac classification (Ajello et al. 2020). It records the state of follow-up, not a
physical class. Coronado-Blázquez (2022), the closest methodological precedent for this pipeline, drops
it for the same reason. Setup A keeps it as a full-catalogue baseline.

**Verdict:** ✅

### A2.2 Dropping `spp`

**Why:** `spp` is the 4FGL class for sources positionally coincident with an SNR or PWN where the
association is ambiguous (Abdollahi et al. 2020). It is therefore the same kind of label as `bcu`. It
was the largest class in `SNR_PWN` (132 rows against 66 `snr`+`pwn`), so that branch was mostly learning
"overlaps a known remnant". It is also one of the classes for which DR3 sets Flag 14 (A1.2). Dropping
it makes the treatment of ambiguous labels consistent.

**Verdict:** ✅

### A2.3 Dropping `gc` and `unk`

**Why:** `gc` is the single Galactic-centre source (n = 1, cannot be stratified, and is not a source
population). `unk` in 4FGL means "unknown-type counterpart" (an association with no class), so it has
no usable target.

**Verdict:** ✅

### A2.4 `ssrq`/`css` merged into `fsrq`, reported as "FSRQ-like"

**What:** 8 sources (2 ssrq, 6 css) are relabelled `fsrq` before any split.

**Arguments for:** these are radio-loud quasars with broad emission lines, like FSRQs, and with 8
sources a separate class cannot be learned or validated.

**Arguments against, which the write-up must state:**
- 4LAC classes CSS and SSRQ among the **non-blazar ("misaligned") AGN**, with radio galaxies (Ajello
  et al. 2020).
- In AGN unification, steep-spectrum radio quasars are the misaligned parent population of FSRQs
  (Urry & Padovani 1995). So "core-dominated, jet-viewed", which the processing design cell still says
  (`fermi_processing.py:145`), is the opposite of the standard picture.
- The empirical argument (a classifier gave about ⅓/⅓/⅓ posteriors) rests on 8 sources, is not
  reproducible from the repository, and near-uniform posteriors are what any classifier gives at that
  n, especially with class weighting.

**Decision taken today:** keep the merge, call the class "FSRQ-like", disclose the 4LAC definition.
That is acceptable if the paper reports it as a convention. The alternative that agrees with 4LAC is
to map them to `AGN_OTHER`. Because these are only 8 rows (0.3% of Setup B), the headline numbers
should not depend on the choice; a one-line sensitivity check (merge vs drop) would settle it.

**Verdict:** 🟡 (a disclosed convention, contradicted by the catalogue's own definition)

### A2.5 Superclass grouping (`TAXONOMY`)

The superclasses are meant to group classes by gamma-ray emission mechanism. The local-classifier-
per-parent-node design that uses them is standard (Silla & Freitas 2011). Per group:

| Superclass | Members | Support | Comment |
|---|---|---|---|
| `BLAZAR_HE` | fsrq (+ssrq, css) | External-Compton-dominated, soft, highly variable FSRQs (Ajello et al. 2020) | The name suggests "high energy", but it maps to FSRQs, which are mostly low-synchrotron-peaked. Define it in the paper. |
| `BLAZAR_LE` | bll | SSC-dominated, harder, less variable BL Lacs (Ajello et al. 2020) | Same naming remark. |
| `AGN_OTHER` | rdg, nlsy1, agn, sey | 4LAC non-blazar AGN (Ajello et al. 2020) | 4LAC also puts css/ssrq here (A2.4). |
| `PSR_ISOLATED` | psr, msp, glc | Magnetospheric emission, curved spectra, steady (Abdo et al. 2013, 2PC; Smith et al. 2023, 3PC). Globular-cluster emission is attributed to MSP populations (Abdo et al. 2010a). | Most MSPs are in binaries, so "isolated" is wrong; rename (e.g. `PULSAR_LIKE`). |
| `BINARY_SYSTEMS` | hmb, lmb, bin, nov | Gamma-ray binaries (Dubus 2013) | Novae emit from shocks in the ejecta, transiently (Ackermann et al. 2014). They are grouped by host system, not by emission mechanism, which contradicts the stated rule. |
| `SNR_PWN` | snr, pwn | Shock-accelerated particles (Acero et al. 2016) | Now without `spp`. |
| `GALACTIC_EXT` | sfr, gal, sbg | Cosmic-ray interactions in star-forming environments (Ackermann et al. 2012b) | `gal` and `sbg` are extragalactic, so "Galactic" is misleading; e.g. `STAR_FORMING`. |

**Verdict:** ✅ as a design; ⚠️ for three names and the novae placement.

---

## A3. Train/test design (`fermi_processing.py:1378`, `run_fermi.py:118`)

### A3.1 Stratified 70/30 split; classes with fewer than 2 rows dropped

**Why:** stratification keeps the class proportions of the test set close to the population's, which
reduces the variance of the estimate when classes are imbalanced (Kohavi 1995). A class with one row
cannot appear in both parts.

**Verdict:** ✅

### A3.2 The three validation designs

This was the main issue of the earlier review: the "untouched" test sets overlapped the data used for
selection. It is now fixed by design.

- **`single_split` (current):** search, feature selection, Stages 1–4 and the family choice read only
  the training part of one split. The test score is the one number that no selection step has seen.
  This is the classic hold-out design (Hastie et al. 2009, §7.2). It avoids the selection bias
  described by Cawley & Talbot (2010) and Varma & Simon (2006).
- **`per_split`:** the whole search and selection procedure is repeated inside each of 5 splits, and
  each split's winner is scored on that split's test set. The mean estimates the performance of the
  *procedure*, which is what Varma & Simon (2006) and Krstajic et al. (2014) recommend reporting when
  selection is part of the procedure. It is not full nested CV, because the 5 splits are random
  re-partitions of one pool, but every selection step is confined to its own split's training data.
- **`shared`:** the legacy scheme, kept for comparison. Its test scores are optimistic and must not be
  reported as estimates.

**Verdict:** ✅ for the logic of `single_split` and `per_split`. The current `single_split` run has a
different problem: one test set of this size gives a very imprecise number, and the code reports no
uncertainty for it (B1.1).

### A3.3 Feature selection outside the model-search loop

**What:** features are selected once per split on its training part, and Modules 2/3 then search
models on those fixed sets.

**Why it is acceptable:** the CV scores used *for selection* are then biased upward, because the same
rows chose the features (Ambroise & McLachlan 2002; Cawley & Talbot 2010). The test score is not
affected, because feature selection never reads test rows. Repeating selection inside every CV fold of
every model search would multiply the cost by (#candidates × #folds).

**Condition:** this argument only holds if the CV scores are never presented as performance estimates.
The fine module currently aliases them as `generalization_*` and says they are "the honest
generalization estimate" (`fermi_fine.py:713`), which is not true (B3.9).

**Verdict:** ✅ as a disclosed simplification.

---

## A4. Feature selection (`fermi_processing.py:551–926`, `fermi_common.py:3849–4082`)

### A4.1 Permutation importance measured on held-out folds, scored by balanced accuracy

**What:** for each CV fold, a HistGBC is fit on the training folds, and the drop in balanced accuracy
when each feature is permuted is measured on the held-out fold.

**Why:**
- Permutation importance (Breiman 2001) measures how much a fitted model relies on a feature. Fisher,
  Rudin & Dominici (2019) formalize it as "model reliance".
- Computing it on held-out data, rather than on training data, avoids rewarding features the model
  merely memorized (Strobl et al. 2007; Molnar et al. 2022).
- Balanced accuracy as the score is consistent with the selection metric and stops the majority
  classes from dominating the ranking (Brodersen et al. 2010).
- Using one model family to rank features follows from the result that relevance is relative to the
  learner (Kohavi & John 1997). The cross-check against RF, logistic regression and SHAP (A4.8) tests
  whether the ranking is specific to that model.

**Problem:** permutation importance is unreliable with correlated features. Permuting one member of a
correlated pair leaves its twin intact, so both look unimportant (Strobl et al. 2008; Gregorutti et al.
2017; Hooker, Mentch & Zhou 2021). This data set contains exact duplicates: `Conf_68_*` and `Conf_95_*`
are fixed rescalings of each other, and the shift table shows identical KS and PSI for them. It also
contains near-duplicates (`PL_*`/`LP_*`/`PLEC_*` indices and fluxes). The `_DECORR` step runs *after*
selection, so it cannot rescue a pair that was already cut (B2.2).

**Verdict:** ⚠️

### A4.2 Repeated CV and the lower-confidence-bound ranking (mean − 1·std)

**What:** the CV importance pass is repeated `STABILITY_N_REPEATS = 10` times with reshuffled folds.
Features are ranked by `max(0, mean − z·std)` with z = 1.

**Why:** repeating CV with different partitions reduces the variance due to the choice of folds
(Kohavi 1995; Bouthillier et al. 2021). Ranking on a lower bound penalizes features whose importance is
unstable.

**Caveat:** the 10 repeats are reshuffles of the *same* training set, so they are strongly correlated.
Their std underestimates the sampling variability of the importance (Bengio & Grandvalet 2004). The
choice z = 1 has no source.

**Verdict:** 🟡 (a reasonable heuristic; say so)

### A4.3 Cumulative-power cut (99 / 90 / 80% of positive importance mass)

**Why:** this replaces a fixed top-N with a threshold on how much of the signal is retained. No paper
proposes this rule, but its role here is only to generate candidate feature sets, and the choice among
them is made by CV in Stage 1 of the model search. That makes the specific thresholds less critical.

**Verdict:** 🟡

### A4.4 "Stability selection" filter (feature kept in ≥ 70% of repeats)

**Cited:** Meinshausen & Bühlmann (2010).

**Departure:** their selection frequencies, and the error bound that justifies the 0.6–0.9 threshold
range, are defined over random **subsamples of half the data**. Here each "repeat" averages
importances over 5 folds covering the **full** training set, so the repeats are nearly identical, the
frequencies are pushed toward 0 or 1, and the 0.7 threshold gives no error control. Complementary-pairs
stability selection (Shah & Samworth 2013) is the citable version and would only need the resampling
changed (B2.3).

**Verdict:** ⚠️

### A4.5 Decorrelated companion sets (`_DECORR`)

**What:** average-linkage hierarchical clustering on 1 − |Spearman ρ|, cut at |ρ| = 0.85, keeping the
most important feature of each cluster.

**Why:** distance- and covariance-based models are hurt by collinear inputs, and trees are not, so both
versions are offered and CV chooses. Dormann et al. (2013) review collinearity handling and use
|r| > 0.7 as the usual rule of thumb. 0.85 is looser and should be stated as a choice.

**Verdict:** ✅

### A4.6 Shift-robust companion sets (`_SHIFT_ROBUST`)

**What:** drops the features flagged by the covariate-shift check (A5).

**Why:** removing the features whose distribution differs most between labelled and unlabelled sources
is a simple form of making the classifier transfer better to the target population. It is a heuristic:
it reduces the reliance on shifted features, but it does not change the evaluation population or the
class priors (B1.3). Since the shift check uses only the X of labelled and unassociated sources, never
labels, it does not leak test labels.

**Verdict:** 🟡

### A4.7 Feature-count cap (`MIN_SAMPLES_PER_FEATURE = 5`) and `ALLPOS`

**Why:** small-sample pattern-recognition theory ties the number of features one can afford to the
number of training rows (Raudys & Jain 1991); clinical prediction uses about 10 events per variable
(Peduzzi et al. 1996). The cap here uses the node's total n, while the binding quantity is the size of
the smallest class. For Setup B (n_train ≈ 2,000) the cap is irrelevant; it only matters for the small
hierarchical branches.

**Verdict:** ⚠️ (B3.10)

### A4.8 Diagnostics: SHAP, RF/LogReg agreement, Jaccard and RBO across splits

**Why:** if two different attribution methods (Lundberg & Lee 2017; Lundberg et al. 2020) and two
unrelated model families agree with the HistGBC ranking, the selected features are not an artefact of
one model. Rank-biased overlap (Webber, Moffat & Zobel 2010) compares top-weighted rankings across
splits.

**Verdict:** ✅ (diagnostic only; part `diag` is not in the current `STEPS`, so these are not produced
in the current run)

### A4.9 Galactic-latitude agreement (`compute_spatial_agreement`)

**What:** compares feature rankings computed in-sample above and below |b| = 10°. The threshold matches
the 4LAC high-latitude sample (Ajello et al. 2020).

**Problem:** the class mix is completely different in the two regions, so the rankings would differ
with no shortcut at all, and a model that uses latitude as a shortcut can show agreement. The check does
not test what it claims to test (B2.6).

**Verdict:** ❌ as a shortcut test; fine as a descriptive plot.

### A4.10 Missingness informativeness (Cramér's V)

Descriptive. Cramér's V (Cramér 1946) on chi-square tables with tiny expected counts for rare classes
is unreliable (Cochran 1954); state that it is diagnostic only.

**Verdict:** 🟡

### A4.11 Optional-features comparison

**Why:** re-running the selection with the manually purged columns restored is a direct empirical test
of the purge. It is a good robustness check. It is not in the current `STEPS`.

**Verdict:** ✅

---

## A5. Covariate-shift diagnostic (`fermi_processing.py:488`)

**What:** for each feature, a two-sample KS test (α = 0.01) and the Population Stability Index (PSI)
binned on the deciles of the associated sources. A feature is flagged only if both KS is significant
and PSI ≥ 0.25.

**Why:**
- Univariate two-sample tests per feature are a published, effective shift detector (Rabanser,
  Günnemann & Lipton 2019).
- Requiring an effect size (PSI) as well as significance avoids flagging trivial differences that are
  significant only because n is large.
- 0.10 and 0.25 are the conventional "moderate" and "major" PSI cut-offs of the credit-scoring
  literature (Siddiqi 2006; Yurdakul 2018).

**Result (full-search reference run's data):** 33 of 71 features flagged; PSI up to 2.25; Signif_Avg, variability, detection and
localization features are the most shifted. Half the unassociated sources are at |b| < 10°.

**Multiple testing:** 71 tests without correction. Here the correction is cosmetic: every flagged
feature has p < 10⁻¹⁰⁰, so Bonferroni (or Benjamini & Hochberg 1995) would change nothing, and PSI is
the criterion that binds. Add one sentence saying so.

**Verdict:** ✅

---

## A6. The model-search engine (`fermi_common.py`)

### A6.1 Class weighting (`class_weight_map`, line 921)

**What:** `none`; `balanced` = n/(K·n_c) capped at 10× the majority weight; `dampened` = balanced^0.5;
`balanced_uncapped`. The reduced profile keeps `balanced_uncapped` and `none`, and Stage 2 of the search
chooses between them per model.

**Why:** inverse-frequency weighting is the standard cost-sensitive correction for imbalance (King &
Zeng 2001; He & Garcia 2009) and is equivalent to optimizing a balanced loss, which matches the
balanced-accuracy objective. The cap and the damping are heuristics; the effective-number-of-samples
weights of Cui et al. (2019) are the principled version of damping. Keeping `none` as a candidate makes
"did weighting help?" an empirical question. The reduction to two modes is justified by the full-search reference run's logs.

**Consequence to state:** the probabilities of a weighted model are tilted toward the rare classes and
are not posteriors under the catalogue's class mix (Elkan 2001; Saerens et al. 2002). With
`balanced_uncapped`, `sey` (3 training rows) gets a weight about 500× that of `bll`.

**Implementation bug:** with `USE_GPU = True`, RandomForest is weighted by a bootstrap resample instead
of sample weights, although it runs on CPU (B1.4).

**Verdict:** ✅ (with the bug fixed)

### A6.2 Weighted resampling for models without `sample_weight` (LDA, KNN)

**Why:** random oversampling in proportion to the class weight is the classical alternative when a
learner cannot take weights (Japkowicz & Stephen 2002). It must happen *after* any internal validation
split; otherwise duplicates of a row land on both sides of the split and the internal score is inflated
(Santos et al. 2018). The MLP was moved off resampling for exactly that reason. LDA has no
internal split, so it is unaffected.

**Verdict:** ✅

### A6.3 CV scoring: pooled out-of-fold predictions, bootstrap std, rare classes train-only, adaptive LOO (`cv_score_estimator`, line 1063)

**Pooling** the out-of-fold predictions and computing the metric once, rather than averaging per-fold
scores, is the recommended practice for skewed classes and small folds (Forman & Scholz 2010). It is
also what makes LOO usable here at all. ✅

**Bootstrap of pooled OOF pairs** gives the variability due to the evaluation rows only, not to the
training sets. It is a lower bound on the true uncertainty, and no unbiased estimator of CV variance
exists (Bengio & Grandvalet 2004). Label it as such. 🟡

**Rare classes (n_c < 5) are put in every training fold and never validated.** This keeps those classes
learnable, but it makes the CV score *optimistic*, not conservative: balanced accuracy is averaged over
the classes present in `y_true`, so the hardest classes are simply missing from the score. It also means
the search optimizes a different quantity from the test metric, which includes them (B2.4). ❌

**LOO for tiny nodes** (n ≤ 150 and a class with ≤ 5 rows) is the only scheme that holds out every row.
It has high variance (Kohavi 1995; Varoquaux 2018); keep the disclosure. ✅

### A6.4 Coarse-to-fine funnel (`coarse_filter`, line 1266)

**What:** each candidate is scored on a single stratified 15% holdout of the training data, the top 25%
(10% in fine, `fermi_fine.py:693`) survive (minimum 3), and survivors get full 5-fold CV.

**Why:** this is successive halving with one round (Jamieson & Talwalkar 2016; Li et al. 2018). Those
methods assume the cheap evaluation ranks candidates roughly correctly. With about one validation row
per rare class, the 15% holdout is very noisy. Under the reduced profile it only bites in Stage 1 (the
~13 feature-set tags); Stages 0 and 2 have ≤ 3 candidates and keep all of them.

**Verdict:** ⚠️ (B2.7)

### A6.5 Staged search: scaler → feature set → weighting

The three axes are optimized one at a time, each conditional on the previous winner (coordinate-wise
greedy search). No paper supports this specific order, and weighting interacts with which features help
the rare classes. With the reduced profile the scaler stage is degenerate (one scaler), so the effective
search is feature set, then weighting.

**Verdict:** 🟡

### A6.6 Fixed hyperparameters

Every model runs at fixed settings (LogReg C = 1, CatBoost depth 6, RF `max_features="sqrt"`, MLP Adam
with weight decay 1e-4). The only capacity chosen from data is the number of iterations or trees
(A6.7). A comparison of model *families* at fixed settings partly measures how good each default is
(Probst, Boulesteix & Bischl 2019). Either say "untuned defaults" in the paper or add a small random
search (Bergstra & Bengio 2012).

**Verdict:** ⚠️

### A6.7 Early stopping and refit on all rows

- **MLP and HistGBC:** chunked training with a per-class stratified 15% validation split, patience 4,
  best epoch restored (MLP) or refit at the best iteration count (HGBC). Early stopping on held-out
  data is the standard capacity control for networks and boosting (Prechelt 1998; Zhang & Yu 2005).
  Stratifying the internal split per class prevents library defaults from dropping rare classes from
  validation. ✅
  The stopping criterion is balanced accuracy on that 15% split, where rare classes have 0–1 rows, so
  it is noisy. Prechelt (1998) analyses stopping on validation *error*; validation cross-entropy is
  smoother and is what CatBoost already uses. ⚠️ (B3.6)
- **CatBoost:** early stopping on class-weighted multiclass log-loss over a stratified 15% split, then
  refit on all rows at the best tree count. ✅
- **Random forest:** trees added in chunks of 50 until the OOB balanced accuracy plateaus. More trees
  never hurt a forest; the number of trees is a compute budget, not a hyperparameter to tune (Probst &
  Boulesteix 2018). The plateau rule is a sensible way to stop spending. ✅
  With the resample bug (B1.4), duplicated rows make the OOB estimate optimistic, which only affects
  when the growing stops.
- **SVM (not in the reduced profile):** Platt-scaled probabilities with `predict = argmax(predict_proba)`
  so the scored decision and the deployed decision agree (Platt 1999; Wu, Lin & Weng 2004). ✅

### A6.8 Imputation and scaling inside each pipeline

`IterativeImputer → scaler → (PCA) → model`, rebuilt per fold. Fitting preprocessing on each fold's
training rows avoids leakage (Hastie et al. 2009, §7.10.2). Chained-equation imputation is the standard
multivariate method (van Buuren & Groothuis-Oudshoorn 2011). Trees and CatBoost take NaN natively (A1.7).

**Caveat at deployment:** the imputer is fit on associated sources and applied to unassociated ones,
whose missingness pattern differs (B3.11).

**Verdict:** ✅

### A6.9 Search-space reduction (`reduced` profile)

The profile was derived from the split-0 CV leaderboards of an earlier full-search run: the top 4–6
models per node were within one std of each other, PCA never won, and the scaler choice was random. This
is a legitimate compute decision. It is also a researcher degree of freedom (Gelman & Loken 2013): the
final split 0 (seed 42 on a pool without `spp`) is a different partition from that run's split 0, so
some rows that informed the narrowing are now in the final test set. The effect is small, because the choice was coarse
(which families to keep) and made on CV scores, but it should be disclosed in one sentence (B3.5).

**Verdict:** ✅ with disclosure.

---

## A7. Hierarchical families (merged 2-level, 3-level) — not in the current run

### A7.1 Local classifier per parent node; branches trained on the true superclass

Training each branch on the rows whose *true* superclass matches is the correct estimator of
P(fine | x, superclass). It is the standard local-classifier-per-parent-node design (Silla & Freitas
2011). Under soft routing (A7.2) this is not a limitation: the router's errors enter through
P(superclass | x), not through the branch.

**Verdict:** ✅

### A7.2 Soft routing: P(fine | x) = Σ_s P(s | x) · P(fine | x, s)

This is the law of total probability, and the multiplicative combination of node probabilities is the
standard hierarchical scoring rule (Dumais & Chen 2000). The same blend is used in chain-CV,
test evaluation and inference, so the deployed classifier is the evaluated one.

The products amplify any miscalibration of the nodes, each tilted differently by its own class weights
(B2.8).

**Verdict:** ✅

### A7.3 Chain cross-validation

The whole router → branch system is refit inside each fold, and the end-to-end fine prediction is
scored. This is the right way to score a pipeline as one unit. The node feature sets were selected on
the full training split before the chain-CV, so the chain-CV scores carry the selection bias of A3.3.

**Verdict:** ✅ (as a selection score, not as an estimate)

### A7.4 Branch reliability tiers ("sweep" / "fallback" / "unvalidatable")

Skipping the model search on tiny branches (fewer than 10 rows) avoids the winner's curse where it is
worst (Cawley & Talbot 2010). Reporting a training-only fit as "unvalidatable" is honest.

**Verdict:** ✅

### A7.5 Top-N combination screening

Combining the top 3 routers with the top 5 branch combinations (merged) or 3 × 3 × 3 gate/subgate/router
(3-level) and choosing end-to-end by chain-CV is a heuristic beam search. It is reasonable, and each extra
candidate adds a little winner's-curse optimism to the selection score.

**Verdict:** 🟡

---

## A8. Validation stages (`fermi_validation.py`)

### A8.1 Stage 1: shortlist re-scored with 5 × 5-fold repeated CV

Repeated CV reduces the partition variance of the ranking (Kohavi 1995; Bouthillier et al. 2021). In
`single_split` mode this re-scores the same training data the Module-2 search used; it refines the
ranking, but it is not independent evidence.

**Verdict:** ✅

### A8.2 Stage 3: cross-split confirmation

In `single_split` mode it re-scores the shortlist once more on split 0 with one seed, so it adds almost
nothing to Stage 1. In `per_split` mode each split confirms its own shortlist.

**Verdict:** ✅ (redundant in the current design, harmless)

### A8.3 Stage 4 and the family choice: Borda average rank over accuracy, balanced accuracy, macro-F1, κ and the overfit gap

**Decision taken today:** kept, as a deliberate multi-metric choice that avoids trading away
big-class performance. The family is now chosen on the training-side (Stage 3) scores, not on the
test scores, which removes the earlier test-set selection.

**What remains unsupported** (B2.1):
- It optimizes balanced accuracy in Modules 2/3 and selects on a different objective in Stage 4.
- Accuracy and κ re-weight toward the majority classes. Delgado & Tibau (2019) argue κ should not be
  used to rank classifiers.
- The overfit gap uses *in-sample* training predictions. Random forests and boosted trees interpolate
  their training data by design (Wyner et al. 2017), so they are penalized regardless of their
  validation score. A train–validation gap is not a generalization criterion in the standard
  references; the validation score already measures generalization.
- The cited source, Demšar (2006), averages ranks across *data sets* to compare classifiers with a
  Friedman test. It is not about combining *metrics* on one data set, and the volume is 7, not 11
  (`fermi_validation.py:1183`, `:3006`).

The defensible alternative is balanced accuracy alone with the one-standard-error rule to prefer the
simpler or more stable of near-tied candidates (Breiman et al. 1984; Hastie et al. 2009, §7.10),
reporting the other metrics descriptively.

**Verdict:** ❌ as justified; 🟡 if presented as a stated preference, with the citation removed.

### A8.4 Stage 5: one evaluation on the test part

Correct and clean in `single_split`. Refitting the winner on the full training part before scoring is
standard (Hastie et al. 2009, §7.10).

**Verdict:** ✅; see B1.1 for the missing uncertainty.

### A8.5 Confidence intervals across splits (`_nb_sem`, line 2674)

In `per_split`/`shared`, the CI on the mean over 5 random re-splits uses the Nadeau–Bengio corrected
variance σ²(1/J + n_test/n_train), with a Student-t multiplier on J − 1 degrees of freedom (Nadeau &
Bengio 2003; Bouckaert & Frank 2004). Here n_test/n_train = 0.43, so the CI is about 1.8× wider than the
naive one. Even the corrected interval is approximate, because no unbiased variance estimator exists for
such designs (Bengio & Grandvalet 2004; Bates, Hastie & Tibshirani 2024). In `single_split` there is
nothing to average, and the code reports no CI (B1.1).

**Verdict:** ✅

### A8.6 Stage 7: deployment refit on all labelled data of split 0 (train + test)

Refitting the selected configuration on all available data before deployment is standard (Hastie et
al. 2009, ch. 7). The Stage-5 number estimates the performance of a model trained on 70% of the data,
so it is slightly pessimistic for the deployed model. The deployed model itself is never evaluated,
which is the correct statement to make.

**Verdict:** ✅

### A8.7 Per-class metrics

Precision, recall, F1 and support per class are necessary because the macro averages hide which rare
classes drive the score. With 1–3 test rows per rare class, each per-class recall needs an exact
interval (Clopper & Pearson 1934) or a Wilson interval (Wilson 1927) next to it.

**Verdict:** ✅ (add intervals)

### A8.8 Shift-aware test metrics (Section 16a, new today)

**What:**
- (a) Test metrics stratified by |b| < 10° vs ≥ 10° and by Signif_Avg tertiles. The tertile edges come
  from the unassociated sources.
- (b) Importance-weighted metrics: a HistGBC domain classifier (associated train vs unassociated) gives
  w(x) = [p(u|x)/p(a|x)] · (n_a/n_u), clipped at the 99th percentile, with Kish's effective sample size
  reported.

**Why:**
- Under covariate shift, an importance-weighted loss is an unbiased estimate of the target-population
  risk (Shimodaira 2000; Sugiyama, Krauledat & Müller 2007).
- A probabilistic classifier separating the two populations gives the density ratio (Bickel, Brückner &
  Scheffer 2009). The same classifier is a two-sample test (Lopez-Paz & Oquab 2017).
- Clipping bounds the variance that large weights cause (Cortes, Mansour & Mohri 2010).
- ESS (Kish 1965) shows how much information is left after weighting.
- Malyshev (2023) applies this framework to 4FGL-DR4 multi-class classification. He finds that shift
  barely changes the predicted probabilities, but lowers precision and recall by 10–20% per class
  relative to unweighted estimates. This is the closest precedent and should be cited as such.
- `balanced_accuracy_score(sample_weight=w)` reweights rows *within* each class and then averages the
  classes equally, which is the right covariate-shift analogue of balanced accuracy.

**Limits the code itself states:** valid only if p(class | x) is the same in both populations; sources
of classes outside the taxonomy are not represented; confidence cuts are not prior-corrected. Two more
to add:
- The domain classifier's probabilities are not calibrated, so the weights are only approximately
  density ratios; fitting it out-of-fold and calibrating it, or reporting the ESS prominently, addresses
  this.
- Strata with ≥ 10 rows can still contain single rows of rare classes.

**Verdict:** ✅

### A8.9 Calibration check (Section 16b)

The multiclass Brier score (Brier 1950) is a strictly proper score (Gneiting & Raftery 2007). The
10-bin reliability table on the top-class probability is the standard diagnostic (Niculescu-Mizil &
Caruana 2005; Guo et al. 2017). It is diagnostic only: nothing recalibrates the probabilities, and
class weighting is known to tilt them (A6.1, B2.8).

**Verdict:** ✅ (as a diagnostic)

### A8.10 Rare-class shrinkage (Section 16c)

**What:** empirical-Bayes Beta–binomial partial pooling of per-class recall, with the concentration
estimated by method of moments with a DerSimonian–Laird-style correction for within-group binomial
variance.

**Why:** shrinking noisy small-sample rates toward a group mean reduces their mean squared error
(Efron & Morris 1975; Gelman et al. 2013, ch. 5; DerSimonian & Laird 1986).

**Departure:** the method assumes the classes in a group are exchangeable. Rare classes are
systematically harder, and the flat (fine) version pools toward a grand mean dominated by bll/fsrq, so
rare-class recall is pulled *up*. In `single_split` mode the earlier double-counting across overlapping
test sets no longer applies.

**Verdict:** ⚠️ (B3.4)

---

## A9. Inference and prospective check (`fermi_inference.py`)

### A9.1 Predictions for unassociated sources with confidence = max probability

`confidence` is the predicted class's probability, and every class probability is exported. It
is a real probability, not a fake 1.0. Cuts such as ≥ 0.8 are reasonable *rankings* of how sure the
model is. They are not "80% correct" on the unassociated population, because:
- the probabilities are tilted by class weighting (Elkan 2001);
- they are not prior-corrected for the very different class mix of unassociated sources (Saerens,
  Latinne & Decaestecker 2002; Lipton, Wang & Smola 2018; Malyshev 2024);
- there is no reject option for sources outside the 18 classes (Chow 1970; Scheirer et al. 2013). DR3
  notes that many soft Galactic unassociated sources may be diffuse-model artefacts (Abdollahi et al.
  2022, §6.2).

**Verdict:** ⚠️

### A9.2 Cross-family agreement

Agreement between architectures is a useful flag for human follow-up. The families share features and
training data, so their errors are correlated, and agreement is weaker evidence than independent votes
(Kuncheva & Whitaker 2003). With `FAMILIES = ["fine"]` it is not produced.

**Verdict:** 🟡

### A9.3 Prospective check against v41

**What:** v35 unassociated sources that v41 newly associates are cross-matched positionally on the
sphere. The match radius is 3× the v41 95% semi-major axis, clipped to 1′–15′. Ground truth is
normalized the same way as training (ssrq/css → fsrq), and `unk`, `bcu`, `spp` are excluded.

**Why:** a temporal validation on labels that did not exist when the model was trained is the strongest
external test available. It is the "temporal validation" level of prediction-model reporting (Collins
et al. 2015, TRIPOD). An adaptive radius tied to the localization error is the standard approach for
catalogue cross-matching.

**Caveats:**
- Newly associated sources are brighter and blazar-heavy, and the excluded `bcu` are the bulk of new
  associations, so the scored subset is not representative of the unassociated population.
- Nearest-neighbour matching does not enforce one-to-one matches.
- Confidence thresholds must not be tuned on v41, or it stops being an external test.

**Verdict:** ✅ (report n per class and the representativeness caveat)

---

# Part B — Review findings

Ordered by effect on the reported numbers. "New" = found in this pass. "Open" = from the earlier
review, still in the code.

## B1. Affects the headline numbers

### B1.1 (New) The current design reports one test score with no uncertainty, and that score is very noisy

`single_split` is clean, but the code has nothing to say about precision ("no CI: n = 1",
`fermi_validation.py:2982`).

A rough calculation for Setup B (861 test rows, 18 classes): assume a recall of 0.9 for the four big
classes, 0.7 for the mid-size ones, and 0.5 for the ten classes with ≤ 3 test rows. The standard error of
balanced accuracy from test sampling alone is then

  SE = (1/18) · sqrt(Σ_c p_c(1 − p_c)/n_c) ≈ 0.064,

that is, a **95% half-width of about ±0.13**. One test row of `sey` moves balanced accuracy by 1/18 =
0.056. Any difference between candidate families or settings smaller than about 0.1 cannot be resolved
by this test set (Varoquaux 2018). This is test-set variance only; training-set variance comes on top of
it.

Fix, cheapest first:
- Report a stratified bootstrap CI over the test rows (Efron & Tibshirani 1993).
- Report per-class Clopper–Pearson intervals.
- Report balanced accuracy also over "classes with ≥ 5 test rows", with the rare classes in a separate
  table.
- For a headline with a CI covering training variance, run `per_split` (about 5× the search cost).

### B1.2 (New) The excess-variance feature is biased for faint sources (`preprocessing.py:318`)

In 4FGL light curves, a bin with TS < 1 has a NULL lower error and an upper-limit-derived upper error
(Abdollahi et al. 2022, Table 6 note c). In `_normalized_excess_variance`, `mean_unc` is then NaN for
that bin, so `np.nanmean(mean_unc**2)` ignores its measurement variance, while `np.nanvar(flux)` still
includes its flux. Faint sources have many such bins, so their noise term is underestimated and their
excess variance inflated. `Flux_History_peak_to_median` also uses those values as if they were measured.

Unassociated sources are fainter, so this creates a shift-correlated artefact in a feature family that
is both highly important for blazar vs pulsar and flagged as shifted (`log_Flux_History_excess_variance`,
PSI 0.77). The size of the effect is unknown until it is measured.

Fix: use the upper error for the bin's σ, or exclude TS < 1 bins from both terms; check the fraction of
such bins per class. The same convention affects `Flux_Band` and the hardness ratios (A1.4). This also
needs the DR4 preprocessing script in the repository (B3.12).

### B1.3 (Open, now measured and disclosed) Evaluated on associated sources, deployed on unassociated ones

The shift-aware metrics (A8.8) now quantify this, which is the fix the earlier review asked for. What is
still missing:
- prior re-estimation before applying confidence cuts (Saerens et al. 2002; Malyshev 2024);
- a reject option or "other" class;
- the uncalibrated domain classifier caveat in A8.8.

Present the importance-weighted numbers next to the plain ones, not in an appendix.

### B1.4 (New) RandomForest is weighted by bootstrap resampling although it runs on CPU (`fermi_common.py:3797`)

`weight_support_for` returns `"resample"` for `"SVM"` and `"RandomForest"` whenever `USE_GPU = True`,
because cuML ignores sample weights. But cuML RF is disabled everywhere: `MODEL_SPECS["RandomForest"]`
has `build_gpu = build_rf`, the CPU `RFPlateauClassifier`. So every weighted RF in the fine search
(Module 2) and in validation (Stages 1, 3, 5, 7) is trained on a weighted bootstrap resample instead of
with native sample weights:
- It is a different and noisier weighting from the declared one.
- It removes about a third of the unique majority-class rows from each fit.
- It duplicates rare-class rows. RF then bootstraps again, so copies of a row are both in-bag and
  out-of-bag, which inflates the OOB score that `RFPlateauClassifier` uses to stop. This is the same
  leakage pattern as bug 1.2 for the MLP (Santos et al. 2018).

The RF candidates' CV scores, and therefore the ranking, are affected. The test score of an RF winner
is still honest, but the model is not the one the docs describe.

Fix: remove `"RandomForest"` from that condition (one line), or condition on the builder actually being
cuML. Bump `NODE_CACHE_VERSION`.

## B2. The method is not justified as implemented

| # | Issue | Where | Fix | Literature |
|---|---|---|---|---|
| B2.1 | Borda over correlated metrics + in-sample overfit gap; wrong Demšar citation | `_borda_rank`, `_overfit_gap` (validation:1158–1210) | Balanced accuracy + one-SE rule; or keep Borda as a stated preference, drop the gap and the citation | Delgado & Tibau 2019; Wyner et al. 2017; Breiman et al. 1984; Demšar 2006 (JMLR **7**) |
| B2.2 | Permutation importance with exact and near-duplicate features | `_fold_importance` (processing:579) | Drop exact duplicates (`Conf_68_*` vs `Conf_95_*`), cluster before importance, grouped permutation importance per cluster | Strobl et al. 2008; Gregorutti et al. 2017; Hooker et al. 2021 |
| B2.3 | "Stability selection" without n/2 subsampling | `compute_repeated_importance`, `stability_select` (processing:600–660); `compute_selection_engine` (common:3899) | 50 complementary-pair half-samples (CPSS), importance on each, keep the 0.7 threshold | Meinshausen & Bühlmann 2010; Shah & Samworth 2013 |
| B2.4 | Rare classes never validated in CV, so CV is optimistic and optimizes a different objective | `cv_score_estimator` (common:1088); `determine_cv_eligibility` (processing:551) | State it; report CV and test balanced accuracy over the same class set; per-class exact intervals for the rare ones | Brodersen et al. 2010; Forman & Scholz 2010 |
| B2.5 | Untuned hyperparameters; coarse filter on a noisy 15% holdout | MODEL_SPECS builders; `coarse_filter` | Say "defaults"; with the reduced profile, give every Stage-1 candidate full CV (cheap now) | Probst et al. 2019; Li et al. 2018 |
| B2.6 | The latitude check does not test shortcut learning | `compute_spatial_agreement` (processing:822) | Report test metrics by \|b\| (now done in 16a); add an ablation without GLAT/GLON; encode GLON as sin/cos | — |
| B2.7 | Greedy scaler → features → weighting staging | `_select_winners` (common:1330) | Label as heuristic, or search features × weighting jointly in Stage 1 (only 2 modes now) | — |
| B2.8 | Probabilities uncalibrated; products amplify it (hierarchical) | all families | Temperature or isotonic scaling on held-out predictions, or a prior correction for the weighting; then the shift prior correction | Guo et al. 2017; Niculescu-Mizil & Caruana 2005; Elkan 2001; Saerens et al. 2002 |

## B3. Minor issues and bugs

| # | Status | Where | Issue | Fix |
|---|---|---|---|---|
| B3.1 | Open | `select_by_cumulative_power` (common:3856) | With no positive importance it returns all features and ignores `max_features`, so the sample-per-feature cap disappears on small branches | Apply the cap on that path too |
| B3.2 | Open | `apply_shift_robust` (processing:685) | If every feature is flagged, returns the unfiltered set under the `_SHIFT_ROBUST` name | Skip the tag |
| B3.3 | Open | `_val_score_fine_candidate` (validation:1334) | Precision/recall come from a second KFold that validates rare classes; balanced accuracy from one that does not. Same row, two OOF definitions | Use `collect_oof=True` |
| B3.4 | Open | Section 16c | Exchangeability of rare and common classes in the shrinkage | Shrink toward a mean modelled on log n_train, or report raw rates with exact intervals |
| B3.5 | New | `reduced` profile | Narrowed using an earlier run's split-0 CV; the final split 0 is a different partition | One sentence of disclosure (Gelman & Loken 2013) |
| B3.6 | New | MLP and HGBC early stopping | Stop on balanced accuracy over a 15% split with 0–1 rows per rare class | Stop on validation cross-entropy (as CatBoost does) |
| B3.7 | Open | design note D3 | "Missingness is never a signal" is false for HistGBC/CatBoost and with `*_is_null` columns | Correct the note; it is legitimate (Josse et al. 2019) |
| B3.8 | New | `preprocessing.py:273` comment | Bits 13/14 called "unused". Flag 14 is association-dependent (SGU/SPP/UNK only), so it is excluded as leakage; Flag 13 (TS < 25 at 12 yr) could be kept; bit 8 is a constant column | Fix the comment; optionally keep 13, drop 8 |
| B3.9 | New | stale text | `three_way_role_disclosure` (validation:2989) still says "five genuine, previously-untouched test partitions" and "corrected_winner_family is sourced from Stage 5"; `ranking_methodology_note` cites Demšar JMLR 11; fine aliases `val_*` as `generalization_*` and calls it "the honest generalization estimate" (fine:713); processing design cell says "core-dominated, jet-viewed" (processing:145) | Update the strings; they end up in the JSON report |
| B3.10 | Open | `MIN_SAMPLES_PER_FEATURE` | Cap uses total n; the binding constraint is the smallest class | Base it on min class count (Raudys & Jain 1991; Peduzzi et al. 1996 suggest ≥ 10) |
| B3.11 | New | deployment imputation | `IterativeImputer` is fit on associated sources and applied to unassociated ones with a different missingness pattern | Add `isna()` rates to the shift diagnostic |
| B3.12 | Open | provenance | The DR4 CSVs cannot be regenerated from this repo (`preprocessing.py` reads DR3 and produces different columns) | Commit the DR4 preprocessing script |
| B3.13 | New | Stage 5/7 vs Module 2 | Validation and deployment fit `MODEL_SPECS["HistGBC"]`, the search proxy (ceiling 500), while Module 2 ranked the "deployable" `StratifiedEarlyStoppingHGBC` (ceiling 2000) (validation:2588, :2744) | Use the same builder in both if HistGBC can win |
| B3.14 | Open | Cramér's V | Chi-square with tiny expected counts | Fisher's exact test, or keep it diagnostic |
| B3.15 | Open | v41 check | Non-representative subset; no one-to-one matching | Report n per class; state the bias |

## B4. Status of the earlier review's issues

| Earlier item | Status in the code now |
|---|---|
| §1.1 test data leaks into selection | **Fixed** by `single_split`/`per_split`; the family is chosen on training-side scores |
| §1.2 CIs too narrow | **Fixed**: Nadeau–Bengio SE in both CI helpers |
| §1.3 associated vs unassociated | **Measured** (Section 16a); prior correction and reject option still missing (B1.3) |
| §1.4 ssrq/css | **Kept as a disclosed convention** ("FSRQ-like") (A2.4) |
| §1.4 spp | **Fixed**: dropped from Setup B |
| §2.1 Borda + overfit gap | **Kept by decision**; justification still needed (B2.1) |
| §2.2 correlated permutation importance | Open (B2.2) |
| §2.3 stability selection design | Open (B2.3) |
| §2.4 rare classes in CV | Open (B2.4) |
| §2.5 latitude check | Partly addressed by the \|b\| strata in 16a; ablation still missing (B2.6) |
| §2.6 missingness claim | Open (B3.7) |
| §2.7 search design | Open (B2.5, B2.7) |
| §2.8 calibration | Open (B2.8) |
| §3.1, §3.2, §3.5, §3.6, §3.9, §3.10, §3.12 | Open (B3.1, B3.2, B3.3, B3.4, B3.10, B3.14, B3.12) |
| §3.3 multiple testing in the shift check | Cosmetic for this data (A5); add one sentence |
| §3.8 v41 class merge | **Fixed** |
| §3.11 flags | Clarified with the DR3 definitions (B3.8) |
| §3.13 bootstrap std is a lower bound | Open (label it) |

---

## B5. Fix order

1. **B1.4** RF weighting (one line, then re-run fine): it changes candidate rankings.
2. **B1.1** test uncertainty: stratified bootstrap CI and per-class exact intervals on the Stage-5
   predictions (no refit needed).
3. **B3.9** stale report strings and the Demšar citation (text only).
4. **B1.2** excess variance with TS < 1 bins (needs the DR4 preprocessing script, then a full re-run).
5. **B2.1** selection criterion: either balanced accuracy + one-SE, or Borda without the in-sample
   gap and without the Demšar citation.
6. **B2.2, B2.3** feature selection: drop exact duplicates, grouped importance, CPSS.
7. **B2.8, B1.3** calibration and prior correction before any confidence cut is interpreted as a
   probability.
8. The rest of B3.

Items 2, 3 and 5 change only reporting. Items 1, 4 and 6 change results and need a re-run.

---

# Part C — References

Verified online during this review: Abdollahi et al. 2022 (flag definitions and the light-curve
upper-limit convention, read from the paper); Malyshev 2023; Coronado-Blázquez 2022 (title and arXiv;
volume/pages not confirmed); Bhat & Malyshev 2022; Ballet et al. 2023; Germani et al. 2021 (arXiv;
volume not confirmed); Kovačević et al. 2019. All others are standard references cited from memory;
check page numbers before submission.

**Fermi-LAT catalogues and source physics**
- Abdo, A. A., et al. 2010a, A&A, 524, A75 — gamma-ray globular clusters (MSP populations).
- Abdo, A. A., et al. 2010b, ApJ, 716, 30 — blazar SEDs, synchrotron-peak classification.
- Abdo, A. A., et al. 2013, ApJS, 208, 17 — Second Fermi-LAT Pulsar Catalog (2PC).
- Abdollahi, S., et al. 2020, ApJS, 247, 33 — 4FGL.
- Abdollahi, S., et al. 2022, ApJS, 260, 53 — 4FGL-DR3 (Flags table; §6.2 SGU sources and Flag 14). arXiv:2201.11184.
- Acero, F., et al. 2016, ApJS, 224, 8 — First Fermi-LAT SNR catalog.
- Ackermann, M., et al. 2012a, ApJ, 753, 83 — statistical classification of 1FGL unassociated sources.
- Ackermann, M., et al. 2012b, ApJ, 755, 164 — GeV observations of star-forming galaxies.
- Ackermann, M., et al. 2014, Science, 345, 554 — classical novae as gamma-ray sources.
- Ajello, M., et al. 2020, ApJ, 892, 105 — 4LAC (classes incl. CSS/SSRQ as non-blazar AGN; |b| > 10° sample).
- Ballet, J., et al. 2023, arXiv:2307.12546 — 4FGL-DR4.
- Dubus, G. 2013, A&ARv, 21, 64 — gamma-ray binaries.
- Nolan, P. L., et al. 2012, ApJS, 199, 31 — 2FGL (hardness-ratio definition).
- Smith, D. A., et al. 2023, ApJ, 958, 191 — Third Fermi-LAT Pulsar Catalog (3PC).
- Urry, C. M., & Padovani, P. 1995, PASP, 107, 803 — AGN unification.
- Vaughan, S., Edelson, R., Warwick, R. S., & Uttley, P. 2003, MNRAS, 345, 1271 — normalized excess variance.

- Dwork, C., Kumar, R., Naor, M., & Sivakumar, D. 2001, WWW — rank aggregation (Borda count).

**Machine learning on Fermi-LAT catalogues**
- Bhat, A., & Malyshev, D. 2022, A&A, 660, A87 — probabilistic Fermi-LAT catalogs. arXiv:2102.07642.
- Coronado-Blázquez, J. 2022, MNRAS — CatBoost classification of 4FGL-DR3 unIDs. arXiv:2207.04725.
- Germani, S., et al. 2021, MNRAS — ANN classification of 4FGL sources. arXiv:2106.08222.
- Kovačević, M., Chiaro, G., Cutini, S., & Tosti, G. 2019, MNRAS, 490, 4770.
- Malyshev, D. V. 2023, RAS Techniques and Instruments, 2, 735 — covariate shift in multi-class Fermi-LAT classification. arXiv:2307.09584.
- Malyshev, D. V. 2024, arXiv:2412.04675 — covariate vs prior shift for unassociated sources.
- Saz Parkinson, P. M., et al. 2016, ApJ, 820, 8 — 3FGL classification (RF, LR).

**Leakage, validation design, selection bias**
- Ambroise, C., & McLachlan, G. J. 2002, PNAS, 99, 6562 — selection bias from feature selection outside CV.
- Bates, S., Hastie, T., & Tibshirani, R. 2024, JASA, 119, 1434 — what cross-validation estimates.
- Bengio, Y., & Grandvalet, Y. 2004, JMLR, 5, 1089 — no unbiased estimator of CV variance.
- Bouckaert, R. R., & Frank, E. 2004, PAKDD, LNCS 3056, 3 — replicability of significance tests.
- Bouthillier, X., et al. 2021, MLSys — accounting for variance in ML benchmarks.
- Cawley, G. C., & Talbot, N. L. C. 2010, JMLR, 11, 2079 — over-fitting in model selection.
- Collins, G. S., et al. 2015, BMJ, 350, g7594 — TRIPOD (temporal/external validation).
- Forman, G., & Scholz, M. 2010, SIGKDD Explorations, 12(1), 49 — pooling OOF predictions.
- Gelman, A., & Loken, E. 2013, "The garden of forking paths" (tech. report, Columbia).
- Hastie, T., Tibshirani, R., & Friedman, J. 2009, The Elements of Statistical Learning, 2nd ed., Springer — ch. 7.
- Kapoor, S., & Narayanan, A. 2023, Patterns, 4, 100804 — leakage and reproducibility.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. 2012, ACM TKDD, 6(4), 15 — leakage in data mining.
- Kohavi, R. 1995, IJCAI, 1137 — cross-validation and bootstrap for accuracy estimation.
- Krstajic, D., et al. 2014, J. Cheminformatics, 6, 10 — CV pitfalls in selection and assessment.
- Nadeau, C., & Bengio, Y. 2003, Machine Learning, 52, 239 — corrected resampled t-test / variance.
- Varma, S., & Simon, R. 2006, BMC Bioinformatics, 7, 91 — bias of CV used for model selection.
- Varoquaux, G. 2018, NeuroImage, 180, 68 — small-sample CV error bars.

**Metrics, model selection, intervals**
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. 1984, Classification and Regression Trees — one-SE rule.
- Brodersen, K. H., et al. 2010, ICPR, 3121 — balanced accuracy.
- Clopper, C. J., & Pearson, E. S. 1934, Biometrika, 26, 404.
- Delgado, R., & Tibau, X.-A. 2019, PLoS ONE, 14, e0222916 — against Cohen's κ.
- Demšar, J. 2006, JMLR, 7, 1 — comparing classifiers over multiple data sets.
- Efron, B., & Tibshirani, R. 1993, An Introduction to the Bootstrap, Chapman & Hall.
- Wilson, E. B. 1927, JASA, 22, 209.
- Wyner, A. J., Olson, M., Bleich, J., & Mease, D. 2017, JMLR, 18(48) — RF/boosting as interpolating classifiers.

**Feature selection and importance**
- Au, Q., Herbinger, J., Stachl, C., Bischl, B., & Casalicchio, G. 2022, Data Mining and Knowledge Discovery, 36, 1401 — grouped feature importance.
- Breiman, L. 2001, Machine Learning, 45, 5 — random forests, permutation importance.
- Dormann, C. F., et al. 2013, Ecography, 36, 27 — collinearity.
- Fisher, A., Rudin, C., & Dominici, F. 2019, JMLR, 20(177) — model reliance.
- Gregorutti, B., Michel, B., & Saint-Pierre, P. 2017, Statistics and Computing, 27, 659 — correlation and grouped importance.
- Hooker, G., Mentch, L., & Zhou, S. 2021, Statistics and Computing, 31, 82 — permutation forces extrapolation.
- Kohavi, R., & John, G. H. 1997, Artificial Intelligence, 97, 273 — wrappers; relevance is learner-relative.
- Lundberg, S. M., & Lee, S.-I. 2017, NeurIPS — SHAP.
- Lundberg, S. M., et al. 2020, Nature Machine Intelligence, 2, 56 — TreeSHAP.
- Meinshausen, N., & Bühlmann, P. 2010, JRSS-B, 72, 417 — stability selection.
- Molnar, C., et al. 2022, xxAI workshop, LNAI 13200, 39 — pitfalls of model-agnostic interpretation.
- Peduzzi, P., et al. 1996, J. Clin. Epidemiol., 49, 1373 — events per variable.
- Raudys, S. J., & Jain, A. K. 1991, IEEE TPAMI, 13, 252 — small-sample effects.
- Shah, R. D., & Samworth, R. J. 2013, JRSS-B, 75, 55 — complementary-pairs stability selection.
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. 2007, BMC Bioinformatics, 8, 25.
- Strobl, C., et al. 2008, BMC Bioinformatics, 9, 307 — conditional permutation importance.
- Webber, W., Moffat, A., & Zobel, J. 2010, ACM TOIS, 28(4), 20 — rank-biased overlap.

**Imbalance, weighting, search, early stopping**
- Bergstra, J., & Bengio, Y. 2012, JMLR, 13, 281 — random search.
- Cui, Y., et al. 2019, CVPR — class-balanced loss (effective number of samples).
- Elkan, C. 2001, IJCAI — foundations of cost-sensitive learning.
- He, H., & Garcia, E. A. 2009, IEEE TKDE, 21, 1263 — learning from imbalanced data.
- Jamieson, K., & Talwalkar, A. 2016, AISTATS — successive halving.
- Japkowicz, N., & Stephen, S. 2002, Intelligent Data Analysis, 6, 429 — the class imbalance problem.
- King, G., & Zeng, L. 2001, Political Analysis, 9, 137 — rare-events logistic regression.
- Li, L., et al. 2018, JMLR, 18(185) — Hyperband.
- Platt, J. 1999, in Advances in Large Margin Classifiers, MIT Press — Platt scaling.
- Prechelt, L. 1998, Neural Networks, 11, 761 — early-stopping criteria.
- Probst, P., & Boulesteix, A.-L. 2018, JMLR, 18(181) — number of trees in RF.
- Probst, P., Boulesteix, A.-L., & Bischl, B. 2019, JMLR, 20(53) — tunability.
- Santos, M. S., et al. 2018, IEEE Computational Intelligence Magazine, 13(4), 59 — oversampling before CV.
- Wu, T.-F., Lin, C.-J., & Weng, R. C. 2004, JMLR, 5, 975 — multiclass probabilities by pairwise coupling.
- Zhang, T., & Yu, B. 2005, Annals of Statistics, 33, 1538 — boosting with early stopping.

**Missing data**
- Josse, J., Prost, N., Scornet, E., & Varoquaux, G. 2019, arXiv:1902.06931 — consistency with missing values.
- Twala, B. E. T. H., Jones, M. C., & Hand, D. J. 2008, Pattern Recognition Letters, 29, 950 — missing incorporated in attributes.
- van Buuren, S., & Groothuis-Oudshoorn, K. 2011, J. Stat. Softw., 45(3) — MICE.

**Dataset shift, calibration, open set**
- Benjamini, Y., & Hochberg, Y. 1995, JRSS-B, 57, 289 — FDR.
- Bickel, S., Brückner, M., & Scheffer, T. 2009, JMLR, 10, 2137 — discriminative learning under covariate shift (classifier-based density ratio).
- Brier, G. W. 1950, Monthly Weather Review, 78, 1.
- Chow, C. K. 1970, IEEE Trans. Inf. Theory, 16, 41 — reject option.
- Cochran, W. G. 1954, Biometrics, 10, 417 — chi-square with small expected counts.
- Cortes, C., Mansour, Y., & Mohri, M. 2010, NeurIPS — learning bounds for importance weighting.
- Cramér, H. 1946, Mathematical Methods of Statistics, Princeton.
- Gneiting, T., & Raftery, A. E. 2007, JASA, 102, 359 — proper scoring rules.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. 2017, ICML — calibration of modern networks.
- Kish, L. 1965, Survey Sampling, Wiley — effective sample size.
- Kuncheva, L. I., & Whitaker, C. J. 2003, Machine Learning, 51, 181 — diversity in ensembles.
- Lipton, Z., Wang, Y.-X., & Smola, A. 2018, ICML — black-box shift estimation.
- Lopez-Paz, D., & Oquab, M. 2017, ICLR — classifier two-sample tests.
- Naeini, M. P., Cooper, G. F., & Hauskrecht, M. 2015, AAAI — expected calibration error (ECE).
- Niculescu-Mizil, A., & Caruana, R. 2005, ICML — predicting good probabilities.
- Rabanser, S., Günnemann, S., & Lipton, Z. C. 2019, NeurIPS — "Failing Loudly".
- Saerens, M., Latinne, P., & Decaestecker, C. 2002, Neural Computation, 14, 21 — adjusting to new priors.
- Scheirer, W. J., et al. 2013, IEEE TPAMI, 35, 1757 — open-set recognition.
- Shimodaira, H. 2000, J. Stat. Plan. Inference, 90, 227 — covariate shift and importance weighting.
- Siddiqi, N. 2006, Credit Risk Scorecards, Wiley — PSI cut-offs.
- Sugiyama, M., Krauledat, M., & Müller, K.-R. 2007, JMLR, 8, 985 — importance-weighted CV.
- Yurdakul, B. 2018, PhD thesis, Western Michigan University — statistical properties of the PSI.

**Hierarchical classification and shrinkage**
- DerSimonian, R., & Laird, N. 1986, Controlled Clinical Trials, 7, 177 — moment estimator of between-group variance.
- Dumais, S., & Chen, H. 2000, SIGIR — hierarchical classification, multiplicative scoring.
- Efron, B., & Morris, C. 1975, JASA, 70, 311 — Stein's estimator.
- Gelman, A., et al. 2013, Bayesian Data Analysis, 3rd ed., CRC — ch. 5.
- Silla, C. N., & Freitas, A. A. 2011, Data Mining and Knowledge Discovery, 22, 31 — hierarchical classification survey.
