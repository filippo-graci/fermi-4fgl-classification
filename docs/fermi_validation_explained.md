# `fermi_validation.py` — What It Does and Why

## What this module is for

This is Module 4 of a four-module pipeline that classifies gamma-ray sources from the Fermi-LAT 4FGL-DR4 catalog into physical classes (blazars, pulsars, supernova remnants, binaries, etc.). The first three modules do the actual model *training*:

- **Module 1** (`fermi_processing.py`) — loads the catalog, engineers features, and produces multiple independent train/test **splits** (so later stages can measure how much a result depends on which rows happened to land in the test set).
- **Module 2** (`fermi_fine.py`) — trains a **flat/fine** classifier: one model that predicts the fine-grained class directly (e.g. `bll`, `fsrq`, `psr`, `nov`, ...).
- **Module 3** (`fermi_merged.py`, 2-level, and `fermi_3levels.py`, 3-level) — train **hierarchical** classifiers. Instead of one flat model, these route a source through a small tree of models: first decide its broad superclass (e.g. "is this a blazar or something else"), then a second (or third) model decides the fine class within that branch. The 2-level version has one router; the 3-level version splits the routing further (a BLAZAR/OTHER gate, then a blazar-subtype subgate and a non-blazar router), because blazars and non-blazars behave very differently as a classification problem and separating them earlier can reduce confusion.

Each of Modules 2 and 3 does its **own internal model search** (which algorithm, which features, which hyperparameters) and leaves behind a "leaderboard" of candidates it screened, plus the single best candidate's artifacts, for every split it processed.

**Module 4 — this file — does not train anything new from raw features.** Its job is to take those already-screened candidates from all three architectures, put them through a rigorous, leakage-free selection and evaluation procedure, and answer three separate questions honestly instead of conflating them:

1. **Which specific configuration (which model, which features, which hyperparameters) should represent each of the three architecture families?** — a *selection* problem.
2. **Once selected, how good is each family really, with an honest uncertainty estimate?** — an *estimation* problem.
3. **What model do we actually ship, and what does it predict for the ~thousands of Fermi sources that have no known class yet?** — a *deployment* problem.

The file is explicit that these are three different questions requiring three different data slices, and it enforces that separation mechanically (never letting a stage that touches test data feed back into a stage that picks the winner). This is the single most important design idea in the file, and almost everything else exists in service of it.

---

## Why this separation matters (the core statistical problem)

If you screen 50 candidate models against the same validation data and report the best one's validation score as its expected real-world performance, that score is optimistically biased — you didn't just measure performance, you also let the validation set tell you which model to pick, so some of what looks like "skill" is actually the model that got lucky on that particular data. This is sometimes called the **winner's curse** or **selection bias**, and it gets worse the more candidates you compare.

The file's answer is a five-stage funnel with a hard rule: **only Stages 1–4 are allowed to look at validation-type data to pick a winner; Stage 5 onward touches test data that no selection step has ever seen, and only to *measure*, never to *choose*.**

- **Stage 1 (Section 4)** — coarse shortlist per family, using only split 0.
- **Stage 2 (Section 5, decoupled)** — an optional diagnostic that explicitly quantifies how big this winner's-curse gap actually is, run separately and never fed into selection.
- **Stage 3 (Section 6)** — re-scores the Stage 1 shortlist across every discovered split's *training* data (still no test data).
- **Stage 4 (Section 7)** — the final architecture is picked, purely from Stage 3's cross-split training-side scores.
- **Stage 5 (Section 8)** — *now, for the first time*, each family's chosen architecture is evaluated on genuine held-out test partitions — five independent ones, one per split — that nothing above this line has ever touched.
- **Stage 6 (Section 9)** — is the difference between families statistically real, given Stage 5's numbers?
- **Stage 7 (Section 10)** — a separate, final refit on more data than any evaluation used, for actual deployment. This model is *never itself re-evaluated* — its expected quality is whatever Stage 5 already measured for that architecture.

Section 11 encodes this explicitly as a `"three_way_role_disclosure"` string written into the final JSON report, so that anyone consuming the report later can't accidentally treat Stage 1–4 numbers as generalization estimates or Stage 7's model as validated on its own.

---

## Why five splits, not one

A single train/test split answers "how did this model do on *this* random partition of the data." Because 4FGL sources are non-i.i.d. in ways that matter (rare classes, spatial/Galactic-latitude structure, etc.), a single split's score can be noisy. Module 1 produces `N_VALIDATION_SPLITS=5` independent splits (different random partitions, same underlying pool of associated sources), and this file uses all 5 for anything that claims to be a genuine performance *estimate* (Stage 3, 5, 6, 14), while Stage 1's cheaper first-pass shortlist deliberately uses only split 0, for cost reasons (explained below).

---

## Fixed comparison ground: "Setup B"

Fine-family classification can be run under different label setups (with vs. without the ambiguous `bcu` "blazar candidate of uncertain type" class). The file hardcodes every stage to **fine Setup B** (`bcu` excluded), because that's the only setup where the fine, merged, and 3-level architectures share the *exact same* class taxonomy and labeled-row pool. Comparing across architectures under different class cardinalities would make every metric in this file (accuracy, macro-F1, etc.) not apples-to-apples, so the file refuses to let that happen by never re-deriving the setup per family.

---

## Section-by-section walkthrough

### Section 0 — Config, constants, output layout, checkpointing

Sets up the directory tree under `outputs_{VERSION_TAG}_{SELECTION_METRIC}/validation/` (logs, results, plots, deployed models, checkpoints, CSV reports), and — importantly — a **per-stage checkpointing system** (`_run_cached`, `_load_checkpoint`, `_save_checkpoint`, `_stage_effectively_forced`).

**Why checkpointing matters here specifically:** this file can run for a long time (chain-CV refits across 5 splits × dozens of candidates × multiple repeats). A checkpoint per named stage (`"stage1"`, `"stage3"`, ... `"stage9d_shrinkage"`) means a crash, a Ctrl-C, or a deliberate stop partway through never throws away work that already finished — the next run just reloads whatever's on disk and continues. `STAGE_DEPENDENCIES` encodes which stages must recompute if an upstream one is forced to recompute (e.g. forcing `stage1` also invalidates `stage3` onward), so you can't accidentally read a stale checkpoint that's now inconsistent with an upstream change. Two stages are *deliberately excluded* from this cascade: Stage 2 (the winner's-curse diagnostic) and, within the automatic driver, nothing forces or is forced by it — because it's a side diagnostic, not part of the selection chain, so forcing it shouldn't force a re-run of the actual selection/estimation pipeline, and vice versa.

`_val_discover_splits` reads Module 1's `multi_split_manifest.json` to find out how many splits actually exist on disk, rather than assuming 5 — if fewer than 5 were generated, the file proceeds with what exists but logs a warning, since Stage 3/5/6's statistics need genuinely independent draws to mean what they claim to mean.

`QUALITY_METRICS`/`REPORT_METRICS`, the Borda-ranking machinery, and the cost-control flags declared here are used throughout the rest of the file — they're described in context below.

---

### Section 1 / 1a / 1b — Re-ported chain-CV helpers and the node-fit cache

**The problem these solve:** to score a *hierarchical* candidate (merged or 3-level) under cross-validation, you can't just call `.fit()`/`.predict()` once — you have to fit a small tree of models (router + per-branch models, or gate + subgate + nonblazar-router + per-branch models) on each CV fold's training rows and route each held-out row through that tree to get its final fine-class prediction. That fitting-and-routing logic already exists inside `fermi_merged.py` and `fermi_3levels.py`, but this file can't just import and reuse it as-is, for two separate reasons:

1. **A name collision in the shared kernel.** Both sibling modules define bare top-level functions with the same names (`build_branch_datasets`, `_fit_and_route_chain_fold`, etc.) in the shared execution namespace this file also runs in. Because `fermi_3levels.py` runs last, its versions "win" — so if this file used the bare names, it would silently get 3-level's chain-routing logic when it meant to call merged's. The fix: this file carries its own faithful copies of merged's versions, prefixed `_m2_*` (Section 1), and uses the live bare names only for 3-level.
2. **A cost-control feature the sibling modules don't have.** This file adds a **node-fit cache** (`node_cache`, Section 1a) that the original functions don't support, so even 3-level's bare functions need a locally-patched copy (`_l3_fit_and_route_chain_fold_cached`, Section 1b) that plumbs the cache through.

**Why the node-fit cache exists and why it's provably safe (Section 1a):** `fermi_merged.py`'s Stage-1 candidate grid is `TOP_N_ROUTERS(3) × TOP_N_BRANCH_COMBOS(5) = 15` candidates built from only 3 distinct router configs and 5 distinct branch-combo configs. `fermi_3levels.py`'s grid is `3×3×3=27` candidates from only 9 distinct node configs, all sharing one fixed branch set. Scoring every candidate independently — as a naive implementation would — refits every shared node from scratch once per candidate that happens to reuse it, which is enormously wasteful. The key insight that makes caching safe rather than an approximation: `StratifiedKFold(n_splits=..., shuffle=True, random_state=seed)`'s fold boundaries depend **only** on `(labels, n_splits, seed)` — never on a candidate's own model/tag/PCA/scaler/weighting configuration. So for a fixed `(context, seed)`, two different candidates that happen to specify the identical `(model_name, tag, pca_variance, scaler_name, weighting)` at a given node will produce a byte-for-byte identical fit on a given fold. `_cfg_key` turns a config dict into a hashable identity, `_cached_fit` fits-and-stores on first request and returns the cached model on repeats, and `node_cache=None` (the default for every single-candidate caller like Stage 5/7/9) degrades to "always fit," reproducing the pre-cache behavior exactly. This is disclosed explicitly as a "free lunch": it changes nothing about *what* is fit or scored, only how many times redundant work is repeated.

`_m2_predict_proba_safe` exists because some backends (GPU/cuML builders) can return `predict_proba` results as DataFrame-like objects rather than plain NumPy arrays; this coerces them uniformly before any positional column indexing, avoiding a class of subtle bugs where column *order* vs. column *label* get confused.

`_m2_build_branch_datasets` guards against a specific bug: a branch superclass with no branch-local feature selection (`local_meta is None`) never got an `"ALL_FEATURES"`/`PCA_BASE_TAG` entry in its per-branch feature dict, because `ctx.tags` excludes `"ALL_FEATURES"` by construction. Any branch candidate whose model used `tag=PCA_BASE_TAG` would then crash with a bare `KeyError: 'ALL_FEATURES'` deep inside routing. The fix adds that entry explicitly whenever it's missing.

---

### Section 2 — Metric primitives, Borda ranking, and end-to-end architecture naming

This section is the shared machinery every later selection stage calls into, so it's worth understanding in detail.

**`QUALITY_METRICS` and why plain accuracy is back.** The four ranking criteria are `accuracy`, `bal_accuracy` (balanced accuracy — averages per-class recall, so it isn't dominated by whichever class has the most rows), `macro_f1` (unweighted mean of per-class F1), and `kappa` (Cohen's kappa — agreement corrected for the agreement you'd expect by chance alone). Under class imbalance — which 4FGL certainly has (a handful of `nov`/`gc`/`sey`-type sources vs. thousands of blazars) — plain accuracy is a poor headline metric because a model that always predicts the majority class scores deceptively well. Versions 12/13 of this pipeline therefore treated accuracy as diagnostic-only, never a ranking criterion. Version 14 reverses that, per explicit instruction, and folds accuracy in as a fourth, equally-weighted Borda criterion — but keeps every other criterion (balanced accuracy, macro-F1, kappa) exactly as before, so the imbalance-robust metrics still dominate 3-out-of-4 votes.

**`REPORT_METRICS`** is a strict superset — `QUALITY_METRICS` plus `precision_macro`/`recall_macro` — used only for CSV/report columns, never for ranking. This split exists so the wider "give me every metric" reporting request doesn't require re-deriving predictions a second time or, more importantly, doesn't accidentally leak two more metrics into what gets *selected*.

**Borda ranking (`_borda_rank`), and why rank-averaging instead of averaging raw metric values.** Given several candidates, each scored on several criteria, `_borda_rank` converts each candidate's value on each criterion into a **rank** (fractional ranks for ties, following Demšar 2006), then averages ranks across the 4 quality criteria plus a 5th criterion, the **train–validation overfitting gap** (`_overfit_gap` — mean, over `QUALITY_METRICS`, of `max(0, train_metric - val_metric)`, clipped at 0 so a model that generalizes *better* than its own training score is never penalized for that). The averaged rank (`borda_score`, lower is better) is the final ranking key.

Why not just average the raw metric values (a weighted sum)? Because these metrics live on different, non-comparable scales — kappa is chance-corrected and can be negative, accuracy is not balanced-class-aware, and all four are mutually correlated by construction. A naive weighted average would silently let whichever metric happens to have the widest numeric spread dominate the ranking. Rank-averaging is scale-free: it only asks "how does this candidate compare to its peers on each criterion," not "how big is the number," so no single metric's arbitrary scale can dominate.

**End-to-end architecture naming (`build_end_to_end_name`, Section 2b).** A hierarchical winner isn't just "RandomForest" — it's a whole tree of model choices (e.g. a `RandomForest` router feeding an `SVM` branch for high-energy blazars, an `RF` branch for low-energy blazars, etc.). This section gives every candidate a compact, human-readable **contracted name** (short abbreviations joined by the routing structure, e.g. `RF->{BHE:SVM,BLE:RF,...}`) for display and CSV columns, plus a `node_models` dict recording the **full, uncontracted** model name at every individual node — the contraction is cosmetic, the dict is the source of truth for "which exact model ran at node X." Both are attached to a candidate the moment it's built (`_attach_end_to_end_name`) and carried through every later stage unchanged — they are never recomputed downstream, always read off whichever row was ranked or selected, so there's no risk of the display name and the actual fitted models drifting apart.

**Per-candidate CV scoring wrappers (`_val_score_fine_candidate`, `_val_score_merged_chain_candidate`, `_val_score_3level_chain_candidate`).** Each takes one candidate configuration and returns `{selection_mean, n, val_metrics, train_metrics}` — the pooled out-of-fold `SELECTION_METRIC` score plus the full metric quartet on both the held-out folds (`val_metrics`) and, by refitting once on the whole training set, the training data itself (`train_metrics`, for the overfit-gap criterion). One documented limitation: for the fine family, `precision_macro`/`recall_macro` on the validation side are left as `NaN` rather than fabricated, because the underlying `cv_score_estimator` (defined in the shared kernel, out of scope to modify here) only returns the aggregated quartet, not the raw out-of-fold predictions those two extra metrics would need — getting them would require re-running the whole CV a second time just for two extra columns, which the file declines to do; it leaves them honestly missing instead of silently omitting the columns (so every family's CSV has the same column set).

**Repeated scoring with a shared cache (`_val_repeated_score`, `_val_repeated_score_shared_cache`).** A single CV pass at one seed is itself noisy (which rows land in which fold matters). Both Stage 1 and Stage 3 therefore repeat the scoring several times at different seeds and average. `_val_repeated_score_shared_cache` is the version used for merged/3level sweeps: it puts the *repeat-seed* loop on the outside and the *candidate* loop on the inside, so every candidate scored at a given repeat's seed shares one `node_cache` — this is the same free-lunch optimization from Section 1a, just applied across the repeated-CV loop instead of within a single pass. The fine family has no shared sub-components to deduplicate, so it keeps using the plain, uncached `_val_repeated_score`.

---

### Section 3 — Loading persisted per-split contexts and leaderboards

Rebuilds, for each split, the three families' in-memory "context" objects (feature frames, label arrays, encoders, branch metadata) by calling each sibling module's own bare context-builder function (`build_fine_context`/`build_merged_context`/`build_3level_context`) against Module 1's persisted per-split exports. It also parses each sibling module's persisted leaderboard/metrics JSON files into candidate configuration dicts.

**The `_nan_to_none` bugfix, and why it matters.** A leaderboard candidate that used no PCA and no scaler round-trips through CSV or JSON as a blank cell / `NaN`, not as Python `None`. Downstream code checks things like `pca_variance is not None`, and `PCA(n_components=nan)` doesn't fail that check — it raises a cryptic `InvalidParameterError` instead. `_nan_to_none` normalizes every `{pca_variance, scaler_name}` pulled off a leaderboard row or a persisted metrics JSON block through one single choke point. The markdown documents a real, previously-shipped bug here: `_leaderboard_row_to_cfg` (the CSV-sourced ingestion path) was already guarded, but `_branch_cfgs_from_metrics` (the JSON-sourced path used to build every 3-level Stage 1 candidate's fixed non-blazar branch configuration) was not — meaning a branch model with no PCA would crash *every single* 3-level Stage 1 candidate, since that fixed branch-config set is shared across all `TOP_N_GATES × TOP_N_BLAZAR_SUBGATES × TOP_N_NONBLAZAR_ROUTERS` combinations. Moving `_nan_to_none` earlier in the file and applying it at both ingestion points fixed this. Merged was never exposed to this specific bug because its own branch-variant path goes through the already-guarded `_leaderboard_row_to_cfg` instead.

---

### Section 4 — Stage 1: coarse shortlist per family [selection, split 0 only]

For each of the three families, builds a shortlist of candidate architectures from split 0's already-persisted leaderboards:

- **fine**: top `TOP_K_SHORTLIST=15` rows of the fine leaderboard by validation score.
- **merged**: top `TOP_N_ROUTERS=3` router candidates × `TOP_N_BRANCH_COMBOS=5` independently-built branch-combo variants = up to 15 whole-system candidates.
- **3level**: top `TOP_N_GATES × TOP_N_BLAZAR_SUBGATES × TOP_N_NONBLAZAR_ROUTERS = 3×3×3 = 27` gate/subgate/router combinations, each paired with one *fixed* set of non-blazar branch models (taken directly from the persisted metrics, not re-searched — 3-level's own branch search already happened inside `fermi_3levels.py`).

Each shortlisted candidate is then **repeated-CV rescored** — `N_REPEATS_SHORTLIST_CV=5` independently-reshuffled `N_CONFIRMATION_FOLDS`-fold passes — using the shared node-fit cache for merged/3level. This produces, per candidate, a `stage1_mean`/`stage1_std` and the full metric quartet with its own cross-repeat error bars. The whole shortlist per family is then Borda-ranked.

**Why split 0 only, and why this two-tier design (cheap Stage 1, expensive Stage 3) exists at all:** doing a full cross-split confirmation (Stage 3, below) against *every* candidate a sibling module's leaderboard produced would be extremely expensive, especially for merged/3level where each candidate score requires a full chain-CV routing fit. Stage 1 uses one split's data to cheaply narrow a potentially large candidate pool down to a short, defensible list; Stage 3 then spends the expensive cross-split budget only on that short list. This is a standard coarse-then-fine model-selection pattern, and it's exactly why Stage 1 is explicitly *not* treated as a generalization estimate anywhere downstream — it's a triage step, not a measurement.

---

### Section 5 — Stage 2: winner's-curse diagnostic [definition only, decoupled]

Defines (but does not automatically run) a nested-CV diagnostic that directly quantifies the winner's-curse effect described above: it re-runs fine's entire Stage-0/1/2-style selection procedure inside `N_NESTED_DIAG_REPEATS × N_OUTER_NESTED` outer folds, each time doing an *inner* model search on the outer-training rows and then scoring the inner winner on outer-*held-out* rows the inner search never saw. The gap between "naive" (best inner-search score) and "corrected" (that same winning config's score on truly held-out outer rows) is the empirical size of the optimism this file is designed to avoid everywhere else. It's expensive (it reruns the entire coarse-search funnel `N_NESTED_DIAG_REPEATS × N_OUTER_NESTED` times) and is a standalone diagnostic that nothing else consumes, so it is kept out of the automatic driver and its cost is paid only when you explicitly ask for it (Section 17).

---

### Section 6 — Stage 3: cross-split confirmation of the shortlist [selection, all splits, train only]

Takes Stage 1's shortlist (optionally trimmed — see `REDUCE_STAGE3_CANDIDATES` below) and re-scores every candidate on **every discovered split's training data**, once per split, with a plain single-seed `StratifiedKFold`. Unlike Stage 1, this never touches any split's test partition — it's still purely a selection-side measurement, just now checked for consistency across 5 independently-drawn splits instead of relying on split 0 alone.

For merged/3level, the loop structure is **split-outer, candidate-inner**, with one shared `node_cache` per split — the same free-lunch reasoning as Stage 1, just organized around splits instead of repeats, since fold boundaries at a given split's seed don't depend on candidate config either.

**Cost control (`REDUCE_STAGE3_CANDIDATES`, `STAGE3_SHORTLIST_TOP_K=10`):** Stage 3 is the most expensive selection stage (merged/3level candidates each need a full chain-CV routing fit, repeated across every split). Since Stage 1's shortlist is already Borda-sorted best-first, `_stage3_shortlist` trims to just the top 10 before this expensive step, when the flag is on (default). This is disclosed as a real, accepted tradeoff, not a free optimization: Stage 1's single-split ranking is itself noisy, so a candidate that would have overtaken the top 10 once genuinely confirmed across splits can be silently dropped here. It's explicitly *not* a reduction in the combo-grid's width, repeat count, or fold count — those stay at full resolution; it's purely a breadth trim on an already-ranked list, and can be disabled entirely by setting the flag `False`.

Each row records both the aggregated cross-split mean/std *and* the raw `per_split_val_metrics`/`per_split_train_metrics` dicts (keyed by split ID) — needed later by Section 12's "with error" CSV columns and Section 14's confidence-interval report, so those don't need to recompute anything, just re-aggregate what's already here.

A candidate is only **eligible** to win if it produced a finite score on at least `STAGE3_MIN_SPLITS_ELIGIBLE=4` of the 5 splits (a candidate that crashes on most splits shouldn't be selectable just because it happened to do well on the one split it survived).

---

### Section 7 — Stage 4: final architecture selection [selection]

For each family, filters Stage 3's rows to the eligible ones, Borda-ranks them by the same 5 criteria (4 quality metrics + overfit gap), and picks the top-ranked candidate — **provided** it has valid scores on all `STAGE3_MIN_SPLITS_SELECTED=5` splits. If the top-ranked candidate doesn't have full split coverage, the file falls back to the best-ranked candidate that does, logging a warning, rather than silently selecting a winner whose "win" might be an artifact of having fewer, easier splits to be scored on. This is the point in the pipeline where each family's single representative architecture is finally locked in — everything from here on evaluates or deploys that one chosen configuration, never re-opens the search.

---

### Section 8 — Stage 5: five genuine test-set evaluations [estimation]

This is where the file crosses the line from selection into estimation. For each family's Stage-4 winner, and for **each** of the 5 splits independently: refit the winning architecture on that split's full training data, then evaluate it **once** on that split's held-out test partition — data that no stage above this line (Stage 1, 2, 3, or 4) has ever seen. Doing this across 5 splits (rather than once) gives 5 independent estimates of test performance, which Section 9 uses for a proper significance test and Section 14 turns into a small-sample confidence interval.

The heavy lifting is in `_fit_predict_merged_core`/`_fit_predict_3level_core` — shared fit-and-route logic used by both the test-side and (for the overfit-gap computation) the train-side evaluators, so the routing math is written once, not duplicated. Both gained a `return_proba` option in this version, which — instead of just the hard predicted label — also returns the full soft-voted class-probability matrix `p_fine`. This is what later lets Section 15 (confidence scoring) and Section 16b (calibration) reuse this exact code path rather than re-implementing the chain-routing math a third time.

`_score_by_name(SELECTION_METRIC, ...)` gives the primary scalar per split; `_repeated_cv_ci` turns the 5 per-split scores into a mean, std, and Student-t 95% confidence interval (small-sample-appropriate, since n is at most 5).

---

### Section 9 — Stage 6: family-vs-family significance testing [estimation]

Answers "is one family's Stage 5 test performance *actually* better than another's, or is the difference just split-to-split noise?" Three architecture pairs are compared: fine-vs-merged, fine-vs-3level, merged-vs-3level.

**Primary test — Nadeau-Bengio corrected paired t-test (`nadeau_bengio_corrected_ttest`).** A naive paired t-test on cross-validation fold differences understates variance, because CV folds share overlapping training data and aren't truly independent draws. The Nadeau-Bengio correction inflates the variance estimate by a `(1/n + n_test/n_train)` factor to account for this, giving a less overconfident p-value than a plain t-test would.

**Supplementary tests, computed only on split 0** (since they need row-aligned predictions, and only split 0's predictions are convenient to compare directly): an exact **McNemar test** (`mcnemar_exact` — compares the two families' *disagreements* on the same held-out rows, using an exact binomial test on the discordant pairs) and a **class-stratified paired bootstrap** (`stratified_paired_bootstrap` — resamples within each class to preserve the class distribution, then looks at how often the sign of the score difference flips).

**New in this version — effect size and multiple-comparison correction.** Two additions, both because comparing 3 pairs from the same 5 splits' test partitions is *not* 3 independent comparisons — reading all 3 p-values against α=0.05 uncorrected would inflate the family-wise false-positive rate.
- **Cohen's d (`cohens_d_paired`)** on the same 5 split-level differences the Nadeau-Bengio test uses — lets you separate "statistically significant" from "practically large," since with only n=5 splits a significant result can still be a small effect, and a non-significant one might simply lack power rather than reflect no real difference.
- **Holm-Bonferroni correction (`holm_bonferroni`)** — a step-down multiple-comparison procedure, less conservative than plain Bonferroni but still exact, applied jointly across the 3 pairwise Nadeau-Bengio p-values to control the family-wise error rate.

---

### Section 10 — Stage 7: deployment refit [deployment]

Refits each family's Stage-4 architecture one final time, on **split 0's train+test rows pooled together** — i.e. on more labeled data than any evaluation step above used. This is the model that would actually be shipped. Critically, **this model is never itself re-evaluated** — the file is explicit that its expected performance is whatever Stage 5 already measured for that architecture, not something newly (and optimistically) re-derived from this refit.

Why split 0 pooled, rather than pooling across all 5 splits? The markdown calls this "the simpler, design-doc-sanctioned alternative" — pooling across all 5 splits would require reconciling different splits' independently-re-selected feature-name sets (since feature selection can differ split to split), which is a materially harder problem the file deliberately avoids.

Every deployed `.joblib` payload (`fine_deployed.joblib`, `merged_deployed.joblib`, `3level_deployed.joblib`) bundles everything needed for later inference: the fitted model(s), the label encoder(s), feature metadata, and — new in this version — the `full_model_name`/`node_models` identity from Section 2b, so later stages (Section 15's inference, Section 18's SHAP) never have to re-derive what architecture they're looking at from raw configs.

---

### Section 11 — Stage 8: persistence / report schema

Aggregates everything Stages 1–7 produced into one JSON report (`cross_pipeline_validation_report.json`). Notably, it also runs its **own** Borda ranking — this time *across the 3 families* rather than within one family's candidates — using Stage 5's genuine test-set metrics, to pick the single `corrected_winner_family`. This is explicitly named "corrected" to underline that it's sourced from Stage 5 (real, held-out test performance), not from Stage 1's split-0-only shortlist score or Stage 2's diagnostic.

The report embeds three long, human-readable disclosure strings as first-class fields, not just comments: `three_way_role_disclosure` (selection vs. estimation vs. deployment, as described above), `ranking_methodology_note` (why rank-averaging rather than raw-value weighting, and the accuracy-inclusion policy change), and `cost_control_note` (the node-fit cache and Stage 3 shortlist trim). Embedding these in the machine-readable report means anyone consuming it later — including a future version of this pipeline — has the methodology caveats attached to the numbers, not living only in source comments that a report consumer might never read.

---

### Section 12 — CSV report export [reporting]

Three CSV deliverables per family, built entirely from data Stages 1/3/4 already computed (no new fitting):

1. **`all_models_validated_{family}.csv`** — every Stage-3 cross-split-scored candidate (not just the eventual winner), with every `REPORT_METRICS` column's cross-split mean *and* std on both train and validation sides, its contracted architecture name, and its Borda score. This is the "show me everything that was tried, with error bars" table.
2. **`per_node_leaderboards/*.csv`** — one CSV per individual hierarchy node (the router; each branch; or the gate, blazar-subgate, non-blazar-router, and each branch for 3-level), dumped straight from the sibling modules' own node-level leaderboards — i.e. the screening that happened *before* Stage 1 ever combined nodes into whole-system candidates.
3. **`end_to_end_performance_{family}.csv`** — a slimmer, presentation-ready version of (1), keyed by the contracted architecture name, with an `is_selected_winner` flag marking which row Stage 4 actually chose (matched via `_candidate_identity`, a config-based identity key — needed because a row reloaded from a checkpoint is a different Python object than the one Stage 4 returned in memory, even when it represents the same candidate).

---

### Section 13 — Driver

The linear script that actually runs the whole pipeline in order — build per-split contexts, then Stages 1, 3, 4, 5, 6, 7, 8, 8b — each wrapped in `_run_cached` so a re-run picks up from wherever a valid checkpoint left off. **Stage 2 is deliberately not called here** — it's available only via the explicit, manual Section 17. This keeps the default run of this file limited to the actual selection/estimation/deployment chain and skips the most expensive diagnostic unless someone asks for it on purpose.

---

### Section 14 — Full train/validation/test metric report, with confidence intervals [reporting]

Builds one wide table (and CSV) with three parallel views of every `REPORT_METRICS` column for each family's chosen architecture:

- **train** — Stage 5's refit, evaluated on the *same* data it was fit on. This is explicitly a **memorization diagnostic**, not a generalization estimate — a model can trivially score high here by overfitting.
- **validation** — Stage 3's cross-split, held-out-fold CV on training rows only (never touching test). This is what Stage 4 actually used to pick the winning architecture.
- **test** — Stage 5's genuine held-out test partitions, 5 splits, never touched before Stage 5.

**Why both a std ("error") and a Student-t confidence interval are reported, and why this specific CI construction** (`_metric_ci_across_splits`): every number in this table is a mean over up to 5 *independent* validation splits — not repeated resamples of one fixed split — so the standard error is `std-across-splits / sqrt(n)`, and with n this small a normal-approximation interval is unreliable; a Student-t interval with `df = n-1` is used instead (the same math `_repeated_cv_ci` already used for the single `SELECTION_METRIC` scalar in Stage 5/6, now applied to every metric column). The std and the CI answer genuinely different questions and neither substitutes for the other: std describes how much one split's score varies, while the CI describes how precisely the *mean* across splits is known. The markdown is also explicit about a road *not* taken: this is deliberately not the bootstrap-CI style used elsewhere in the pipeline for pooled out-of-fold predictions — bootstrap-resampling one fixed set of predictions answers "how much would this vary if I resampled these same rows," which is a different and, for a design with 5 genuinely independent splits, less relevant question than "how consistent was performance across truly different splits."

---

### Section 15 — Stage 9: unassociated-source inference [reporting, deployment]

This is the actual scientific payoff of the whole pipeline: applying the Stage-7 deployed models to the thousands of Fermi sources that have **no known class** (`CLASS1` is unknown/blank in the catalog), to produce candidate classifications for follow-up.

Loads the unassociated-source catalog (checked against several candidate paths, since Module 1's data layout has moved between versions), then, per family, produces predictions plus a **confidence score**:

- **fine** — top-1 `predict_proba` value.
- **merged** — `min(router_confidence, branch_confidence)` — the weakest link in the two-stage chain.
- **3level** — `min(gate_confidence, superclass_confidence, branch_confidence)` — weakest link across all three stages.

Using the *minimum* across the chain (rather than, say, the product) means a source is only called "confident" if every stage that had to make a decision about it was confident — a source can't look falsely confident just because one stage was very sure while another stage further down the chain was actually guessing.

**New in this version:** a per-family **confidence-threshold summary** (one row per value in `CONFIDENCE_SUMMARY_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]`, so "how many sources clear ≥0.8 confidence" is a single table lookup rather than a fresh filter every time), a **high-confidence subset CSV** (everything above `HIGH_CONFIDENCE_EXPORT_THRESHOLD=0.80`), and a **cross-family agreement table** comparing all three families' predicted label for the same unassociated sources, flagging where they disagree. This last one is called out as "the single most useful signal for picking which unassociated sources deserve a human look" — a source where three architecturally different models independently agree is a far stronger candidate for follow-up than one where they split three ways, since agreement across genuinely different model structures is harder to get by chance than agreement within one model family's own noise.

---

### Section 16 — Per-class metrics export [reporting]

The macro-averaged metrics used everywhere else hide *which specific classes* are dragging a low macro score down. This section computes precision/recall/F1/support **per fine class** for each family's Stage-5 winner, mean ± std pooled across the 5 validation splits, reusing Stage 5's already-cached predictions (no refitting). Sorted worst-F1-first, this is the table you'd actually read to find out whether, say, novae (`nov`) or Seyfert galaxies (`sey`) are the specific classes a given architecture struggles with, rather than just knowing the overall macro-F1 is mediocre.

---

### Section 16b — Calibration / Brier-score check [reporting, opt-in]

Confidence scores (Section 15) are only meaningful as a filter if the model is actually **calibrated** — i.e., among sources predicted at ~0.8 confidence, roughly 80% really should be correct. This section checks that directly on the labeled test partitions: a **multiclass Brier score** (mean squared distance between the predicted probability vector and the one-hot true label; 0 is a perfect, fully-confident-and-correct forecaster, 2 is the worst possible) per split, plus a **10-bin reliability table** (mean predicted confidence vs. empirical accuracy within that confidence bin) — the reliability table is what actually answers "is 0.8 confidence really ~80% correct for this family," since Brier score alone is a single summary number that can't distinguish over- from under-confidence.

This section is **opt-in** (`RUN_CALIBRATION_CHECK_NOW`, default `False`), unlike Sections 16/16c, because unlike those two — which only re-read Stage 5's already-cached predictions — this one genuinely **refits** Stage 4's winning architecture from scratch, once per family per split (3 × 5 = 15 refits, with merged/3level each being a full chain-CV routing fit). It was, as of the previous version, the one section past Stage 5 still silently wired into the automatic driver despite being this expensive; it's now made explicit opt-in, matching the pattern already used for Stage 2 and SHAP, after confirming nothing downstream (Stage 8's report, Section 17/18) actually reads its output — so skipping it costs nothing.

---

### Section 16c — Rare-class hierarchical partial-pooling shrinkage [reporting-only]

This section implements a specific, previously-proposed statistical fix for a real problem: some fine classes (novae, globular clusters, certain Seyferts) have so few sources in the whole catalog that their raw per-class recall (Section 16's `k/n`) is a nearly meaningless number — with, say, 8 total novae in 4FGL-DR4, a recall of "2 out of 3 held out correctly classified" swings wildly depending on which 3 happened to be held out.

**The method — Beta-Binomial empirical-Bayes shrinkage.** Instead of reporting each rare class's raw `k/n` recall in isolation, this treats it as a noisy draw from a Beta prior distribution whose mean and concentration (how tightly classes cluster around that mean) are estimated from the class's *sibling* classes, then combines the raw count with that prior into a shrunk posterior estimate plus a credible interval. This is the closed-form, fast alternative to a full Bayesian MCMC fit — a "James-Stein-type" shrinkage estimator, in the sense that it pulls noisy individual estimates toward a more stable group mean, with the amount of pulling determined automatically by how much evidence (`n`) each class has.

**Grouping mirrors each architecture's own taxonomy, never invented ad hoc:**
- **fine** — 1 level: every fine class shrinks toward one grand mean (there's no natural sub-grouping in a flat classifier beyond the taxonomy itself).
- **merged** — 2 levels: fine class nested inside its `SUPERCLASS_ORDER` group (so a rare class like `nov` shrinks toward `BINARY_SYSTEMS`'s own already-shrunk mean, not straight to a grand mean spanning pulsars and blazars alike — matching the actual 2-level routing structure).
- **3level** — 3 levels: fine class nested inside superclass nested inside the gate's BLAZAR/OTHER split, matching `fermi_3levels.py`'s own routing exactly.

**Handling thin groups (`_mom_beta_concentration`).** Estimating a group's own between-class variance needs enough member classes to be trustworthy; below `SHRINKAGE_MIN_GROUPS_FOR_OWN_VARIANCE=3` members, the concentration is instead **borrowed** from the 7-superclass level (the level with the most groups, hence the most stable variance estimate), rather than trusting a near-worthless 1- or 2-point variance estimate. The 3-level gate (only 2 groups: BLAZAR/OTHER) *always* borrows for this reason.

**Zero held-out data is treated honestly, not hidden.** A class that happened to get zero held-out test rows across all pooled splits collapses *exactly* to its group's prior mean, with that prior's typically wide credible interval, and is flagged `zero_held_out=True` in the output — explicitly framed as "the honest thing to report... not new information recovered from nowhere." The underlying sample-size floor (e.g. 8 total novae) is untouched by any of this — no statistical technique manufactures data that doesn't exist; it only reports the resulting uncertainty honestly instead of either omitting the class or reporting a spurious point estimate.

**Scope discipline — reporting-only, by explicit design.** This is the section most insistent, in its own comments, about *not* overreaching: it reads only Stage 5's already-cached predictions (no new CV folds, no new refits), and its output is **never** read by `QUALITY_METRICS`, `SELECTION_METRIC`, or any Borda ranking anywhere in the file — Stage 1/3/4/8's architecture selection is completely untouched by it. The reasoning is that using shrinkage-adjusted metrics as an actual selection criterion would be a materially larger methodological change requiring its own separate validation (e.g., checking whether it would even change which architecture wins) before being trusted for that role — so this section deliberately stops short of that, and says so both in code comments and in a machine-readable caveats file (`rare_class_shrinkage_caveats.json`) that also documents the method's other genuine limitations (the prior-choice subjectivity isn't swept across alternatives; oracle branch training — branches still train only on rows whose *true* superclass matches — is unaffected by any of this and remains a separate, unresolved limitation of the underlying architecture).

---

### Section 17 — Optional second pass: Stage 2 winner's-curse diagnostic [manual]

The explicit, opt-in trigger for Stage 2 (Section 5). Setting `RUN_WINNERS_CURSE_DIAGNOSTIC_NOW=True` and re-running this cell runs the expensive nested-CV diagnostic, then automatically forces Stage 8's report to regenerate so the diagnostic's results get folded into `cross_pipeline_validation_report.json` without needing to rerun the entire driver from scratch.

---

### Section 18 — Optional: SHAP explainability [manual]

Also entirely opt-in. Explains **only** each family's Stage-7 **deployed** winner — never a Stage 1/3 candidate — one hierarchy node at a time (the single flat model for fine; the router plus every trained branch for merged; the gate, blazar-subgate, non-blazar-router, plus every trained branch for 3-level). The framing is explicit: this is a "why does the model I'm actually shipping behave this way" tool, not a model-selection tool — it answers questions about the chosen deployment artifact, not about choosing between candidates.

**Domain payoff:** SHAP values let you check whether a model's decisions track physically sensible features — e.g., does the router rely heavily on `PL_Index` (photon spectral index) and `Signif_Avg` (detection significance), as the physics would predict, or is it leaning on something spurious? This directly serves the project's stated goal of mapping physical logic to features rather than treating the classifier as an unexamined black box.

**Cost control, without approximating the deployed model itself:** tree-based winners (`SHAP_TREE_MODEL_NAMES = {RandomForest, GBC, HistGBC, CatBoost, DecisionTree}`) get `shap.TreeExplainer`, which is exact and fast — no sampling needed. Every other model type (SVM, KNN, LDA, LogReg, MLP) gets a sampling-based `shap.Explainer` wrapped around `predict_proba`, with the background set capped at `SHAP_MAX_BACKGROUND_SAMPLES=100` rows and the evaluated set capped at `SHAP_MAX_EVAL_SAMPLES=200` rows — chosen because mean-|SHAP| feature importance stabilizes well below 1000 samples. The markdown is careful to note these caps only affect the *precision* of the SHAP value estimate, not what gets fit or selected anywhere upstream — unlike the Stage 3 shortlist trim, this cost control touches nothing about model selection at all.

Each node's SHAP result is checkpointed independently (`shap_<node>.joblib`), separate from the Section-0 stage-checkpoint machinery, specifically so a failure explaining one node (say, a branch with too little data) never invalidates another node's already-computed result — with a hierarchical model made of many independent estimators, an all-or-nothing checkpoint would be far too fragile.

---

## How the pieces fit together, end to end

1. Module 1 makes 5 independent splits. Modules 2/3 each independently search for good configurations per split and leave leaderboards behind.
2. This file's Stage 1 cheaply triages those leaderboards down to a short list per family, using only split 0.
3. Stage 3 re-confirms that short list across all 5 splits' *training* data only — still no peeking at test data.
4. Stage 4 locks in one final architecture per family, based purely on Stage 3.
5. Stage 5 evaluates each locked-in architecture, for the first time, on 5 genuinely held-out test partitions.
6. Stage 6 asks whether the differences between families in Stage 5 are statistically real, correcting for both CV's inherent non-independence (Nadeau-Bengio) and running three related comparisons at once (Holm-Bonferroni).
7. Stage 7 refits the winning architectures one more time on more data, purely for shipping — never re-evaluated itself.
8. Stage 8 persists everything as one auditable JSON report, including which family "corrected-wins" and why, with the selection/estimation/deployment distinction spelled out in the report itself.
9. Everything past Stage 8 (Sections 12–18) is reporting, deployment application, or optional diagnostics — CSV exports for humans, application of the deployed models to real unclassified Fermi sources, per-class and calibration diagnostics, a statistically principled treatment of rare classes, and (opt-in) the two most expensive analyses — the winner's-curse size check and SHAP explainability — available on demand without being forced into every routine run.

The throughline across nearly every design choice documented in this file's own comments is the same: never let a number that was used to *choose* something also be reported as a *measurement* of that thing's quality, and when a shortcut is taken for cost reasons (the node-fit cache, the Stage 3 shortlist trim, the SHAP sampling caps), disclose explicitly whether it changes results (it doesn't, for the first two) or only their precision (it does, for the third) — never leave that distinction implicit.
