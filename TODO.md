# To do

Future steps that the final run did not cover. Items marked with a reference (B1.3, B3.2, ...) are
explained in [docs/scientific_review_with_citations.md](docs/scientific_review_with_citations.md),
which also lists the issues already fixed.

## Validation design

- [ ] Run the `per_split` design. The final run uses `single_split`: one 70/30 split, with the search,
  the selection and one test score on it, so the intervals in `RESULTS.md` cover the sampling of test
  rows only. `per_split` repeats the whole search and selection on 5 splits and reports the mean test
  score with a Nadeau–Bengio interval. It costs about five times the search time. Set
  `"VALIDATION_DESIGN": "per_split"` in `pipeline/run_fermi.py`.
- [ ] Remove the researcher degree of freedom in the search space. The `reduced` profile was narrowed
  on an earlier run's partition, and some of those rows are in the final test set (A6.9). Either run
  the `full` profile, or derive the `reduced` profile again from the training part of the final
  split.
- [ ] Tune hyperparameters. Every model uses default settings, and the coarse filter drops candidates
  on a noisy 15% holdout. With the reduced profile it is affordable to give every Stage-1 candidate
  full cross-validation (B2.5).
- [ ] Search feature sets and class weighting together. The search fixes the scaler, then the feature
  set, then the weighting, one at a time; either search the last two jointly or describe the staging
  as a heuristic (B2.7).
- [ ] Justify the selection rule in the paper, or replace it. The Borda count over correlated
  metrics, with an in-sample overfitting penalty, is a stated preference rather than a published
  rule (B2.1).

## Predictions for unassociated sources

- [ ] Calibrate the hierarchical families. Probability calibration and prior correction exist for
  the flat family only (B2.8).
- [ ] Correct for covariate shift. Adjust the class priors to the unassociated population and add a
  reject option for low-confidence predictions (B1.3).
- [ ] Check the imputation shift. `IterativeImputer` is fit on associated sources and applied to
  unassociated ones with a different missingness pattern; add the missing-value rates to the shift
  diagnostic (B3.11).
- [ ] Test whether the models rely on sky position. Run an ablation without `GLAT`/`GLON`; the test
  metrics by Galactic latitude are already reported (B2.6).
- [ ] Report the v41 check per class and state its selection bias. The sources the later catalog
  associated are not a random sample of the unassociated ones (B3.15).

## Code fixes

- [ ] `apply_shift_robust` (`fermi_processing.py`): when every feature is flagged, it returns the
  unfiltered set under the `_SHIFT_ROBUST` name. Skip the tag instead (B3.2).
- [ ] `MIN_SAMPLES_PER_FEATURE` (`fermi_common.py`, `fermi_processing.py`): the feature cap uses the
  total row count, but the binding constraint is the smallest class (B3.10).
- [ ] HistGBC in validation and deployment: Stages 5 and 7 build `MODEL_SPECS["HistGBC"]`, the
  search proxy, while the model search ranked the deployable `StratifiedEarlyStoppingHGBC`. Use the
  same builder in both (B3.13).
- [ ] Early stopping of the MLP and HistGBC estimators: they stop on the selection metric over a 15%
  split that holds 0 or 1 rows of each rare class. Stop on validation cross-entropy, as CatBoost does
  (B3.6).
- [ ] Rare-class shrinkage: it assumes rare and common classes are exchangeable. Shrink toward a mean
  modelled on the training-set size, or report raw rates with exact intervals (B3.4).
- [ ] Cramér's V in the shift diagnostic uses a chi-square test with tiny expected counts. Use
  Fisher's exact test, or keep it as a diagnostic only (B3.14).
- [ ] Label the bootstrap standard deviation in the CV leaderboards as a lower bound.

## Analyses available but switched off

These exist in the code and were not part of the final run. Each one is a setting in
`pipeline/run_fermi.py`.

- [ ] Processing diagnostics and the optional-features comparison: add `processing_diag` and
  `processing_optional` to `STEPS`.
- [ ] Setup A (`bcu` sources included): `"RUN_FINE_SETUP_A": True`.
- [ ] Winner's-curse nested-CV diagnostic (Stage 2): `"RUN_WINNERS_CURSE_DIAGNOSTIC_NOW": True`.
- [ ] Significance tests between families (Stage 6): `"RUN_STAGE6_SIGNIFICANCE_NOW": True`.
- [ ] SHAP explanations of the deployed models: `"RUN_SHAP_EXPLAINABILITY_NOW": True`.
