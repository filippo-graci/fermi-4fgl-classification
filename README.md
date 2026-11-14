# Fermi-AI: classifying Fermi-LAT 4FGL-DR4 gamma-ray sources

This project classifies the sources in the Fermi-LAT 4FGL-DR4 catalog (`gll_psc_v35.fit`) into
physical classes such as BL Lacs, FSRQs, pulsars, supernova remnants and binaries. It then predicts a
class for each of the roughly 2,560 sources that the catalog leaves unassociated.

Three model architectures are searched, validated and compared on the same held-out test set:

| Family | Structure |
|---|---|
| fine | One flat classifier over 18 classes |
| merged | A router into 7 superclasses, then one model per superclass |
| 3level | A blazar/non-blazar gate, then a blazar sub-gate and a non-blazar router, then one model per branch |

The results are in [RESULTS.md](RESULTS.md). All three families reach a test macro-F1 of about 0.31
to 0.33, with overlapping confidence intervals, so the hierarchies do not beat a flat logistic
regression.

For an overview of the whole project in 22 slides, see the
[presentation](docs/Fermi_classification.pdf).

## How the pipeline works

1. Preprocessing (`preprocessing.py`) turns the FITS catalog into tables ready for machine learning.
   It computes hardness ratios, the SED peak, detection summaries and the sine and cosine of Galactic
   longitude. It drops columns derived from the association process, because they leak the label, and
   features that mostly measure photon-counting noise.
2. Processing (`fermi_processing.py`) makes a stratified 70/30 split. On the training part only, it
   runs stability-based feature selection, once globally and once for every node of the two
   hierarchies.
3. Model search (`fermi_fine.py`, `fermi_merged.py`, `fermi_3levels.py`) tries several models at every
   node (logistic regression, LDA, random forest, HistGBC, CatBoost and MLPs), each with different
   feature sets, scalers and class weightings. Candidates are scored by 5-fold stratified
   cross-validation. The hierarchies are scored end to end with chain cross-validation and soft
   probability routing.
4. Validation (`fermi_validation.py`) shortlists and confirms candidates on the training data. It
   picks one model per family with a Borda count over four metrics, plus a penalty for the gap between
   training and validation scores. The winners are then scored once on the test set, with confidence
   intervals, per-class metrics, a calibration check and a covariate-shift check.
5. Inference (`fermi_inference.py`) predicts the unassociated sources and checks the predictions
   against a later catalog release (`gll_psc_v41`).
6. Analysis (`fermi_analysis.py`, `fermi_candidates.py`) produces the plots and computes test metrics
   for every validated candidate, including those that were not deployed.

## Documentation

| File | Contents |
|---|---|
| [docs/fermi_pipeline_design_choices_explained.md](docs/fermi_pipeline_design_choices_explained.md) | Plain-language walkthrough of the design choices |
| [docs/scientific_review_with_citations.md](docs/scientific_review_with_citations.md) | Each step with its justification and literature, and the status of every known issue |
| [docs/fermi_validation_explained.md](docs/fermi_validation_explained.md) | What the validation module does, stage by stage |
| [TODO.md](TODO.md) | Future steps: open issues and analyses not yet run |
| [docs/Fermi_classification.pdf](docs/Fermi_classification.pdf) | Presentation of the project and its final results. The LaTeX source and figure script are in `docs/presentation/`. |

## Repository layout

```
pipeline/                       pipeline code
  preprocessing.py              FITS catalog -> ML-ready CSVs (feature engineering)
  run_fermi.py                  master script: settings, steps, resume, logging
  fermi_common.py               shared models, CV, metrics, I/O
  fermi_processing.py           splits, feature selection (global + per hierarchy node)
  fermi_fine.py                 flat-model search
  fermi_merged.py               router + branch search
  fermi_3levels.py              gate / sub-gate / router + branch search
  fermi_validation.py           shortlist, confirmation, test evaluation, calibration, CIs
  fermi_inference.py            unassociated-source predictions + v41 prospective check
  fermi_analysis.py             plots (calibration, distributions, feature importance, ...)
  fermi_candidates.py           test metrics for every validated candidate
data/processed/                 processed input CSVs (output of preprocessing.py)
outputs_4FGL_DR4_macro_f1_final_single_split/
                                final run: metrics, leaderboards, plots, predictions
docs/                           design notes, scientific review, presentation
README.md                       this file
RESULTS.md                      results report
TODO.md                         future steps
```

Trained models, checkpoints and run logs are not committed because they take several hundred MB.
Rerunning the pipeline regenerates them.

## Running the pipeline

Set up the environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The processed CSV files are included. To rebuild them, download `gll_psc_v35.fit` (4FGL-DR4) from the
[Fermi Science Support Center](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/) into
`data/catalogs/` and run:

```bash
python pipeline/preprocessing.py
```

Then run the pipeline. All settings are at the top of `pipeline/run_fermi.py`.

```bash
python pipeline/run_fermi.py --dry-run          # show plan and resolved settings
python pipeline/run_fermi.py --profile smoke    # quick end-to-end plumbing check
python pipeline/run_fermi.py                    # the final run (profile "reduced")
python pipeline/run_fermi.py --resume           # continue after an interruption
```

Notes:

- A full run takes many hours on one GPU. To run on the CPU, set `"USE_GPU": False` in
  `run_fermi.py`. The CPU run is slower, and some models handle sample weights differently on GPU and
  CPU (see `fermi_common.py`).
- The check against the later catalog runs only if `data/catalogs/` contains `gll_psc_v41.fit` and
  `gll_psc_v35.fit`. Otherwise it is skipped.
- The committed results are in the default output folder. Change `RUN_NAME` before rerunning, so that
  a new run does not resume from them or mix with them.

## License

MIT, see [LICENSE](LICENSE). The 4FGL catalog data are public products of the Fermi-LAT
collaboration.

## Author

Filippo Graci, Università di Pavia
