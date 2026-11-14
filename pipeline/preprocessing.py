# -*- coding: utf-8 -*-
"""
preprocessing.py

Module 0 — Fermi 4FGL catalog ingestion & feature engineering
====================================================================
Converts a 4FGL FITS catalog (default: 4FGL-DR4 gll_psc_v35.fit) into two
leakage-clean, ML-ready CSVs per dataset variant, consumed by Module 1
(fermi_processing.py):
    - full_processed_dataset_with_string_labels.csv   (labeled sources)
    - unknown_sources_full_processed.csv               (CLASS1 == 'unk')

Usage (from anywhere; paths are relative to the project folder):
    python preprocessing.py                                   # DR4 v35 -> ./data/processed/
    python preprocessing.py --fits ./data/catalogs/gll_psc_v31.fit --out ./data/processed_DR3/
Everything printed is also written to <out>/preprocessing_<timestamp>.log.

Runs on DR3 and DR4 alike: the band count (7 or 8) and the presence of the
2-month light curves (DR3 only) are read from the file.

Feature-engineering choices
----------------------------------------------------------------------
Fermi-LAT sources are photon-starved (often a handful of detected photons
per year, fewer still per 2-month bin), so any statistic computed directly
on raw per-bin flux/TS values is dominated by Poisson counting noise once
the bins get fine or the source gets faint. An engineered feature is
therefore kept only if it is (a) not redundant with another retained
feature or an existing catalog scalar, and (b) not primarily measuring
shot noise. The reasoning is repeated inline in
engineer_band_history_features(), next to the code it affects.

Engineered and kept:
    * HR_0_1 ... HR_6_7 (hardness ratios; up to HR_5_6 for DR3) — core
      spectral-shape signal;
      each ratio probes a different segment of the spectrum.
    * Sqrt_TS_Band_max/_argmax/_nbands_detected (3) — compact detection-
      quality summary across bands, without 7-8 raw per-band columns.
    * nuFnu_Band_peak/_peak_band (2) — SED peak location, the basis for
      blazar LSP/ISP/HSP sub-classification; nothing else captures it.
    * Sqrt_TS_History_frac_years_detected (1) — persistence/duty cycle of
      detection across years, at a cadence coarse enough to be robust to
      per-interval photon noise.
    * GLON_sin / GLON_cos — GLON is an angle (359 deg is next to 1 deg).
      GLON itself stays in the CSV for cross-matching; Module 1 does not
      use it as a feature.

Deliberately NOT engineered / dropped:
    * Sqrt_TS_Band_0..7 (raw per-band columns) — redundant with the
      3-feature summary above and with the catalog's own Signif_Avg.
    * Unc_Flux_Band features (SNR max/argmax, asymmetry) — in the photon-
      limited regime TS approximately tracks (signal/noise)^2, so these
      duplicate the Sqrt_TS_Band summary, computed a more fragile way
      (Unc_Flux_Band axis-order ambiguity).
    * Flux(2)_History_excess_variance — in 4FGL a light-curve bin with
      TS < 1 has a NULL lower error (Abdollahi et al. 2022, Table 6 note c);
      ignoring that inflates the value for faint sources (19% of yearly
      bins for associated, 29% for unassociated sources). Corrected, it has
      Spearman rho = 0.98 with the catalog's own Frac_Variability, which is
      already a feature.
    * Flux(2)_History_peak_to_median — for faint sources the median of the
      best-fit yearly fluxes is near 0, so the ratio is noise (max 4.9e5 in
      DR4); flares are covered by Signif_Peak / Flux_Peak /
      Variability_Index.
    * Flux_History_std_over_median — does not deconvolve measurement noise
      from source variability; mostly shot noise for faint sources.
    * Sqrt_TS_History_max — redundant with Signif_Avg / Variability_Index.
    * Flux2_History_von_neumann — a second-order shape statistic on the
      finest (2-month) cadence, where it measures bin-to-bin counting noise
      far more than real serial structure.
    * Sqrt_TS2_History_frac_bins_detected / _max — at 2-month binning,
      "TS > 3 in this bin" mostly reflects per-bin exposure; persistence is
      captured more robustly by Sqrt_TS_History_frac_years_detected.
    * Conf_95_* — exactly 1.62146 x Conf_68_* (same position angle). Exact
      duplicates split permutation importance between them (Gregorutti et
      al. 2017).
    * Flag columns never set in the catalog (DR4: Flag_8, Flag_12) —
      constant columns carry nothing. Flags bits 13/14 are excluded for
      the reasons given at FLAG_BITS_USED.

Design principles enforced throughout:
    - No association-process or external cross-match columns retained
      (ASSOC_*, CLASS2, TEVCAT_FLAG, counterpart astrometry) — they are
      by-products of the procedure that assigned CLASS1 (label leakage).
      Conf_68/95_* and PL/LP/PLEC_Flux_Density ARE kept (Tier 3 below):
      they are redundancy/selection-effect arguments, left to Module 1.
    - Multidimensional FITS columns (7/8-band spectral arrays, yearly/2-month
      light curves, and their companion uncertainty/SED arrays) are
      engineered into scalar hardness-ratio / SED-peak / detection-
      persistence features BEFORE the scalar-only table conversion
      discards them — but only where the resulting statistic is not
      redundant with an existing feature and not primarily a measurement
      of Poisson counting noise.
    - No global (pre-split) statistic is baked into any imputed value —
      uncertain infinities are left as NaN and deferred to Module 1's
      native/per-fold missing-value handling.
    - SpectrumType is one-hot encoded (nominal, non-ordinal) — not
      LabelEncoder, which would impose a false ordinal relationship for
      any distance/linear model downstream.

"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

warnings.filterwarnings("ignore")

# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FITS_PATH = "./data/catalogs/gll_psc_v35.fit"   # set from --fits in main()
OUTPUT_DIR = "./data/processed"                   # set from --out in main()


def build_output_paths(variant_name: str):
    """Each variant gets its own subdirectory so both CSVs can coexist
    on disk with identical filenames, ready to be pointed at independently
    from Module 1 (just swap DATA_PATH)."""
    variant_dir = os.path.join(OUTPUT_DIR, variant_name)
    labeled_path = os.path.join(variant_dir, "full_processed_dataset_with_string_labels.csv")
    unknown_path = os.path.join(variant_dir, "unknown_sources_full_processed.csv")
    return variant_dir, labeled_path, unknown_path

# Columns encoding leakage, instrumental artifacts, or pure identifiers.
# See module docstring for physical rationale of each block.
# --- Tier 1: Hard-drop before Module 1's feature selection ever runs. ---
# These are not "possibly low-importance" — they are literal restatements
# of CLASS1, or direct byproducts of the SAME process that assigned CLASS1
# (association-confidence scores, counterpart name strings). Permutation
# importance / SHAP cannot distinguish "legitimate physical signal" from
# "artifact of how the label itself was generated" — it will rank these
# features at the TOP, not prune them. They must be removed here, upstream
# of any selection logic, precisely because selection has no mechanism to
# self-correct for this failure mode.
LEAKAGE_COLUMNS_TO_DROP = [
    "CLASS2",                                          # associator's 2nd-choice class guess
    "ASSOC_4FGL", "ASSOC_FGL", "ASSOC_FHL",             # cross-DR "was this already known" bias
    "ASSOC_GAM1", "ASSOC_GAM2", "ASSOC_GAM3",           # pre-Fermi gamma-ray catalog cross-match
    "ASSOC1", "ASSOC2",                                 # counterpart name strings, near-literal label
    "ASSOC_TEV", "TEVCAT_FLAG",                         # ground-based TeV target-selection bias (collinear pair)
    "ASSOC_PROB_BAY", "ASSOC_PROB_LR",                  # confidence score of the SAME process that set CLASS1
]

# --- Pure identifiers / bookkeeping — zero generalizable content regardless
# of what feature selection would say about them. ---
IDENTIFIER_COLUMNS_TO_DROP = [
     "Extended_Source_Name",
     "ROI_num",
    "Time_Peak", "Peak_Interval",
    "DataRelease",                           # catalog-vintage bookkeeping tag, not source-intrinsic
]

# --- External-instrument astrometry for the multiwavelength counterpart,
# not this source's own gamma-ray position (RAJ2000/DEJ2000 handled
# separately). Populated only for confidently-associated sources — same
# label-generation-process leakage pattern as ASSOC_PROB_BAY/LR above. ---
COUNTERPART_ASTROMETRY_COLUMNS_TO_DROP = [
    "RA_Counterpart", "DEC_Counterpart", "Unc_Counterpart",
]

# Exact duplicates: Conf_95_* = 1.62146 x Conf_68_* (Gaussian 95%/68% radius ratio), same PosAng.
DUPLICATE_COLUMNS_TO_DROP = ["Conf_95_SemiMajor", "Conf_95_SemiMinor", "Conf_95_PosAng"]

COLUMNS_TO_DROP = (
    LEAKAGE_COLUMNS_TO_DROP
    + IDENTIFIER_COLUMNS_TO_DROP
    + COUNTERPART_ASTROMETRY_COLUMNS_TO_DROP
    + DUPLICATE_COLUMNS_TO_DROP
)

# --- Tier 2: Empirically-contestable exclusions — deliberately NOT
# hard-dropped above. Cutting these on an a-priori "redundant/undefined"
# argument, before any model ever sees them, is exactly the kind of
# judgment call that isn't a scientifically defensible thesis on its own.
# Instead, BOTH variants of the dataset are written to disk: one with these
# columns present, one with them absent. Module 1's permutation importance +
# SHAP + 99/90/80% cumulative-power cut is the actual arbiter — if a column
# lands near the bottom and gets cut at 90/80% in the "with" variant, the
# redundancy argument is empirically confirmed at zero cost; if it shows
# independent importance, that's a real finding, not a mistake to pre-empt
# on intuition. ---

# Equatorial RA/DEC — argued redundant with the Galactic GLAT/GLON already
# retained (same sky position, different coordinate frame).
REDUNDANT_SPATIAL_COLUMNS = ["RAJ2000", "DEJ2000"]

# Mathematically undefined for most of the sample, not merely "possibly
# low-importance." E_peak has no interior SED solution for a monotonic
# (unbroken) power law; PLEC_Exp_Index is a team-fixed constant (1.0 or 2/3)
# for ~all rows, not a freely fitted physical quantity. Argued
# redundant-by-construction with SpectrumType/LP_SigCurv/PLEC_SigCurv, which
# are retained directly regardless of variant.
UNDEFINED_BY_CONSTRUCTION_COLUMNS_TO_DROP = [
    "LP_EPeak", "Unc_LP_EPeak",
    "PLEC_Exp_Index", "Unc_PLEC_Exp_Index",
    "PLEC_EPeak", "Unc_PLEC_EPeak",
]

OPTIONAL_COLUMNS = REDUNDANT_SPATIAL_COLUMNS + UNDEFINED_BY_CONSTRUCTION_COLUMNS_TO_DROP

# variant_name -> whether OPTIONAL_COLUMNS are dropped for that variant
DATASET_VARIANTS = {
    "without_optional_features": True,
    "with_optional_features": False,
}

# --- Tier 3: Also deliberately NOT hard-dropped. ---
# Conf_68_* and PL/LP/PLEC_Flux_Density are NOT removed here: the case
# against them is a redundancy/confound ARGUMENT (correlated with GLAT/Signif_Avg/
# SpectrumType respectively) rather than a leakage argument. That
# distinction matters: redundancy is exactly what Module 1's
# permutation-importance + SHAP + 99/90/80% cumulative-power cut is built
# to measure empirically. Pre-dropping them here removes potential real
# signal from the pool before that machinery gets to evaluate it.
# DataRelease is NOT part of this tier — it is hard-dropped above
# in IDENTIFIER_COLUMNS_TO_DROP as pure catalog-vintage bookkeeping, not
# kept for empirical arbitration.
# They are left in x_all/x_raw and allowed to flow into Module 1; check
# feature_importance_{A,B,C}.csv and shap_importance_{A,B,C}.csv after a
# run — if they land near the bottom and get cut at 90/80%, the
# redundancy argument is empirically confirmed at zero cost; if they show
# independent importance, that's a real finding, not a mistake to
# pre-empt on intuition.

# Columns to apply safe-clipped log10 transform (preserve zero via clip floor)
LOG_TRANSFORM_COLUMNS = [
    "Flux1000", "Unc_Flux1000", "Energy_Flux100", "Unc_Energy_Flux100",
    "Npred", "Variability_Index", "Flux_Peak", "Unc_Flux_Peak",
    "Pivot_Energy",
    # The only engineered feature still spanning orders of magnitude is
    # nuFnu_Band_peak (an energy-flux-like SED value).
    # NOTE: nuFnu_Band_peak_band is intentionally EXCLUDED — it is a
    # band-index label (0-6, or 0-7 for DR4), not a magnitude, consistent with
    # Sqrt_TS_Band_argmax above (also never log-transformed).
    "nuFnu_Band_peak",
]

SAFE_LOG_FLOOR = 1e-18

# Fermi 4FGL 'Flags' bitmask (Abdollahi et al. 2022, ApJS 260:53, Table 4). Bits 7 and 11 are
# not defined. Bit 14 ("Soft Galactic Unassociated", §6.2) is set only for sources classed SGU,
# SPP or UNK, i.e. it depends on the association outcome -> excluded as label leakage. Bit 13
# ("TS < 25 at 12 yr") is a property of the source and could be kept; it is excluded to keep the
# flag feature set small.
FLAG_BITS_USED = [n for n in range(1, 13) if n not in (7, 11)]


# =====================================================================
# Step 1 — FITS ingestion + multidimensional-array feature engineering
# =====================================================================

def _safe_nanargmax(arr_2d):
    """np.nanargmax raises on all-NaN rows; return -1 there instead."""
    out = np.full(arr_2d.shape[0], -1.0)
    valid = ~np.all(np.isnan(arr_2d), axis=1)
    out[valid] = np.nanargmax(arr_2d[valid], axis=1)
    return out


def engineer_band_history_features(table: Table) -> pd.DataFrame:
    """Scalar features derived from the 7- or 8-band spectral arrays and the
    yearly (and, DR3 only, 2-month) light curves. These are multidimensional FITS columns
    that get silently discarded by the scalar-only Table->DataFrame
    conversion; without this step, none of the hardness-ratio / light-curve
    engineering the catalog documentation calls for ever executes.

    Deliberately minimal. Fermi-LAT gamma-ray sources are photon-starved: most have
    only a handful of detected photons per year, fewer still per 2-month
    bin. Any statistic computed directly on raw per-bin flux/TS values is
    increasingly dominated by Poisson counting noise as the bins get
    finer or the source gets fainter. Only features are kept that are (a)
    not redundant with another retained feature or an existing catalog
    scalar (Signif_Avg, Variability_Index, Frac_Variability, ...), and
    (b) not primarily measuring shot noise rather than source physics.
    See the module docstring for the full keep/drop table;
    the reasoning is repeated inline below at the point of each cut so it
    stays next to the code it affects."""
    n = len(table)
    feats = {}

    flux_band = None
    if "Flux_Band" in table.colnames:
        flux_band = np.asarray(table["Flux_Band"], dtype=np.float64)  # (n, 7)
        for i in range(flux_band.shape[1] - 1):
            num = flux_band[:, i + 1] - flux_band[:, i]
            den = flux_band[:, i + 1] + flux_band[:, i]
            with np.errstate(divide="ignore", invalid="ignore"):
                feats[f"HR_{i}_{i+1}"] = np.where(den != 0, num / den, 0.0)

    # Unc_Flux_Band (Flux_Band_SNR_max/_argmax, unc_asymmetry_mean)
    # not engineered. In the photon-limited regime Fermi-LAT operates
    # in, TS approximately tracks (signal/noise)^2 -- so this would be a
    # near-duplicate of the Sqrt_TS_Band summary below, computed a second,
    # more fragile way (via the documented Unc_Flux_Band axis-order
    # ambiguity in _split_asymmetric_uncertainty). flux_band above is
    # used only for the hardness-ratio block.

    # nuFnu_Band: SED peak location — KEPT. This is DIFFERENT
    # physical information from the Flux_Band hardness ratios above: HR
    # captures the SHAPE between adjacent bands, while nuFnu_Band_peak_band
    # captures WHERE the spectral energy distribution actually peaks --
    # the standard basis for the synchrotron-peak blazar sub-classification
    # (low/intermediate/high-synchrotron-peaked, LSP/ISP/HSP) used
    # throughout the 4LAC/4FGL AGN literature. Nothing else retained in
    # this function captures this axis. -inf entries follow the same "no
    # significant detection in that band" sentinel convention as
    # Sqrt_TS_Band below.
    if "nuFnu_Band" in table.colnames:
        nufnu_band = np.asarray(table["nuFnu_Band"], dtype=np.float64)
        n_inf_nufnu = np.isneginf(nufnu_band).sum()
        if n_inf_nufnu:
            print(f"nuFnu_Band: replacing {n_inf_nufnu} -inf non-detection "
                  f"sentinel(s) with 0.")
        nufnu_band = np.where(np.isneginf(nufnu_band), 0.0, nufnu_band)
        feats["nuFnu_Band_peak"] = np.nanmax(nufnu_band, axis=1)
        feats["nuFnu_Band_peak_band"] = _safe_nanargmax(nufnu_band)

    if "Sqrt_TS_Band" in table.colnames:
        sqrt_ts_band = np.asarray(table["Sqrt_TS_Band"], dtype=np.float64)  # (n, n_bands)
        # sqrt(TS) is mathematically non-negative; -inf appearing in this
        # column is a catalog-pipeline sentinel for "no significant
        # detection in that band" (TS <= 0 floored via a log/sqrt path
        # upstream), not a real magnitude. Treat as zero significance,
        # same convention as the Signif_Peak/Flux_Peak non-convergence fix.
        n_inf_sentinel = np.isneginf(sqrt_ts_band).sum()
        if n_inf_sentinel:
            print(f"Sqrt_TS_Band: replacing {n_inf_sentinel} -inf non-detection "
                  f"sentinel(s) with 0.")
        sqrt_ts_band = np.where(np.isneginf(sqrt_ts_band), 0.0, sqrt_ts_band)

        # The raw per-band Sqrt_TS_Band_{i} columns are not kept:
        # redundant with the 3-feature summary below and
        # with the catalog's own Signif_Avg. Only the compact summary
        # (max/argmax/nbands_detected) is retained.
        feats["Sqrt_TS_Band_max"] = np.nanmax(sqrt_ts_band, axis=1)
        feats["Sqrt_TS_Band_argmax"] = _safe_nanargmax(sqrt_ts_band)
        feats["Sqrt_TS_Band_nbands_detected"] = (
            (sqrt_ts_band > 3.0).sum(axis=1).astype(np.float64)
        )

    # Flux_History_peak_to_median and Flux_History_excess_variance are not
    # engineered — see the module docstring (noise-driven for faint sources /
    # duplicate of the catalogue's Frac_Variability once corrected for the
    # NULL lower errors of TS < 1 bins).

    if "Sqrt_TS_History" in table.colnames:
        sqrt_ts_hist = np.asarray(table["Sqrt_TS_History"], dtype=np.float64)
        n_inf = np.isneginf(sqrt_ts_hist).sum()
        if n_inf:
            print(f"Sqrt_TS_History: replacing {n_inf} -inf non-detection sentinel(s) with 0.")
        sqrt_ts_hist = np.where(np.isneginf(sqrt_ts_hist), 0.0, sqrt_ts_hist)
        # Sqrt_TS_History_max is not kept: redundant with the catalog's own
        # Signif_Avg/Variability_Index. frac_years_detected (a distinct,
        # persistence/duty-cycle concept) is kept.
        feats["Sqrt_TS_History_frac_years_detected"] = (sqrt_ts_hist > 3.0).mean(axis=1)

    # Flux2_History_peak_to_median / _excess_variance are not engineered for the
    # same reasons (DR3 has Frac2_Variability for the 2-month cadence).

    # Sqrt_TS2_History (frac_bins_detected, max) is not engineered.
    # Same redundancy argument as
    # Sqrt_TS_History_max above, one level worse: at 2-month binning,
    # "was TS>3 in this bin" mostly reflects per-bin exposure/photon
    # count rather than intrinsic source behavior for photon-starved
    # sources. The persistence concept is already captured more robustly
    # at the yearly cadence by Sqrt_TS_History_frac_years_detected.

    engineered = pd.DataFrame(feats)
    assert len(engineered) == n, "Row-count mismatch during array-column engineering"
    return engineered


def load_fits_catalog(fits_path: str) -> pd.DataFrame:
    """Loads the 4FGL FITS table, engineers scalar features from array
    columns, converts to a flat DataFrame, and strips FITS string padding."""
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
    astropy_table = Table(data)
    print(f"FITS file loaded: {len(astropy_table)} rows, "
          f"{len(astropy_table.colnames)} raw columns.")

    engineered = engineer_band_history_features(astropy_table)

    scalar_names = [
        name for name in astropy_table.colnames
        if len(astropy_table[name].shape) <= 1
    ]
    dataset = astropy_table[scalar_names].to_pandas()
    dataset = pd.concat(
        [dataset.reset_index(drop=True), engineered.reset_index(drop=True)],
        axis=1,
    )

    # Fixed-width FITS char arrays are space-padded; strip before any
    # exact-string matching (CLASS1 standardization, SpectrumType, etc.)
    for col in dataset.select_dtypes(include=["object"]).columns:
        dataset[col] = dataset[col].astype(str).str.strip()

    print(f"Post-engineering shape: {dataset.shape}")
    return dataset


# =====================================================================
# Step 2 — Target standardization
# =====================================================================

def standardize_classes(df: pd.DataFrame, target_col: str = "CLASS1") -> pd.DataFrame:
    """Collapses case distinctions and maps blank/unknown markers to 'unk'."""
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in the DataFrame.")

    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    null_and_unknown_mappings = {"": "unk", "nan": "unk", "none": "unk", "unk": "unk"}
    df[target_col] = df[target_col].replace(null_and_unknown_mappings)
    return df


# =====================================================================
# Step 3 — Column dropping, row filtering, extended-source flag
# =====================================================================

def drop_leakage_and_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts Is_Extended before dropping the raw identifier string, then
    drops all leakage/instrumental/identifier columns in one pass."""
    df = df.copy()
    # FITS string columns are space-padded, not NaN-padded — after stripping,
    # non-extended sources read back as '' (empty string), not NaN. notna('')
    # is True, so a naive notna() check flags every row as extended. Compare
    # against the stripped empty string explicitly instead.
    df["Is_Extended"] = (
        df["Extended_Source_Name"].notna()
        & (df["Extended_Source_Name"].astype(str).str.strip() != "")
    ).astype(int)
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    return df


def drop_galactic_center_row(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the single 'gc' (galactic center) source — not a resolvable
    point/extended source class, and n=1 is unstratifiable regardless."""
    gc_index = df[df["CLASS1"] == "gc"].index
    df = df.drop(gc_index)
    print(f"Dropped {len(gc_index)} 'gc' source(s). "
          f"CLASS1 value counts:\n{df['CLASS1'].value_counts()}")
    return df


# =====================================================================
# Step 4 — Infinity remediation (documented physical boundary conditions)
# =====================================================================

def remediate_infinities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols_to_zero_fill = [c for c in ["Signif_Peak", "Flux_Peak", "Unc_Flux_Peak"]
                         if c in df.columns]
    for col in cols_to_zero_fill:
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        df[f"{col}_is_null"] = (df[col] == 0).astype(int)

    variability_cols = [c for c in
        ["Variability_Index", "Frac_Variability",
         "Variability2_Index", "Frac2_Variability"]
        if c in df.columns]
    for col in variability_cols:
        df[col] = df[col].replace(-np.inf, 0)

    uncertainty_cols = [c for c in
        ["Unc_Frac_Variability", "Unc_Frac2_Variability"]
        if c in df.columns]
    for col in uncertainty_cols:
        df[col] = df[col].replace([-np.inf, np.inf], np.nan)

    missing = set(["Signif_Peak", "Flux_Peak", "Unc_Flux_Peak",
                    "Variability_Index", "Frac_Variability",
                    "Variability2_Index", "Frac2_Variability",
                    "Unc_Frac_Variability", "Unc_Frac2_Variability"]) - set(df.columns)
    if missing:
        print(f"WARNING: expected columns not found in catalog, skipped: {sorted(missing)}")

    remaining_inf = {
        c: int(np.isinf(df[c]).sum())
        for c in df.select_dtypes(include=[np.float64, np.float32]).columns
        if np.isinf(df[c]).any()
    }
    assert not remaining_inf, f"Unremediated infinities remain: {remaining_inf}"
    print("Infinity remediation verified clean.")
    return df


def report_nan_columns(df: pd.DataFrame, stage: str = "") -> None:
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print(f"--- NaN columns ({stage}) ---\n{nan_cols}")
    else:
        print(f"--- No NaN values found ({stage}) ---")


# =====================================================================
# Step 5 — Categorical encoding (Flags bitmask, SpectrumType one-hot)
# =====================================================================

def deconvolve_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Decomposes the 'Flags' bitmask into individual binary indicators (bits in FLAG_BITS_USED —
    see the comment there for why 7, 11, 13 and 14 are skipped)."""
    df = df.copy()
    for n in FLAG_BITS_USED:
        bit_val = 2 ** (n - 1)
        df[f"Flag_{n}"] = ((df["Flags"] & bit_val) > 0).astype(int)
    never_set = [f"Flag_{n}" for n in FLAG_BITS_USED if df[f"Flag_{n}"].sum() == 0]
    if never_set:
        print(f"Flags never set in this catalog (constant columns, dropped): {never_set}")
    return df.drop(columns=["Flags"] + never_set)


def encode_glon(df: pd.DataFrame) -> pd.DataFrame:
    """GLON is an angle: 359 deg and 1 deg are neighbours, which a raw GLON
    column hides from linear/MLP models. Adds sin/cos; GLON itself is kept
    (cross-matching needs it) and Module 1 leaves it out of the features."""
    df = df.copy()
    rad = np.radians(df["GLON"].astype(float))
    df["GLON_sin"], df["GLON_cos"] = np.sin(rad), np.cos(rad)
    return df


def encode_spectrum_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes SpectrumType (nominal, non-ordinal category).
    LabelEncoder would impose a false ordinal relationship harmless to tree
    ensembles but corrupting to any downstream distance/linear model."""
    df = df.copy()
    dummies = pd.get_dummies(df["SpectrumType"], prefix="SpectrumType", dtype=int)
    df = pd.concat([df.drop(columns=["SpectrumType"]), dummies], axis=1)
    print(f"SpectrumType one-hot columns: {dummies.columns.tolist()}")
    return df


# =====================================================================
# Step 6 — Safe-clipped log10 transforms
# =====================================================================

def apply_log_transforms(df: pd.DataFrame,
                          columns=LOG_TRANSFORM_COLUMNS,
                          floor: float = SAFE_LOG_FLOOR) -> pd.DataFrame:
    """log10(clip(x, floor, None)) -> new log_<col>, original column dropped.
    Floor prevents log10(0)/log10(negative) on near-zero/negative sentinel
    values while preserving the qualitative 'near-zero' signal."""
    df = df.copy()
    valid_cols = [c for c in columns if c in df.columns]
    for col in valid_cols:
        clipped = np.clip(df[col], a_min=floor, a_max=None)
        df[f"log_{col}"] = np.log10(clipped)
        df = df.drop(columns=[col])
    print(f"Log-transformed columns: {valid_cols}")
    return df


def print_feature_ranges(df: pd.DataFrame) -> None:
    print("--- Range of Values for Each Feature ---")
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):
            print(f"{column:<35} | min={df[column].min():^15.4g} | "
                  f"max={df[column].max():^15.4g}")
        else:
            print(f"{column:<35} | non-numeric")


# =====================================================================
# Step 7 — Unknown/labeled split + persistence
# =====================================================================

def split_unknown_labeled(features: pd.DataFrame, class_labels: pd.Series):
    combined = features.copy()
    combined["CLASS1"] = class_labels
    unk_mask = combined["CLASS1"] == "unk"
    unk_df = combined[unk_mask].copy()
    labeled_df = combined[~unk_mask].copy()
    print(f"Unknown sources : {unk_df.shape}")
    print(f"Labeled sources : {labeled_df.shape}")
    return unk_df, labeled_df


def save_outputs(unk_df: pd.DataFrame, labeled_df: pd.DataFrame,
                  output_dir: str, labeled_path: str, unknown_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # --- Terminal sanity assertions — guarantee Module 1's preconditions ---
    assert labeled_df["CLASS1"].isna().sum() == 0, "NaN in CLASS1 label column"
    assert "unk" not in labeled_df["CLASS1"].unique(), "'unk' leaked into labeled set"
    assert "Source_Name" in labeled_df.columns and "Source_Name" in unk_df.columns, (
        "Source_Name missing — traceability tag was dropped upstream"
    )
    assert labeled_df["Source_Name"].is_unique, "Duplicate Source_Name in labeled set"
    assert unk_df["Source_Name"].is_unique, "Duplicate Source_Name in unknown set"
    numeric_cols = labeled_df.select_dtypes(include=[np.float64, np.float32]).columns
    inf_check = {
        c: int(np.isinf(labeled_df[c]).sum())
        for c in numeric_cols if np.isinf(labeled_df[c]).any()
    }
    assert not inf_check, f"Residual infinities in labeled dataset: {inf_check}"
    assert "GLAT" in labeled_df.columns and "GLON" in labeled_df.columns, (
        "Galactic coordinates missing — required by Module 1's stratification"
    )

    # Cosmetic: put the tag first so it visually reads as metadata, not a
    # feature buried mid-table.
    def _lead_source_name(df):
        cols = ["Source_Name"] + [c for c in df.columns if c != "Source_Name"]
        return df[cols]

    unk_df = _lead_source_name(unk_df)
    labeled_df = _lead_source_name(labeled_df)

    unk_df.to_csv(unknown_path, index=False)
    labeled_df.to_csv(labeled_path, index=False)
    print(f"Unknown sources saved -> {unknown_path}")
    print(f"Labeled sources saved -> {labeled_path}")
    print("All terminal sanity assertions passed.")


# =====================================================================
# Orchestration
# =====================================================================

class _Tee:
    """Writes everything printed to the console AND to the run log."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    global FITS_PATH, OUTPUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", default=FITS_PATH, help="4FGL FITS catalog (relative to the project folder)")
    ap.add_argument("--out", default=OUTPUT_DIR, help="output folder (one subfolder per dataset variant)")
    args = ap.parse_args()
    os.chdir(PROJECT_ROOT)
    FITS_PATH, OUTPUT_DIR = args.fits, args.out
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, f"preprocessing_{datetime.now():%Y%m%d_%H%M%S}.log")
    with open(log_path, "w", buffering=1) as log:
        sys.stdout = _Tee(sys.__stdout__, log)
        sys.stderr = _Tee(sys.__stderr__, log)
        try:
            print(f"preprocessing — {datetime.now():%Y-%m-%d %H:%M:%S}  fits={FITS_PATH}  out={OUTPUT_DIR}")
            _run()
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    print(f"Run log -> {log_path}")


def _run():
    # 1) Ingest + engineer array-derived features
    dataset = load_fits_catalog(FITS_PATH)
    report_nan_columns(dataset, stage="raw ingestion")

    # 2) Standardize target labels
    dataset = standardize_classes(dataset, target_col="CLASS1")
    print("Standardized CLASS1 value counts:")
    print(dataset["CLASS1"].value_counts())

    # 3) Drop leakage/identifier columns + galactic-center row
    dataset = drop_leakage_and_identifier_columns(dataset)
    dataset = drop_galactic_center_row(dataset)

    # 4) Remediate documented infinities, keeping NaN where no principled
    #    fill exists (deferred to Module 1's leakage-safe handling)
    dataset = remediate_infinities(dataset)
    report_nan_columns(dataset, stage="post infinity-remediation")

    # 5) Categorical encoding: Flags bitmask + SpectrumType one-hot
    dataset = deconvolve_flags(dataset)
    dataset = encode_spectrum_type(dataset)
    dataset = encode_glon(dataset)

    # 6) Safe-clipped log10 transforms on heavy-tailed flux/variability metrics
    dataset = apply_log_transforms(dataset)
    print_feature_ranges(dataset)

    non_numeric_cols = dataset.select_dtypes(include=["object"]).columns.tolist()
    print("Remaining non-numeric columns after drop:", non_numeric_cols)

    # 7) Split unknown vs. labeled once, then fan out into both OPTIONAL_COLUMNS
    #    variants. Everything upstream (engineering, encoding, log transforms)
    #    is identical between variants — only this last cut differs — so the
    #    two CSVs are directly comparable for a feature-selection ablation,
    #    nothing else confounds the comparison.
    class_labels = dataset["CLASS1"]
    base_features = dataset.drop(columns=["CLASS1"])

    print("=" * 70)
    print(f"Writing {len(DATASET_VARIANTS)} dataset variant(s): "
          f"{list(DATASET_VARIANTS)}")
    print(f"Optional columns under test: {OPTIONAL_COLUMNS}")

    written_paths = {}
    for variant_name, drop_optional in DATASET_VARIANTS.items():
        print("-" * 70)
        print(f"Variant '{variant_name}' (drop_optional={drop_optional})")

        if drop_optional:
            variant_features = base_features.drop(columns=OPTIONAL_COLUMNS, errors="ignore")
        else:
            variant_features = base_features.copy()

        unk_df, labeled_df = split_unknown_labeled(variant_features, class_labels)
        variant_dir, labeled_path, unknown_path = build_output_paths(variant_name)
        save_outputs(unk_df, labeled_df, variant_dir, labeled_path, unknown_path)
        written_paths[variant_name] = (labeled_path, unknown_path)

    print("=" * 70)
    print("Module 0 (Preprocessing) complete.")
    for variant_name, (labeled_path, unknown_path) in written_paths.items():
        print(f"  [{variant_name}]")
        print(f"    Labeled dataset -> {labeled_path}")
        print(f"    Unknown dataset -> {unknown_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
