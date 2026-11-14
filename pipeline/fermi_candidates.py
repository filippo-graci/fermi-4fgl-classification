# %% [markdown]
# # fermi_candidates.py — every validated fine candidate, not only the winner
#
# Runs after fermi_validation.py and fermi_inference.py in the shared namespace
# (run_fermi.py step "candidates"). For each Stage-3 fine candidate — the Borda-ranked shortlist
# the winner was chosen from — on split 0:
#   1. TEST: refit on the training part, score the held-out test part: every REPORT_METRICS column
#      with its row-level 95% CI (`_metric_ci`); calibration, raw vs calibrated (calibrator fit on
#      out-of-fold TRAINING predictions only); per-class recall and precision with Clopper-Pearson
#      intervals; accuracy by confidence threshold, for all / predicted-blazar / predicted-non-blazar
#      sources, at fine-class and superclass level.
#   2. DEPLOYMENT: refit + calibrator on train+test pooled (as Stage 7 does for the winner) and
#      predictions for the unassociated sources.
#   3. v41 (FY16): the Stage-10 prospective check, by threshold, per class, and blazar vs non-blazar.
# One folder per candidate under validation/all_candidates/, plus candidates_summary.csv.
#
# Nothing here selects a model. These are descriptive comparisons on the same test rows; picking
# the best of them by its TEST score would be model selection on the test set. To deploy another
# candidate, set WINNER_OVERRIDE in run_fermi.py on training-side grounds.

# %%
import joblib
import numpy as np
import pandas as pd

CANDIDATES_DIR = VALIDATION_DIR / "all_candidates"
CANDIDATE_THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
BLAZAR_CLASSES = ["bll", "fsrq"]


def _superclass(labels) -> np.ndarray:
    return np.array([TAXONOMY.get(c, "OTHER") for c in labels])


def threshold_table(true_str, pred_str, conf, thresholds=CANDIDATE_THRESHOLDS) -> pd.DataFrame:
    """Accuracy (fine class and superclass, exact Clopper-Pearson 95% CIs) of the predictions with
    confidence >= threshold, for all predictions, predicted blazars and predicted non-blazars."""
    true_str, pred_str, conf = np.asarray(true_str), np.asarray(pred_str), np.asarray(conf, dtype=float)
    hits = {"fine": true_str == pred_str, "superclass": _superclass(true_str) == _superclass(pred_str)}
    is_blazar = np.isin(pred_str, BLAZAR_CLASSES)
    groups = {"all": np.ones(len(pred_str), dtype=bool), "predicted_blazar": is_blazar,
              "predicted_nonblazar": ~is_blazar}
    rows = []
    for t in thresholds:
        for group, mask in groups.items():
            sel = mask & (conf >= t)
            n = int(sel.sum())
            row = {"confidence_threshold": t, "group": group, "n": n}
            for level, ok in hits.items():
                k = int((sel & ok).sum())
                lo, hi = clopper_pearson(k, n)
                row.update({f"{level}_accuracy": k / n if n else float("nan"),
                            f"{level}_ci_low": lo, f"{level}_ci_high": hi})
            rows.append(row)
    return pd.DataFrame(rows)


def _at(table: pd.DataFrame, group: str, t: float, col: str):
    r = table[(table["group"] == group) & (table["confidence_threshold"] == t)]
    return r[col].iloc[0] if len(r) else float("nan")


def evaluate_candidate(rank: int, cand: dict, fine_ctx, df_unk, v41_df, is_winner: bool, logger) -> dict:
    cfg = cand["cfg"]
    name = cand.get("full_model_name") or cfg["model_name"]
    out_dir = CANDIDATES_DIR / f"{rank:02d}_{_safe_filename(name, cfg['tag'], cfg['weighting'])}"
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = fine_ctx.label_encoder.classes_
    x_tr, y_tr = fine_ctx.x_train_by_tag[cfg["tag"]], np.asarray(fine_ctx.y_train_enc)
    x_te, y_te = fine_ctx.x_test_by_tag[cfg["tag"]], np.asarray(fine_ctx.y_test_enc)

    # 1. Held-out test split
    model = fit_component(cfg, MODEL_SPECS, get_builder, x_tr, y_tr)
    pred = np.asarray(model.predict(x_te)).reshape(-1)
    raw = np.asarray(model.predict_proba(x_te))
    calibrator = fit_calibrator(cfg, MODEL_SPECS, get_builder, x_tr, y_tr)
    calibrated = calibrator.transform(raw)
    rows_idx = np.arange(len(pred))
    conf = calibrated[rows_idx, pred]
    test_ci = _metric_ci(y_te, pred)
    pd.DataFrame([{"metric": m, **v} for m, v in test_ci.items()]).to_csv(out_dir / "test_metrics_ci.csv", index=False)
    calib = {variant: {"brier": _brier_with_ci(y_te, P)["value"],
                       "ece": _ece(P[rows_idx, pred], (pred == y_te).astype(float))}
             for variant, P in (("raw", raw), ("calibrated", calibrated))}
    true_str, pred_str = classes[y_te], classes[pred]
    # Per-row test predictions, so any later breakdown (latitude, significance, ...) needs no refit.
    pd.DataFrame({"GLAT": fine_ctx.x_test_by_tag["ALL_FEATURES"]["GLAT"].to_numpy(), "true": true_str,
                  "predicted": pred_str, "confidence": conf, "confidence_raw": raw[rows_idx, pred],
                  **{f"proba_{c}": calibrated[:, i] for i, c in enumerate(classes)}}
                 ).to_csv(out_dir / "test_predictions.csv", index=False)
    test_thr = threshold_table(true_str, pred_str, conf)
    test_thr.to_csv(out_dir / "test_accuracy_by_threshold.csv", index=False)
    per_class = []
    for cls in sorted(set(true_str)):
        n, k, n_pred = int((true_str == cls).sum()), int(((true_str == cls) & (pred_str == cls)).sum()), int((pred_str == cls).sum())
        lo, hi = clopper_pearson(k, n)
        per_class.append({"fine_class": cls, "superclass": TAXONOMY.get(cls), "n_test": n, "k_correct": k,
                          "recall": k / n, "recall_ci_low": lo, "recall_ci_high": hi, "n_predicted": n_pred,
                          "precision": k / n_pred if n_pred else float("nan")})
    pd.DataFrame(per_class).to_csv(out_dir / "test_per_class.csv", index=False)

    # 2. Deployment refit on all labelled rows of split 0 + unassociated-source predictions
    x_pool = pd.concat([x_tr, x_te], ignore_index=True)
    y_pool = np.concatenate([y_tr, y_te])
    deployed = {"model": fit_component(cfg, MODEL_SPECS, get_builder, x_pool, y_pool),
                "calibrator": fit_calibrator(cfg, MODEL_SPECS, get_builder, x_pool, y_pool),
                "train_priors": np.bincount(y_pool, minlength=len(classes)) / len(y_pool),
                "label_encoder": fine_ctx.label_encoder, "features": list(x_pool.columns),
                "cfg": cfg, "full_model_name": name}
    joblib.dump(deployed, out_dir / "deployed.joblib")
    unk = predict_unknown_sources_fine(df_unk, deployed, csv_dir=out_dir)
    unk.to_csv(out_dir / "predictions_unknown_sources.csv", index=False)
    unk_hi = unk[unk["confidence"] >= 0.8]

    # 3. v41 (FY16) prospective check
    v41_thr = None
    if v41_df is not None:
        v41_by_threshold, merged = evaluate_best_model_thresholds(unk, v41_df, name)
        v41_by_threshold.to_csv(out_dir / "v41_by_threshold.csv", index=False)
        v41_per_class(merged).to_csv(out_dir / "v41_per_class.csv")
        merged.to_csv(out_dir / "v41_crossmatch_per_source.csv", index=False)
        ev = merged[_resolved_eval_mask(merged)]
        v41_thr = threshold_table(ev["real_class1_norm"], ev["predicted_label_norm"], ev["confidence"])
        v41_thr.to_csv(out_dir / "v41_accuracy_by_threshold.csv", index=False)

    row = {"borda_rank": rank, "model": name, "tag": cfg["tag"], "weighting": cfg["weighting"],
           "deployed_winner": is_winner,
           "cv_macro_f1": cand.get("cross_split_mean"),
           "cv_accuracy": (cand.get("val_metrics") or {}).get("accuracy"),
           "temperature": calibrator.temperature_}
    for m in ("accuracy", "macro_f1", "bal_accuracy", "kappa"):
        row.update({f"test_{m}": test_ci[m]["value"], f"test_{m}_ci_low": test_ci[m]["ci_low"],
                    f"test_{m}_ci_high": test_ci[m]["ci_high"]})
    row.update({"test_brier_raw": calib["raw"]["brier"], "test_brier_calibrated": calib["calibrated"]["brier"],
                "test_ece_raw": calib["raw"]["ece"], "test_ece_calibrated": calib["calibrated"]["ece"]})
    for group in ("all", "predicted_nonblazar"):
        for t in (0.7, 0.8):
            row[f"test_{group}_conf{t}_n"] = _at(test_thr, group, t, "n")
            row[f"test_{group}_conf{t}_fine_acc"] = _at(test_thr, group, t, "fine_accuracy")
            row[f"test_{group}_conf{t}_superclass_acc"] = _at(test_thr, group, t, "superclass_accuracy")
    row.update({"unassoc_conf0.8_n": len(unk_hi),
                "unassoc_conf0.8_nonblazar_n": int((~unk_hi["predicted_label"].isin(BLAZAR_CLASSES)).sum())})
    if v41_thr is not None:
        for group in ("all", "predicted_nonblazar"):
            row[f"v41_{group}_conf0.8_n"] = _at(v41_thr, group, 0.8, "n")
            row[f"v41_{group}_conf0.8_fine_acc"] = _at(v41_thr, group, 0.8, "fine_accuracy")
            row[f"v41_{group}_conf0.5_superclass_acc"] = _at(v41_thr, group, 0.5, "superclass_accuracy")
    logger.notice(f"  [{rank:02d}] {name} ({cfg['tag']}, {cfg['weighting']}){' <== deployed winner' if is_winner else ''}: "
                  f"test acc {row['test_accuracy']:.3f}  macro-F1 {row['test_macro_f1']:.3f}  "
                  f"ECE {calib['raw']['ece']:.3f}->{calib['calibrated']['ece']:.3f}  "
                  f"unassociated conf>=0.8: {len(unk_hi)}  -> {out_dir}")
    return row


# %%
if "fine" in FAMILIES:
    _cand_logger = get_section_logger("candidates", VALIDATION_LOGS_DIR)
    _cand_logger.notice(f"\n{'=' * 70}\nCANDIDATES — every validated fine candidate on {SPLIT0}\n{'=' * 70}")
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    _fine_ctx = SPLIT_CONTEXTS[SPLIT0][0]
    _df_unk = pd.read_csv(_locate_unknown_catalog())
    _v41_df = load_v41_catalog(V41_FITS_PATH) if V41_FITS_PATH.exists() else None
    _winner_cfg = STAGE4_WINNERS["fine"]["cfg"]
    _rows = []
    for _rank, _cand in enumerate(pbar(STAGE3_RESULTS["fine"], desc="[candidates]"), start=1):
        try:
            _rows.append(evaluate_candidate(_rank, _cand, _fine_ctx, _df_unk, _v41_df,
                                            _cand["cfg"] == _winner_cfg, _cand_logger))
        except Exception as _e:
            _cand_logger.warning(f"  ⚠ candidate {_rank} ({_cand.get('full_model_name')}) failed: "
                                 f"{type(_e).__name__}: {_e}")
    CANDIDATES_SUMMARY = pd.DataFrame(_rows)
    CANDIDATES_SUMMARY.to_csv(CANDIDATES_DIR / "candidates_summary.csv", index=False)
    _cand_logger.notice(f"  Summary ({len(_rows)} candidate(s)) -> {CANDIDATES_DIR / 'candidates_summary.csv'}")
