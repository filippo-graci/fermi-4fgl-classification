"""Master script for the Fermi 4FGL classification pipeline.

Runs the fermi_*.py files in one shared namespace (each file sees the globals the
earlier ones defined), in the fixed order listed in _RF_STEPS. All
settings live in _RF_PROFILES / _RF_CONFIG below; every file reads them through
`_cfg(...)` (defined in fermi_common.py), so a value set here can't be
silently re-assigned by a later file.

Usage (from anywhere; it chdirs to the project folder itself):
    python run_fermi.py                       # run the default steps with the default profile
    python run_fermi.py --profile full        # the full, unrestricted search
    python run_fermi.py --resume              # continue after a crash/interruption
    python run_fermi.py --from validation     # re-run from this step on
    python run_fermi.py --only inference      # re-run a single step
    python run_fermi.py --dry-run             # show the plan and resolved settings, run nothing

Steps: processing, processing_hier, processing_diag, processing_optional, fine, merged,
3levels, validation, inference, analysis, candidates. The four processing steps are parts of
fermi_processing.py that can run separately (each resumes from its own caches):
processing = splits + global feature selection + exports (always needed),
processing_hier = hierarchical local selections (merged/3levels need it),
processing_diag = diagnostics, processing_optional = optional-features comparison.
candidates = test metrics, calibration, unassociated predictions and the v41 check for EVERY
validated fine candidate, not only the winner (fermi_candidates.py).
Steps run in the order above; to run a part later, use --only, e.g.
    python run_fermi.py --only processing_hier

Resuming works at every level:
  * steps     — --resume skips steps marked done in <OUTPUT_ROOT>/_run_logs/run_state.json
  * splits    — fine/merged/3levels skip splits whose outputs already exist
  * nodes     — every model search (fine setup, router, gate, branch) is checkpointed in
                <OUTPUT_ROOT>/_checkpoints/nodes and reloaded instead of recomputed
  * stages    — validation stages and inference already checkpoint themselves
                (force a stage with FORCE_RECOMPUTE_STAGES in the config)

Logs, per run, in <OUTPUT_ROOT>/_run_logs/<timestamp>/:
  run.log     — everything printed to the console (Python and C/C++ libraries alike)
  errors.log  — only the problem lines (⚠, ✗, errors, warnings, crashes, tracebacks), with step
                and time, plus the full traceback of any step that fails
"""

import argparse
import ast
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

_RF_HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
_RF_PROFILES = {
    # The full search: every model x scaler x tag x PCA x weighting.
    "full": {},
    # Reduced search, justified by the CV leaderboards of a full-profile DR4 run (split 0):
    #  - models: the top 4-6 per node were within 1 std; this keeps the ones that win or
    #    tie somewhere. LDA only where it won (small branches).
    #  - scalers: the Stage-0 winner was effectively random -> Standard only.
    #  - PCA: never won at any node (0.00 std loss when removed).
    #  - weighting: all four modes. (The full-run evidence for keeping only {balanced_uncapped,
    #    none} was about balanced accuracy; under macro-F1 the milder modes can win.)
    #  - feature-set tags: all kept (dropping 80/90/SHIFT_ROBUST cost up to 2 std).
    # Disclose in the paper that the search space was narrowed using those CV results. That run
    # used a different partition (pool still with spp), so some rows that informed the narrowing
    # are in the final test set — a small, coarse researcher degree of freedom.
    "reduced": {
        "SEARCH_MODELS": ["RandomForest", "HistGBC", "CatBoost", "LogReg",
                          "MLP_1x64", "MLP_1x256", "MLP_3x256_128_64"],
        "SEARCH_MODELS_EXTRA": {"BINARY_SYSTEMS": ["LDA"], "GALACTIC_EXT": ["LDA"]},
        "SEARCH_SCALERS": ["Standard"],
        "PCA_VARIANTS": {},
        "WEIGHTING_MODES": ["none", "balanced", "dampened", "balanced_uncapped"],
        "TOP_K_SHORTLIST": 5,
        "STAGE3_SHORTLIST_TOP_K": 5,
    },
    # Plumbing check: every step end to end on a tiny search (CPU-only LogReg, the 90% feature
    # sets, 2 stability repeats, 1 shortlist repeat). Its numbers mean nothing, so it gets its own
    # output AND processing folders: its 2-repeat feature selections must never be reused.
    "smoke": {
        "RUN_NAME": "smoke",
        "PROCESSING_ROOT": Path("./fermi_module1_processing_smoke/"),
        "STABILITY_N_PAIRS": 2,
        "SEARCH_MODELS": ["LogReg"],
        "SEARCH_TAGS": ["90", "router_90", "gate_90", "subgate_90", "nonblazar_90", "branch_90"],
        "SEARCH_SCALERS": ["Standard"],
        "PCA_VARIANTS": {},
        "WEIGHTING_MODES": ["balanced_uncapped"],
        "TOP_K_SHORTLIST": 2,
        "STAGE3_SHORTLIST_TOP_K": 2,
        "N_REPEATS_SHORTLIST_CV": 1,
        "RUN_CALIBRATION_CHECK_NOW": False,
    },
}

_RF_CONFIG = {
    "PROFILE": "reduced",
    "RUN_NAME": "final",             # becomes part of the output folder name
    # All three families, single split: processing (+ hierarchical selections), the three model
    # searches, validation, unassociated-source inference, analysis plots, per-candidate comparison.
    # Full set: ["processing", "processing_hier", "processing_diag", "processing_optional", "fine",
    # "merged", "3levels", "validation", "inference", "analysis", "candidates"].
    "STEPS": ["processing", "processing_hier", "fine", "merged", "3levels", "validation", "inference",
              "analysis", "candidates"],
    # Architecture families validated/deployed. ["fine"] skips merged/3levels everywhere
    # (their steps are dropped from STEPS); they also need the processing_hier step.
    "FAMILIES": ["fine", "merged", "3level"],
    "WORKDIR": _RF_HERE.parent,         # the modules use paths relative to the project folder

    # How the test estimate is protected from model selection (see _RF_DESIGNS):
    #   "single_split" — one 70/30 split; search + selection on its train, one clean test score.
    #   "per_split"    — 5 splits, the whole search + selection repeated inside each split
    #                    (~5x the search cost); mean test score with a Nadeau-Bengio CI.
    #   "shared"       — legacy: 5 splits, search on split 0, Stage 3 on all splits' train;
    #                    test scores are optimistic (the splits overlap).
    "VALIDATION_DESIGN": "single_split",

    # Deploy a different Stage-3 candidate than the Borda winner, e.g. {"fine": "CatBoost"} (matched
    # against the full model name or the model name). Forces validation Stage 4+ and inference to
    # recompute in this output folder. The "candidates" step evaluates every candidate anyway.
    "WINNER_OVERRIDE": {},

    # Fine Setup A (all classes incl. bcu). Off: only Setup B is used downstream
    # (validation, inference, merged/3-level all use B), so A only feeds the A-vs-B comparison.
    "RUN_FINE_SETUP_A": False,

    # Catalog / metric / hardware
    # New tag -> new processing folder: Module 1's caches have no invalidation, so change the
    # tag after any change to the input data or to feature selection.
    "VERSION_TAG": "4FGL_DR4",
    # Input data from pipeline/preprocessing.py. Read by processing, validation and inference.
    "DATA_PATH": Path("./data/processed/without_optional_features/"),
    "DATA_PATH_WITH_OPTIONAL": Path("./data/processed/with_optional_features/"),
    # macro-F1, not balanced accuracy: balanced accuracy (= macro recall) ignores false positives,
    # so with 18 classes (10 with <= 3 test rows) and class weights it rewarded sending hundreds of
    # bll/fsrq into rare classes (balanced-accuracy run: test accuracy 0.59, 72 bll -> sbg). Macro-F1
    # still weights every class equally but penalises those false positives (Sokolova & Lapalme 2009).
    "SELECTION_METRIC": "macro_f1",
    "USE_GPU": True,
    "SEED": 42,

    # CatBoost (early stopping on a held-out slice, not a fixed tree count)
    "CATBOOST_MAX_ITERATIONS": 5000,
    "CATBOOST_EARLY_STOPPING_ROUNDS": 100,
    "CATBOOST_LEARNING_RATE": 0.05,

    # Resume / recompute
    "FINE_FORCE_RECOMPUTE": False,      # True would delete and recompute finished fine splits
    "FORCE_RECOMPUTE_STAGES": [],       # validation stages to recompute, e.g. ["stage1"]
    "FORCE_RECOMPUTE_INFERENCE": False,
    "FORCE_RECOMPUTE_V41_VALIDATION": False,
    "NODE_CACHE_VERSION": 4,            # bump after a code change that should invalidate node checkpoints (4: fp32 MLP inference)

    # Optional / expensive validation extras
    "RUN_CALIBRATION_CHECK_NOW": True,
    "RUN_WINNERS_CURSE_DIAGNOSTIC_NOW": False,
    "RUN_STAGE6_SIGNIFICANCE_NOW": False,
    "RUN_SHAP_EXPLAINABILITY_NOW": False,

    # Catalogs for the v41 prospective check (Stage 10)
    "V41_FITS_PATH": Path("./data/catalogs/gll_psc_v41.fit"),
    "DR4_V35_FITS_PATH": Path("./data/catalogs/gll_psc_v35.fit"),
}

_RF_STEPS = [  # (name, file, settings for that step); "common" always runs first and in full
    ("common", "fermi_common.py", {}),
    ("processing", "fermi_processing.py", {"PROCESSING_PARTS": ["core"]}),
    ("processing_hier", "fermi_processing.py", {"PROCESSING_PARTS": ["hier"]}),
    ("processing_diag", "fermi_processing.py", {"PROCESSING_PARTS": ["diag"]}),
    ("processing_optional", "fermi_processing.py", {"PROCESSING_PARTS": ["optional"]}),
    ("fine", "fermi_fine.py", {}),
    ("merged", "fermi_merged.py", {}),
    ("3levels", "fermi_3levels.py", {}),
    ("validation", "fermi_validation.py", {}),
    ("inference", "fermi_inference.py", {}),
    ("analysis", "fermi_analysis.py", {}),
    ("candidates", "fermi_candidates.py", {}),
]
_RF_STEP_FILES = {name: (file, settings) for name, file, settings in _RF_STEPS}
# Steps that checkpoint everything internally: when already done, they are replayed
# (cheap) instead of loaded as definitions only, because later steps read their
# in-memory results (e.g. analysis reads STAGE5_RESULTS).
_RF_REPLAY_WHEN_DONE = {"validation", "inference"}  # "candidates" reads their in-memory results
_RF_SPLIT_LOOP_STEPS = {"fine", "merged", "3levels"}  # report per-split crashes via FAILED_SPLITS

_RF_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # cursor moves from nested progress bars
_RF_PROBLEM = re.compile(r"⚠|✗|Traceback|Error|ERROR|WARNING|\[warning\]|CRASH|FAILED|failed|Exception")


_RF_DESIGNS = {  # VALIDATION_DESIGN -> settings the modules read through _cfg
    "single_split": {"N_VALIDATION_SPLITS": 1, "MODULE_MULTI_SPLIT": False, "PER_SPLIT_SELECTION": False},
    "per_split": {"N_VALIDATION_SPLITS": 5, "MODULE_MULTI_SPLIT": True, "PER_SPLIT_SELECTION": True},
    "shared": {"N_VALIDATION_SPLITS": 5, "MODULE_MULTI_SPLIT": False, "PER_SPLIT_SELECTION": False},
}


def _rf_resolve_config(profile_name):
    cfg = dict(_RF_CONFIG)
    cfg.update(_RF_PROFILES[profile_name])
    cfg["PROFILE"] = profile_name
    cfg["FINE_SETUP_KEYS"] = ["A", "B"] if cfg["RUN_FINE_SETUP_A"] else ["B"]
    cfg.update(_RF_DESIGNS[cfg["VALIDATION_DESIGN"]])
    if cfg["WINNER_OVERRIDE"]:  # a new winner -> its deployment and predictions must be redone
        cfg.update(FORCE_RECOMPUTE_INFERENCE=True, FORCE_RECOMPUTE_V41_VALIDATION=True)
    skip = {"merged"} - set(cfg["FAMILIES"]) | ({"3levels"} if "3level" not in cfg["FAMILIES"] else set())
    cfg["STEPS"] = [s for s in cfg["STEPS"] if s not in skip]
    # One output folder per design and family set: the modules skip splits/stages already on
    # disk, and the checkpoints of different designs/family sets mean different things.
    families = "" if len(cfg["FAMILIES"]) == 3 else "_" + "-".join(cfg["FAMILIES"])
    cfg.setdefault("OUTPUT_ROOT", Path(f"./outputs_{cfg['VERSION_TAG']}_{cfg['SELECTION_METRIC']}_"
                                       f"{cfg['RUN_NAME']}_{cfg['VALIDATION_DESIGN']}{families}/"))
    cfg.setdefault("PROCESSING_ROOT", Path(f"./fermi_module1_processing_{cfg['VERSION_TAG']}/"))
    cfg.setdefault("NODE_CACHE_DIR", Path(cfg["OUTPUT_ROOT"]) / "_checkpoints" / "nodes")
    return cfg


def _rf_fingerprint(cfg):
    keep = {k: v for k, v in cfg.items() if not k.startswith(("FORCE_", "FINE_FORCE", "WORKDIR"))}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# Logging: capture fd 1/2 (Python prints AND C/C++ library output) into run.log/errors.log
# --------------------------------------------------------------------------------------
class _RfCapture:
    def __init__(self, run_dir):
        self.step = "-"
        self.run_log = open(run_dir / "run.log", "a", buffering=1)
        self.err_log = open(run_dir / "errors.log", "a", buffering=1)
        self.n_problems = 0

    def problem(self, text):
        self.n_problems += 1
        self.err_log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{self.step}] {text}\n")

    def __enter__(self):
        sys.stdout.flush(); sys.stderr.flush()
        self.console = os.dup(1)
        self.read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1); os.dup2(write_fd, 2); os.close(write_fd)
        sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        return self

    def _pump(self):
        buf = b""
        while chunk := os.read(self.read_fd, 65536):
            os.write(self.console, chunk)
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                line = _RF_ANSI.sub("", raw.decode("utf-8", "replace").split("\r")[-1])  # bar's final state
                self.run_log.write(line + "\n")
                if _RF_PROBLEM.search(line):
                    self.problem(line.strip())

    def __exit__(self, *exc):
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(self.console, 1); os.dup2(self.console, 2)  # closes the pipe's write ends -> pump ends
        self.thread.join(timeout=10)
        os.close(self.read_fd); os.close(self.console)
        self.run_log.close(); self.err_log.close()


# --------------------------------------------------------------------------------------
# Running one file in the shared namespace
# --------------------------------------------------------------------------------------
_RF_CHEAP_CALLS = {"_cfg", "Path", "LabelEncoder", "fit", "list", "dict", "set", "tuple", "sorted",
                   "range", "frozenset", "str", "int", "float", "len", "keys", "items", "values"}


def _rf_definitions_only(tree):
    """Keeps imports, defs, classes and constant settings (ALL_CAPS assignments that only
    call cheap constructors) — everything later steps need from a skipped step, nothing
    that computes or writes results."""
    def cheap(node):
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            f = call.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name not in _RF_CHEAP_CALLS:
                return False
        return True

    def setting_target(t):
        base = t.value if isinstance(t, ast.Subscript) else t
        return isinstance(base, ast.Name) and base.id.lstrip("_")[:1].isupper() and base.id.upper() == base.id

    body = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            body.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None and cheap(n.value):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if all(setting_target(t) for t in targets):
                body.append(n)
    tree.body = body
    return tree


def _rf_exec(path, ns, defs_only):
    tree = ast.parse(path.read_text(), filename=str(path))
    if defs_only:
        tree = _rf_definitions_only(tree)
        # A setting may depend on something only the full run computes; skip it quietly.
        for stmt in tree.body:
            code = compile(ast.Module(body=[stmt], type_ignores=[]), str(path), "exec")
            try:
                exec(code, ns)
            except NameError:
                pass
    else:
        exec(compile(tree, str(path), "exec"), ns)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def _rf_plan(args, cfg, state):
    names = [n for n, *_ in _RF_STEPS[1:]]
    selected = [s for s in names if s in cfg["STEPS"]]
    if args.only:
        selected = [args.only]
    elif args.from_step:
        later = names[names.index(args.from_step):]
        selected = [args.from_step] + [s for s in later[1:] if s in cfg["STEPS"]]
    done = {s for s, v in state["steps"].items() if v.get("status") == "done"}
    last = max(names.index(s) for s in selected)
    plan = []
    for s in names[:last + 1]:
        if s in selected and not (args.resume and s in done):
            plan.append((s, "run"))
        elif s in _RF_REPLAY_WHEN_DONE and s in done:
            plan.append((s, "replay"))  # re-executed; its own checkpoints make it fast
        else:
            plan.append((s, "definitions"))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=_RF_CONFIG["PROFILE"], choices=sorted(_RF_PROFILES))
    ap.add_argument("--resume", action="store_true", help="skip steps already marked done")
    ap.add_argument("--from", dest="from_step", choices=[n for n, *_ in _RF_STEPS[1:]])
    ap.add_argument("--only", choices=[n for n, *_ in _RF_STEPS[1:]])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = _rf_resolve_config(args.profile)
    os.chdir(cfg["WORKDIR"])
    logs_root = Path(cfg["OUTPUT_ROOT"]) / "_run_logs"
    state_path = logs_root / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"steps": {}}
    fp = _rf_fingerprint(cfg)
    plan = _rf_plan(args, cfg, state)

    print(f"Profile {cfg['PROFILE']}  |  output {Path(cfg['OUTPUT_ROOT']).resolve()}  |  config {fp}")
    for s, mode in plan:
        print(f"  {s:11s} {mode}")
    if state.get("config_fingerprint") not in (None, fp):
        print(f"⚠ settings differ from the previous run in this output folder ({state['config_fingerprint']}); "
              "finished splits/stages there were produced with the old settings. Use a new RUN_NAME "
              "unless that is intended.")
    if args.dry_run:
        print(json.dumps(cfg, indent=2, default=str))
        return 0

    run_dir = logs_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    state["config_fingerprint"] = fp
    ns = sys.modules["__main__"].__dict__  # same namespace the notebook used: pickles stay compatible
    ns["_FERMI_CONFIG"] = cfg

    def save_state():
        state_path.write_text(json.dumps(state, indent=2, default=str))

    # A stop request (Ctrl-C, `kill`, `timeout`) becomes a clean interruption: the running step is
    # recorded as "interrupted" and the run resumes with --resume. SIGINT is installed explicitly
    # because a process started in the background (`nohup ... &`) inherits it as IGNORED.
    def _stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signal.Signals(signum).name}")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    exit_code = 0
    with _RfCapture(run_dir) as cap:
        print(f"Run logs -> {run_dir}")
        for s, mode in [("common", "run")] + plan:
            cap.step = s
            step_file, step_settings = _RF_STEP_FILES[s]
            path = _RF_HERE / step_file
            ns["_FERMI_CONFIG"] = {**cfg, **step_settings}
            t0 = time.time()
            entry = state["steps"].setdefault(s, {}) if s != "common" else {}
            prev_status = entry.get("status")
            entry.update(started=datetime.now().isoformat(timespec="seconds"), mode=mode, status="running")
            if s != "common":
                save_state()
            print(f"\n{'#' * 78}\n# STEP {s} ({mode}) — {path.name}\n{'#' * 78}")
            try:
                ns.pop("FAILED_SPLITS", None)
                _rf_exec(path, ns, defs_only=(mode == "definitions"))
                failed = ns.get("FAILED_SPLITS") if (s in _RF_SPLIT_LOOP_STEPS and mode == "run") else None
                if failed:
                    raise RuntimeError(f"{len(failed)} split(s) crashed: {failed} — see CRASH_traceback.txt "
                                       "under results_aggregated/<split>/")
                status = "done" if mode != "definitions" else (prev_status or "not run")
            except KeyboardInterrupt:
                status, exit_code = "interrupted", 130
                cap.problem(f"STEP {s} interrupted (Ctrl-C / stop signal) — resume with --resume")
            except BaseException as e:
                status, exit_code = "failed", 1
                entry["error"] = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                cap.problem(f"STEP {s} FAILED — {type(e).__name__}: {e}\n{tb}")
                cap.run_log.write(tb)  # full traceback in run.log too, without re-matching it into errors.log
                print(f"STEP {s} FAILED — {type(e).__name__}: {e}  (traceback in errors.log)")
            dt = time.time() - t0
            if s != "common":
                entry.update(status=status, finished=datetime.now().isoformat(timespec="seconds"),
                             seconds=round(dt, 1), run_dir=str(run_dir))
                if status == "done":
                    entry.pop("error", None)
                save_state()
            print(f"# STEP {s}: {status} in {dt / 60:.1f} min")
            if exit_code:
                print(f"\nStopped at step '{s}'. Fix the problem, then resume with:\n"
                      f"    python {Path(__file__).name} --profile {cfg['PROFILE']} --resume")
                break
        cap.step = "-"
        print(f"\nProblems logged: {cap.n_problems} -> {run_dir / 'errors.log'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
