# Why the Fermi pipeline does what it does — a plain-language walkthrough

This covers every design choice in `fermi_processing.py` / `fermi_common.py` that isn't self-explanatory: the ones where a reasonable person might ask "wait, why not just do the obvious thing instead?" For each one: what it looks like on the surface, why the obvious alternative is actually worse, and what's honestly still imperfect about it.

---

## A. Decisions about the labels themselves

### A1. Merging `fsrq`/`ssrq`/`css` into one label

**Looks odd because:** you're throwing away information the catalog already gives you — why not let the model try to tell three real classes apart?

**Why it's actually right:** the three labels differ by a property the model never sees. They're distinguished by *radio-band* measurements (how steep the radio spectrum is, how compact the radio structure is) — not by anything in the gamma-ray features being fed to the classifier. All three are believed to shine via the same physical process. So this isn't "the model isn't good enough to separate them" — it's "the information needed to separate them literally isn't in the input." This was also checked empirically, not just assumed: a test model given all three labels separately guessed close to 1/3-1/3-1/3 for every source — a flat, structureless guess, which is the specific signature of "truly can't tell," as opposed to a skewed-but-uncertain guess (which is what a real-but-rare class usually looks like).

**Honest limitation:** this is a design decision baked into the taxonomy, not something a downstream user can second-guess — if radio data were ever added as a feature, this merge should be revisited.

### A2. Dropping `bcu` from the main pipeline

**Looks odd because:** `bcu` is a real label in the catalog — why exclude legitimate data?

**Why it's actually right:** `bcu` doesn't mean "a class of object" — it means "we're confident this is a blazar, but the follow-up spectrum wasn't good enough to tell if it's the FSRQ flavor or the BL Lac flavor." Training a model to predict `bcu` would mean training it to predict *which sources got a deep enough follow-up observation* — an artifact of telescope time allocation, not astrophysics. A separate baseline ("Setup A") keeps `bcu` in, purely so there's still an apples-to-apples comparison against older pipeline versions and naive baselines; the primary pipeline ("Setup B") excludes it, matching the precedent set by the closest published comparable study (Coronado-Blázquez 2022, which does the same for the same reason).

### A3. Dropping RA/Dec but keeping Galactic latitude/longitude

**Looks odd because:** they're both "just coordinates" — why treat them differently?

**Why it's actually right:** equatorial coordinates (RA/Dec) encode nothing physical — they're an accident of which way Earth's axis happens to point. Galactic coordinates encode real structure: pulsars and supernova remnants cluster near the galactic plane; blazars, being outside our galaxy, are scattered roughly evenly across the sky. Position is genuinely informative here, so throwing it away would be discarding real signal.

**The risk this creates, and how it's checked:** if position is a real feature, a model could "cheat" by learning sky position as a shortcut for the class label, instead of learning genuine spectral/temporal physics — and from the accuracy number alone you couldn't tell the difference. That's exactly what the spatial-agreement check (A3-adjacent, described in section D below) exists to catch.

### A4. Running a whole separate comparison for "optional" (manually purged) columns

**Looks odd because:** it sounds like duplicated work — the pipeline already has an automated way to pick good features, so why also do a manual purge and then compare the two?

**Why it's actually right:** the manual purge and the automated statistical selection could disagree, and each failure mode is worth catching. Maybe a column was manually removed as "redundant" but the data-driven ranking would actually still find it useful (worth re-examining the manual call). Or maybe the automated ranking likes a column only because it's a near-duplicate of something better — something the manual purge had already correctly caught. Running the identical selection engine on both column sets and measuring the overlap turns "trust me, I purged the right things" into a checkable claim.

---

## B. Picking which features to use

### B1. "Cover 90% of the importance" instead of "keep the top 15 features"

**Looks odd because:** a fixed number is simpler and easier to reason about.

**Why it's actually right:** how concentrated the useful signal is varies wildly by classification problem. For one branch of the taxonomy, 5 features might carry 95% of everything separable; for another, those same 5 might carry only 40%, with real signal spread across 30 more features. A fixed "top 15" rule either wastes features on one problem or starves another. A percentage-of-importance-mass rule adapts automatically — at the cost of the *size* of the selected set no longer being predictable in advance.

### B2. Capping the feature count at roughly one feature per 5 training rows

*(Covered in depth in the earlier part of this conversation — the short version:)* Letting a small class pick as many features as the importance-coverage rule wants would let a low-sample branch essentially guarantee it can perfectly separate its own training data with pure noise, since with enough dimensions relative to few points, some accidental combination of features will always look perfectly discriminating. The cap is what stops the model from mistaking a coincidence in a few dozen rows for a real pattern — it discards the *least reproducible* candidate features first, since the ranking that decides which features to cap away is itself sorted by how consistently each feature showed up as useful.

### B3. The feature cap gets tighter deeper in the classification hierarchy

*(Also covered earlier.)* Nodes near the bottom of the taxonomy tree naturally see fewer training rows (only the sources already routed to that specific physical family), so their feature budget shrinks accordingly. There's a floor below which the pipeline doesn't even attempt a bespoke local selection — it falls back to the full, well-supported global feature set instead, and that floor is deliberately set differently for the shallower 2-level tree (40 rows) versus the deeper 3-level tree (8 rows), because requiring the stricter 40-row floor everywhere would force almost every deep node in the 3-level tree to fall back, defeating the entire point of building a deeper hierarchy in the first place.

---

## C. Making the selection trustworthy, not just accurate-looking

### C1. Running the whole importance calculation 10 times, then scoring by "mean minus a penalty for inconsistency" (LCB) instead of by raw mean

**Looks odd because:** if you've already computed a good importance estimate once, why redo it 10 times and then, oddly, actively *punish* features for having high variance instead of just trusting the average?

**Why it's actually right:** a single train/validation split's importance ranking is a noisy read — reshuffle which rows land in which fold, and the ranking can shift meaningfully, especially between features with genuinely similar effect sizes. Repeating the whole process 10 times with independently reshuffled folds gives you both a mean *and* a sense of how much that mean can be trusted. Scoring by `mean − z·std` (the lower-confidence-bound) rather than plain mean means a feature that looks great on average but wildly inconsistent across the 10 repeats gets ranked below a feature with the same average but tighter, more reproducible importance. That's specifically the property you want for something that has to generalize to genuinely new, unlabeled sources — not just to reproduce whichever lucky split you happened to draw.

### C2. A second filter on top of that: a feature must show up in most of the 10 individual repeats, not just score well on the combined LCB ranking

**Looks odd because:** doesn't the LCB score in C1 already account for consistency?

**Why it's actually right:** LCB penalizes *how much* a feature's importance varies, but a feature could still clear that bar by chance in a couple of lucky repeats without being reliably useful overall. Requiring a feature to have *actually been the kind of feature that gets separately selected* in at least 70% of the 10 individual repeats is a second, independent form of evidence — this is the direct application of a published statistical technique (stability selection) built for exactly this purpose: using selection *frequency* across resampled data as its own error control, on top of whatever the importance score itself says. If applying this filter would wipe out an entire feature set (possible when no single feature dominates), the code falls back to the unfiltered selection rather than shipping something unusable — a deliberate, disclosed departure from the strict version of the technique.

### C3. Building two versions of every feature set — one with correlated features removed, one without

**Looks odd because:** it sounds like doubling the work for no clear benefit.

**Why it's actually right:** different model types react to redundant, correlated features completely differently. Tree-based models (Random Forest, gradient boosting) don't care — splitting on either of two nearly-identical features gets you the same partition. Distance-based models (KNN, SVM, LDA) *do* care — redundant features distort the distance calculations those methods depend on. Rather than pick one philosophy for every downstream model, the pipeline produces both: the full set for tree ensembles, and a "decluttered" companion (one representative feature per cluster of mutually-correlated features, picked via hierarchical clustering on rank correlation) for anything distance-based.

**A smaller but deliberate choice inside this:** the correlation used is *Spearman* (rank correlation), not the more familiar Pearson (linear correlation). Several of the catalog's spectral measurements are heavy-tailed and only monotonically related to each other, not linearly — rank correlation catches "these two features carry the same information" even when a straight-line correlation coefficient would understate it.

---

## D. Checking the pipeline's own blind spots

### D1. Cross-checking the main model's feature ranking against SHAP, a Random Forest, and a plain logistic regression

**Looks odd because:** if you already trust the main model's importance ranking enough to use it, why go verify it against three other, weaker methods?

**Why it's actually right:** which features look "important" is not a universal, model-independent fact — it's relative to the specific algorithm doing the ranking (this is a well-established result in the feature-selection literature). A ranking that only one model family has ever seen is not automatically valid evidence for how a *different* kind of model (say, a linear one) will use those same features. Of the three cross-checks, agreement with the plain logistic regression is treated as the strongest evidence, precisely because it's the most different in kind from the tree-based model doing the primary ranking — nonlinear-vs-linear agreement is much stronger proof the signal is real than nonlinear-vs-nonlinear agreement would be.

**What doesn't happen with a disagreement:** none of these three checks actually change which features get selected. They're diagnostic, not corrective — the intended use is: if a paper's best downstream model later turns out to be linear or distance-based, you can point to the matching cross-check number as evidence the feature selection didn't handicap it. A poor agreement score is a signal to go investigate that specific case by hand, not something the pipeline automatically reacts to.

### D2. Re-running the whole importance ranking separately for sources near vs. far from the galactic plane

**Looks odd because:** why recompute the same thing twice on subsets of the same data?

**Why it's actually right:** this is the direct empirical test of the concern raised in A3 — that a model might be using sky position as a shortcut instead of learning real physics. If a feature's importance ranking looks similar whether you're looking at galactic-plane-adjacent sources or far-from-the-plane sources, that's evidence the model isn't quietly substituting "is this near the plane" for genuine spectral discrimination. A large disagreement between the two would be the trigger to go inspect that specific feature, not to distrust the whole feature set.

**A deliberate departure from the pipeline's usual "always held-out" rule:** this specific check is computed in-sample (fit and score on each hemisphere's *entire* subset), not via cross-validation. That's intentional — the goal here is comparing two rankings to each other, not estimating how well either would generalize, so held-out evaluation isn't actually the right tool for this particular question.

### D3. Measuring whether *missingness itself* is informative, but never using that information

**Looks odd because:** if a feature's missingness pattern is informative, why compute that and then throw it away?

**Why it's actually right:** there's a real scientific finding buried in "which values are missing" — a spectral-curvature parameter, say, might only be well-measured (and hence non-missing) for bright sources of one particular class, which is a genuine astrophysical selection effect worth knowing about. But building that pattern into the model as a usable signal is a separate engineering decision with its own pitfalls (it would need to generalize to the unassociated sources the model is eventually applied to, whose missingness pattern might differ for unrelated reasons). So the check is reported for awareness and future work, deliberately without acting on it yet — surfacing a candidate improvement rather than silently baking in an unvetted one.

### D4. Requiring *both* a statistical significance test (KS) and an effect-size measure (PSI) before flagging a feature's distribution as having shifted between labeled and unlabeled sources

**Looks odd because:** doesn't a significant statistical test already tell you there's a real difference?

**Why it's actually right:** with thousands of rows, a KS test becomes significant for almost any difference at all, including ones too small to matter in practice — statistical significance and practical importance come apart at this sample size. PSI (Population Stability Index), by contrast, measures the *size* of the shift but has no built-in significance test and can be noisy on its own. Requiring both — a stricter-than-usual significance threshold *and* a PSI value past a conventional "severe shift" cutoff — is a deliberate belt-and-suspenders combination: catch shifts big enough to actually matter for deployment, without drowning in false alarms from trivial differences in a large catalog.

**What happens with a flagged feature:** nothing automatic. Flagged features aren't dropped from the main feature sets — instead, a separate "shift-robust" companion feature set is produced with them removed, and it's left to the downstream modeling stage to decide, case by case, whether protecting against this specific risk is worth trading away some accuracy on the known population.

**Disclosed gap:** running one significance test per feature, across many features, without a formal multiple-comparisons correction (like Bonferroni) is a known simplification — the stricter-than-usual threshold compensates informally, not rigorously.

---

## E. Handling small and imbalanced classes honestly

### E1. Rare classes are never held out during cross-validation — they only ever appear in training folds

**Looks odd because:** doesn't that mean the model's reported accuracy never actually gets *tested* on those classes?

**Why it's actually right, and yes, this is a real trade-off:** with only a handful of examples of some class, holding even one out for testing removes a meaningful fraction of the model's only chances to ever learn that class — you'd be trading away training signal for a held-out test that, with so few points, would mostly be noise anyway (a single held-out example scores as a degenerate right-or-wrong coin flip, not a meaningful accuracy estimate). Keeping rare-class rows in every fold's training set means the model still benefits from seeing them, at the acknowledged cost that the reported accuracy is conservative — it doesn't reflect any genuine held-out test of the model's ability to recognize those classes. The pipeline doesn't hide this: it separately tracks which classes never got a genuine held-out test, so the conservatism is visible rather than silently smoothed into one aggregate number.

### E2. Below a certain total sample size, the whole cross-validation scheme switches to Leave-One-Out

**Looks odd because:** LOO is usually described as the most expensive, "last resort" CV method.

**Why it's actually right for this specific situation:** for a genuinely tiny classification problem (some branches have training sets in the tens of rows), an ordinary k-fold split would leave each fold with only a couple of examples per class — not enough to learn from or evaluate against sensibly. LOO — holding out exactly one row at a time — is the only scheme that still uses every other row for training on each pass, which matters enormously when there's so little data to begin with.

### E3. Scores are computed by pooling every out-of-fold prediction together, rather than averaging a metric per fold

**Looks odd because:** averaging fold scores sounds like the standard, simpler approach.

**Why it's actually right:** under Leave-One-Out specifically, each "fold" is a single held-out row — so a per-fold accuracy score is either exactly right or exactly wrong, a 0-or-1 coin flip. Averaging a pile of individual coin flips conflates genuine model performance with pure sampling noise from having just one row per fold. Pooling all the individual (true, predicted) pairs first, then computing one metric over the whole pool, sidesteps that problem entirely and stays well-defined no matter how many folds there are — including the extreme case where the fold count equals the number of rows.

### E4. Everything is scored by balanced accuracy, not plain accuracy

This is arguably the least surprising choice here, but worth stating: with a catalog where some classes have thousands of members and others have a few dozen, plain accuracy can look excellent while completely ignoring the rare classes (a model that always guesses the majority class can still score well on raw accuracy). Balanced accuracy — averaging the per-class accuracy rather than the overall hit rate — forces rare classes to actually count.

---

## F. The hierarchical taxonomy design

### F1. Physical superclasses grouped by emission mechanism, not by how the source was discovered or what it looks like

The seven broad groupings (blazars split into two physically distinct types, pulsars, binary systems, supernova remnants, etc.) are organized around *how each type of object actually produces gamma rays* — jet physics, neutron-star spin-down, shock acceleration — rather than more superficial groupings like discovery method or sky appearance. This matters because it's the reason a hierarchical classifier makes sense at all: each of these physically distinct groups has its own characteristic spectral/temporal signature, so letting each branch of the classification tree specialize its own feature set (see B3, F2) is chasing something real, not an arbitrary partition.

### F2. Every branch of the hierarchy gets to pick its own features, instead of one feature set being forced on every decision

Rather than run one global feature-selection pass and hand the same feature list to every node of the classification tree, the pipeline runs a separate, local feature-selection pass for each router/gate/branch decision. This exists because different physical distinctions plausibly hinge on different measurements — distinguishing two blazar subtypes might hinge on spectral curvature, while distinguishing pulsar subtypes might hinge on variability — and forcing one feature set onto every decision would dilute or hide those decision-specific signals.

### F3. Branch-level feature selection is computed once and shared between the two hierarchical designs, not recomputed separately for each

The pipeline builds two different hierarchical trees (a simpler 2-level design and a deeper 3-level design), and both need "branch" classifiers for splitting within a physical superclass. Rather than run that selection twice — once per hierarchical design — the underlying per-superclass selection is computed a single time and then filtered into each design's own cache. This isn't just an efficiency shortcut: re-deriving feature selection more times than necessary multiplies the number of independent statistical decisions being made on the same data, which is exactly the kind of "researcher degrees of freedom" the pipeline's stability-selection machinery is trying to keep in check elsewhere.

---

## G. Validating the whole procedure, not just one result

### G1. The entire split → select-features → export pipeline is re-run 5 times on independent data splits, rather than validating one fixed feature set 5 times

**Looks odd because:** if you already have a good feature set, why not just check that its downstream accuracy holds up on a few different train/test splits, which sounds cheaper?

**Why it's actually right:** that cheaper version would only tell you whether one particular *chosen* feature set generalizes — it says nothing about whether the *selection process itself* is stable, i.e. whether a different random draw of the data would have led you to select a meaningfully different set of features in the first place. The more expensive version — re-running the entire selection from scratch on 5 independently-drawn splits — is what actually tests the thing worth testing: does this whole recipe reliably arrive at similar answers, not just "does one particular answer hold up."

### G2. Feature selection isn't re-derived inside every fold of every downstream model's hyperparameter search (no full nested cross-validation)

**Looks odd because:** the textbook-correct, most rigorous approach would nest feature selection inside every single model-selection fold, for every model, across every pipeline shape.

**Why the cheaper approach was chosen anyway, and what it costs:** doing the fully nested version here would multiply an already expensive repeated-CV procedure by every hyperparameter configuration and fold tried by the later modeling stages — described as genuinely unaffordable on the hardware available, and judged not to be worth it anyway, since the whole point of the stability-selection machinery (section C) is to make the chosen feature set already insensitive to which particular split produced it. The actual, disclosed cost of skipping full nesting: the internal cross-validated scores used *inside* the later modeling stages to pick a model or hyperparameters carry a small, well-understood optimism (a documented statistical phenomenon — tuning and scoring on overlapping data biases the internal score upward). Critically, this optimism only touches the internal scores used for picking a model; it does **not** touch the pipeline's final held-out test-set number, because that test set is never touched by feature selection at any point and is used exactly once, after all model selection is already finished.

---

## H. Smaller engineering choices that are easy to misread as bugs

### H1. Every downstream export includes a "does nothing" scaling-parameters file

`scaling_parameters.json` records the mean/std/median/IQR the pipeline observed for every feature — but nothing in this module actually applies those numbers to the exported data. This is intentional: later modeling stages fit their own scalers inside their own cross-validation pipelines, so this file exists purely as a diagnostic record ("here's what the training data's scale looked like at this point"), not as a transformation anyone should apply.

### H2. Diagnostic plots are only generated for the first ("primary") split, not for every one of the 5 validation splits

Since the 5 splits exist purely to confirm the *procedure* is stable (see G1), regenerating a full set of plots for every confirmation split would just be visual clutter with no additional scientific value — the plots exist to let a human eyeball the primary split's results, not to be produced mechanically five times over.

### H3. The label encoder is refit from scratch for every split, not built once for the whole dataset

Because rare classes can occasionally fall below the minimum sample count needed to stratify a train/test split, the exact set of class labels present in a given split's training data can shift split to split. Refitting the encoder per split (rather than reusing one global encoder) keeps each split's numeric label codes consistent with whatever classes actually survived into that split's training set.

### H4. A defensive patch exists for columns that are 100%-missing within a single cross-validation fold

Some catalog columns are only populated for certain classes, so a column that has *some* data overall can still be entirely empty within one particular fold's training rows. Left unhandled, this breaks the standard imputation method in a way that's easy to miss: an all-missing column's imputed "average" value is itself undefined, and that undefined value can then leak into and corrupt the imputation of every *other* column that uses it as a predictor — silently poisoning the whole fold's data, not just the one bad column. The fix is narrow and deliberately limited in scope: only fully-empty-within-this-fold columns get a simple constant fill; any column with even partial data is left to the normal, more sophisticated imputation method.

### H5. A specific historical bug fix: filtering both the test features and test labels together, not just the features

An earlier version of this pipeline filtered out test-set rows whose label was one the encoder never saw during training — but only applied that filter to the feature matrix, not to the label series itself. This didn't cause visible failures before, because a class that survives the earlier rare-class filtering step is, by construction of a stratified split, virtually guaranteed to appear in both halves of the split — so the mismatch was latent, not actively triggered. It's fixed now by filtering both together, unconditionally, closing off a subtle way the two could have silently drifted out of alignment under different data.

### H6. A disclosed, unfixed limitation: no cache invalidation by content hash

Every expensive computation in this module is cached to disk so it isn't repeated needlessly. But the cache has no mechanism to detect that an *upstream* constant changed and the cached result is now stale — if a threshold or setting changes, the cache must be manually cleared, or a re-run will silently reuse outdated results. This is documented as an accepted trade-off for a single-operator research pipeline (not something a shared/production pipeline could get away with), rather than an oversight nobody noticed.

---

## I. Benchmarking against the published literature

### I1. A CatBoost model is included in the model lineup specifically as a literature benchmark

This isn't there because CatBoost is expected to win — it's there because it's the primary model used by the closest comparable published study on this exact classification problem. Including the same model family as a baseline means any difference between this pipeline's results and that published study's results can be attributed to genuine differences in feature engineering, taxonomy, or validation methodology — rather than being muddled together with "we just used a different underlying algorithm."

---

## The throughline

Almost every "why would they do it that way" moment in this pipeline traces back to the same handful of underlying concerns, applied over and over in different places:

- **Don't trust a single noisy estimate** — repeat it, and penalize inconsistency (C1, C2, G1).
- **Don't let too little data support too confident a claim** — cap what small samples are allowed to assert (B2, B3), and don't score what wasn't genuinely tested (E1, E3).
- **Don't let one method's blind spot become the whole pipeline's blind spot** — cross-check across model families (D1), across subpopulations (D2), across time-of-observation populations (D4).
- **Be honest about what's checked versus what's actually acted on** — several diagnostics (D1, D3, D4) are explicitly informational, not automatically corrective, and the documentation is careful not to claim otherwise.
- **Disclose the corners that were cut, rather than hide them** — selection-bias in internal scores (G2), stale-cache risk (H6), no multiple-comparisons correction (D4), and the engineering-judgment-call thresholds throughout are all stated plainly as limitations, not presented as settled science.
