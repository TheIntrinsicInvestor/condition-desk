# Model report: methods, benchmarking and model selection

NEBULA X, Problem Statement 3 (Train Condition Monitoring). All four subsystems.

This document covers what we built, what we compared it against, what we rejected and why, and
how much we actually trust each number. It is deliberately explicit about uncertainty: three of
our four validation scores are optimistic for different structural reasons, and we would rather
say so than quote them as solid.

---

## 0. The rule we worked to

One rule governs every decision below:

> Refit every choice inside the validation fold. When two estimates tie, take the simpler model.

This is not a slogan. It is the rule that decided four of the results in this document. On SHM,
pinning the S-N exponent to its textbook value beat fitting it. On Rail, a single-split sweep of
about 35 configurations once read macro F1 0.739 against a true 0.721, purely because the
configuration was chosen on the folds it was scored on. Everything here is validated by a shared
harness (`src/experiments/validate.py`) that scores every candidate on identical folds, so
differences between candidates are paired rather than compared across independent runs.

A second rule was needed for Door, where the training data is perfectly separable and so
cross-validation scores every model 1.000 and can rank nothing. There, a change ships only if it
has a stated physical mechanism, leaves the training score unchanged, and flips few enough test
predictions that every flip can be inspected by hand. As it turned out, nothing needed to ship.

---

## 1. Summary

| Subsystem | Method | Validation score | Changed this round? |
|---|---|---|---|
| SHM | Rainflow + Miner, m = 5 fixed, single fitted C | LOO MAPE 2.71%, score 0.973 | No |
| ACV | Rank by mean(indoor minus cooling setpoint) | 0.979 over 6 cases | No |
| Door | `vol_area` threshold (equivalently, a 33-feature GBM) | Not measurable, see §4 | No |
| Rail | Logistic regression over frequency **and** wavelength bands | macro F1 0.745 +/- 0.027 | **Yes** |

One model changed. Three were tested against named alternatives and kept, which is a result in
itself: in each case we can now say what the alternative was and why it lost, rather than not
having looked.

---

## 2. Rail corrugation: the one model we changed

### 2.1 What we found before modelling anything

Train speed across the 272 labelled recordings spans 0 to 19.48 m/s (mean 8.99, sd 5.88). Of the
133 files below 9.7 m/s, **all 133 are Normal**, and 44 were recorded stationary. All 38
corrugated files lie above 9.7 m/s.

So "slow means Normal" is a perfect rule on half the training set. Two things follow. First, the
68 held-out files have a near-identical speed distribution (mean 9.39, sd 5.62, 10 stationary), so
this shortcut probably transfers and our score is not built on sand. Second, the real task is much
narrower than the headline: among the 139 files moving fast enough to be corrugated, the split is
101 Normal, 14 Side I, 24 Side II, and that is where all the difficulty lives.

We quantified how much of the score is the shortcut. Speed alone scores macro F1 0.437 over all
files but only 0.301 on the fast subpopulation, while the full model scores 0.721 and 0.708
respectively. The model is therefore doing real work, not just reading a speedometer.

### 2.2 The physical argument for wavelength

Corrugation is a periodic wear pattern with a characteristic **wavelength**. An axle box measures
a **frequency**, and frequency equals speed divided by wavelength. A fixed 200 Hz band therefore
covers a completely different wavelength at every speed, and even within the fast subpopulation
speed varies by a factor of two. The original frequency bands could not have been reading a
consistent wavelength signature.

The standard fix is order tracking: resample each channel at constant **distance** increments
rather than constant time increments, using the 90-tooth tacho pulse train on column 0 as the
phase reference. The FFT of the resampled signal is then a spatial spectrum in cycles per metre,
speed-invariant by construction. This is strictly better than dividing band edges by a mean speed,
because it also removes smearing caused by speed changing within a single one-second recording.

Implementation is in `src/rail_order.py`: integrate the pulse train to distance, low-pass to match
the decimation ratio (otherwise resampling aliases), interpolate onto a 2 mm grid (Nyquist 250
cycles per metre, which is the worst-case raw resolution at 19.5 m/s), then band into 25
log-spaced bands from 1 to 250 cycles per metre. 54 of the 340 files travel less than 1 metre and
carry an explicit `order_valid = 0` flag rather than a fabricated spectrum.

### 2.3 Benchmarking

Repeated stratified 5-fold, 20 seeds, identical folds for every candidate:

| Feature set | macro F1 | Side I F1 | Side II F1 | Normal F1 |
|---|---|---|---|---|
| Frequency bands (previous model) | 0.7218 +/- 0.0180 | 0.412 | 0.804 | 0.949 |
| Wavelength bands alone | 0.7314 +/- 0.0350 | 0.456 | 0.788 | 0.951 |
| **Frequency + wavelength (selected)** | **0.7454 +/- 0.0269** | **0.447** | **0.831** | **0.958** |
| Frequency + wavelength + p90 aggregation | 0.7426 +/- 0.0189 | 0.466 | 0.803 | 0.959 |
| Wavelength contrast bands only | 0.5769 +/- 0.0377 | 0.331 | 0.495 | 0.904 |

The means overlap within one standard deviation, so the mean alone is weak evidence. The paired
comparison is much stronger, because every candidate saw identical folds:

- **Union versus frequency alone: +0.0236 mean, winning on 16 of 20 seeds** (sign test p ~ 0.006).
- Wavelength alone versus frequency alone: +0.0096, winning on 12 of 20 seeds. Not significant.

So the two banks are complementary rather than redundant, which is what the physics predicts: the
frequency bank is what the sensor measures directly, the wavelength bank is what the rail actually
has, and neither subsumes the other.

### 2.4 What we rejected, and why

- **Wavelength bands as a replacement.** Only 12 of 20 paired wins. Kept both banks instead.
- **p90 aggregation across a side's 32 channels** (the hypothesis that corrugation appears on only
  a few axle boxes). Lost 9 of 20 against the union for -0.003. Ties go to the simpler model.
- **Two-binary decomposition** ("is Side I corrugated?" and "is Side II corrugated?" separately, so
  each model learns from all 272 files). Physically well motivated, since the info kit states the
  rails are judged independently. It helped slightly on frequency features alone (13 of 20 wins)
  but lost on the union (-0.011, 5 of 20). Rejected.
- **Dropping the explicit `speed_mps` feature** to remove the shortcut. Costs almost nothing
  (0.7193 versus 0.7218), which tells us the shortcut is carried by the band energies rather than
  by that one column. Kept, since removing it neither helps nor cleans anything up.

### 2.5 Honest uncertainty

Nested cross-validation, with both the feature set and the regularisation strength chosen inside
each outer fold, gives **0.7230 +/- 0.0358**. That is the honest number for the *selection
procedure*, and it is well below the 0.7454 of the fixed union, because the inner selection is
unstable: across 25 outer folds it chose the wavelength set 9 times, the union 11 and the
frequency set 5. That instability is the signature of a flat optimum where the three sets are
close.

We report both numbers rather than the flattering one. The fixed union is justified by the paired
sign test, not by the nested search, and we expect the held-out score to land between 0.72 and
0.75. Side I remains the ceiling: 14 training examples carrying one third of macro F1.

Effect on the submission: 10 of 68 predictions changed. The predicted corrugation rate moved from
21% to 15%, against a training prevalence of 14%.

---

## 3. SHM: tested, and deliberately left alone

The model is a calculation rather than a fit. Rainflow counting plus Miner's rule gives
`D = (1/C) * sum(n_i * S_i^m)`, with m pinned to 5 (the textbook S-N exponent for welded steel)
and C the single fitted parameter. Leave-one-out MAPE is 2.71%, essentially equal to in-sample
2.67%, so variance is not the problem here. Model form is.

**m = 5 is confirmed by a sharp minimum**, with C refit inside every fold:

| m | 4.0 | 4.5 | **5.0** | 5.5 | 6.0 |
|---|---|---|---|---|---|
| LOO MAPE | 25.36% | 12.48% | **2.71%** | 13.35% | 25.69% |

### 3.1 Diagnostic: the residual tail is real structure

54 of 64 files land within 5% but the worst is 13.1%, which is a tail rather than symmetric noise.
Regressing the leave-one-out residual on file statistics found one real correlation: **signal skew
correlates +0.537 with the signed log residual**. That is a genuine unmodelled term, not noise.

### 3.2 But exploiting it does not survive validation

| Candidate | LOO MAPE | Score | Change |
|---|---|---|---|
| Current, single C | 2.714% | 0.9729 | baseline |
| Plus a linear skew term (1 extra parameter) | 2.555% | 0.9744 | +0.0016 |
| Plus skew and skew squared | 2.289% | 0.9771 | +0.0042 |
| C fitted per cluster, k = 3, clustering refit in-fold | 2.217% | 0.9778 | +0.0050 |
| C fitted per cluster, k = 2 | 2.748% | 0.9725 | -0.0003 |
| Residual half cycles counted as full instead of 0.5 | 6.240% | 0.9376 | **-0.0353** |

Every apparent gain is between 0.002 and 0.005 on a score of 0.973, and the skew correction
improves only **35 of 64 files** (sign test p ~ 0.35, indistinguishable from a coin flip). The
clustering result is also non-monotonic in k (k = 2 is worse than baseline, k = 3 better, k = 4
middling), which is what noise looks like, and k was itself chosen by reading this table.

By our own rule these are ties, and ties go to the simpler model. **SHM is unchanged.**

The half-cycle variant is a clean rejection worth recording: counting rainflow residual half
cycles as full cycles more than doubles the error, confirming the ASTM E1049-85 convention already
in use.

Previously tested and rejected, not repeated here: a fatigue cut-off stress, a two-slope Eurocode
S-N curve, and a Goodman mean-stress correction. All three scored worse.

**Confidence: high.** One free parameter against 64 files, leave-one-out equal to in-sample, and
the exponent fixed on physical grounds rather than fitted.

---

## 4. ACV: the weakest evidence base, and what we did about it

### 4.1 The problem

Only six labelled cases exist. The current method ranks cars by mean(indoor temperature minus
cooling setpoint) while cooling, and it was chosen because it widened margins **on the same cases
it is validated against**. With six observations, "correct on 5 of 6" cannot distinguish a strong
method from a lucky one, and the held-out set is a single file.

### 4.2 What we tested

Rather than tune further on six cases, we built five features that are independently motivated by
refrigeration physics and asked whether they agree. A unit low on refrigerant cannot reject heat,
so it should sit above its setpoint, pull the cabin down more slowly, run harder for the same
load, sit above its siblings, and track ambient temperature rather than its setpoint.

| Feature | What it measures | Score over 6 cases | Top-1 |
|---|---|---|---|
| `excess` | mean(indoor minus setpoint) while cooling | **0.9792** | 5 of 6 |
| `vs_fleet` | mean indoor minus the median across sibling cars | 0.9583 | 4 of 6 |
| `pulldown` | how slowly indoor temperature falls while cooling | 0.8125 | 4 of 6 |
| `ambient` | correlation of indoor with outdoor temperature | 0.7917 | 4 of 6 |
| `duty` | fraction of time in the hardest cooling mode | 0.6250 | 0 of 6 |
| Equal-weight aggregate of all five | | 0.9167 | 3 of 6 |

**Equal-weight rank aggregation over all five is worse than the single current feature**, because
`duty` and `ambient` are weak and drag the average down. Aggregating only the two strong, directly
physical features (`excess` and `vs_fleet`) ties the current method exactly at 0.9792, with an
identical per-case rank profile and a **byte-identical ranking on the test file**. It is therefore
a no-op on the submission, and ties go to the simpler model.

**ACV is unchanged.** But the evidence behind it is now much better than it was.

### 4.3 What improved is the confidence, not the score

- **`acv_case_04` is recoverable.** It uses a completely different vocabulary (483 columns, 63
  parameters per car, sharing exactly one parameter name with the other files). Mapping it
  properly gives a sixth honest check, on which the method ranks the true car 2nd (score 0.875).
  The reported 0.979 now rests on six cases rather than five.
- **The two features that work independently agree on the test file.** Both `excess` and
  `vs_fleet` put car 01 first, which is what the submission says.
- **The three features that fail on training also disagree with each other on the test file**
  (they pick 02, 04 and 04). That is what noise looks like, and it is reassuring that the signal
  and the noise separate so cleanly.

**Confidence: low, and unavoidably so.** Six cases, a single held-out file, and a scoring rule
where slipping from rank 1 to rank 2 costs only 0.125 but a catastrophic miss to rank 7 costs
0.75. We always emit a full eight-car ranking, since an incomplete list is the only way to score
zero outright.

---

## 5. Door: where cross-validation cannot help, and what we did instead

### 5.1 Segmentation is not a modelling problem

The recording contains only the cycles, not the quiet periods between them, so consecutive rows
are either exactly 20 ms apart or separated by a multi-second jump. Splitting on any gap above
50 ms reproduces **all 110 official training segments exactly**, boundaries and row counts
included. `Test.csv` yields 38 cycles.

This matters for the metric. Because the boundaries are exact, every correctly labelled cycle
scores IoU 1.0 and the predicted segment count equals the true count, so the official IoU-weighted
F1 collapses to plain accuracy over 38 binary decisions. We therefore tuned for accuracy, and
specifically did **not** bias toward flagging, which would be the right move under a recall-leaning
F1 but is wrong here.

### 5.2 The real situation is blindness, not overfitting

The training set is **perfectly separable** on `vol_area` (the voltage integral): Normal runs 15252
to 15890, Abnormal 16142 to 18230, with a clean empty gap. Every reasonable model therefore scores
1.000 in cross-validation. That is not a passing grade, it is an uninformative test. We cannot
rank models, tune anything, or detect a problem.

So instead of a comparison CV cannot give us, we asked what could actually go wrong and tested for
it directly.

### 5.3 Three independent methods, one answer

| Method | Abnormal predicted | Disagreements with current |
|---|---|---|
| Submitted 33-feature gradient boosting model | 9 of 38 | baseline |
| Single `vol_area` threshold at the gap midpoint | 9 of 38 | **0** |
| Stream-relative features (each divided by its own stream median) | 9 of 38 | **0** |
| Separate thresholds for Open and Close cycles | 9 of 38 | **0** |

All four agree on all 38 cycles. Three specific findings came out of this:

- **The feared stream offset does not exist.** If the test stream had a different door, supply
  voltage or sensor gain, an absolute threshold fitted on the training stream would land in the
  wrong place and nothing in CV would reveal it. Measured, the `vol_area` median ratio between the
  streams is **1.005**. Stream-relative normalisation was built and tested, remains perfectly
  separable on training, and changes no prediction. We did not ship it, because it is a no-op that
  adds a moving part.
- **The 33-feature model is exactly equivalent to a one-line threshold**, matching on 38 of 38.
  For the app we recommend showing the threshold, because it is equally accurate and vastly more
  explainable to a non-technical user.
- **Open and Close cycles genuinely differ.** Open Normal runs 15676 to 15890, Close Normal 15252
  to 15654, so opens sit about 400 units higher. Each operation is separately perfectly separable,
  and the per-operation thresholds happen to reproduce the global answer exactly.

### 5.4 Cycle 32, and the cycles that are genuinely ambiguous

Cycle 32 had been flagged as not fitting the clean two-cluster story. It does: it sits at
`vol_area` 15472, a margin of -2.16 training gap widths below the threshold, and falls inside
neither per-operation gap. It is unambiguously Normal.

The honest concern lies elsewhere. **Seven test cycles fall inside the training gap**, a region
containing zero of the 110 training cycles: cycles 4, 9, 14, 18, 27, 29 and 35, with cycle 4 only
10 units below a threshold sitting on a gap 252 wide. These seven are where Door's entire score
variance lives. We keep them Normal, because the training base rate is 27% Abnormal and our
prediction is 24%, whereas flipping all seven would give 42%, well outside the observed rate.

**Confidence: unmeasurable, but better evidenced than before.** We cannot put a number on Door.
What we can now say is that every alternative we could construct gives the same answer, and that
the one failure mode CV could not have caught has been measured and is absent.

---

## 6. What we would do with more time

1. **Rail Side I.** F1 of roughly 0.45 from 14 training examples carries a third of macro F1, and
   everything else is near its ceiling (Normal 0.96, Side II 0.83). Window augmentation, splitting
   each one-second recording into overlapping sub-windows with a grouped split so windows from one
   file never straddle a fold, is the obvious next lever. It must be grouped, or it silently
   becomes leakage.
2. **ACV.** Nothing will fix six cases except more cases. The useful work is defensive: confirm
   the full ranking is always emitted and that the schema traps stay handled.
3. **Door.** The seven gap cycles would repay a look at their raw current traces against typical
   Normal and Abnormal cycles, which would also serve as an explainability panel in the app.
