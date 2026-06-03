# Evaluation Results

*Generated 2026-05-31 18:42*

## Dataset

- **Total rows:** 389 (Normal = 320, Anomaly = 69)
- **Split:** 70 / 30 stratified, `random_state=42`
- **Train:** 50 rows (capped at 50 to mirror production)
- **Test:** 117 rows

The dataset covers four behavioural archetypes seeded by `seed_training_data.py`:
normal humans, bots, malicious users (seven attack sub-types), and a mixed
"bot + attack" group. Labels are rule-derived; the rule-only configuration
therefore achieves a synthetic ceiling and the meaningful comparison is
between the **behavioural** and **attack-signal** feature layers, and between
**TabPFN-only** and the deployed **Hybrid** configuration.

## Layer comparison — the two detection sources side-by-side

The same train/test split was used to train three TabPFN models on three
feature subsets: behavioural biometrics alone (mouse / keyboard timing),
attack-signal features alone (hacking-string detection, paste behaviour,
DevTools usage, unauthorised-attempt counter), and the combined feature set.

| Feature set | # features | Accuracy | Precision (Anom) | Recall (Anom) | F1 (Anom) | ROC-AUC |
|---|---|---|---|---|---|---|
| (a) Behavioural-only | 13 | 0.889 | 1.000 | 0.381 | 0.552 | 0.911 |
| (b) Attack-signal-only | 7 | 0.897 | 0.667 | 0.857 | 0.750 | 0.915 |
| (c) Combined | 20 | 0.923 | 0.700 | 1.000 | 0.824 | 0.976 |


**Figure: `evaluation_results/confusion_matrices_three_way.png`** —
three-panel confusion matrix showing how each layer performs in isolation
versus combined.

## Configuration comparison — model variants

| Configuration | Accuracy | F1 (Anomaly) | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| Rules-only baseline | 1.000 | 1.000 | 1.000 | — |
| TabPFN combined | 0.923 | 0.824 | 0.887 | 0.976 |
| Hybrid (deployed) | 0.923 | 0.824 | 0.887 | 0.976 |


**Figure: `evaluation_results/confusion_matrices_hybrid_vs_rules.png`** —
the deployed Hybrid configuration combines the combined-feature TabPFN
output with a hard-rule floor (`models/tabpfn_model.py:77-105`) that
guarantees a minimum confidence on definitive attack signals so TabPFN
cannot underpredict them due to the 50-row training cap.

### Confusion matrices

**Behavioural-only**

|              | Pred Normal | Pred Anomaly |
|--------------|-------------|--------------|
| **Actual Normal**  | 96 | 0 |
| **Actual Anomaly** | 13 | 8 |

**Attack-signal-only**

|              | Pred Normal | Pred Anomaly |
|--------------|-------------|--------------|
| **Actual Normal**  | 87 | 9 |
| **Actual Anomaly** | 3 | 18 |

**Combined**

|              | Pred Normal | Pred Anomaly |
|--------------|-------------|--------------|
| **Actual Normal**  | 87 | 9 |
| **Actual Anomaly** | 0 | 21 |

**Hybrid (deployed)**

|              | Pred Normal | Pred Anomaly |
|--------------|-------------|--------------|
| **Actual Normal**  | 87 | 9 |
| **Actual Anomaly** | 0 | 21 |


## Attack-pattern strings caught in the test set

The frontend `BehaviorTrackerService` records the specific regex label
(e.g. `[SQL Injection] UNION SELECT`) that triggered every hacking-string
detection. The chart below counts those labels by OWASP category for the
rows that landed in the test split.

| Attack category | Count in test set |
|---|---|
| SQL Injection | 5 |
| XSS | 3 |
| Command Injection | 3 |
| Path Traversal | 2 |
| Other | 1 |
| XXE | 1 |


**Figure: `evaluation_results/attack_pattern_frequency.png`**

## Per-archetype breakdown (Hybrid configuration)

The malicious group is split into its seven seeded attack types so per-attack
recall is visible — not just an aggregate "malicious" number.

| archetype                            |   n |   tp |   fn |   tn |   fp |   accuracy |
|:-------------------------------------|----:|-----:|-----:|-----:|-----:|-----------:|
| bot                                  |   3 |    3 |    0 |    0 |    0 |        1   |
| malicious — command                  |   3 |    3 |    0 |    0 |    0 |        1   |
| malicious — path                     |   1 |    1 |    0 |    0 |    0 |        1   |
| malicious — probe (challenge bypass) |   1 |    1 |    0 |    0 |    0 |        1   |
| malicious — sql                      |   2 |    2 |    0 |    0 |    0 |        1   |
| malicious — xss                      |   2 |    2 |    0 |    0 |    0 |        1   |
| mixed (bot+attack)                   |   4 |    4 |    0 |    0 |    0 |        1   |
| normal                               |  11 |    0 |    0 |   11 |    0 |        1   |
| unknown                              |  90 |    5 |    0 |   76 |    9 |        0.9 |

## ROC curves

- Behavioural-only AUC: **0.911**
- Attack-signal-only AUC: **0.915**
- Combined AUC: **0.976**
- Hybrid AUC: **0.976**

**Figure: `evaluation_results/roc_curve.png`**

## Calibration

The deployed system clips raw TabPFN probabilities into [0.05, 0.95] via
`calibrate_confidence` (`routes/predict.py:14`). The reliability diagram
verifies this clip removes overconfident extremes without distorting the
central probability range where verdicts are most often ambiguous.

**Figure: `evaluation_results/calibration_plot.png`**

## Feature ablation (top contributors)

Positive Δ = feature carried unique signal the remaining features could not
reconstruct. Behavioural features in blue, attack-signal features in red on
the figure.

| Feature dropped | Category | F1 without it | Δ vs baseline |
|---|---|---|---|
| HackingStringDetected | attack-signal | 0.571 | +0.252 |
| AvgMouseSpeed | behavioural | 0.824 | +0.000 |
| MouseMoveCount | behavioural | 0.824 | +0.000 |
| StdMouseSpeed | behavioural | 0.824 | +0.000 |
| AvgMouseIdle | behavioural | 0.824 | +0.000 |
| AvgClickDuration | behavioural | 0.824 | +0.000 |
| AvgClickInterval | behavioural | 0.824 | +0.000 |
| ClickCount | behavioural | 0.824 | +0.000 |


Full table: `evaluation_results/feature_ablation.csv`
**Figure: `evaluation_results/feature_ablation.png`**

## Latency

| Operation | Time |
|---|---|
| Cold fit (combined model, 50 rows) | 412.9 ms |
| Single-sample predict P50 | 3017.8 ms |
| Single-sample predict P95 | 4016.0 ms |
| Single-sample predict P99 | 4070.9 ms |
| Batched predict (117 samples, per sample) | 44.27 ms |

Snapshots are emitted every 30 s by the frontend, so P95 single-sample
latency of 4016 ms leaves ample headroom. The Groq LLM analysis adds
~1–3 s but runs on a separate 8-second timeout with rule-based fallback
(`app/groq_service.py:96-99`), so it never blocks the verdict path.

**Figure: `evaluation_results/latency_distribution.png`**

## Threats to validity

1. **Synthetic data ceiling.** Labels are rule-derived; rule-only achieves
   near-perfect score by construction. Real-world deployment will encounter
   patterns outside both the training rules and the held-out sample.
2. **TabPFN training cap.** TabPFN's zero-shot regime is calibrated for
   ≤1000 rows; the production system caps at 50 for latency. The evaluation
   honours this constraint rather than reporting an unrealistic best case.
3. **Small test set (117 rows).** Confusion-matrix cells with
   single digits are sensitive to a single misclassification; confidence
   intervals are wide. Larger live-traffic evaluation is left as future
   work.
4. **Population-level evaluation.** Per-user baselines are deliberately not
   measured — the design argument (behavioural convergence in repetitive
   government tasks, no cold-start, harder to spoof) is in the Design
   chapter; per-user accuracy would test a model the system does not aim
   to be.

## Files produced

- `confusion_matrices_three_way.png` — behavioural / attack-signal / combined
- `confusion_matrices_hybrid_vs_rules.png`
- `attack_pattern_frequency.png`
- `roc_curve.png`
- `calibration_plot.png`
- `feature_ablation.png` and `.csv`
- `latency_distribution.png`
- `per_attack_breakdown.csv`
- `summary_report.md` (this file)
