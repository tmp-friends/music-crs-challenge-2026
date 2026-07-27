# Evaluation-alignment audit: additional analysis

Last updated: 2026-07-24

## Run configuration

- Train sessions: 15,199
- Cluster bootstrap: 5,000 resamples of whole sessions, seed=20260720
- Dataset access: official TalkPlayData-Challenge train + official track metadata
- Response audit: 43/44 eligible local
  Blind-A artifacts joined to `results_ledger.jsonl`; 1
  exclusions are recorded in a separate CSV.
- Leak policy: train-only progress/future music rows are used for retrospective
  analysis only and never enter inference, prompt, fallback, or validation.

## RQ1: sequential artist-pivot audit with turn adjustment

The correct temporal ordering is `music[t-1] -> listener feedback/message[t] -> music[t]`.
Therefore `goal_progress[t]` evaluates the previous recommendation, not the current relevant track.

- Primary artist-targeted pivots: n=15,126; raw next-target
  prior-session-artist rate 0.763. After direct standardization
  to the pooled turn-2--8 distribution, pivot/non-pivot rates are
  0.760/0.631,
  a risk difference of 0.129
  [0.118,
  0.140].
- The turn-fixed-effects logistic sensitivity check gives OR
  1.904 [1.806,
  2.007] with session-cluster robust SE.
- For t>=3 primary pivots, prior same-artist recommendations receive MOVES at
  0.563, versus 0.785
  after prior different-artist recommendations: gap 0.221
  [0.197, 0.245].
- High-conflict primary slice (artist-targeted pivot after prior same-artist +
  DOES_NOT): n=5,278; the next target repeats the
  immediately prior artist in 0.858
  [0.847,
  0.868].
- The broad discovery/change cue remains a sensitivity slice: n=10,982,
  raw prior-session-artist rate 0.770, turn-standardized
  risk difference 0.126, and
  high-conflict immediate-artist repeat rate 0.859.
- An alternative core-change operationalization gives n=7,147
  and raw next-target prior-session-artist rate 0.801;
  turn-standardized risk difference 0.160;
  and high-conflict immediate-artist repeat rate 0.862.
  Because the three regex definitions are alternative rather than nested filters,
  this is a sensitivity check, not a strict conservative subset comparison.

## RQ2: fixed-ranking response audit

- Initial -> grounded-long composite delta: 0.0843.
- Judge contribution: 0.0638 (75.6%);
  lexical contribution: 0.0206 (24.4%).
- The response-only delta has the same composite weight as an nDCG increase of
  0.1687, with ranking held fixed.
- Exact canonical prediction resubmission: judge 4.15--4.25 (range 0.10), composite range 0.0075. One pair does not estimate
  the full judge variance, but proves that a 0.10 judge difference need not reflect a content change.

Across 42 unique scored
response--ranking pairs, the strongest Pearson correlations are:

| Feature | Pearson r | Spearman r |
|---|---:|---:|
| opening_unique_rate | 0.722 | 0.516 |
| mean_words | 0.512 | 0.489 |
| question_end_rate | 0.509 | 0.648 |
| mean_sentences | -0.285 | -0.250 |
| exclamation_per_response | -0.398 | -0.596 |
| apology_rate | -0.724 | -0.577 |

These are post-hoc submission-level associations. In particular, signs for sentence count and
exclamation frequency are not stable relative to the earlier 13-variant slice, so they must not
be described as causal judge preferences.
