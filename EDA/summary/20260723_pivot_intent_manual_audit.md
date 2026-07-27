# Single-annotator artist-pivot detector audit

Last updated: 2026-07-23

## Scope

- Annotators: 1
- Sample: 200 stratified train turns
- Labels: ARTIST_PIVOT / NOT_ARTIST_PIVOT / UNCERTAIN
- Sessions represented: 198 (2 additional rows
  share a session)
- Bootstrap: 5,000 common session-cluster resamples across
  strata with within-stratum weight normalization, seed=20260723
- Weighting: population turns / sampled turns within each sampling stratum
- Agreement: not applicable (single-annotator audit)

## Unweighted sample label counts

- ARTIST_PIVOT: 127
- NOT_ARTIST_PIVOT: 67
- UNCERTAIN: 6

These raw counts describe the deliberately stratified sample and are not prevalence
estimates. The population-weighted uncertain rate is 4.598%; binary
metric coverage is 95.402%.

## Detector metrics

| Detector | Eval. n | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) |
|---|---:|---:|---:|---:|
| broad_discovery | 194 | 0.904 [0.841, 0.957] | 0.218 [0.179, 0.273] | 0.351 [0.298, 0.420] |
| artist_explicit | 194 | 0.992 [0.970, 1.000] | 0.322 [0.225, 0.435] | 0.486 [0.366, 0.606] |
| core_change | 194 | 0.983 [0.942, 1.000] | 0.160 [0.106, 0.234] | 0.276 [0.191, 0.379] |

`UNCERTAIN` rows are excluded from binary metrics. `Eval. n` is the unweighted
number of classifiable sampled rows. Confidence intervals use session clusters
within each sampling stratum; they quantify sampling uncertainty but do not include
annotator uncertainty. Because there is only one annotator, the result must be
described as a manual audit rather than inter-annotator validation.
