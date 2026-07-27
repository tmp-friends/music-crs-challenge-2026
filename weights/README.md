# Trained GBDT weights (final submission)

Trained LightGBM models behind the selected final submission `submission_exp116_lgbm6_nocs_B.zip` (Blind B composite 0.4998 / nDCG@20 0.3528). Both stages are trained only on the official challenge datasets (`talkpl-ai/TalkPlayData-Challenge-*`); no external data is used. The models are fully retrainable from scratch with the commands in the [repository README](../README.md); these files are published for transparency and for direct verification without retraining.

## Files

| File | Role |
|---|---|
| `stage1_anchor/lgbm_B_ll_top500.txt` | Stage-1 BM25-anchor ranker. LightGBM binary logloss over the top-500 RRF union of 4 BM25 views + 4 exact-match fields (config [exp015_B_ll_t500_B.yaml](../mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_B_ll_t500_B.yaml), source profile B). This is the file referenced by `reranker_model_name` in that config. |
| `stage2_reranker_lgbm6_nocs/lgbm_lambdarank_seed2026061{6..21}.txt` | Stage-2 reranker of the final submission: 6 LightGBM LambdaRank boosters (seeds 20260616 to 20260621). Trained on the Dev-split candidate union (12 sources, cross_session dropped): 2,949,717 candidate rows, 4,333 positive groups, 73 rank-based features, `num_boost_round=450`, `num_leaves=31`, `min_data_in_leaf=80`, `lambda_l2=8.0`, `learning_rate=0.03`, `deterministic=true`. Raw scores of the 6 boosters are averaged, then the top-20 per session-turn is taken. |

Feature names are embedded in each model file (LightGBM text format). The 73 features are rank-derived only (anchor rank flags, per-source rank features, aggregate hit/RRF statistics, turn position); see `build_features` in [rank_source_lgbm_mixed_topk.py](../mcrs/experiments/exp027_wide_source_lgbm/rank_source_lgbm_mixed_topk.py).

## Usage

```python
import lightgbm as lgb

booster = lgb.Booster(model_file="weights/stage2_reranker_lgbm6_nocs/lgbm_lambdarank_seed20260616.txt")
scores = booster.predict(x)  # x: [n_candidates, 73] built by build_features()
```

## Provenance and verification

Stage-2 training is deterministic (`deterministic=true`, fixed seeds, fixed thread count). The published boosters were regenerated with the exact final-run configuration and verified against the original run artifacts:

- Dev full-fit nDCG@20 matches the original run report value `0.25870485648368946` exactly (see [report_ens_blindB_lgbm6_nocs.json in the exp116 README](../mcrs/experiments/exp116_gbdt_family_ensemble/README.md)).
- The Blind B top-20 ranking produced by these boosters is identical, on all 80 of 80 rows, to the canonical ranking that produced the final submission; reloaded files reproduce in-memory scores with zero difference.
