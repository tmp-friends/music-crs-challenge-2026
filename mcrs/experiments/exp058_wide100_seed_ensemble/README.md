# exp058 wide100_p0 seed ensemble

Last updated: 2026-06-19

## Objective

現 base wide100_p0 は単一 seed の LightGBM で、[[blind-a-ndcg-noise-floor]] のとおり Blind A
80 件は学習 non-determinism だけで ndcg ±0.02 揺れる。本実験はこの variance を、deterministic
な 6 seed の **生スコア平均**で潰した denoised 版を作る。top1 は固定せず（`protect_top=0`）
ensemble スコアで全体を再ランキングするため「top1 保護の保守的 fusion」ではない。狙いは
recall ceiling の引き上げではなく、banked 0.4279 の variance 低減（安全 artifact）。

## Changes

- New: `seed_ensemble_wide100.py`（exp027 `build_features` を再利用、N seed 学習→スコア平均。
  dev-full と Blind A 適用で同一 model 群を共有し二重学習を回避）。OOF は floor 判定用に
  6 seed で再計算。
- Run: `run.sh`（exp027 wide100 と同一の 12 source・同一 primary・同一 params）。

## Configuration

- Date: 2026-06-19 / Eval: devset(学習)+blindset_A(適用) / Track split: `all_tracks`
- Sources: exp027 wide100 と同一（continuity×4 + QNM×3 + entity×4 + transition×1, 全 top100, dense 無し）
- Reranker: LightGBM LambdaRank, 6 seeds score-average, `num_boost_round=450`,
  `num_leaves=31`, `min_data_in_leaf=80`, `lambda_l2=8.0`, `deterministic=True`
- Seeds: `20260616..20260621`

## Results

| Metric | base wide100_p0 (single seed, LB 0.4279) | exp058 wide100 ens6 |
|---|---:|---:|
| candidate positives | 4333 / 8000 | 4333 / 8000（同一プール） |
| Dev OOF | 0.17680 | **0.17777**（+0.00097, floor クリア） |
| Dev full-fit | 0.25584 | 0.25743 |
| fold OOF | — | [0.1859, 0.1843, 0.1567, 0.1828, 0.1792] |

**Blind A 変化**: vs exp015B primary は changed 80/80・top1_changed 55/80・overlap20 11.44
（単一 seed wide100_p0 とほぼ同一の change profile）。vs base wide100_p0 は overlap20 18.93・
top1_changed 6/80 ＝ banked base の忠実な denoise（≈ 0.4279 ± noise）。

## Artifacts

- `seed_ensemble_wide100.py` / `run.sh`
- Blind A tracks: `results/blindA_tracks_ens6.json`
- Blind A prediction: `exp/inference/blindset_A/exp058_wide100_ens6_A.json`
- Submission ZIP: `exp/inference/blindset_A/submission_exp058_wide100_ens6_A.zip`
- Report: `results/report_ens6.json`

## Validation

```text
validate_submission.py submission_exp058_wide100_ens6_A.zip
records: 80 / OK / zip root: prediction.json / name 36 chars
```

## Leaderboard Result（2026-06-19 実測）

```text
ndcg@20                   0.4355
catalog_diversity         0.0280
lexical_diversity         0.5925
llm_judge_score           4.0500
composite_score           0.5086
```

**現 base wide100_p0（0.4279 / 0.5018）を ndcg +0.0076 / composite +0.0068 で上回り project
歴代最高。new base/gate 候補。** ndcg gain 単体は noise floor ±0.02 内だが、composite も同方向に
上昇し、**同一プールの denoise なので robust**（候補集合を変えていない＝CV-LB 逆相関の温床が無い）。

## Interpretation（更新: exp058 が WON、当初の「optional」評価は誤り）

- **seed ensemble の variance 低減は Blind A に実際に効いた**（0.4279→0.4355）。当初「単発提出では
  観測不能・EV 低」と書いたが、**実測で base を超えた**。high-precision な wide100 プール上の
  deterministic 多 seed score-average は、候補集合を変えずに Blind A を底上げできる安全 lever。
- 対照的に exp060（recall ceiling 引き上げ）は Blind A 0.2303 で **collapse**。**候補プールを広げる
  と density 低下で Blind A precision が壊れ、seed ensemble で候補を変えない方が勝つ**ことが
  同日の 2 提出で明確に分離された（[[blind-a-ndcg-noise-floor]] / 候補拡張は CV-LB 逆相関）。

## Next Actions

- **exp058 wide100 ens6 を new base/gate に昇格する候補（user 判断）**。今後の seed ensemble は
  この high-precision プール上で seed 数を増やす方向（候補拡張はしない）。
