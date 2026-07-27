# exp116 GBDT family ensemble（LightGBM + XGBoost + CatBoost + TabM）

Last updated: 2026-06-23（Blind A SCORED：lxc nDCG 0.4431 / composite 0.5758、lxct nDCG 0.4420 / composite 0.5789。両者とも nDCG は noise 内で base 同等。**★user 判断で頑健性の観点から lxct を新 bankable ranking base に採用**＝exp090-fix を superseded。採用根拠は point-estimate gain でなく model-family 多様化による robustness 仮説。lxc は not adopted）

## Summary

best nDCG モデル **exp090-fix**（exp058 wide100 + cross_session source の 6-seed LightGBM
seed ensemble、Blind A nDCG 0.4396 / composite 0.5657）への **reranker の model 層だけの
addon**。候補プール・特徴量(78 次元 rank feature)・exclude_map・source は exp090-fix と
**byte-identical**（`candidate_rows=3,078,934` / `candidate_positive_count=4,528` が一致、
LightGBM full-fit Dev nDCG=0.259737 が exp090-fix と完全一致で確認）に固定し、LightGBM 単一
family の seed-average を **LightGBM + XGBoost + CatBoost（+ optional TabM）の family
ensemble** に置き換えた。

ユーザ依頼に従い 2 つの submission を作成:

- **lxc**: GBDT 3 種（LightGBM + XGBoost + CatBoost）等重み blend。
- **lxct**: lxc に TabM（別関数族の MLP ensemble）を加えた 4 family 等重み blend。

### 結論（Dev OOF=信頼 ruler、同一 fold 比較）

| family / blend | Dev OOF nDCG@20 | vs LightGBM |
|---|---:|---:|
| **LightGBM 単独（=exp090-fix の family）** | **0.18090** | — |
| XGBoost 単独 | 0.18031 | -0.0006（parity, noise 内） |
| CatBoost 単独 | 0.17056 | **-0.0103（弱い）** |
| TabM 単独（別関数族 MLP） | 0.17788 | -0.0030（弱いが decorrelated） |
| **lxc blend（lgbm+xgb+catboost 等重み）** | **0.17954** | **-0.0014（base 未満）** |
| **lxct blend（+tabm 等重み）** | **0.17951** | **-0.0014（lxc とほぼ同値）** |
| lgbm+xgb blend（parity-gate, 診断） | 0.18106 | **+0.00016（noise 内、唯一の非負）** |

full-fit Dev nDCG: lgbm 0.259737（=exp090-fix と完全一致）/ xgb 0.243255 / catboost 0.176151
/ tabm 0.180702 / lxc blend 0.221400 / lxct blend 0.219070。TabM は pointwise なので full-fit
と OOF がほぼ同値（=過学習しない代わりに弱い）。

**同一の 78 次元特徴量を全 family が共有するため、GBDT 3 種は同一関数族で blend は本質的に
variance 低減にしかならず（advisor 事前予測どおり）、point-nDCG は伸びない。** むしろ
CatBoost standalone が LightGBM より明確に弱い（YetiRank が本特徴量・極端な class imbalance
下で振るわない）ため、**等重み blend は LightGBM 単独を Dev OOF で下回る**（advisor が予告した
「weak family を等重みで入れると blend が base を割る」罠そのもの）。XGBoost のみ parity。
TabM（唯一の別関数族）は standalone 0.17788 で decorrelated だが、これも LightGBM より弱く、
4 family 等重み blend(lxct 0.17951) は lxc(0.17954) と実質同値＝**TabM 追加でも base を超えない**。

**結論の範囲（重要・advisor 指摘で穴埋め）**: 上の「base 未満」は *等重み* blend が弱 family
希釈で負けただけで、それ自体は「ensemble が効かない」証明ではない。そこで **唯一 parity の
2 family= lgbm+xgb だけの blend**（standalone parity を満たす family のみ入れる binary gate。
連続重みの Dev OOF tune ではないので overfit しない）を同一 fold で測った → **0.18106、lgbm
0.18090 比 +0.00016**。すなわち non-dilutive な parity-gate blend でも gain は **Dev OOF noise
内**で、[[blind-a-ndcg-noise-floor]] の Blind A ±0.02 floor を遥かに下回る＝ **usable な nDCG
gain は無い**。XGB と LightGBM の LambdaRank は同一 78 特徴量で誤差が高相関なので variance
低減が極小なため。

[[dev-blinda-drift-decomposition]] の「0.44→0.50 残差は within-pool ordering の signal
律速（順位付け不能）であって model 律速ではない」という既往結論とも整合: 同特徴量で関数族を
増やしても ordering ceiling は動かない。**model-family ensemble（GBDT 3 種 / +TabM / parity-gate
lgbm+xgb のいずれも）は本データの reranker nDCG を usable に改善しない**
（[[test-tracks-lever-and-measurement]] の reranker-bound を model 軸で再確認）。
候補 recall と response 軸が依然レバー。

## なぜ作ったか / 採否

- ユーザ明示依頼（「xgboost, catboost も含めた ensemble にできないか、best nDCG への addon
  実験として」「GBDT 三種と TabM 系それぞれで zip を作って」）に応えるための実験。
- 採否 gate は Blind A（[[test-tracks-lever-and-measurement]] / exp060 で Dev gain が Blind A
  で crash した前例があるため、Dev OOF は floor 判定専用）。だが **Dev OOF で既に base 未満**
  なので、Blind A での逆転は期待薄。AGENTS.md ルールに従い submission zip は両方とも必ず作成
  （比較・再現用）するが、**bankable base は exp090-fix のまま据え置き**を推奨。
- blend 重みは Dev OOF で tune しない（CV-LB 逆相関で overfit）。等重みを既定にし、weight gate
  をかけるなら「standalone が LightGBM と parity の family のみ採用」= 実質 lgbm(+xgb) に縮退。

## TabPFN を入れなかった理由（実現不能）

TabPFN は in-context 学習の構造上 **train 行 ~10,000・特徴 ~500 が実用上限**。本 reranker の
学習行は **3,078,934 行**（上限の約 300 倍）かつ正例率 ~0.15%（4,528 / 3.08M）。
10k 行に subsample すると 99.7% の学習データを捨て、正例は期待 ~15 件で ranking を学習できない。
さらに TabPFN は group 構造を無視する pointwise classifier。よって **本データ規模・imbalance
では reranker family として成立しない**ため不採用（pip 導入自体は可能だが scale が壁）。

## Design

- 共有 wrapper:
  - `mcrs/rerank_modules/multi_family.py`: XGBoost LambdaRank(`rank:ndcg`) / CatBoost
    YetiRank の fit/predict、group 内 z-score 正規化、**family 単位等重み** blend
    （6 LightGBM seed でも 1 票。per-model でなく per-family 等重みで XGB/CatBoost の埋没を防ぐ）。
  - `mcrs/rerank_modules/tabm_ranker.py`: TabM（k-MLP ensemble）を pos_weight 付き
    BCEWithLogitsLoss の pointwise scorer として学習。特徴量は train 統計で z-score 標準化。
- orchestration: `ensemble_rerank.py`（exp058 `seed_ensemble_wide100` の LightGBM 経路を
  そのまま再利用し、XGB/CatBoost/TabM family を追加）。OOF は各 family 1 seed で**同一 fold**
  比較し、blend 効果を attributable にする。
- XGB/CatBoost/TabM は GPU(RTX4090)、LightGBM は CPU(8 threads)。

## Command

```bash
# GBDT 3 種（lxc）
bash mcrs/experiments/exp116_gbdt_family_ensemble/run_ens.sh
# + TabM（lxct）
bash mcrs/experiments/exp116_gbdt_family_ensemble/run_ens_tabm.sh
# submission（response 全行新規生成 + zip + validate）
bash mcrs/experiments/exp116_gbdt_family_ensemble/build_submission.sh <ranking.json> <tag>
```

## Artifacts

- ranking: `results/ranker/blindA_tracks_ens_lxc.json` / `..._lxct.json`
- submission: `exp/inference/blindset_A/exp116_lxc_A.json` + `submission_exp116_lxc_A.zip`
  （80 件 / top-20 / root=prediction.json / 名前 27 字、validator OK）、`..._lxct_A.*` 同様。
- report: `results/ranker/report_ens_lxc.json` / `report_ens_lxct.json`

## Results (Blind A) — SCORED 2026-06-23

両 submission を Codabench / Blind A で採点。LB score block:

```text
submission_exp116_lxc_A.zip
  ndcg@20                   0.4431
  catalog_diversity         0.0277
  lexical_diversity         0.7395
  llm_judge_score           4.7000
  composite_score           0.5758
```

```text
submission_exp116_lxct_A.zip
  ndcg@20                   0.4420
  catalog_diversity         0.0277
  lexical_diversity         0.7388
  llm_judge_score           4.7500
  composite_score           0.5789
```

```text
submission_exp116_lxct6_A.zip   (4 family 各 6-seed 対称・採用 base 推奨デプロイ)
  ndcg@20                   0.4410
  catalog_diversity         0.0282
  lexical_diversity         0.7361
  llm_judge_score           4.9500
  composite_score           0.5932
```

| variant | Blind A nDCG | cat_div | lexical | judge | composite | vs exp090-fix nDCG | status |
|---|---:|---:|---:|---:|---:|---:|---|
| exp090-fix（base, 参考） | 0.4396 | 0.0281 | 0.7679 | 4.55 | 0.5657 | — | base |
| exp116 lxc（lgbm+xgb+catboost, 2-seed） | **0.4431** | 0.0277 | 0.7395 | 4.70 | 0.5758 | +0.0035 | candidate |
| exp116 lxct（+tabm, 2-seed 非対称） | **0.4420** | 0.0277 | 0.7388 | 4.75 | 0.5789 | +0.0024 | candidate |
| **exp116 lxct6（+tabm, 4 family 各 6-seed 対称）** | **0.4410** | 0.0282 | 0.7361 | **4.95** | **0.5932** | +0.0014 | **adopted-base deploy** |

**lxct6 採点の解釈（結論不変・composite 最高更新は response 軸）**:

- **nDCG 0.4410 は exp090-fix 0.4396 比 +0.0014＝[[blind-a-ndcg-noise-floor]] の ±0.02 floor 内**で
  base 同等。2-seed lxct 0.4420 とも −0.0010 で noise 内（6-seed の効果は variance のみで ranking は
  ほぼ不変＝top1 2/80 しか動かなかった件と整合）。**「model 多様化は usable nDCG gain を出さない」は
  実測でも不変**（Dev OOF lxct6 0.17953 < lgbm 0.18173 とも整合）。
- **composite 0.5932 は project 歴代最高**（旧最高 exp063 V-Claude-long 0.5929 を僅かに更新）だが、
  内訳は judge **4.95**（exp090-fix 4.55 / 2-seed lxct 4.75 から上振れ）が主因で、nDCG は noise・
  lexical はむしろ 0.7361 と低め。すなわち composite 更新は **submission 都度の response 全行新規生成
  （[[response-full-regen-rule]]）の judge variance による良い draw** であり、reranker model 層
  （ranking）の寄与ではない。judge 4.95 は単一サンプルの上振れで、±0.02 nDCG floor と同様に
  judge も run 間 variance が大きい点に留意（再採点で 4.7 前後へ戻り得る）。
- 採用根拠は変わらず **robustness 仮説**（4 family を等しく 6-seed で束ね variance 最小化）。
  measured nDCG cost は実測でもゼロ（base と noise 内同等）で、composite は上振れ。

### 解釈（Blind A 実測＝Dev OOF 結論を追認）

- **nDCG: 両 variant とも base を僅かに上回るが delta（lxc +0.0035 / lxct +0.0024）は
  [[blind-a-ndcg-noise-floor]] の ±0.02 floor を大きく下回る＝確定改善ではなく noise**。
  Dev OOF で等重み blend が base 未満（lxc 0.17954 / lxct 0.17951 vs lgbm 0.18090）だった結論と
  矛盾せず、**model-family ensemble は Blind A nDCG に usable gain を出さない**ことを実測で追認。
  parity-gate lgbm+xgb の Dev OOF +0.00016 も含め、同一 78 特徴量を共有する関数族を増やしても
  ordering ceiling は動かない（[[dev-blinda-drift-decomposition]] / [[test-tracks-lever-and-measurement]]
  の reranker-bound を model 軸で再確認）。
- **composite は base 比 lxc +0.0101 / lxct +0.0132 で上振れしたが、内訳は response 軸**:
  judge 4.55→4.70/4.75（+0.15/+0.20）が主因で、lexical はむしろ 0.7679→0.7395/0.7388 と低下、
  nDCG は noise 内。すなわち composite gain は **submission 都度の response 全行新規生成
  （[[response-full-regen-rule]]）の judge variance** によるもので、reranker model 層の寄与ではない。
  歴代最高 composite は exp063 V-Claude-long 0.5929（response 軸）のままで lxct 0.5789 はそれ未満。
- **lxct > lxc（composite +0.0031）も judge 4.75 vs 4.70 の response variance** で、TabM 追加が
  ranking を改善した証拠ではない（nDCG はむしろ lxct < lxc）。

### 採否（2026-06-23 user 判断）

**★ lxct を新 bankable ranking base に採用（exp090-fix を superseded）／ response base は exp063 据え置き。**

- 採用根拠は **頑健性**: prior bankable exp090-fix は 6-seed でも単一 family（同一 GBDT・同一 HP）で、
  その family の系統的 overfit に脆い。lxct は XGBoost / CatBoost（実装・正則化の異なる GBDT）に加え
  **別関数族の TabM（MLP ensemble、最も decorrelated）** を等重みで束ね、final ranking の variance を
  下げて単一 family の失敗モードをヘッジする。特に Blind B（final eval、未知分布）への transfer で
  diverse ensemble が単一 family の良い draw 頼みよりロバスト、という仮説に基づく判断。
- **measured nDCG cost はゼロ**: Blind A nDCG 0.4420 は exp090-fix 0.4396 と noise floor ±0.02 内で
  同等、composite はむしろ +0.0132。lxc(0.4431) ともほぼ同値だが、別関数族 TabM の decorrelation を
  理由に lxct を優先（lxc は not adopted・残置）。
- **重要な留保（empirical 結論は不変）**: 採用は point-estimate gain ではなく robustness 仮説に基づく。
  Dev OOF（信頼 ruler）では等重み blend lxct 0.17951 < lgbm 単独 0.18090 で **むしろ微減**であり、
  composite 上振れも judge 4.55→4.75 の response 全行新規生成 variance 由来（ranking 寄与ではない）。
  「同一 78 特徴量を共有する model 多様化は usable nDCG gain を出さない」という
  [[model-family-ensemble-no-headroom]] の結論はそのまま生きており、変わったのは bankable base の
  選択（robustness 判断）のみ。
- **実務含意**: 今後の ranking addon 実験は OOF ruler / Blind ranking 生成を 4-family
  `ensemble_rerank.py` 上で組む（lgbm 単独より compute 増・OOF 比較基準も ensemble に変わる）。
  残る nDCG レバーは候補 recall、composite レバーは response 軸。

## 6-seed 対称版（lxct6 = 採用 base の推奨デプロイ）— 2026-06-23

当初の lxct は **seed 数が family 間で非対称**（lgbm 6 / xgb 2 / catboost 2 / tabm 1）だった。
robustness 採用の趣旨（多 family で variance を下げる）を最大化するため、**4 family すべてを
6 seed で seed-ensemble**（full-fit / OOF とも対称）し直したのが lxct6。これを採用 base の
**推奨デプロイ artifact** とする（全 family が等しく seed 平均され、final ranking の variance が
最小）。

| | seed 数 | Dev OOF nDCG@20 |
|---|---|---:|
| LightGBM 単独（6-seed） | 6 | 0.18173 |
| XGBoost 単独（6-seed） | 6 | 0.18012 |
| CatBoost 単独（6-seed） | 6 | 0.17032 |
| TabM 単独（6-seed） | 6 | 0.17765 |
| **lxct6 blend（4 family 各 6-seed 等重み）** | 6×4 | **0.17953** |

- **結論は 6-seed でも不変（むしろ明確化）**: lgbm 単独は 6-seed で 0.18090→**0.18173**（variance
  低減で微増）する一方、等重み blend は 0.17953 のままで、gap は 1-seed の −0.0014 から
  **−0.0022 に拡大**。seed ensemble は variance を下げるが弱 family（catboost 0.17032 /
  tabm 0.17765）の mean=bias は動かさないので、希釈で base 未満という構図は seed 数を揃えても同じ。
  「同一 78 特徴量を共有する model 多様化は usable nDCG gain を出さない」
  （[[model-family-ensemble-no-headroom]]）は seed 対称化後も生きている。lgbm full-fit 0.259737 も
  再び exp090-fix と完全一致＝候補/特徴量は byte-identical のまま。
- ranking は 2-seed lxct から top1 が 2/80 しか動かない（6-seed の効果は variance のみで ranking は
  ほぼ不変）。採用根拠は変わらず **robustness 仮説**（point gain ではない）。
- artifact: `exp/inference/blindset_A/exp116_lxct6_A.json` +
  `submission_exp116_lxct6_A.zip`（80 件 / top-20 / root=prediction.json / 名 29 字、validator OK、
  response 80 行は lxct6 top1 に合わせ全行新規生成）。
- 実行: `bash mcrs/experiments/exp116_gbdt_family_ensemble/run_ens_lxct6.sh`（lxct のみ・6-seed 対称・
  full-fit + OOF）→ `build_submission.sh <blindA_tracks_ens_lxct6.json> lxct6`。
- **★ SCORED 2026-06-23（Blind A）**: nDCG **0.4410** / cat_div 0.0282 / lexical 0.7361 /
  judge **4.95** / composite **0.5932**。nDCG は exp090-fix 0.4396・2-seed lxct 0.4420 と noise 内同等
  （6-seed でも ranking ほぼ不変を実測追認）。composite 0.5932 は歴代最高（旧 exp063 0.5929 を僅か更新）
  だが judge 4.95 の response variance 由来で ranking 寄与ではない。詳細は上の Results 節を参照。

## Blind B 適用（2026-06-24・final phase・未提出）

- Blind B（cold 40/80・user_id/profile 欠落・goal/thought 削除）へ bankable lxct6 を本番適用。
- 候補 source は **Blind B 用に全 13 種を再生成**（`--blind_dataset_name Blind-B`、置き場 `exp/smoke_blindB/sources/`）。
  cross_session は空 user_id を pool 除外する **leak-safe 修正**版（`cross_session_candidates.py`、Blind A は canonical と
  byte-identical を確認＝Dev/Blind A 無影響）。within-session exclude も Blind B 分を regenerate。
- 実行: `run_ens_lxct6_blindB.sh`（dev `--source` / params は `run_ens_lxct6.sh` と byte-identical、blind 側のみ
  Blind B）。`REDUCED=1` で軽量 dry-run（1 seed / OOF skip）し **13 source の join・80 行・top-20 を検証**してから本番 6-seed
  を実行。応答は `generate_responses_v2.py --backend claude --model claude-opus-4-8 --test_dataset_name ...Blind-B` で
  80 行全新規生成（leak-safe・最終 ranking top1 整合）。
- 成果物: ranking `results/ranker/blindB_tracks_ens_blindB_lxct6.json`、submission `exp/inference/blindset_B/exp116_lxct6_B.json`
  / `submission_exp116_lxct6_B.zip`（**validate OK**: 80records / 全 top-20 / catalog 外 0 / 空 resp 0 / 名 29 字）。
- Dev full-fit blend nDCG `0.219`（lgbm 0.260 / xgb 0.243 / catboost 0.176 / tabm 0.181）はあくまで**学習 sanity**で、
  full-fit ゆえ楽観・[[model-family-ensemble-no-headroom]] 同様 lgbm 単独 ≥ blend。**Blind B 実 score は hidden LB のみ**。
- robustness 根拠: cross_session の Dev 生成では `true-cross nonempty 5208/8000`＝学習行の ~35% が元々 cross_session 空。
  cold Blind B 行（修正後＝空 cross）は **in-distribution** で新規入力ではない。詳細・smoke は [exp/smoke_blindB/REPORT.md](../../../exp/smoke_blindB/REPORT.md)。

### LightGBM 6-seed 単独 Blind B hedge（2026-06-28・Codabench scored）

`lxct6_B` を primary submission として提出済み。残り 2 枠の 1 つ用に、同じ Blind B 再生成 source / 修正版
cross_session / within-session exclude を使い、model family だけを **LightGBM 6-seed 単独**へ縮退した
`lgbm6_B` を作成した。狙いは `lxct6` の頑健性採用に対する nDCG hedge。Dev OOF は既存 `lxct6` report で
`lgbm 0.18173 > lxct6 blend 0.17944` と分かっているため、この run は `--skip_oof` で full-fit ranking
生成に絞った。

Command:

```bash
OUT=mcrs/experiments/exp116_gbdt_family_ensemble/results
CS=mcrs/experiments/exp090_cross_session_source/results
D21=mcrs/experiments/exp021_candidate_fusion/results
D23=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100
D24=mcrs/experiments/exp024_transition_memory/results/transition_memory_top100
BB=exp/smoke_blindB/sources
.venv/bin/python mcrs/experiments/exp116_gbdt_family_ensemble/ensemble_rerank.py \
  --families "lgbm" \
  --seeds "20260616,20260617,20260618,20260619,20260620,20260621" \
  --skip_oof \
  --primary "exp015B=mcrs/experiments/exp015_candidate_rules_ablation/results/top500/B_exact_split/lgbm_B_ll_top500_dev_predictions.json" \
  --blind_primary "exp015B=exp/inference/blindset_B/exp015_B_ll_t500_B.json" \
  --source "cont_aat=${D21}/continuity_candidates/dev_tracks_exp021_cont_album_artist_tag.json" \
  --source "cont_aa=${D21}/continuity_candidates/dev_tracks_exp021_cont_album_artist.json" \
  --source "cont_album=${D21}/continuity_candidates/dev_tracks_exp021_cont_album.json" \
  --source "cont_artist=${D21}/continuity_candidates/dev_tracks_exp021_cont_artist.json" \
  --source "qnm_rp=${D21}/query_neighbor_memory/dev_tracks_exp021_qnm_recent_profile.json" \
  --source "qnm_hist=${D21}/query_neighbor_memory/dev_tracks_exp021_qnm_history_music.json" \
  --source "qnm_cur=${D21}/query_neighbor_memory/dev_tracks_exp021_qnm_current.json" \
  --source "ent_cur=${D23}/dev_tracks_exp023_entity_current.json" \
  --source "ent_tight=${D23}/dev_tracks_exp023_entity_current_tight.json" \
  --source "ent_r2=${D23}/dev_tracks_exp023_entity_recent2.json" \
  --source "ent_r4=${D23}/dev_tracks_exp023_entity_recent4.json" \
  --source "trans_r1=${D24}/dev_tracks_exp024_transition_recent1.json" \
  --source "cross_session:50=${CS}/nopad/dev_tracks_cross_session.json" \
  --blind_source "cont_aat=${BB}/continuity/blindA_tracks_exp021_cont_album_artist_tag.json" \
  --blind_source "cont_aa=${BB}/continuity/blindA_tracks_exp021_cont_album_artist.json" \
  --blind_source "cont_album=${BB}/continuity/blindA_tracks_exp021_cont_album.json" \
  --blind_source "cont_artist=${BB}/continuity/blindA_tracks_exp021_cont_artist.json" \
  --blind_source "qnm_rp=${BB}/qnm/blindA_tracks_exp021_qnm_recent_profile.json" \
  --blind_source "qnm_hist=${BB}/qnm/blindA_tracks_exp021_qnm_history_music.json" \
  --blind_source "qnm_cur=${BB}/qnm/blindA_tracks_exp021_qnm_current.json" \
  --blind_source "ent_cur=${BB}/entity/blindA_tracks_exp023_entity_current.json" \
  --blind_source "ent_tight=${BB}/entity/blindA_tracks_exp023_entity_current_tight.json" \
  --blind_source "ent_r2=${BB}/entity/blindA_tracks_exp023_entity_recent2.json" \
  --blind_source "ent_r4=${BB}/entity/blindA_tracks_exp023_entity_recent4.json" \
  --blind_source "trans_r1=${BB}/transition/blindA_tracks_exp024_transition_recent1.json" \
  --blind_source "cross_session:50=${BB}/cross_session/blindA_tracks_cross_session.json" \
  --exclude_json "${BB}/within_session_exclude_blindB.json" \
  --allow_short_sources \
  --protect_top 0 \
  --num_boost_round 450 --num_leaves 31 --min_data_in_leaf 80 --lambda_l2 8.0 --num_threads 8 \
  --output_report "${OUT}/ranker/report_ens_blindB_lgbm6.json" \
  --output_dev_full "${OUT}/ranker/dev_full_ens_blindB_lgbm6.json" \
  --output_blind_tracks "${OUT}/ranker/blindB_tracks_ens_blindB_lgbm6.json"

.venv/bin/python mcrs/experiments/exp063_response_quality/generate_responses_v2.py \
  --backend claude \
  --model claude-opus-4-8 \
  --tracks_json mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/blindB_tracks_ens_blindB_lgbm6.json \
  --output exp/inference/blindset_B/exp116_lgbm6_B.json \
  --test_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-B \
  --batch_size 6
```

Artifacts:

- ranking: `mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/blindB_tracks_ens_blindB_lgbm6.json`
- report: `mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/report_ens_blindB_lgbm6.json`
- response/submission JSON: `exp/inference/blindset_B/exp116_lgbm6_B.json`
- submission ZIP: `exp/inference/blindset_B/submission_exp116_lgbm6_B.zip`（root=`prediction.json`, name 29 chars）
- logs: `exp/smoke_blindB/logs/ensemble_lgbm6.log`, `exp/smoke_blindB/logs/resp_lgbm6_blindB.log`

Report summary:

```json
{
  "families": ["lgbm"],
  "candidate_rows": 3078934,
  "candidate_positive_count": 4528,
  "base_ndcg": 0.14441804967129152,
  "full_fit_blend_ndcg": 0.2597372988789493,
  "blind_change_stats_vs_primary": {
    "changed": 80,
    "top1_changed": 57,
    "avg_overlap20": 11.725
  }
}
```

Diff vs submitted `exp116_lxct6_B.json`:

```json
{
  "rows": 80,
  "changed": 80,
  "top1_changed": 10,
  "avg_overlap20": 17.1875
}
```

Validation:

```text
=== validate_submission: exp/inference/blindset_B/exp116_lgbm6_B.json ===
records: 80
OK: 提出制約をすべて満たす。

=== validate_submission: exp/inference/blindset_B/submission_exp116_lgbm6_B.zip ===
records: 80
OK: 提出制約をすべて満たす。
```

catalog 照合込み:

```text
[info] catalog 照合: 47071 track_ids を読み込み。
records: 80
OK: 提出制約をすべて満たす。
```

Compact check:

```json
{
  "records": 80,
  "empty_responses": 0,
  "min_tracks": 20,
  "max_tracks": 20,
  "unique_top1": 78,
  "unique_tracks": 1300
}
```

Leaderboard Result:

- Submission id: `816453`
- File name: `submission_exp116_lgbm6_B.zip`
- Date: `2026-06-28 11:22`
- Status: `Finished`
- Codabench Score: `0.47`
- Per-metric breakdown: unavailable from the displayed table.

```text
File name                              Date                Status    Score
submission_exp116_lgbm6_B.zip          2026-06-28 11:22    Finished  0.47
```

Interpretation:

- `lgbm6_B` は `lxct6_B` と同じ candidate/source/exclude を使い、弱 family 希釈を外した nDCG hedge。
- Blind B Codabench score は `0.47`。同日提出の `lgbm6_nocs_B` (`0.50`) を下回ったため、Blind B では cross_session を完全に抜いた hedge の方が良い。
- response は `lgbm6_B` の最終 top1 に合わせて 80 行すべて Claude backend で新規生成済み。

### LightGBM 6-seed / cross_session 完全 drop Blind B hedge（2026-06-28・Codabench scored）

提出済み `lxct6_B` と作成済み `lgbm6_B` は cold pollution 修正済み `cross_session` を含む。一方
Blind B は cold user 40/80 かつ warm 側も cross-session pool が sparse なので、`cross_session` source
自体を dev/blind 両方から完全に抜いた分布シフト hedge として `lgbm6_nocs_B` を作成した。family は
ユーザー指定どおり **LightGBM 6-seed 単独**、response は最終 ranking top1 に合わせて 80 行すべて
Claude backend で新規生成。

Command:

```bash
OUT=mcrs/experiments/exp116_gbdt_family_ensemble/results
D21=mcrs/experiments/exp021_candidate_fusion/results
D23=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100
D24=mcrs/experiments/exp024_transition_memory/results/transition_memory_top100
BB=exp/smoke_blindB/sources
.venv/bin/python mcrs/experiments/exp116_gbdt_family_ensemble/ensemble_rerank.py \
  --families "lgbm" \
  --seeds "20260616,20260617,20260618,20260619,20260620,20260621" \
  --skip_oof \
  --primary "exp015B=mcrs/experiments/exp015_candidate_rules_ablation/results/top500/B_exact_split/lgbm_B_ll_top500_dev_predictions.json" \
  --blind_primary "exp015B=exp/inference/blindset_B/exp015_B_ll_t500_B.json" \
  --source "cont_aat=${D21}/continuity_candidates/dev_tracks_exp021_cont_album_artist_tag.json" \
  --source "cont_aa=${D21}/continuity_candidates/dev_tracks_exp021_cont_album_artist.json" \
  --source "cont_album=${D21}/continuity_candidates/dev_tracks_exp021_cont_album.json" \
  --source "cont_artist=${D21}/continuity_candidates/dev_tracks_exp021_cont_artist.json" \
  --source "qnm_rp=${D21}/query_neighbor_memory/dev_tracks_exp021_qnm_recent_profile.json" \
  --source "qnm_hist=${D21}/query_neighbor_memory/dev_tracks_exp021_qnm_history_music.json" \
  --source "qnm_cur=${D21}/query_neighbor_memory/dev_tracks_exp021_qnm_current.json" \
  --source "ent_cur=${D23}/dev_tracks_exp023_entity_current.json" \
  --source "ent_tight=${D23}/dev_tracks_exp023_entity_current_tight.json" \
  --source "ent_r2=${D23}/dev_tracks_exp023_entity_recent2.json" \
  --source "ent_r4=${D23}/dev_tracks_exp023_entity_recent4.json" \
  --source "trans_r1=${D24}/dev_tracks_exp024_transition_recent1.json" \
  --blind_source "cont_aat=${BB}/continuity/blindA_tracks_exp021_cont_album_artist_tag.json" \
  --blind_source "cont_aa=${BB}/continuity/blindA_tracks_exp021_cont_album_artist.json" \
  --blind_source "cont_album=${BB}/continuity/blindA_tracks_exp021_cont_album.json" \
  --blind_source "cont_artist=${BB}/continuity/blindA_tracks_exp021_cont_artist.json" \
  --blind_source "qnm_rp=${BB}/qnm/blindA_tracks_exp021_qnm_recent_profile.json" \
  --blind_source "qnm_hist=${BB}/qnm/blindA_tracks_exp021_qnm_history_music.json" \
  --blind_source "qnm_cur=${BB}/qnm/blindA_tracks_exp021_qnm_current.json" \
  --blind_source "ent_cur=${BB}/entity/blindA_tracks_exp023_entity_current.json" \
  --blind_source "ent_tight=${BB}/entity/blindA_tracks_exp023_entity_current_tight.json" \
  --blind_source "ent_r2=${BB}/entity/blindA_tracks_exp023_entity_recent2.json" \
  --blind_source "ent_r4=${BB}/entity/blindA_tracks_exp023_entity_recent4.json" \
  --blind_source "trans_r1=${BB}/transition/blindA_tracks_exp024_transition_recent1.json" \
  --exclude_json "${BB}/within_session_exclude_blindB.json" \
  --allow_short_sources \
  --protect_top 0 \
  --num_boost_round 450 --num_leaves 31 --min_data_in_leaf 80 --lambda_l2 8.0 --num_threads 8 \
  --output_report "${OUT}/ranker/report_ens_blindB_lgbm6_nocs.json" \
  --output_dev_full "${OUT}/ranker/dev_full_ens_blindB_lgbm6_nocs.json" \
  --output_blind_tracks "${OUT}/ranker/blindB_tracks_ens_blindB_lgbm6_nocs.json"

.venv/bin/python mcrs/experiments/exp063_response_quality/generate_responses_v2.py \
  --backend claude \
  --model claude-opus-4-8 \
  --tracks_json mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/blindB_tracks_ens_blindB_lgbm6_nocs.json \
  --output exp/inference/blindset_B/exp116_lgbm6_nocs_B.json \
  --test_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-B \
  --batch_size 6
```

Artifacts:

- ranking: `mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/blindB_tracks_ens_blindB_lgbm6_nocs.json`
- report: `mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/report_ens_blindB_lgbm6_nocs.json`
- response/submission JSON: `exp/inference/blindset_B/exp116_lgbm6_nocs_B.json`
- submission ZIP: `exp/inference/blindset_B/submission_exp116_lgbm6_nocs_B.zip`（root=`prediction.json`, name 34 chars）
- logs: `exp/smoke_blindB/logs/ensemble_lgbm6_nocs.log`, `exp/smoke_blindB/logs/resp_lgbm6_nocs_blindB.log`

Report summary:

```json
{
  "families": ["lgbm"],
  "source_count": 12,
  "has_cross_session": false,
  "candidate_rows": 2949717,
  "candidate_positive_count": 4333,
  "base_ndcg": 0.14441804967129152,
  "full_fit_blend_ndcg": 0.25870485648368946,
  "blind_change_stats_vs_primary": {
    "changed": 80,
    "top1_changed": 56,
    "avg_overlap20": 12.0125
  }
}
```

Diff:

```json
{
  "nocs_vs_lgbm6_with_cross": {
    "rows": 80,
    "changed": 80,
    "top1_changed": 10,
    "avg_overlap20": 18.1625
  },
  "nocs_vs_lxct6_submitted": {
    "rows": 80,
    "changed": 80,
    "top1_changed": 12,
    "avg_overlap20": 16.8
  }
}
```

Validation:

```text
=== validate_submission: exp/inference/blindset_B/exp116_lgbm6_nocs_B.json ===
records: 80
OK: 提出制約をすべて満たす。

=== validate_submission: exp/inference/blindset_B/submission_exp116_lgbm6_nocs_B.zip ===
records: 80
OK: 提出制約をすべて満たす。
```

catalog 照合込み:

```text
[info] catalog 照合: 47071 track_ids を読み込み。
records: 80
OK: 提出制約をすべて満たす。
```

Compact check:

```json
{
  "records": 80,
  "empty_responses": 0,
  "min_tracks": 20,
  "max_tracks": 20,
  "unique_top1": 78,
  "unique_tracks": 1296
}
```

Sample Prediction Output:

`k = 3`

```bash
jq --argjson k 3 '.[0:$k] | map({session_id, user_id, turn_number, predicted_track_ids, predicted_response: (.predicted_response[0:500])})' exp/inference/blindset_B/exp116_lgbm6_nocs_B.json
```

```json
[
  {
    "session_id": "ff76b679-7d7a-4796-8f6c-929efef45428",
    "user_id": "",
    "turn_number": 1,
    "predicted_track_ids": [
      "3c03ae26-ff7d-4295-b1eb-79630d5a318c",
      "eae46469-483b-4219-bbde-1c159c165353",
      "189f6360-072e-4bcf-a2e6-60b8ddbc583e",
      "80801d29-2556-4952-98ae-be91231b5b0b",
      "99c31027-3ce6-4bc8-9caa-1be90edb9f84",
      "f204633b-885f-4bb4-a586-32bf3596e350",
      "9aeed52a-cdab-497f-a722-c481fbc3c149",
      "dd9afb63-db6e-40d7-84ba-4f0160f5bb00",
      "ef35b0a5-b024-4603-8dd8-4eb024cd3dc7",
      "0efc3e6f-17e1-461f-bf78-88a5ec8126ca",
      "2e0cad9b-6c1c-41ed-8e16-eae0d8c31a83",
      "6058fb7e-3a0c-4512-8240-a3d67c08f8df",
      "bfae69c6-9627-4829-b673-718b1065ee32",
      "3b08891f-8ba3-4d9b-b0ab-4f9afa5b58dd",
      "010211af-6f11-4379-991d-74867a58df0d",
      "e37e90ec-64a2-46ad-977c-93d74bd463df",
      "18bc31bc-acb9-446a-a5bf-029d071c7674",
      "90daf37a-9905-4d8e-85fe-1be22f4f2977",
      "914bf9cb-3f2c-4c58-a6fd-ebcdbe772d3d",
      "fd035677-fcce-466b-84cf-0d39d208ac9d"
    ],
    "predicted_response": "Alternative rock with a playful, brainy streak — that's exactly the lane \"If I Had $1,000,000\" by Barenaked Ladies lives in, pulled from their 1992 debut Gordon. While I can't speak to the album's cover art specifically, what I can promise is a track every bit as bold and vivid as the abstract, color-splashed painting you're picturing: its nerd-rock wit and bright alternative-rock bounce paint in big, daydreamy strokes, all cheeky imagination and warmth. You wanted something striking and a littl"
  },
  {
    "session_id": "68993adf-60d9-42f9-b21f-cdd5bf851cb1",
    "user_id": "",
    "turn_number": 1,
    "predicted_track_ids": [
      "f700abf0-8600-4509-8fcd-b040539342ad",
      "6e0dcd0d-46a2-43b8-9804-db8555a6a3a2",
      "f2613dc1-314a-4980-be28-836bb463bbb1",
      "58f968de-875b-4cd1-9c6e-5364010c1203",
      "f204633b-885f-4bb4-a586-32bf3596e350",
      "bfae69c6-9627-4829-b673-718b1065ee32",
      "9aeed52a-cdab-497f-a722-c481fbc3c149",
      "ef35b0a5-b024-4603-8dd8-4eb024cd3dc7",
      "799b3223-9867-4c6e-892e-239c9715d352",
      "f4dc4244-9eff-44bc-942d-9dc2ae546142",
      "bbd5c74b-2732-43cc-bf1f-6eda4ba57dc8",
      "7a631803-8f7f-47b0-b54d-b7d5bca5d678",
      "189f6360-072e-4bcf-a2e6-60b8ddbc583e",
      "f53eced1-8cac-47bb-a69c-9bb664502fb6",
      "010211af-6f11-4379-991d-74867a58df0d",
      "a9e37ddd-a793-4fb4-9103-1f9912ab9f0e",
      "2e0cad9b-6c1c-41ed-8e16-eae0d8c31a83",
      "e13d8f66-d93f-43ef-8c72-962ca1d807f4",
      "97b7e3c7-945f-4283-abcf-ef1190b4d17c",
      "c27aeae0-8d82-46cb-abd9-35d9937b1ce9"
    ],
    "predicted_response": "Landing in 1996, \"3AM\" by Matchbox Twenty arrives from a peak moment in storytelling rock, exactly the kind of late-'90s songwriting built to carry a life unfolding! As a cornerstone of that era's modern rock and alternative rock movement, it pairs an energetic, driving pulse with a warmth that pulls you straight into its world. The easy-listening accessibility means the narrative never gets lost in noise — every emotional beat lands clearly, which is just what you want when you're chasing a son"
  },
  {
    "session_id": "ab87371b-9ebc-4fb1-9eea-b984a5b76258",
    "user_id": "",
    "turn_number": 1,
    "predicted_track_ids": [
      "189f6360-072e-4bcf-a2e6-60b8ddbc583e",
      "9aeed52a-cdab-497f-a722-c481fbc3c149",
      "ffc14c3d-21b1-4938-b58e-28c21c4d39a3",
      "010211af-6f11-4379-991d-74867a58df0d",
      "f204633b-885f-4bb4-a586-32bf3596e350",
      "e16e2363-a173-471e-953b-0a4fa7cf7ea3",
      "a4af2660-a86a-4b15-b580-4729dfe0d8c0",
      "a9e37ddd-a793-4fb4-9103-1f9912ab9f0e",
      "6e257dc1-fe8b-4f33-b9b7-b5ef058dd111",
      "f1869385-f814-454c-9843-95b2433aee7e",
      "0efc3e6f-17e1-461f-bf78-88a5ec8126ca",
      "914bf9cb-3f2c-4c58-a6fd-ebcdbe772d3d",
      "fd035677-fcce-466b-84cf-0d39d208ac9d",
      "ef35b0a5-b024-4603-8dd8-4eb024cd3dc7",
      "bbd5c74b-2732-43cc-bf1f-6eda4ba57dc8",
      "c770326a-69af-4a93-ac5a-eb460f8d9b1e",
      "d0dd75dd-3b6d-450b-9fde-b57a012a1902",
      "2e0cad9b-6c1c-41ed-8e16-eae0d8c31a83",
      "cb8efcb8-ea64-472d-9ad3-ccfe4a4e5759",
      "90b48e8c-8f19-4f2f-893f-bb1b722722ef"
    ],
    "predicted_response": "You clearly thrive on music with real intensity and emotional grip, so let me hand you \"Iris\" by The Goo Goo Dolls — a 1998 alternative rock ballad that hits with surprising force! That bittersweet, slow-building energy is its own kind of focus fuel: the kind of swelling, raw emotional pull that quiets the noise in your head and locks you into a single feeling when a creative block has you stuck. Its soaring alternative rock dynamics give you momentum to push through, while the aching, heartfelt"
  }
]
```

Leaderboard Result:

- Submission id: `816456`
- File name: `submission_exp116_lgbm6_nocs_B.zip`
- Date: `2026-06-28 11:37`
- Status: `Finished`
- Codabench Score: `0.50`
- Per-metric breakdown: unavailable from the displayed table.

```text
File name                                   Date                Status    Score
submission_exp116_lgbm6_nocs_B.zip          2026-06-28 11:37    Finished  0.50
```

Interpretation:

- `lgbm6_nocs_B` は `cross_session` を完全に抜いた 12 source 構成で、Blind B の cold/warm sparse shift への hedge。
- Dev full-fit は `lgbm6_B` の 0.259737 から 0.258705 へ小差低下するが、これは label 付き Dev sanity で Blind B の真 score は hidden LB のみ。
- Blind B Codabench score は `0.50` で、`lgbm6_B` (`0.47`) を `+0.03` 上回った。Blind B では、Dev では有効だった cross_session source が cold 40/80 と warm sparse pool でノイズ化していたという smoke 診断を支持する結果。
- response は `lgbm6_nocs_B` の最終 top1 に合わせて 80 行すべて新規生成済み。過去 response の流用はない。
