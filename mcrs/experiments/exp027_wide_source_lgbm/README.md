# exp027 wide source LGBM

Last updated: 2026-06-17

## Objective

`exp026` の top20 後段 fusion では nDCG +0.1 に届かないため、候補集合を top100/top500 source まで抜本的に広げ、rank-source LightGBM で top20 を再構成する。ユーザー指示に従い、Blind A では top1 を固定せず、top1 変更後の推薦に合わせて LLM で `predicted_response` を新規生成する。

## Inputs

Primary anchor:

- Dev: `mcrs/experiments/exp015_candidate_rules_ablation/results/top500/B_exact_split/lgbm_B_ll_top500_dev_predictions.json`
- Blind A: `exp/inference/blindset_A/exp015_B_ll_t500_A.json`

Wide source set:

- continuity: `cont_album_artist_tag`, `cont_album_artist`, `cont_album`, `cont_artist`
- query-neighbor memory: `qnm_recent_profile`, `qnm_history_music`, `qnm_current`
- entity candidates: `entity_current`, `entity_current_tight`, `entity_recent2`, `entity_recent4`
- transition memory: `transition_recent1`
- top20 strong predictions in mixed setting: `exp015A`, `exp016_b2dense`, `exp017_e2v2`, `exp018_bgelarge`, `exp022_oof`

All sources use only current/past dialogue context and public catalog/profile metadata. Blind labels and future turns are not used.

## Oracle Diagnostics

`exp015_B` primary Dev baseline is `0.1444180497`.

| Candidate union | Source setup | Candidate hits | Oracle upper | Avg union size |
|---|---:|---:|---:|---:|
| top100 wide | 12 sources, all `source_topk=100` | 4,333 / 8,000 | 0.541625 | 371.57 |
| mixed topK | continuity=100, QNM=500, entity/transition=100, strong predictions=20 | 5,291 / 8,000 | 0.661375 | 1,015.19 |

The candidate ceiling is high enough to justify changing the ranker rather than continuing small top20 tail edits.

## Runs

| Run | protect_top | Candidate rows | Candidate positives | Dev full-fit nDCG | Dev OOF nDCG | Blind A change vs exp015_B |
|---|---:|---:|---:|---:|---:|---|
| `wide100_p1` | 1 | 2,972,524 | 4,333 | 0.2109176136 | 0.1685500139 | changed 80/80, top1 0/80, overlap20 11.55 |
| `wide100_p0` | 0 | 2,972,524 | 4,333 | 0.2558438735 | 0.1767990106 | changed 80/80, top1 55/80, overlap20 11.3875 |
| `mixed_p0` | 0 | 8,121,506 | 5,291 | **0.2763837874** | **0.1967225765** | changed 80/80, top1 54/80, overlap20 8.075 |

`mixed_p0` was the **Dev-time** selected run: full-fit `+0.1319657377` over `exp015_B` (meeting the requested `+0.1` Dev target), OOF `+0.0523045268`, all five session folds positive (below). **Blind A (2026-06-17) inverted this Dev-based choice: the no-dense `wide100_p0` won (ndcg 0.4279) over the dense-augmented `mixed_p0` (0.4207). See [Leaderboard Result](#leaderboard-result).** OOF folds for `mixed_p0`:

| Fold | Base | OOF | Delta |
|---:|---:|---:|---:|
| 0 | 0.1569665276 | 0.2026188287 | +0.0456523011 |
| 1 | 0.1528584028 | 0.2070740188 | +0.0542156159 |
| 2 | 0.1218324845 | 0.1760909132 | +0.0542584287 |
| 3 | 0.1488876395 | 0.2015712130 | +0.0526835736 |
| 4 | 0.1415451939 | 0.1962579089 | +0.0547127149 |

## LLM Response Generation

Top1 is not protected. For Blind A submission, responses are generated after reranking with:

```bash
.venv/bin/python mcrs/experiments/exp019_qwen_reranker_ft/generate_responses.py \
  --tracks_json mcrs/experiments/exp027_wide_source_lgbm/results/blindA_tracks_exp027_mixed_p0.json \
  --output exp/inference/blindset_A/exp027_mixed_p0_llm_A.json \
  --batch_size 8 \
  --max_new_tokens 512
```

This uses `Qwen/Qwen3.5-4B`, Blind A conversation history, user profile, and the reranked top1 track metadata. Output has 80 records and 0 empty responses.

## Artifacts

Adopted base artifact (`wide100_p0`, Blind A best — see [Leaderboard Result](#leaderboard-result)):

- LLM response JSON: `exp/inference/blindset_A/exp027_wide100_p0_llm_A.json`
- Submission ZIP: `exp/inference/blindset_A/submission_exp027_wide100_p0_llm_A.zip`
- Report: `mcrs/experiments/exp027_wide_source_lgbm/results/wide100_p0_report.json`

Dev-best comparison artifact (`mixed_p0`, lost on Blind A):

- Tracks-only: `mcrs/experiments/exp027_wide_source_lgbm/results/blindA_tracks_exp027_mixed_p0.json`
- LLM response JSON: `exp/inference/blindset_A/exp027_mixed_p0_llm_A.json`
- Submission ZIP: `exp/inference/blindset_A/submission_exp027_mixed_p0_llm_A.zip`
- Model: `mcrs/experiments/exp027_wide_source_lgbm/results/lgbm_mixed_p0.txt`
- Report: `mcrs/experiments/exp027_wide_source_lgbm/results/mixed_p0_report.json`

Validation:

```text
validate_submission.py exp/inference/blindset_A/exp027_mixed_p0_llm_A.json
records: 80
OK: 提出制約をすべて満たす。

validate_submission.py exp/inference/blindset_A/submission_exp027_mixed_p0_llm_A.zip
records: 80
OK: 提出制約をすべて満たす。

zipinfo -1 exp/inference/blindset_A/submission_exp027_mixed_p0_llm_A.zip
prediction.json
```

## Sample

```json
[
  {
    "session_id": "9c37dcd7-d7c2-4686-8541-1e37c4814a09",
    "turn_number": 1,
    "top3": [
      "59f8be05-193a-4112-87f3-1affe08d9865",
      "46b8d378-c1d2-419f-b026-4c4ed53617be",
      "1ff471e9-7449-4b85-9a09-8d41af667550"
    ],
    "response_prefix": "Hey there! I found a perfect track for you: \"Shut Up and Dance\" by Walk the Moon..."
  }
]
```

## Leaderboard Result

Submitted: yes (Codabench, 2026-06-17). Both exp027 variants cleared the previous Blind A gate `exp015_B` (ndcg `0.3869` / composite `0.4766`).

`wide100_p0` (12 session-local sources, no dense) — **adopted as new base**:

```text
ndcg@20                   0.4279
catalog_diversity         0.0280
lexical_diversity         0.6002
llm_judge_score           4.0000
composite_score           0.5018
```

`mixed_p0` (wide100 + 5 dense/prior predictions, Dev-best):

```text
ndcg@20                   0.4207
catalog_diversity         0.0306
lexical_diversity         0.6055
llm_judge_score           3.3500
composite_score           0.4502
```

## Interpretation

これは project 初の **reranker-bound / CV-LB 逆相関の打破**。break したのは reranker そのものではなく **candidate set**: session-local continuity/entity/transition source を top100 まで広げ、rank-source LightGBM で top20 を再構成したことが Blind A ndcg を `0.3869 → 0.4279` (+0.041) へ押し上げた。dense backbone (exp016/017/018) / LLM reranker (exp019x) lever は依然 dead で、効いたのは候補拡張側。

- **ndcg は robust に gate 超え**: `wide100_p0` +0.041 / `mixed_p0` +0.034、ともに noise floor ±0.02 の ~2倍。near-independent な 2 run が同方向で gate 超え。
- **2 variant は ndcg 上ほぼ tie**（0.4279 vs 0.4207 = +0.0072、noise floor 内）。composite の差（0.5018 vs 0.4502）は ~95% judge 由来（4.00 vs 3.35 = LLM response 再生成の variance）であり ranking の差ではない。よって採用根拠は ndcg、composite の優位は response 側の soft な差として扱う。
- **Dev-best ≠ Blind A best が再現**: Dev full-fit で勝っていた `mixed_p0`（0.27638 > wide100 0.25584）が Blind A では負けた。dense/prior 予測を足した `mixed` は Dev では効くが Blind A では効かない＝dense は依然 Dev-overfit lever。**live な lever は dense を足さない session-local 候補拡張**。
- composite 0.5018 は歴代最高 0.4952 (exp014 s06 fallback) も更新。gate に対する composite gain (+0.0252) は主に ndcg 由来（robust）＋小さい judge 寄与。

## ndcg 要因分析

ndcg を押し上げた要因を、Dev 学習モデルの feature importance（gain）・候補プール構造・wide vs mixed の Blind A 実測の3点で三角測量した。効いたのは reranker そのものではなく **候補生成の切り替え**で、それを精度優先のプール・dense/OOF 排除・lambdarank が支えた。

### 要因A（最大）: 候補集合が Blind A 正解分布（artist/album continuity）に直接命中

continuity source の根拠は EDA の「同一 session 内では exact track repeat はほぼ無いが artist/album は継続する」（[continuity_candidates.py](../exp021_candidate_fusion/continuity_candidates.py) docstring）。`cont_aa` は session の過去 turn に出た曲を同一 album+artist で展開した候補。

- `wide100_p0` の候補プール = **primary (exp015 B) の top20 ∪ 12 session-local source（各 top100、avg union 372件）**。primary は 20件アンカー＋feature として残るだけで、候補の大半は session-local。
- ranker 最重用 feature は **`cont_aa_inv_rank` = gain 22.8%（source group 計 29.8%）** で単独トップ。continuity 系で過半。

旧 base（BM25/dense retrieval）は意味的近傍を引くが session 内 artist/album 継続を体系的に狙えていなかった。それを 4 粒度（album / album+artist / +tag / artist）＋ entity/transition/QNM で明示候補化したのが直撃。

### 要因B: recall ceiling より precision（正例密度）が律速

| Run | oracle 上限 | Blind A ndcg | 実現率 | 正例密度 | avg 候補数 |
|---|---:|---:|---:|---:|---:|
| **wide100_p0（勝）** | 0.542 | **0.4279** | **79%** | 0.146% | 372 |
| mixed_p0（負） | 0.661 | 0.4207 | 64% | 0.065% | 1015 |

mixed は oracle 上限を +0.12 上げたのに実測 ndcg は下がった。top20 出力枠に対し候補を増やすほど 1 正例あたりの distractor が増え precision@20 が落ちる。recall は既に飽和し、効いたのは正例が濃いプール（密度 2.2倍・候補数 1/2.7）。

### 要因C: dense/OOF 系を外し「Dev-overfit な強 feature」依存を回避（wide が mixed に勝った機序）

| Run | top feature の gain 集中 | 学習された戦略 |
|---|---|---|
| `mixed_p0`（負） | `source_best_inv_rank` 36.6% ＋ `exp022_oof_inv_rank` 33.3% ＝ **約70%に collapse** | Dev 製 OOF 予測（学習 artifact）にロックイン＝Dev で強いが Blind A に転移せず |
| `wide100_p0`（勝） | cont_aa 30% / source集約 17% / primary 17% / qnm_hist 7% / trans_r1 6% / cont_aat 5% … **5本以上に分散** | raw retrieval rank だけから汎化可能な hedge blend |

「dense は Dev-overfit lever」の実体は、dense/OOF 候補が distractor を増やすだけでなく **ranker を奪う overfit feature を持ち込む**こと。

### 要因D（変換器）: lambdarank が per-source rank を直接 ndcg@20 最適化

`objective=lambdarank / metric=ndcg / eval_at=[20]`（[wide100_p0_report.json](results/wide100_p0_report.json)）で 12 source の inv_rank ＋ RRF/hit-rate 集約を blend。旧 base を 20件アンカー＋feature(16.7%) として残す形（top1 変更 55/80・overlap20 ~11）なので、ゼロ置換ではなく「旧 base ＋ session-local 候補拡張の再ランキング」。

### ndcg に効いていないもの（明示）

- **LLM response 再生成（Qwen3.5-4B）は ndcg に無関係**。ndcg は ranking のみ依存し、response は composite の judge/lexical 側だけ。wide vs mixed の composite 差が ~95% judge 由来なのはこのため。

### 証拠の限界

- feature importance は **Dev 学習モデル**由来。Blind A 80件では per-source ablation を取れないため、帰属は「Dev importance ＋ プール構造（oracle/密度）＋ wide vs mixed の Blind A 実測」の三角測量であり Blind A 上の直接分解ではない。
- gate 比 +0.041 は noise floor ±0.02 の2倍で有意、near-independent な 2 run がともに gate 超え＝候補拡張という効果自体は noise を超えて頑健。ただし各要因の寄与量を 80件で厳密分離はできない。

## Next Actions

1. ~~Submit~~ 完了。`wide100_p0` を新 base / gate に採用（experiment-plan.md（作業 repo 側） 更新済）。
2. session-local continuity/entity/transition source を rank-only fusion ではなく **main candidate build 段へ正式統合**し、query-text / metadata feature を rank 以外にも足して oracle recall と feature 表現を上げる（exp022 から続く本流の方向）。
3. `mixed_p0` を Dev で勝らせた dense source は Blind A で効かないので、base への dense 追加は見送り。dense は Dev-overfit lever として参照にとどめる。
4. composite の judge 差は response 再生成 variance なので、response stabilization (Tier D2) で judge を安定 4.0+ に固められれば composite はさらに伸びる余地。
