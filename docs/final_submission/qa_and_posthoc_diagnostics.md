# 最終提出の実装 Q&A と事後診断実験

Last updated: 2026-07-20（本ドキュメントは 2026-07-16〜20 の実装 Q&A セッションの内容を整理したもの。
事後診断実験の artifacts は `exp/diag_*` 参照。全体像は [architecture.md](architecture.md)、
第 1 段の詳細は [anchor_pipeline.md](anchor_pipeline.md) を参照）

## 1. 最終提出の概要

- **Final 選択 submission**: `submission_exp116_lgbm6_nocs_B.zip`（Codabench id 816456、2026-06-28 提出）
- **公式最終結果**: composite **0.4998** / nDCG@20 **0.3528** / catalog_diversity 0.0275 /
  lexical_diversity 0.7315 / LLM judge 4.30 — **40 チーム中 16 位**
- **構成の要約**: LightGBM 6-seed 単独 reranker / 12 candidate source（cross_session 完全 drop）/
  rank-only 特徴量 / Claude backend による応答 80 行全新規生成
- **Blind B 3 提出枠の使い方**:

| 提出 | 構成 | Codabench score |
|---|---|---:|
| `lxct6_B`（primary 提出） | 4-family（LGBM+XGB+CatBoost+TabM）各 6-seed ensemble | （per-metric 記録なし） |
| `lgbm6_B` | LightGBM 6-seed 単独へ縮退（nDCG hedge） | 0.47 |
| **`lgbm6_nocs_B`（final 選択）** | 上記 + cross_session source 完全 drop | **0.50** |

Blind B は cold user 40/80（user_id / profile 空）+ メタ削除（goal/thought/session_date）の分布シフト
セットであり、「最もシンプルな構成へ縮退した hedge」が最高スコアとなった。

## 2. パイプライン実装（4 段構成）

### 2.1 候補生成

**Stage-1 anchor（exp015 profile B）**: 語彙 8 source を RRF（k=60、score=Σ1/(60+rank)）で union し
**候補プール top500** を作り、その上の学習済み LightGBM で並べ替えて **top-20 の ranked list** を出力する。
（注意: 実験名の「top500」は候補プール深さで、出力リスト長は 20。詳細 §4 Q5）

| source | index する catalog field | query | raw→cap |
|---|---|---|---:|
| bm25_recent_turns | track/artist/album/release_date | 直近 2 turn + 現在 user 発話 | 1000→250 |
| bm25_full_context | 同上 | 全履歴 + 現在 user 発話 | 1000→250 |
| metadata_bm25_full_context | track/artist/album/**tag_list** | 全履歴 + 現在 user 発話 | 1500→400 |
| metadata_bm25_current_user_turn | 同上 | 現在 user 発話のみ | 1000→200 |
| exact_track_current | track_name 完全一致 | 現在 user 発話 | 500→100 |
| exact_artist_current | artist_name 完全一致 | 同上 | 500→150 |
| exact_album_current | album_name 完全一致 | 同上 | 500→100 |
| exact_tag_current | tag_list 完全一致 | 同上 | 500→80 |

- exact 系は catalog field 値を正規化 n-gram index 化し、発話 n-gram（長い一致優先）と照合。
  マッチ集合内は「一致長 → field priority（track>artist>album>tag）→ popularity 降順」。
- 履歴中の music turn は track_id を metadata 文字列に展開してから query に連結（leak-safe）。

**Stage-2 の追加候補 source（各 top100、12 種）**:

| 系統 | source | 内容 |
|---|---|---|
| continuity ×4 (exp021) | cont_aat / cont_aa / cont_album / cont_artist | session 内既出 music の artist/album/tag から catalog 展開 |
| query-neighbor memory ×3 (exp021) | qnm_rp / qnm_hist / qnm_cur | train task の BM25 近傍から正解 label / artist / tag 展開を伝播 |
| entity ×4 (exp023) | ent_cur / ent_tight / ent_r2 / ent_r4 | 発話中の metadata entity phrase の catalog 照合 |
| transition ×1 (exp024) | trans_r1 | train の「直前 music → 次の正解 track」遷移統計 |

- **within-session exclude**: 同一 session 既出 music を union から除外（Blind B 用に再生成）。
- **cross_session（exp090）は final 版で完全 drop**（cold 40/80 + warm sparse の分布シフト hedge。
  実測で drop 版 0.50 > 込み版 0.47 と正解）。
- union = primary top20 + 12×top100 ≈ **372 行/group、Dev 全体 2,949,717 行 / pool 正例 4,333（recall 0.542）**。
- 候補は常に全 catalog 47,071 tracks（`track_split_types=all_tracks`）。

### 2.2 特徴量

**Stage-1（61 次元 `f_*`）**: source 別 rank/RRF/hit、union 集約（candidate_rank, rrf_score 等）、
exact match フラグ（f_artist_name_exact_full 等）、session 履歴一致数（same_artist/same_tag/
history_music_count）、track metadata（popularity/tag_count/release_year）、query 統計、intent フラグ。
embedding 系 5 列は `include_embedding_features: false` で**全行ゼロ（未使用）**。
importance 上位は `f_artist_name_exact_full`（発話中の artist 名指し）。

**Stage-2（「5 + source 数×5 + 8」次元 = 12 source で 73 次元、rank-only）**:

- primary 系 5: inv_rank / is_top1 / in_top5 / in_top10 / in_top20
- 各 source 5: inv_rank / in_top5 / in_top20 / rank_clipped / rank_pct
- 集約 + turn 系 8: source_hit_rate / hit_count / best_inv_rank / RRF sum(k60,k200) /
  turn_number / is_early_turn / is_followup_turn

content・embedding・train 統計特徴量は**意図的に排除**（Dev で効いて Blind で反転する
CV-LB 逆相関の実測: exp015e / exp060 / train 統計 leak）。副次効果として user_id 非依存 =
Blind B cold user でもそのまま機能。

### 2.3 Reranking

- **Stage-1 学習**: train split **121,592 task**（= 15,199 session × 8 music turn）。GT が候補 500 に
  入らない 35% の group を除外し 79,245 group。負例は「rank 上位 80 hard negative + random」で
  120 行/group にサンプリング（9,509,400 行、正例率 0.83%）。objective = **binary logloss**
  （pointwise、exp014d 比較で選択）。early stopping は Dev（best_iter 141）。
  Dev all-tasks nDCG@20 = 0.1444（素の RRF union top-20 recall 0.259 → LGBM 後 0.304）。
- **Stage-2 学習**: Development split **8,000 task**（= 1,000 session × 8 music turn）の候補 union 上で
  **LightGBM LambdaRank**。label は「候補 == その turn の GT track（turn ごとに 1 曲）」の二値。
  負例サンプリングなし・全候補行使用。params: `num_boost_round=450, num_leaves=31,
  min_data_in_leaf=80, lambda_l2=8.0, learning_rate=0.03, deterministic=True`。
  **6 seed（20260616–20260621）の生スコア平均**（Blind 80 行は seed 非決定性だけで nDCG ±0.02
  揺れるため variance 低減）。full-fit model を Blind B 80 行に適用、`protect_top=0` で全再ランク、
  top-20 出力。
- **評価 ruler**: Dev は leave-session-out 5-fold OOF（8,000 task）で測る（full-fit 値は in-sample 楽観）。
  最終提出構成の Dev OOF = **0.17803**（事後計測）。
- 4-family ensemble（lxct6）は Dev OOF で LGBM 単独 0.18173 > blend 0.17944 のため、最終選択は
  LGBM 単独縮退版。

### 2.4 応答文生成

- `generate_responses_v2.py --backend claude --model claude-opus-4-8`
  （claude-agent-sdk 経由、`mcrs/lm_modules/claude_agent_backend.py`、batch 6）。
- **最終 ranking の top1 に合わせて 80 行すべて新規生成**（過去 response の流用・splice 禁止ルール）。
- プロンプト設計 lever（exp063、Blind A judge 4.90 の書き方）: ①謝罪排除 ②top1 の tag_list
  （genre/mood/style）注入 grounding ③user_profile + 対話内嗜好への personalization（cold user は
  対話内のみ）④決定的な書き出しスタイル rotation（Distinct-2 対策）⑤長め・熱量・疑問文締め
  （平均 ~114 words）。
- **Leak-safe**: current turn までの対話 + 入力側 user_profile + catalog metadata のみ。
  conversation_goal / goal_progress / thought 不使用。
- 提出前に `validate_submission.py` で機械チェック（80 行 / 全行 top-20 / catalog 照合 /
  ZIP root=prediction.json / ファイル名 ≤63 字）。

## 3. データ split の整理

| split | 規模 | label | 役割 |
|---|---|---|---|
| Train | 15,199 session × 8 turn = 121,592 task | あり | stage-1 学習 + QNM/transition memory 構築元 |
| Development（Dev、HF split 名は "test"） | 1,000 session × 8 turn = 8,000 task | あり | stage-2 学習 + OOF ruler + 公式 evaluator 対象 |
| Blind A | 80 行（最終 user turn のみ） | hidden | interim LB |
| Blind B | 80 行（cold 40/80） | hidden | final LB |

- **label は music turn 単位で 1 件**（session 単位ではない）。Blind は「最終 turn の 1 行だけ提出」
  という仕様のため session あたり 1 行に見えるだけ。
- **なぜ stage-2 は Dev で学習するか**: QNM/transition が train から構築された memory であるため、
  train で fusion を学習すると self-leak（実測: Dev 0.175→0.053 崩壊）。ラベルのある split は
  Train と Dev のみなので、消去法で fusion は Dev。二段構成は「2 つのラベル付き split を
  それぞれ leak なく使える場所へ割り当てて使い切る」設計。また Dev への primary 予測は
  out-of-sample なので、stage-2 が見る primary 特徴量の分布が推論時（Blind）と一致する
  （train 上で学習すると primary が in-sample 楽観 — train recall@20 0.835 vs 未見 0.571 — で過信を学ぶ）。

## 4. 設計 Q&A（本セッションで議論した質問と結論）

**Q1. Primary と追加候補 source はなぜ分けている？**
primary は全 task を必ずカバーする学習済み anchor（BM25 系なのでどんな行でも候補を返せる。
`build_features` は primary 欠落 key で ValueError）、追加 source は条件付きで発火する精度シグナル
（cold/単一 turn では 12 source 全滅があり得る）。特徴量設計も非対称（primary は専用 5 特徴量、
source は各 5 特徴量で信頼度を学習で判定）。報告の `base_ndcg` / `blind_change_stats_vs_primary` の
基準線でもある。

**Q2. Primary は LGBM で何をどう学習している？**
§2.3 参照。8 source RRF union top500 → binary logloss で「候補 = その turn の GT か」を分類、
上位 20 件を出力。

**Q3. 学習なしで RRF union top500 を stage-2 に直接渡さない理由は？**
① stage-2 は Dev 8,000 group でしか学習できず（leak 制約）、語彙証拠の混ぜ方を学ぶ 121,592 task の
教師信号を捨てることになる。② 素の RRF は学習済み primary より頭が弱い（top-20 recall 0.259 vs
0.304）。③ 500 深の tail は noise で precision gate を壊す前科（exp060/061）。
→ ただし事後実験（§5.3）で「61 特徴量を伴えば」この設計が Dev では勝つことが判明。

**Q4. 61 特徴量 + 追加 source で 1 つの LGBM を学習した方が良いのでは？**
事前予測は「exp101 (+0.007) / exp015d (0 変化) から noise 級」だったが、**事後実験で Dev OOF
+0.0077 の全 fold 勝ちが実測された**（§5.3）。二段蒸留は lossy だった。Blind 転移は検証不能。

**Q5. primary の top-20 は少なすぎないか？根拠は？**
実験的根拠なし（submission 形式 artifact の再利用 + `ensemble_rerank.py` が primary を出力用
`--topk`=20 で切り詰める実装だった）。recall 会計では top500 化で pool recall 0.542→0.607 の
+6.5pt が「捨てられて」いたが、事後実験（§5.1）で **nDCG には全く変換されない**ことを実証
（漏れた GT は全 source 不発火・primary rank median ~200 の tail で、rank-only 特徴量に識別信号なし）。

**Q6. Dev を学習に使いたくて今の構成？ / train のみで学習できないか？**
因果は逆（memory 系 source を使う帰結として fusion の学習先が Dev になった）。train-only 化は
LOO source 再構築 + k-fold primary を実装すれば原理的に可能（教科書的 stacking 衛生はむしろそちら）
だが未実施。label 数の論点は §5.2 の学習曲線で「まだ微増が残る（完全飽和ではない）」と判明。

**Q7. session ごとに 1 label とする学習は試した？**
文字通りには未実施。近傍: exp040 turn×qlen reweight 学習 = +0.0007（noise）、exp112 = Dev↔Blind の
差は turn 位置の covariate shift ではない。学習曲線から 1,000 task への削減は −0.01 級の損失見込みで
負けが濃厚。

## 5. 事後診断実験（2026-07-16〜20、コンペ終了後・論文用）

すべて最終提出と同一の候補構成・特徴量・fold split（5-fold leave-session-out, seed 20260616）・
LightGBM params・OOF 手順（1 seed / 240 rounds）で計測。artifacts: `exp/diag_primary_depth/`,
`exp/diag_label_curve/`, `exp/diag_no_primary/`, `exp/diag_embed_features/`。

### 5.1 primary 深さ（top20 → 100 → 500）

| run | union 行数 | pool 正例（recall） | Dev OOF |
|---|---:|---:|---:|
| p20（最終提出） | 2,949,717 | 4,333 (0.542) | **0.17803** |
| p100 | 3,311,335 | 4,597 (0.575) | 0.17839 (+0.0004) |
| p500 | 6,088,988 | 4,896 (0.612) | 0.17830 (+0.0003) |

**pool recall +7pt が nDCG に変換されない**（oracle recall ≠ reachable nDCG の直接実証）。
偶然だった top-20 primary は実測で無コストと事後正当化された。

### 5.2 label 学習曲線（学習側 session を nested 間引き）

| 学習側 | 正例/fold | Dev OOF |
|---:|---:|---:|
| 25%（200 session） | 869 | 0.16972 |
| 50%（400） | 1,731 | 0.17479 |
| 75%（600） | 2,588 | 0.17629 |
| 100%（800） | 3,466 | 0.17803 |

100% でも +0.0017（75→100）の伸びが残り**完全飽和ではない**。ただし doubling あたり
+0.0051→+0.0032 と逓減。log 外挿で train-only 化（~15 倍、要 LOO source + k-fold primary）に
+0.005〜0.01 OOF の余地の示唆。

### 5.3 primary 廃止（user 提案の flattened 構成）

候補 = lexical RRF union top500 + behavioral 12 source（§5.1 p500 と同一プール）。

| 構成 | Dev OOF | 対最終提出 |
|---|---:|---:|
| p20 = 最終提出（学習済み primary + rank 73 次元） | 0.17803 | — |
| A: primary 廃止・RRF anchor・rank-only 73 次元 | 0.17949 | +0.0015 |
| **B: A + 61 語彙特徴量 join（134 次元）** | **0.18577** | **+0.0077（5 fold 全勝）** |

- A ≈ baseline: 学習済み primary の蒸留順位を捨てるコストはほぼゼロ。
- B − A = +0.0063: **生の 61 特徴量には rank 特徴量に冗長でない順序付け信号が残っていた**
  （= 二段蒸留は lossy）。pool 外の behavioral 候補は NaN（LightGBM native missing、
  join coverage ~65%）。
- §5.1 との整合: 深い pool は「特徴量が伴って初めて」nDCG に変換される。
- **限界**: Blind 提出は不可能で転移は検証不能。Dev→Blind 換算（×~2.35）+0.018 は Blind noise
  floor ±0.02 と同オーダー。rich feature の Blind 反転前科（exp060/exp015e）は embedding/content 系で、
  語彙 61 次元が同様に反転するかは未知。**「Dev では勝ち・Blind では未知」が正確な結論**。

### 5.4 embedding 特徴量の追加（variant C）

B の上に user/item embedding 特徴量 13 次元を追加: track 6 空間（cf-bpr / audio CLAP /
image SigLIP2 / attributes / lyrics / metadata Qwen3）で cos(候補, 履歴 music) の mean/max = 12 次元
+ cos(user cf-bpr, 候補 cf-bpr) = 1 次元。欠損（履歴なし / cold user / embedding なし）は NaN。

| 構成 | Dev OOF | 対 B | 対最終提出 |
|---|---:|---:|---:|
| B（rerun、同一 run 内統制） | 0.18577 | — | +0.0077 |
| C_cf（+CF 3 次元） | 0.18821 | +0.0024 | +0.0102 |
| C_content（+content 10 次元） | 0.18724 | +0.0015 | +0.0092 |
| **C_all（+13 次元）** | **0.18947** | **+0.0037（5 fold 全勝）** | **+0.0114** |

- CF 系と content 系はそれぞれ単独でも正、合算でほぼ加算的。flattened 構成の累積で
  **最終提出比 +0.0114** まで Dev OOF が伸びた。
- B rerun が前回値と完全一致（決定性確認済み）。
- **重要な留保**: この特徴量 class（履歴/user CF cosine）は **exp015e で Blind A 反転
  （Dev best → 0.2988 vs base 0.3705）の直接の前科がある**。§5.3 の語彙 61 次元より
  Blind 転移リスクは明確に高く、「Dev では効く・Blind では反転前科あり・検証不能」が結論。
（補足: stage-1 parquet の embedding 列は全行ゼロ = 本番 pipeline を通じ lyrics / image / attributes
embedding は一度も特徴量化されていなかった）

### 5.5 サンプル 1 件のトレース（Blind B `ff76b679`、cold・単一 turn）

「abstract cover art の album を探している」という固有名なしクエリ（profile 空・履歴なし）。

- stage-1 primary: "colors" 等の語彙断片マッチ（True Colors ×2 など）で頭は noise。
- 12 source: ent×4 と qnm×3 は session 固有の実信号、**cont×4 + trans は履歴なしのため全 source
  同一の global popularity padding**（8 つの cold 単一 turn session で同一リストと確認）。
- reranker の選択: final#1 = primary p18 × entity 4 本一致（If I Had $1,000,000）、#2 = p6 × ent、
  #3 = padding 由来の偽の 8-source 合意で popularity 曲（Iris）、#7 以降は qnm 伝播の人気 alt-rock。
  primary top5 の語彙断片マッチは全 demote。
- 応答は top1 に整合しつつ検証不能な視覚情報を明示的に回避（"I can't speak to the album's cover
  art specifically..."）。
- 発見: cont/trans の popularity padding が「偽の cross-source 合意」を作る残存 wart
  （exp090 の no-pad 修正は cross_session のみだった）。

## 6. 論文への示唆

- 「Simple yet Strong」本文: rank-only 特徴量 + LightGBM seed-ensemble + 分布シフトに対する
  source 削減 hedge（cross_session drop が Blind B で +0.03）。
- 事後 ablation 節: §5.1（oracle recall ≠ realized nDCG）と §5.3（二段蒸留の損失 +0.008 を定量化、
  ただし blind 検証機会なし）は negative/positive results として一級の材料。
- 検証可能性の議論: Dev OOF 8,000 = 信頼 ruler、Blind 80 行 = ±0.02 noise floor、
  full-fit = in-sample 楽観、の三層を明示する。
