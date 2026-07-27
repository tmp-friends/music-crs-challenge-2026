# 補助 candidate source 4 種の実サンプル walkthrough

最終提出（`submission_exp116_lgbm6_nocs_B.zip`、12 source + LightGBM 6-seed LambdaRank）で使った
補助 candidate source 4 系統を、train split の実サンプル 1 session で具体的に説明する。

- 再現スクリプト: [EDA/20260720_candidate_sources_demo.py](../../EDA/20260720_candidate_sources_demo.py)
  （`.venv/bin/python EDA/20260720_candidate_sources_demo.py`。QNM は bm25s を token overlap で近似した簡易版）
- source 実装:
  - continuity: `mcrs/experiments/exp021_candidate_fusion/continuity_candidates.py`
  - QNM: `mcrs/experiments/exp021_candidate_fusion/query_neighbor_memory.py`
  - entity: `mcrs/experiments/exp023_entity_candidates/entity_candidates.py`
  - transition: `mcrs/experiments/exp024_transition_memory/transition_memory.py`
- 特徴量化と union: `mcrs/experiments/exp027_wide_source_lgbm/rank_source_lgbm_mixed_topk.py`
  （`build_features` / `feature_names`）

## デモに使う train サンプル

train row 0（`session_id=9c337a02-…`）。8 ターンの会話で、システムが Alesana を出し続ける session。

| turn | role | 内容（要約） |
|---|---|---|
| t1 | user | 「新しいアーティストを開拓したい。intense か dramatic なものある?」 |
| t1 | music | 「The Fiend」/ Alesana |
| t2 | user | 「Alesana は大好きだけど**新しい**アーティストが聴きたい」 |
| t2–t7 | music | 「A Forbidden Dance」→「The Temptress」→「Circle VII」→「Beyond The Sacred Glass」→「Welcome To The Vanity Faire」→「And Now For The Final Illusion」（すべて Alesana） |
| t3–t8 | user | 「*new* artists と何度も言っている」を段々強く繰り返す |
| **t8** | **music (GT)** | **「The Best Laid Plans Of Mice And Marionettes」/ Alesana** |

ポイント: ユーザーは「Alesana 以外」を懇願しているのに GT はまた Alesana。GT は hidden listening
pool から選ばれるため会話上の要望と乖離する（会話 EDA・exp073 で確認済みの「nDCG 追求とユーザー
満足が両立しない」構造）。この構造ゆえに、以下の「履歴をなぞる」系 source が強く効く。

予測タスクとしては t8 の music turn が GT、入力は t8 の user 発話までの対話履歴
（観測済み music 履歴 = Alesana 7 曲）。

## 1) Session-local continuity（cont_artist / cont_album / cont_aa / cont_aat、exp021）

- **入力**: セッション内で既に提示された music track（このデモでは Alesana 7 曲）。
- **処理**: その artist（/album/tag）を key に catalog 全体を引き、同 artist の未提示 track を
  popularity 順に候補化。within-session 既出は除外（既出 exact repeat は GT にならないことを
  Dev 分析で確認済み）。

デモ実行結果:

```text
既出 artist: ['alesana']
cont_artist 候補: 31 件（catalog の Alesana 38 曲 − within-session 既出 7 曲）
  - 「Apology」/ Alesana (Try This With Your Eyes Closed)
  - 「The Thespian」/ Alesana (The Thespian / The Emptiness)
  - 「Ambrosia」/ Alesana (On Frail Wings Of Vanity And Wax)
  - ...
→ GT を含むか: True（popularity 順で rank 28）
```

「ユーザーの発話は無視してでも直前の artist を続ける」だけで正解を拾える。4 source 中この系統が
最強である理由が分かる例（Dev OOF の bimodal 分解で continuity slice nDCG 0.39 vs cold 0.05）。

## 2) Query-neighbor memory（qnm_cur / qnm_rp / qnm_hist、exp021）

- **入力**: query view 3 種 — 現在 request（qnm_cur）/ 直近対話+profile（qnm_rp）/ 既出 music
  （qnm_hist）。
- **処理**: query を bm25s で **train split の全 task に照合**し、近傍 train task の GT track を
  候補として伝播（`top_neighbors=240` + artist/tag expansion）。**train label のみ index**、
  Dev/Blind label は index しない（leak-safe）。

デモ実行結果（t8 発話「This isn't working at all. I've asked multiple times… discover *new*
artists…」を query に、token overlap 近似）:

```text
[近傍 train task] session#3630
  発話: "This isn't working at all. I have asked you multiple times, very clearly,
         to play music by *different* pop punk and alternative ro..."
  → その task の GT を候補化: 「Paper Chase」/ The Academy Is...

[近傍 train task] session#2803
  発話: "This isn't working. I am NOT on an Asking Alexandria kick. I have repeatedly,
         explicitly asked for a *new band*..."
  → その task の GT を候補化: 「Situations」/ Escape the Fate

[近傍 train task] session#11093
  発話: "This is unacceptable. You have given me seven Korn tracks in a row..."
  → その task の GT を候補化: 「Reclaim My Place」/ Korn   ← また同 artist
```

「似た文句を言っている会話では、結局どんな track が GT だったか」という **label memory**。
この例では「同じバンドを聴かされ続けて怒っている人の GT は、結局そのバンドか近縁バンド」という
パターン自体を候補に変換している。

## 3) Entity candidates（ent_cur / ent_tight / ent_r2 / ent_r4、exp023）

- **入力**: 現在（variant により直近 2/4 ターン）の user 発話の**表層テキスト**。
- **処理**: catalog の track/artist/album 名と文字列照合し、ヒットした entity の track を候補化。

デモ実行結果:

```text
t8 発話 "...you keep playing Alesana..." から artist entity "alesana" を検出
→ catalog の Alesana 全 38 曲を候補化（GT を含む）
```

continuity と違い「会話履歴の music turn」ではなく「ユーザーが名前を口にしたか」だけを見るので、
**履歴が無い cold session でも発話に固有名詞さえあれば動く**。Blind B smoke test で cold の
load-bearing source が primary + entity だったのはこのため（`exp/smoke_blindB/REPORT.md`）。

## 4) Transition memory（trans_r1、exp024）

- **入力**: 直前に観測された music entity（デモでは t7「And Now For The Final Illusion」/ Alesana）。
- **処理**: train 全会話から「track X の直後に GT として来た track」の遷移統計を作り、候補化。

デモ実行結果:

```text
「And Now For The Final Illusion」の直後遷移: train 中 1 件
  → 「The Best Laid Plans Of Mice And Marionettes」/ Alesana   ← ★まさに GT

artist Alesana の直後遷移 48 件の上位:
  2 回: 「Circle VII: Sins Of The Lion」
  2 回: 「Welcome To The Vanity Faire」
  2 回: 「The Lover」 ...
```

アルバム曲順どおりの遷移がそのまま当たった例。カバレッジは薄い（standalone nDCG@20 0.0037）が、
当たるときはピンポイントで当たる tail source。

## reranker での合流

このサンプルでは GT が **cont_artist(rank 28)・ent_cur・trans_r1(rank 1)** の 3 source に同時に
出現する。2 段目の LightGBM LambdaRank は「どの source の何位に出たか」だけを特徴量化した
73 次元（anchor 5 + 12 source × 5 + 集約 8: `source_hit_count` / `source_best_inv_rank` /
`rrf_sum_k60` / `rrf_sum_k200` / turn 位置）で並べ替えるため、**複数 source が合意した track**
= GT を上位へ押し上げられる。BM25 anchor 単独では "new artists" という表層発話に引っ張られて
外しやすいところを、構造 source が救う典型例。
