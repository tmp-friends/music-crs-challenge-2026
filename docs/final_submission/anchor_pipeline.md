# BM25 アンカー（primary）の中身

最終提出パイプラインの土台になる第一段階の候補生成＋順位付け。論文で "BM25-based anchor"、
コード上では 2 段目 reranker への `--primary`（特徴量名 `primary_*`）と呼ばれているもの。

実体は exp015 の source profile B pipeline
（config: [exp015_B_ll_t500_B.yaml](../../mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_B_ll_t500_B.yaml)、
Blind B 推論: `python run_inference_blindset.py --tid exp015_B_ll_t500_B --eval_dataset blindset_B`）。

## 8 つの lexical signal → RRF union → 1 段目 LightGBM

`enabled_sources` の 8 source を RRF（`union_mode: rrf`）で統合し、最大 500 候補を作る:

| source | 内容 |
|---|---|
| `bm25_recent_turns` | 直近対話ターンを query にした BM25（BM25S、corpus = track/artist/album/release） |
| `bm25_full_context` | 対話全体を query にした BM25 |
| `metadata_bm25_full_context` | metadata corpus（track_name/artist_name/album_name/tag_list）への BM25、対話全体 query |
| `metadata_bm25_current_user_turn` | 同上、現在 user turn のみ query |
| `exact_track_current` | 現在発話と track_name の完全一致 |
| `exact_artist_current` | 現在発話と artist_name の完全一致 |
| `exact_album_current` | 現在発話と album_name の完全一致 |
| `exact_tag_current` | 現在発話と tag の完全一致 |

exact match を field 別に分離しているのは、broad な tag match が「明示的に名指しされた
track/artist」の強い signal を圧倒しないようにするため（exp015 の candidate rule ablation の主眼）。

この union 500 候補に対し、**1 段目 LightGBM（binary logloss、
`lgbm_B_ll_top500.txt`）** が BM25 スコア・exact 一致フラグ等を特徴量に順位を学習し、
anchor ranking を出力する。

## 「アンカー」と呼ぶ理由（2 段目での扱い）

- anchor の候補は捨てられず**そのまま候補 union の先頭に入る**
  （`build_features` で `primary[key]` が最初に union へ入る）
- anchor 上での順位が 73 次元中 5 次元（`primary_inv_rank` / `primary_is_top1` /
  `primary_in_top5` / `primary_in_top10` / `primary_in_top20`）として特徴量化される
- その上に補助 12 source が**追加で** union され、全体を 2 段目 LambdaRank が並べ替える

つまり「土台として常に存在し、他 source はそれを補強・追加するだけで置き換えない」という
位置づけ。lexical に確信度の高いマッチ（名指しされた track 等）を保持しつつ、
follow-up turn（「もっと似たの」等、lexical 手がかりが薄い turn）を構造 source が救う設計。

## 性能上の位置づけ

| 段階 | Blind A nDCG@20 |
|---|---:|
| BM25 素朴 baseline（公式） | 0.1938 |
| rule-aware アンカー単体（exp015 B 系） | 0.3869 |
| + 12 source + LambdaRank 6-seed | 0.4355–0.4396 |

アンカー単体で公式 baseline の約 2 倍に達しており、最終システムの nDCG の大半はこの
lexical + exact-match 段が支えている。補助 source + reranker はその上に +0.05 前後を積む。
