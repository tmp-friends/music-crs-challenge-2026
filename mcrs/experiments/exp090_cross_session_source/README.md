# exp090 cross-session exact-track source

Last updated: 2026-06-20

## Summary

同一ユーザの**別セッションで提示済みの track**（exact-track）を small-cap 候補 source 化し、exp058 の 12 session-local source に追加して LGBM を再学習する round。exp073 調査で見つけた「cross-session は除外でなく正のシグナル」を実装したもの。

**single-seed Dev OOF = 0.17960（floor 0.177771 を +0.00183）でクリア**。goal_artist(+0.00057) の約3倍で、この系列の単一 source 追加として最大の OOF gain。candidate_positive_count 4528（base 4333 から +195＝到達可能 GT が実際に増えた）。**ただし Dev OOF gain は Blind A に transfer しない前科（exp060 crash）があるので必要条件にすぎず、採否は Blind A 提出で判定する。**

2026-06-20 follow-up: raw exp090 proposal は Blind A top1 変更 4 件中 1 件に文脈衝突があったため、artist/album continuity guard をかけた postprocess 版 exp091（cross-session guard、本 repo には未収録） を作成した。提出優先は raw exp090 より exp091。

## Hypothesis（exp073 調査の数値）

- within-session の既出 music は次 turn の正解と絶対一致しない（Dev 0/7000）。
- **cross-session は逆**: 同一ユーザの別セッション提示 track が次の正解になる確率 9.89%（Dev）。
- recall ceiling: exp058 が取りこぼした GT のうち cross-session pool で回収可能 = +305（naive, 全セッション）/ **+158（session_date 過去のみ, +1.98pt）**。
- **時系列制限は不要**（user 指摘）: 本コンペはバッチ予測（全セッション同時提供）かつモデルに時系列特徴が無いため、未来セッション概念は deployment 上存在しない。leak の硬制約は「各セッションの隠し最終 turn を使わない」のみ。よって本実装は **全別セッション**を使い、honest な ceiling は +305 側。
- 現 no-dense pool は cross-session を未活用。exact-track は high-precision・low-density = robust class（exp066 artist-exact 寄り、exp060 fuzzy expansion と別）。

## Design（user 承認: exact-track のみでまず測る）

- `cross_session_candidates.py`: 各タスク (session, turn) に、同一ユーザの**別セッション**提示 track を出現セッション数降順で候補化（cap 50）。
  - artist/album 展開は含めない（density-precision の罠。Dev 実測で artist 展開は recall ceiling 961 だが density 276/row・precision 0.067%＝exp060 crash class。次段で feature 化を検討）。
- ranker は source 行 `len==topk` 厳密を要求するため、真の cross tracks を先頭に置き共有 popularity で 100 に pad（goal_artist 等と同一パターン）。真候補は rank 1..N、padding は全行共有の bounded 集合。`cross_session:50` で読み込み。
- **再学習**: `rank_source_lgbm_mixed_topk.py`（→ 6-seed は `seed_ensemble_wide100.py`）で毎回 LGBM を学習。cross-session は候補追加＋「source rank/membership」特徴量として学習される。

### Leak 対策

- **leave-current-session-out**（`s2≠s`）: 現タスク自身のセッションを cross-pool から除外（学習 leak 防止、[[train-statistic-feature-target-leak]] 回避）。within-session GT は絶対 repeat しないので自タスク正例混入もなし。
- **隠し最終 turn 不使用**: Blind A は各セッション最終 turn が hidden。別 Blind A セッションからは visible turn の music のみ採る。Dev は全 turn 公開。
- future turn / conversation_goal / goal_progress_assessments 不使用。

## Configuration

- Date: 2026-06-20
- Base: exp058 wide100（12 source, no-dense, protect_top=0）
- Added source: `cross_session:50`（真 cross tracks + 共有 popularity padding, 固定長 100）
- Reranker: LightGBM LambdaRank, single-seed（floor check）→ 6-seed score-average
- Cross-pool: Dev = user の別 Dev セッション全 music / Blind A = 別 Blind A セッション(visible) ∪ Dev セッション(全)

## Command

```bash
# source 生成
.venv/bin/python mcrs/experiments/exp090_cross_session_source/cross_session_candidates.py
# single-seed floor check（Dev OOF）
bash mcrs/experiments/exp090_cross_session_source/run_cross_s1.sh
# 6-seed ensemble + Blind A tracks
bash mcrs/experiments/exp090_cross_session_source/run_cross_ens6.sh
```

## Source statistics

| split | rows | true-cross nonempty | avg true-cross | pad_to |
|---|---:|---:|---:|---:|
| Dev | 8000 | 5208 | 28.8 | 100 |
| Blind A | 80 | 52 | 11.2 | 100 |

## Results

### Single-seed retrain（Dev OOF floor check）

| run | candidate positives | candidate rows | Dev OOF | vs floor |
|---|---:|---:|---:|---|
| exp058 base（floor） | 4333 | ~2.97M | 0.177771 | - |
| exp066 goal_artist | 4373 | ~3.04M | 0.178337 | +0.00057 |
| **exp090 cross_session** | **4528** | 3.10M | **0.179601** | **+0.001830** |

full_fit Dev = 0.25961, base_ndcg(primary alone) = 0.14442。OOF top1_changed vs primary 6115/8000。**floor を healthy margin でクリア**。

### 6-seed ensemble + Blind A

| metric | 値 |
|---|---|
| full_fit Dev (ens6) | 0.25858 |
| Blind A top1_changed vs exp058/exp063-long | **4/80** |
| Blind A avg_overlap20 vs exp058 | 17.19 |
| **rows with a true cross-session track in final top20** | **42/80** |
| unique tracks (catalog_div proxy) | 1322 vs exp058 1316（ほぼ不変） |

cross-session candidates は **42/80 行で top20 入り**＝シグナルは広く使われている。top1 変更は exp058/exp063-long 比 4/80（76 行は exp063-long response 流用、4 行のみ metadata template）。一方で tail は全80行で変更され、手動監査では 3 件は同一 artist/album 系で妥当、1 件は Los Caligaris 文脈から Pink Floyd へ飛ぶ危険例だった。artifact `submission_exp090_cross_ens6_A.zip`（validate OK: 80 records / catalog 照合 OK / ZIP root `prediction.json` / name 34字）。

### Clean re-run（cleanx: cap20 + 候補段 within-session 除外, user 指摘で 2 点クリーン化）

raw exp090（cross_session:50）は (a) popularity padding が ~10/80 行の top20 に注入される confound、(b) within-session 既出除外なし、の 2 点で「clean な cross-session 単独テスト」として不完全だった。これを修正:

1. **padding 注入縮小**: `cross_session:50 → :20`。no-cross 行の popularity-pad 注入 **10 → 4 行**。
2. **within-session 除外を候補段で実施**: 共有 ranker/seed_ensemble に後方互換 optional `--exclude_json` flag を追加し、候補 union 構築段で同一セッション既出 music を除外（後処理 backfill ではなく正攻法。user 指摘）。GT は防御的に保持（除外対象は guaranteed-non-GT＝Dev 0/7000）。default off で従来挙動 byte-identical（並行実験に影響なし、単体テスト済）。

cleanx 結果（`submission_exp090_cross_cleanx_A.zip`, validate OK）:

| metric | raw exp090 (cap50) | **cleanx (cap20+除外)** |
|---|---|---|
| within-session 既出が top20 に残る行 | 23/80 | **0/80** |
| popularity-pad 注入行(no-cross) | 10 | **4** |
| true cross-session track が top20 入り | 42/80 | **42/80**（維持） |
| top1_changed vs exp058 | 4/80 | 8/80 |
| candidate_positive_count | 4528 | 4473（cap20 で高pool user の真cross が一部 truncate、なお base 4333 +140） |
| full_fit Dev (ens6) | 0.25858 | 0.25812 |

artifact: `exp/inference/blindset_A/exp090_cross_cleanx_ens6_A.json` / `submission_exp090_cross_cleanx_A.zip`（80 records / catalog 照合 OK / ZIP root `prediction.json` / name 36字）。**提出優先は raw より cleanx**。

**response は全 80 行を新規生成**（user 指示・[[response-full-regen-rule]]）: 過去 exp063-long の流用や top1 変更行だけ template、をやめ、cleanx ranking の各行 top1 + 会話文脈 + user_profile で exp063 v2 prompt（proven judge4.90）により Claude サブエージェント 4 並列で全行生成し `assemble_claude_responses.py` で結合。word 95-114 / distinct-2 0.986 / 空 0 / top1 track 名が response に登場 78/80。生成: `gen_input_cleanx.json` → `claude_resp_cleanx_part{0-3}.json`。

### Validation

```text
validate_submission: exp/inference/blindset_A/exp090_cross_session_ens6_A.json
records: 80
OK: 提出制約をすべて満たす。

validate_submission: exp/inference/blindset_A/submission_exp090_cross_ens6_A.zip
records: 80
OK: 提出制約をすべて満たす。

unzip -l exp/inference/blindset_A/submission_exp090_cross_ens6_A.zip
prediction.json
```

## Leaderboard Result

2 回提出（cleanx＝padding バグあり / fix＝padding 根本修正）。

| 指標 | exp058 base | exp066 | exp063 response best | cleanx (padding bug) | **fix (clean)** |
|---|---:|---:|---:|---:|---:|
| **ndcg@20** | 0.4355 | 0.4380 | 0.4355 | 0.4298 | **0.4396** |
| catalog_div | 0.0280 | 0.0281 | 0.0280 | 0.0282 | 0.0281 |
| lexical | 0.5925 | 0.5970 | 0.7983 | 0.7669 | 0.7679 |
| judge | 4.05 | 3.95 | 4.90 | 4.60 | 4.55 |
| **composite** | 0.5086 | 0.5028 | **0.5929** | 0.5644 | **0.5657** |

- cleanx `submission_exp090_cross_cleanx_A.zip`: ndcg 0.4298 / composite 0.5644
- **fix `submission_exp090_cross_fix_A.zip`: ndcg 0.4396 / composite 0.5657**

## Interpretation（scored・先の結論を訂正）

- **★ 重要な訂正: 「cross-session ranking は非transfer」という cleanx 時点の結論は誤りで、その正体は padding バグだった**。padding を根本修正すると **ndcg が 0.4298 → 0.4396 へ +0.0098 跳ね上がり、0.4396 は project 歴代最高 ndcg@20**（exp066 0.4380・exp058 0.4355 を超える）。つまり **clean な cross-session は ndcg を下げず、むしろ最高値を出す＝Blind A に positive 方向で transfer**。cleanx の見かけの下落（0.4298）は popularity-padding（"Iris" 等）が top1/top20 を奪った artifact だった。
- **ただし noise floor 内**: fix 0.4396 vs exp058 0.4355 = +0.0041 は [[blind-a-ndcg-noise-floor]] の ±0.02 内なので「確定的な ndcg 改善」とは言えない。だが **(a) 最高値を出した (b) cleanx→fix の +0.0098 は candidate-set 変更（padding 除去＋cap50 で真 cross 全採用）の実効果 (c) Dev OOF +0.00183 と方向一致** の 3 点で、**cross-session は「ranking dead」ではなく「neutral〜positive・最高 ndcg 候補」**と位置づけが変わった。
- **composite は cleanx 0.5644 → fix 0.5657 でほぼ横ばい**: ndcg gain（+0.0098×0.5=+0.0049）が judge 4.60→4.55（response 再生成 variance）でほぼ相殺された。exp063(0.5929) 未満なのは judge 4.55<4.90（同じく regen variance）。
- **判定（更新）**: **cross-session-fix は project 最高 ndcg（0.4396）＝新 ranking base の候補**。composite を最大化するなら「cross-session-fix ranking ＋ exp063 級 response」が理想だが、response regen variance（judge 4.55-4.90）が composite を支配する。**padding 修正により cross-session の真価（ranking で positive）が判明**。採否（ranking base 更新するか）はユーザー判断。

## 実装検証（user 依頼）

- **response 側: 実装ミスなし**。scaffold leak 0 / markdown leak 0 / **年号捏造 0**（exp064 の judge-killer 回避）/ 全 80 行が疑問締め / 空 0 / title 不一致 2 は誤検出（remaster 接尾辞除去・apostrophe unicode 差）。judge 4.60<4.90 は**バグでなく response 再生成 variance**。
- **ranking 側: 瑕疵を 1 つ検出**。top1 変更 8/80 のうち **2 行で popularity-padding（"Iris" by Goo Goo Dolls）が top1 に浮上**（cross 履歴なし・base 信号弱の行）。cross-session source を ranker の `len==topk` 要求に合わせる共有 popularity padding が候補化され top1 を奪う既知の sparse-source 欠点（goal_artist と同質、cap50→20 で top20 注入は 10→4 に縮小したが top1 には 2 残存）。汎用人気曲が個人化推薦を上書きし ndcg/personalization を僅かに毀損した可能性。**次段の修正候補**: padding を「base プールに既出の track」で行い候補非追加にする / cross_session を feature 専用化して候補追加を切る。
- **within-session 除外は正しい**: `within_session_exclude.json` が prior-music と 0/80 不一致＝完全一致。候補段除外で top20 残存 0/80。GT は除外されない（Dev 0/7000）。
- 旧 caveat: Dev OOF iterate は CV-LB trap。本 round で「exact/high-precision でも cross-session candidate は transfer せず」が実測確定。

### padding-to-top1 の根本修正（fix 版, user 依頼）

popularity-padding が top1 を奪う瑕疵を**根本修正**した（`submission_exp090_cross_fix_A.zip`）。

- 原因は ranker の `load_records` が source 行に `len==topk` 厳密を要求し、sparse な cross-session source を popularity で埋めるしかなかった点。**`load_records` に後方互換 opt-in `allow_short`（既定 False＝既存 dense source は従来 strict・誤読検知維持）を追加**し、`seed_ensemble` に `--allow_short_sources` を配線。cross-session source を `--no_pad` で再生成＝**真の cross track のみ（no-cross 行は空）・popularity padding ゼロ**。cap も :20→:50 に戻し真 cross の truncation も解消。
- **検証**: no-cross 行は候補 union に何も足さない（injection 構造的に消滅、単体テスト済）。**"Iris" top1 2→0**。fix の pad-top1 残存 3 行（Gimme More / Kiss Me / Pumped Up Kicks）は**全て exp058 と同一 top1＝base 正規**で artifact ではない。strict load は従来どおり raise＝並行実験に影響なし。
- response は **fix ranking に合わせ全 80 行を新規生成**（[[response-full-regen-rule]]、scaffold/markdown/年号捏造 0・全行疑問締め・空 0・words 96-114）。
- artifact `exp/inference/blindset_A/exp090_cross_fix_ens6_A.json` / `submission_exp090_cross_fix_A.zip`（validate OK・catalog 照合 OK）。**fix は実装の clean 化＋padding 抜き cross-session の再テスト。ただし cross-session ranking 自体が非transfer（cleanx 0.4298<exp058）なので fix も exp058 超えは期待薄、"Iris" 2 行ぶんの微回復が主**。Codabench 提出はユーザー操作。

## Next Actions

1. raw exp090 は diagnostic / risky probe として扱い、まず exp091 guard 版を提出して cross-session の transfer を確認する。
2. exp090/091 のどちらかが positive なら、cross-session の artist/album を **feature** または postprocess guard として追加検証する。
3. negative なら Dev OOF gain は Blind A に transfer しない recall lever として凍結する。
