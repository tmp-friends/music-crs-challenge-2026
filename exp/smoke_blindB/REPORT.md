# Blind B smoke test レポート (2026-06-23)

**目的**: bankable パイプライン(exp116 lxct6 / exp090-fix)が Blind B(cold 40/80・空 user_id・conversation_goal/profile/thought 欠落)で **crash せず、cold/single-turn が候補プールを silent に失わないか、全 80 件が top-20 を持つか** を実測で確認する(EDA で挙げた open TODO の検証)。

再現:
- primary backbone: `python run_inference_blindset.py --tid exp015_B_ll_t500_B --eval_dataset blindset_B --batch_size 8`(config: [exp015_B_ll_t500_B.yaml](../../mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_B_ll_t500_B.yaml))
- 候補 source 群: [run_blindB_smoke_sources.sh](run_blindB_smoke_sources.sh)
- 分析は本レポート末尾の手順(EDA inline script)。

## 結論(TL;DR)

✅ **crash なし**(primary 推論 + continuity/qnm/entity/cross_session の 4 generator すべて exit 0)。EDA で懸念した空 user_id の `id_to_profile` KeyError は **bankable 経路では発生しない**(guard 済み経路のみ通る、Explore のコード調査とも一致)。primary は 80/80 で valid な top-20 submission を生成(validate_submission OK・catalog 外参照 0・応答空 0)。**Blind B submission のビルドは可能**。

✅ **cold/single-turn も「実信号」の候補を得る(popularity collapse ではない)**。candidate content の across-session distinct-ratio で検証(§3): **primary(distinct 0.87–0.95)+ entity 4 variant(0.46–0.74, 共通∩=0)** が cold・single-turn 含め全 group で per-session に varied な候補を出す。これらが cold ユーザの load-bearing source。

⚠️ **ただし「15–16 source がカバー」の過半は実信号でなく shared popularity padding**(当初の coverage 計数だけでは過大評価だった)。continuity 7 variant と qnm 3 variant は低履歴ケースで共有 popularity 列に飽和する。特に **single-turn の continuity は 10 セッション全てで同一 100 件(distinct 0.10, 共通∩=100)= 純粋な padding**。ただしこれは構造的(single-turn は履歴ゼロ→continuity は定義上 pad、Blind A の single-turn 20 件でも同じ)で、ranker は rank feature で padded 列を弱信号として扱う。**Blind B 固有の破綻ではない**。

⚠️ **Blind B 固有の silent degradation = cross_session source のみ**。crash でなく品質劣化。Blind A で正だった cross_session が Blind B では機能不全(§4)。bankable の 13 source 中 1 つで submission は壊れないが、cold で noise・warm で sparse になり ranking 寄与は Blind A から劣化。**leak-safe な修正(空 user_id を no-pool 扱い)を本番ビルド前に入れるか、Blind B では drop するかを判断推奨**。

## 詳細

### 1. primary backbone(exp015 B)end-to-end

| 指標 | 値 |
|---|---|
| records | 80/80 |
| exactly top-20 | 80/80 |
| out-of-catalog track 参照 | 0 |
| 応答空 | 0 |
| validate_submission | OK |

group 別 top-20 健全性(popularity collapse なら distinct-ratio が落ちる):

| group | n | top20 distinct-ratio | uniq_top1 |
|---|---|---|---|
| single_turn | 10 | 0.950 | 10/10 |
| cold_multi | 32 | 0.909 | 31/32 |
| warm_multi | 38 | 0.871 | 38/38 |

→ cold/single-turn も高 diversity で popularity に潰れていない。cold 応答も生成される(中身は Qwen の apology 調だが、bankable response は exp063 Claude 経路なので smoke の対象外)。

### 2. 候補カバレッジ(80 final-turn ターゲット)

| source | カバレッジ |
|---|---|
| primary (exp015 B) | 80/80 |
| continuity ×7 variant | 各 80/80 |
| entity ×4 variant | 各 80/80 |
| qnm ×3 variant | 各 80/80 |
| **cross_session** | **58/80** |

各ターゲットあたりの平均カバー source 数: single_turn 15.9 / cold_multi 16.0 / warm_multi 15.4。**「primary 以下しか候補が無い」ターゲットは全 group で 0**。

ただし「100 件カバー」の多くは padding 充足で、実信号とは限らない。candidate content の **across-session distinct-ratio**(group 内で候補が session ごとに varied か / 共有 popularity 列に潰れているか)と **common∩**(group 内全 session に共通の track 数 = 共有 padding の署名)で実信号を判別:

| source | single_turn distinct (∩) | cold_multi distinct (∩) | warm_multi distinct (∩) | 判定 |
|---|---|---|---|---|
| primary_exp015B (top20) | 0.95 (0) | 0.91 (0) | 0.87 (0) | **実信号(全 group)** |
| entity ×4 variant | 0.72–0.74 (0) | 0.46–0.50 (0) | 0.42–0.46 (0) | **実信号(全 group)** |
| continuity ×7 variant | **0.10 (∩=100)** | 0.11–0.50 | 0.09–0.40 | single-turn は純 padding / multi は部分的 |
| qnm ×3 variant | 0.19–0.27 (∩=57–72) | 0.20–0.40 | 0.22–0.38 | 大半 padding(single-turn 特に) |
| cross_session | 0.14 | **0.03** | 0.66(但 2 件のみ) | §4 参照 |

→ **cold・single-turn の load-bearing な実信号は primary + entity**。continuity/qnm は低履歴で共有 popularity に飽和するが、これは single-turn=履歴ゼロという構造由来で Blind A でも同様(Blind A も single-turn 20 件)。ranker は rank feature で padded 列を弱信号として扱うため、これ自体は Blind B 固有の劣化ではない。

### 3. ⚠️ cross_session の Blind B 機能不全(silent degradation)

`--no_pad` で実候補数を計測(generator stdout: Blind 80 行, true-cross nonempty **58/80**, avg 35.9):

| group | n | mean 実候補 | #zero |
|---|---|---|---|
| single_turn | 10 | 40.6 | 1 |
| cold_multi | 32 | **50.0 (cap 上限)** | 0 |
| warm_multi | 38 | **2.0** | **21/38** |

- **cold(空 user_id)**: cross_session は `str(user_id)` をキーにするため、空 user_id が全 cold セッションで `""`(→`"None"`)に集約され、**他 cold ユーザの track が cap=50 まで混入**。crash しないが意味的に noise。
- **warm**: Blind B の warm ユーザは pool(Dev + Blind B)に他セッションをほとんど持たないため **21/38 がゼロ**、平均 2 件。
- Blind A では 80/80 warm かつユーザが他セッションを持つ構造だったため cross_session が正に効いた([[exp090-cross-session-source]])。Blind B はこの前提が崩れる。

→ **leak-safe な対処案**: 空 user_id を「cross-pool 無し」として扱う(`""`/`"None"` 集約をスキップ)。これで cold の noise は消えるが、warm の sparsity は構造的に解消されない(Blind B にユーザの他セッションが無い)。判断は user に委ねる(下記)。

## 本番 Blind B submission ビルド結果（2026-06-24、user 承認: cross_session 修正後ビルド）

smoke を step-1 として、bankable(exp116 lxct6 = 4-family 6-seed ensemble)を Blind B に適用し submission を生成した。

### cross_session leak-safe 修正
- [cross_session_candidates.py](../../mcrs/experiments/exp090_cross_session_source/cross_session_candidates.py) の `build_user_sessions` で空/None user_id を pool から除外。
- 検証: **Blind A は canonical と byte-identical**(Dev/Blind A は user_id 非空なので no-op)。Blind B は cold pollution 除去(cold_multi 50→0、warm 不変 2.0、true-cross 58→18/80)。

### ビルド手順（再現）
1. 13 blind source 全再生成(Blind B): primary `exp015_B_ll_t500_B`(run_inference_blindset)+ continuity/qnm/entity/transition(各 generator `--blind_dataset_name Blind-B`)+ cross_session(修正版)+ exclude_json(Dev+Blind B)。置き場 `exp/smoke_blindB/sources/`。
2. ensemble: [run_ens_lxct6_blindB.sh](../../mcrs/experiments/exp116_gbdt_family_ensemble/run_ens_lxct6_blindB.sh)(dev `--source`/params は lxct6 と byte-identical、blind 側のみ Blind B)。**REDUCED=1 dry-run で統合検証**(13 source join OK・80 行・top-20)→ 本番 6-seed 実行。
3. response: `generate_responses_v2.py --backend claude --model claude-opus-4-8 --test_dataset_name ...Blind-B`(80 行全新規生成、最終 ranking top1 整合、leak-safe)。
4. zip 化 + validate。

### 成果物 / 検証
- ranking: `mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/blindB_tracks_ens_blindB_lxct6.json`(Dev full-fit blend nDCG 0.219 / lgbm 0.260・xgb 0.243・catboost 0.176・tabm 0.181 = 学習 sanity。Blind B は label 無しで実 score 測定不可)。
- submission JSON: `exp/inference/blindset_B/exp116_lxct6_B.json`。
- submission ZIP: `exp/inference/blindset_B/submission_exp116_lxct6_B.zip`(root=prediction.json、名前 29 字)。
- **validate_submission: JSON / ZIP とも OK**(80 行・全 top-20・catalog 外参照 0・(session,turn) 80/80 unique・空 response 0・words 平均 114)。
- leak-safe 確認: cold single-turn の応答も current turn の user query + catalog metadata のみで top1 整合(GT/future-turn 不使用)。Claude 経路(従来 Blind B 未実行)も正常動作を実測。

### なぜ cold-user の空 cross_session が安全か（generalization 論拠）
- cross_session の Dev 生成 log: `Dev rows 8000 (true-cross nonempty 5208)` = **Dev 学習行の ~35%(2792/8000)は元々 cross_session が空**。つまり「cross_session 候補ゼロ」は ranker が学習時に約 1/3 経験している in-distribution パターンで、Blind B の cold(修正後=空 cross)も新規入力ではない。これが「rank feature で down-weight」より firm な根拠。
- f_missing_user_embedding フラグ等の欠損シグナルも Dev の test_cold で学習済み。

### 注意
- **Dev full-fit blend 0.219 < lgbm 単独 0.260** は (a) full-fit=train 上採点で楽観/noise、(b) [[model-family-ensemble-no-headroom]] が既に記録した「Dev では 4-family blend ≤ lgbm」パターンそのもの。**lxct6 採否は再検討不要**(user は頑健性論拠で採用、それは不変)。LB 時にこの数字で誤解しないこと。`oof_ndcg` は Dev 専用診断で blind 成果物に無関係(full-fit models で blind ranking 生成)。
- Dev metrics はあくまで学習 sanity。**Blind B の真の score は hidden leaderboard**でしか分からない。提出枠は team 最大 3 回なので submit 判断は別。
- ranking 中間物の dryrun 出力は整理済み。
- 安価なヘッジ案(任意): 同一 trained models から `--secondary_families lgbm --output_blind_tracks2` で lgbm 単独 Blind B ranking を near-free 生成可(response は別途 Claude pass)。lxct6 を primary に据えたまま 3 枠の 1 つ用の代替札になり得る。

## 未実行 / 残作業（明示）

以下は reasoning で crash を否定したが **Blind B 上で実行はしていない(unexercised)**。本番ビルド = local 検証時に必ず通すこと(submit 時でなく local で表面化させる)。

- **ensemble apply は Blind B 未実行**。78 特徴量は rank-only(user_id 非依存)なので **crash は否定**できるが、**join/shape の不整合は未検証**: source は全 360 turn の行を出すのに対し予測ターゲットは 80 final-turn のみ。この Blind B 固有の突合は reasoning では潰せない。本番ビルドの reduced ensemble で必ず確認。
- **Claude response path(exp063 build_profile_block)は Blind B の null profile 上で未実行**。code-read 上は `if not user_profile` で guard 済みだが実走はしていない(low risk・unexercised)。
- **transition source は smoke 除外**(train index 構築が重く、conversation 駆動で collapse リスク低)。本番ビルドでは要再生成。
- **本番 Blind B submission ビルド = 別作業**(13 source 全再生成 + 4-family ensemble 再学習 + 80 行 response を exp063 Claude で全再生成)。**提出枠は team 単位で最大 3 回**(AGENTS.md「Blind B Phase」)なので、ビルド実行は user 承認後。cross_session の扱い(修正 / drop / 据置)もビルド前に決める。
