# 会話フロー EDA サマリー（goal_progress と pivot honor）

Last updated: 2026-06-20

> **重要（2026-07-20）:** 本文書の Section 4 は
> `goal_progress[t]` を `music[t]` の評価として集計しているが、TalkPlayData 2 の
> 生成順では `goal_progress[t]` は直前の `music[t-1]` に対する評価である。
> したがって、44.6% 対 62.4% の同一ターン比較とそこから導いた解釈は使用しないこと。
> 正しい逐次分析は [20260720_evaluation_alignment_audit.md](20260720_evaluation_alignment_audit.md)
> と `EDA/20260720_evaluation_alignment_audit.py` を参照する。

## 実行条件
- run scope: full train（train split, 15,199 sessions）
- metadata join: `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` の all_tracks（artist/album）
- output scope: `EDA/` only。`mcrs/experiments/` は更新していない。
- **リーク注意**: `thought` / `goal_progress_assessments` / `listener_goal` は train 専用の
  god's-eye 情報。本 EDA は**分析専用**であり、これらを推論時の feature / prompt / fallback /
  validation に流用しないこと。

## いちばん大事なフレーミング
`role=music` の content は **nDCG@20 の正解 track_id そのもの**（[parse_turn](../../mcrs/ltr/data_utils.py) の label 抽出）。
したがって本 EDA の「artist 継続率」は "system が pivot を無視した量" ではなく、
**正解ターゲット自体がどれだけ pivot しないか**の測定である。これが exp073 の
artist-demote（別 artist 要求 turn で同 artist を top20 から降格）が **net-negative** だった
理由を定量的に説明する: 同 artist を下げる＝GT から離れる＝nDCG が下がる。

## 主要結果

### 1. goal_progress（dataset 正解ラベル）の動態
- 全 assessment 106,393 件のうち **MOVES_TOWARD_GOAL = 50.6%** / DOES_NOT = 49.4%。
  baseline 会話の**約半分は「ゴールに近づいていない」とラベルされている**。
- turn 別 MOVES 率は turn2 55.2% から中盤に向けて低下し
  （会話が進むほど袋小路になりやすい）、最終 turn で回復する。詳細は
  `tables/conv_flow_goal_progress_by_turn.csv`。
- goal category 別では **`I` が最難（MOVES 38.3%）**、
  最も達成しやすいのは `A`（56.7%）。
  category × specificity の全量は `tables/conv_flow_goal_progress_by_category.csv` /
  `..._by_specificity.csv`。

### 2. session 単位の達成度
- **never_moves（全 turn で DOES_NOT）= 2,193 session
  (14.4%)**。
  最後まで一度もゴールに近づかない session が一定数ある。
- last_turn が DOES_NOT で終わる session = 45.4%。
- 詳細は `tables/conv_flow_session_outcome.csv`。

### 3. GT の artist 継続率（隣接 vs 累積）
- **隣接（turn N vs N-1）平均 = 53.0%**。
- **累積（union(1..N-1)）= 65.0%**（history 有り turn 分母）。
- 全 music event(8/△session)を分母にすると **56.9%** で、既存
  `relisten-eda-summary.md` の「同一 session 内 artist 継続率 train=56.88%」と整合する
  （差は turn1=prior 空を分母に含めるか否かの定義差）。turn 別は
  `tables/conv_flow_artist_continuity_by_turn.csv`。

### 4. ★ pivot × GT継続 × goal_progress クロス集計（本 EDA の核心）
history 有り・goal_progress ラベル有りの turn を 4 セルに分けた MOVES 率:
（前提: `assessment[t]` は turn t の推薦 `music[t]` を評価しているとみなす。assessment は
turn2 開始・1 turn 内は user→music→assistant 順なので "t の推薦" か "t-1 への反応" か原理的に
曖昧だが、下の 18pt gap（pivot+別 62.4% vs pivot+同 44.6%）が `music[t]` の artist に強く連動する
事実は、もし label[t] が `music[t-1]` を評価していたら現れ得ない＝この alignment を裏付ける
（連続 rec の autocorrelation でやや減衰するため "強い傍証" 扱い）。）

| | GT 同 artist | GT 別 artist |
|---|---|---|
| **pivot 要求あり** | n=8,455, MOVES 44.6% | n=2,527, MOVES 62.4% |
| **pivot 要求なし** | n=60,711, MOVES 50.8% | n=34,700, MOVES 51.1% |

- **全体（pivot 問わず）では同 artist 50.0% vs 別 artist 51.9% と
  ほぼ差が無い**＝同 artist 継続それ自体は「悪」ではない（GT の半分強は同 artist だが MOVES 率は
  別 artist とほぼ同じ）。
- **差は pivot 要求 turn に局在する**: pivot turn では 同 artist GT の MOVES = **44.6%** に対し
  別 artist GT は **62.4%**（約 18pt 差）。
  ところが pivot 要求 turn（history 有り 10,982 件）の GT は
  **77.0% が同 artist**（累積定義=過去いずれかの turn と artist
  共有。隣接定義ならより低い／"別 artist" を直前以外なら可とするユーザもいる）。
  pivot かつ GT 同 artist の turn の DOES_NOT 率 =
  **55.4%**。
- 解釈: **衝突は pivot turn に局在する**。そこでだけ「ユーザ満足（別 artist→MOVES）」と
  「nDCG 正解（同 artist 77%）」が両立しない。pivot turn で nDCG を取りに行く＝dataset が
  DOES_NOT とラベルする同 artist track を 55.4% の
  確率で推薦することになる。これが「pivot を honor すれば nDCG が上がる」という直感が成り立たない
  理由であり、exp073 の artist-demote が net-negative だった原因そのもの。LLM-Judge は blind の
  response 側で測られるので、**ranking は GT(同 artist 含む)に素直に寄せ、満足度の改善は
  response 軸（説明・personalization）で取りに行く**のが整合的。
- ただし pivot かつ GT 別 artist のセル（n=2,527, MOVES 62.4%）は、**switch が
  正解(GT)かつ満足の両立する pivot turn の少数派（pivot turn の約 23%）**。
  「いつ switch すべきか」を当てる signal は将来 lever になり得る（exp073 の blanket artist-demote
  とは別物＝全 pivot で同 artist を下げるのではなく、switch が GT のサブ集合だけを狙う）。
  ranking が pivot に一切効かない訳ではない。
- 注意: pivot 検出は語彙ヒューリスティックで noise を含むため、headline は authoritative な
  goal_progress 側（同 artist vs 別 artist の MOVES 率差）に置き、regex slice は補助とする。

### 5. 発話統計
- user / assistant の発話語数の turn 別推移は `tables/conv_flow_utterance_by_turn.csv`。
  user 発話は袋小路 turn で長くなる（懇願・繰り返しの言い直し）傾向。

## 代表 session のフロー（walkthrough）
- session_id: `9c337a02-15b1-408f-8103-c2f9459b3bed`
- goal(category=H, specificity=LL): The listener wants to explore different artists and discover new songs from a broad collection, focusing on general moods and styles, without specific genre constraints. They are open to exploring the discographies of artists they find interesting.

| turn | user 発話(要約) | GT track / 推薦理由 | goal_progress |
|---|---|---|---|
| 1 | I want to discover some new artists. Do you have anything that's a bit intense or dramatic | You enjoyed the last Alesana track, so I'm sticking with your favorite | - |
| 2 | Oh, Alesana is amazing, I love them! "The Fiend" is definitely intense. But I'm actually l | You're clearly enjoying Alesana, so I'm diving deeper into their catal | MOVES_TOWARD_GOAL |
| 3 | Okay, I really appreciate the Alesana tracks, they're fantastic! But I *really* want to di | Since you're enjoying the heavier side of Alesana, "The Temptress" is  | DOES_NOT_MOVE_TOWARD_GOAL |
| 4 | I *really* need to hear something by a *completely different artist*. I love Alesana, but  | You're really into the Alesana tracks from 'A Place Where The Sun Is S | DOES_NOT_MOVE_TOWARD_GOAL |
| 5 | Look, I appreciate that Alesana fits the intensity, but I've asked multiple times for *new | Since you're enjoying the Alesana tracks from 'A Place Where The Sun I | DOES_NOT_MOVE_TOWARD_GOAL |
| 6 | This is getting a bit frustrating. I need to discover *new* artists. I love Alesana, but I | You're still enjoying Alesana and their 'A Place Where The Sun Is Sile | DOES_NOT_MOVE_TOWARD_GOAL |
| 7 | I'm sorry, but this isn't working. I've asked multiple times for *new* artists, not more A | Since you're still enjoying Alesana and the 'A Place Where The Sun Is  | DOES_NOT_MOVE_TOWARD_GOAL |
| 8 | This isn't working at all. I've asked multiple times, very clearly, to discover *new* arti | You're still enjoying Alesana and the 'A Place Where The Sun Is Silent | DOES_NOT_MOVE_TOWARD_GOAL |

## 示唆
- **ranking 軸**: GT は同 artist 継続が支配的（隣接 53.0%）。継続系 source
  （artist/album continuity・entity）を主軸にするのは GT 構造と一致しており、artist-demote の
  ような pivot 迎合は GT を壊す。改善余地は continuity ではなく **recall（GT が候補に入らない
  crashed class）**側にある（[[exp073-refinement-saturation]] と整合）。
- **response 軸**: goal_progress が約半数 DOES_NOT である以上、ranking だけでユーザ満足は
  上がりにくい。説明・personalization で「同 sound だが要望も理解している」ことを示す response
  設計（exp063 V-Claude-long 系）が満足度 lever として正しい。
- **category 難易度**: 最難 category を feature/bucket として持ち、bucket 別 nDCG で source 採否
  を判断する（strategy-eda の decision gate と整合）。

## 生成物
- `tables/conv_flow_goal_progress_by_turn.csv`
- `tables/conv_flow_goal_progress_by_category.csv`
- `tables/conv_flow_goal_progress_by_specificity.csv`
- `tables/conv_flow_session_outcome.csv`
- `tables/conv_flow_artist_continuity_by_turn.csv`
- `tables/conv_flow_pivot_crosstab.csv`
- `tables/conv_flow_pivot_request_stats.csv`
- `tables/conv_flow_utterance_by_turn.csv`
