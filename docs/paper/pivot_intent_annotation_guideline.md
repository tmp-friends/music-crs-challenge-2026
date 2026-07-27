# Artist-pivot detector: single-annotator guideline

Last updated: 2026-07-23

## Purpose

論文の lexical intent detector が、current user request に含まれる
「新しい／別の artist へ変更したい」という意図をどの程度捉えるか、一人で監査する。
これは detector の manual audit であり、response quality の人手評価ではない。

## Important blinding rule

判定が200件すべて完了するまで、次の manifest を開かない。

`EDA/annotation/pivot_intent_annotation_200_manifest.csv`

manifest には detector 出力と分析用 annotation が含まれる。先に見ると判断が
detector や論文結果へ引っ張られる。編集対象は次のファイルだけである。

`EDA/annotation/pivot_intent_annotation_200.csv`

annotation form に含まれる情報は、current user message とその直前までの対話である。
current relevant track、goal progress、future turn、conversation goal、thought は含まない。

## Primary question

各行について、次の質問だけに答える。

> Current user message は、直前までの会話を踏まえて、新しい／別の
> artist・band・singer・musician を推薦するよう明示的または明確に要求しているか？

「別の曲」や一般的な novelty ではなく、**artist identity の変更または未知 artist の発見**
が要求されているかを判定する。

## Labels

### `ARTIST_PIVOT`

次のいずれかが current request から明確に読み取れる。

- different / another / new artist, band, singer, musician を要求する。
- “someone else” のように、別の演者を明確に要求する。
- “an artist I have not heard before” のように、未知 artist の発見を要求する。
- artist 名を明示せずとも、文脈上 “not them; try somebody else” が演者変更を指す。

Examples:

- “Could you try a different artist with the same energy?”
- “I like this track, but I want to discover someone I have not heard before.”
- “Not this band again—please play somebody else.”

### `NOT_ARTIST_PIVOT`

次のいずれかに当てはまる。

- new / different **song or track** を求めるだけで、artist 変更は要求しない。
- “something new” の対象が曖昧で、artist 変更を示す文脈がない。
- 同じ artist の別曲、同じ album、同じ performer の継続を求める。
- genre、tempo、mood、era だけを変更し、artist identity には触れない。
- 推薦への不満だけを述べ、別 artist の要求まではしていない。

Examples:

- “Can you play another song by them?”
- “Something more upbeat, please.”
- “I have not heard this song before.”（song の novelty のみ）

### `UNCERTAIN`

文脈を読んでも artist 変更か track／style 変更か決められない場合に使う。
無理に二値へ寄せない。

Examples:

- “Give me something completely different.”（文脈にも artist の手掛かりがない）
- 代名詞の参照先が不明で、artist と track のどちらか判断できない。

## Confidence

label と同時に一つ入力する。

- `HIGH`: 明示的な artist 語または明確な言い換えがある。
- `MEDIUM`: 文脈を使えば一意に判断できるが、current message 単独では弱い。
- `LOW`: label は選べるが、合理的な別解釈が残る。

`UNCERTAIN` にも confidence を入力する。通常は `HIGH` または `MEDIUM` で
「曖昧であること」を確信できる。

## Annotation procedure

1. CSV を UTF-8 対応の spreadsheet で開く。
2. `context_before_current` と `current_user_message` を読む。
3. `label` に3種類のいずれかを正確に入力する。
4. `confidence` に `HIGH` / `MEDIUM` / `LOW` を入力する。
5. 境界事例や判断理由を `notes` に短く記録する。
6. session_id、turn_number、context、message、annotation_id は変更しない。
7. 25件ごとに保存し、次の command で入力値と進捗を確認する。

```bash
.venv/bin/python EDA/20260723_pivot_intent_annotation.py status
```

行順を変えても annotation_id で結合できるが、行の追加・削除はしない。
`status` は manifest を機械的に参照し、ID、session-turn、context、message が
生成時から変わっていないことも検査する。detector 判定は画面へ表示しない。

## Completion and aggregation

200件すべてを入力後、まず status が `200/200` になることを確認する。

```bash
.venv/bin/python EDA/20260723_pivot_intent_annotation.py status
```

その後に初めて manifest と突合して集計する。

```bash
.venv/bin/python EDA/20260723_pivot_intent_annotation.py summarize
```

出力:

- `EDA/tables/pivot_intent_manual_metrics.csv`
- `EDA/annotation/pivot_intent_annotation_errors.csv`
- `EDA/summary/20260723_pivot_intent_manual_audit.md`

precision / recall / F1 は sampling stratum の母集団サイズで重み付けする。
95% CI は全 sampling stratum で共通の session multiplicity を使う
session-cluster bootstrap 5,000回で求め、各層の総 weight を母集団サイズへ
再正規化する。
`UNCERTAIN` は二値 metrics から除外し、重み付き uncertain rate と
classifiable coverage を別に報告する。

## Reporting constraints

一人での監査なので Cohen's kappa や inter-annotator agreement は報告しない。
論文では “single-annotator manual audit” と記述し、human satisfaction の検証や
detector の完全な妥当性確認とは表現しない。主要結果の方向が維持されても、
200件の監査から人間の効用や因果効果を主張しない。
