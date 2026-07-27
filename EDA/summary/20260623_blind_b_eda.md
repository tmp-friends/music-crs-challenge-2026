# Blind B EDA サマリ (2026-06-23)

- 対象: `talkpl-ai/TalkPlayData-Challenge-Blind-B`（2026-06-23 公開、final 評価セット）
- 比較対象: Blind A / catalog (`Track-Metadata`)
- 再現: `python EDA/20260623_blind_b_eda.py`

## TL;DR

Blind B は Blind A と **同一スキーマ・同一 catalog（track 100% overlap）・1 session 1 予測（最終 user turn）= 80 行** で、提出フォーマットは変えなくてよい。ただし **意図的に「より blind」かつ「より口語」に作られた robustness テスト** で、3 つの分布シフトがある:

1. **メタデータ大幅削除**: `conversation_goal`・`session_date`・`goal_progress`・`thought` は **全件 null/削除**。`user_profile` と `user_id` は **40/80 のみ**（残り 40 は cold start）。
2. **会話が長く・cold が多い**: 平均 11.5 turn（A は 8.88）、turn 分布は `{1,4,7,10,13,16,19,22}` 各 10 件の均一サンプリング。
3. **口語 register シフト**: Blind A の session を口語調に paraphrase したものが多数（slang marker 約 3 倍、平均クエリ長やや増）。

> ⚠️ **未検証 / 締切前必須**: 空 `user_id`（40/80）で bankable pipeline が **KeyError で落ちる懸念**あり（下記 §5）。Blind B 提出（締切 2026-06-30）前に smoke test 必須。

## 1. 構造・提出フォーマット

| 指標 | Blind B | Blind A |
|---|---|---|
| sessions | 80 | 80 |
| 予測対象 | 最終 user turn（80/80 が user 終端） | 同左 |
| turns/session mean (median) | 11.50 (11.5) | 8.88 (7.0) |
| turn count 分布 | `{1,4,7,10,13,16,19,22}` 各 **10** | `{1:20,4:15,7:10,10:5,13:8,16:9,19:8,22:5}` |
| music turn 総数（文脈） | 280（70 session に存在） | 210（60 session） |
| catalog overlap | **274/274 = 100%** | — |

- 1 turn_number = `user`→`music`→`assistant` の 3 役。**過去の music turn には推薦 track_id が可視**（in-session の継続/cross-session 候補 source = exp090 系の信号は Blind B でも生きている）。
- 推論ループ（`run_inference_blindset.py`）は `conversations`/`user_id`/`session_id` のみ読む → **提出構造の変更は不要**。
- 10 session は単一 user turn（cold start・履歴ゼロ・8/10 が profile 無し）。

## 2. メタデータ削除（A→B）

| フィールド | Blind B | Blind A |
|---|---|---|
| `user_profile` 充足 | **40/80** | 80/80 |
| `user_id` 非空 | **40/80** | 80/80 |
| `conversation_goal` | **0/80** | 80/80 |
| `session_date` | **0/80** | 80/80 |
| `goal_progress_assessments` | **0/80** | 80/80 |
| `thought` 非空 turn | **0** | 423（`user`213/`assistant`210 turn、god's-eye。music turn には無い） |

- `conversation_goal`(category/specificity/listener_goal) は **全削除** → 当該フィールドに依存する feature/prompt は Blind B で完全に無効（exp107 で既に -0.0026 棄却済なので機能影響は無い見込み）。
- `thought` は user/assistant turn の god's-eye 推論で、我々は元々不使用。Blind B は念のため全削除。leak-safe な可視 content ではない。
- candidates/ltr/db_user を grep した限り `conversation_goal`/`goal_progress`/`session_date` は **未参照** → これらの null は crash 源にならない。

## 3. Blind A との重複・paraphrase 関係（leak は無い）

- **session_id 43/80 が Blind A と重複**（UUID なので偶然衝突ではなく構造的）。うち **41 が同 turn 長・2 のみ B で予測ターンが後ろ**。
- 重複 43 のうち **user-turn 完全一致は 22**、残り 21 は **同一 session_id・同一 user・同一意図を口語調に paraphrase**（例: A "I'm looking for music with album art that really stands out…" → B "Cheers, I'm after some music where the album art actually stands out, y'know?"）。
- B-only 37 session も **新規 user は 10 のみ**（空 uid 18 / Blind A の uid 再利用 9）。
- **leak は無い**: 予測対象（最終 user turn の次 music）の GT は A/B とも hidden。重複の価値は「Blind A LB が Blind B の約半分の partial proxy になる」「organizer が paraphrase + メタ欠落への robustness を測っている」と読める点のみ。

## 4. register / 人口統計シフト

- **口語化**: slang marker（cheers/ngl/omg/mate/gonna…） Blind B 0.09/turn vs A 0.03/turn（約 3 倍）、平均クエリ語数 42.1 vs 39.0。BM25/dense の lexical match を僅かに削る方向のシフト。継続は in-session の可視 track_id が anchor になるため致命的ではない。
- **populated profile は narrower**: Blind B (40件) は 20s 82%/male 85%/English 100%。Blind A は 20s 60%・10s 22% とやや多様。ただし profile 自体が半数欠落。

## 5. ⚠️ 頑健性リスク（open TODO・締切前に必須）

- [user_profile.py:35](../../mcrs/db_user/user_profile.py#L35) `id_to_profile` / `:47` `id_to_profile_str` は **無ガードで `self.user_profiles[user_id]`** → 空 `user_id`（Blind B 40/80）や DB 未登録 uid で **KeyError**。同様に [user_embeddings.py:33](../../mcrs/db_user/user_embeddings.py#L33) も uid キー lookup。
- bankable pipeline（exp116 lxct6 = candidate-gen → 78-feature build → 4-family ensemble）がこれらを cold session で呼ぶと落ちる可能性。**コード読みだけでは silent degradation を取り逃すので smoke test 必須**:
  - bankable chain を Blind B の **(a) 空 user_id の cold session, (b) 単一 user turn session** を含む数件で end-to-end 実行し、crash と劣化を確認。
  - 落ちる場合は cold 用 fallback（profile 無し prompt / user feature 欠損埋め）を leak-safe に追加。

## 6. 既存判断への含意

- この regime change（cold 増・口語化・メタ欠落）は [[prefer-robust-ensemble-on-noise-tie]] が hedge していた当のシナリオ。EDA は **lxct6（model-family 多様 ensemble）採用を validate** する（再検討は不要）。
- レバーの優先は不変: 候補 recall は in-session 可視 track anchor で維持、response 軸は引き続き headroom。conversation_goal 系は Blind B で無効化されるため依存しないこと。
