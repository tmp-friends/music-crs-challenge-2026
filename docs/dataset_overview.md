# データセット概要

本コンペ (RecSys Challenge 2026: Music-CRS) で使用するデータセットの論理区分とフィールド定義をまとめる。

---

## 論理区分

### 1. 会話データ (Conversation)

| Split | 行数 | Sessions | Users | 用途 |
|---|---|---|---|---|
| **train** | 15,199 | 15,199 | 8,591 | 学習・パターン分析用 |
| **dev_test** | 1,000 | 1,000 | 500 | ローカル評価・チューニング用 |
| **blind_a** | 80 | 80 | 58 | interim leaderboard 提出用 |
| **blind_b** | 80 | 80 | 29 (40件のみ profile 有) | 最終評価用 (2026-06-23 公開) |

各レコードは 1 session = 1 行で、train/dev は平均 24 turns/session・8 music推薦、blind_a は平均 8.88 turns・2.62 music推薦と短い。**blind_b は平均 11.5 turns（turn 分布 {1,4,7,10,13,16,19,22} 各 10 件）と長め**。提出構造（最終 user turn 1 予測 = 80 行 / catalog 100% overlap）は blind_a と同じだが、**40/80 が Cold Start Users で user_id / user_profile が無く**、`conversation_goal` / `goal_progress_assessments` / `thought` フィールドも削除されている。詳細は [../EDA/summary/20260623_blind_b_eda.md](../EDA/summary/20260623_blind_b_eda.md) と AGENTS.md「Blind B Phase」節を参照。

---

### 2. Track Metadata

| Split | 行数 | 説明 |
|---|---|---|
| **all_tracks** | 47,071 | 推薦候補の全カタログ（提出時はここから選ぶ） |
| **test_tracks** | 7,405 | all_tracks の部分集合（dev/blind の ground truth が含まれる track のみ） |

---

### 3. Track Embeddings

all_tracks / test_tracks それぞれに対して 6 種類の事前抽出ベクトル:

| Embedding | 次元 | ソース |
|---|---|---|
| `audio-laion_clap` | 512 | 音声特徴 (CLAP) |
| `image-siglip2` | 768 | アルバムアート画像 (SigLIP2) |
| `cf-bpr` | 128 | 協調フィルタリング (BPR) |
| `attributes-qwen3_embedding_0.6b` | 1024 | 属性テキスト埋め込み |
| `lyrics-qwen3_embedding_0.6b` | 1024 | 歌詞テキスト埋め込み |
| `metadata-qwen3_embedding_0.6b` | 1024 | メタデータテキスト埋め込み |

※ 492〜616 tracks はゼロベクトル（データ欠損）

---

### 4. User Metadata

| Split | 行数 |
|---|---|
| **all_users** | 8,772 |

---

### 5. User Embeddings (cf-bpr)

| Split | 行数 | 説明 |
|---|---|---|
| **train** | 8,591 | train会話に登場するユーザー |
| **test_warm** | 371 | test/blind の warm-start ユーザー（CF学習済み） |
| **test_cold** | 129 | test/blind の cold-start ユーザー（ゼロベクトル） |

次元は 128 (cf-bpr)。cold ユーザーはゼロベクトルで協調フィルタリング情報なし。

---

### まとめ図

```
TalkPlayData-Challenge
├── Conversation: train / dev_test / blind_a / blind_b
├── Track-Metadata: all_tracks (47k) ⊃ test_tracks (7.4k)
├── Track-Embeddings: 6種 × {all_tracks, test_tracks}
├── User-Metadata: all_users (8.7k)
└── User-Embeddings: train / test_warm / test_cold (cf-bpr 128d)
```

推薦候補は常に **all_tracks (47,071)** 全体から選ぶ必要があり、test_tracks はあくまで評価対象 track の部分集合として提供されている。

---

## フィールド定義

### Conversation Dataset

1レコード = 1 session。

- **`session_id`** (string) — セッションの一意識別子
- **`user_id`** (string) — ユーザーの一意識別子
- **`session_date`** (string) — セッションの日付
- **`user_profile`** (struct) — セッション時点のユーザー属性（inline 埋め込み）
  - `age` (int) — 年齢
  - `age_group` (string) — 年齢層
  - `country_code` (string) — 国コード
  - `country_name` (string) — 国名
  - `gender` (string) — 性別
  - `preferred_language` (string) — 希望言語
  - `preferred_musical_culture` (string) — 音楽文化圏の嗜好
  - `user_id` (string) — ユーザーID（冗長）
  - `user_split` (string) — ユーザーの split 区分（train / test_warm / test_cold）
- **`conversation_goal`** (struct) — そのセッションで対話が達成しようとしている目的
  - `category` (string) — ゴールのカテゴリ
  - `listener_goal` (string) — リスナー視点のゴール記述
  - `specificity` (string) — ゴールの具体性レベル
- **`conversations`** (list of struct) — multi-turn 対話の各発話
  - `turn_number` (int) — ターン番号（1〜8程度）
  - `role` (string) — 発話者ロール: `"user"` / `"assistant"` / `"music"`
    - `user`: ユーザーの発話テキスト
    - `assistant`: システムの応答テキスト
    - `music`: 推薦された track_id（ground truth）
  - `content` (string) — 発話内容または track_id
  - `thought` (string) — システム側の内部推論（推薦の理由付け）
- **`goal_progress_assessments`** (list of struct) — ターンごとのゴール達成度
  - `turn_number` (int) — 評価対象のターン番号
  - `goal_progress_assessment` (string) — 進捗評価テキスト

---

### Track Metadata

1レコード = 1 track。

- **`track_id`** (string) — トラックの一意識別子
- **`ISRC`** (list[string]) — 国際標準レコーディングコード（複数可）
- **`track_name`** (list[string]) — 楽曲名（表記揺れ対応で複数可）
- **`artist_name`** (list[string]) — アーティスト名（複数可）
- **`album_name`** (list[string]) — アルバム名（複数可）
- **`tag_list`** (list[string]) — ジャンル・ムード等のタグ（中央値 17 個）
- **`popularity`** (double) — 人気度スコア（中央値 38.0）
- **`release_date`** (string) — リリース日
- **`duration`** (int) — 再生時間（秒）、中央値 ≈ 3.78 分
- **`artist_id`** (list[string]) — アーティストの一意識別子（複数可）
- **`album_id`** (list[string]) — アルバムの一意識別子（複数可）

---

### Track Embeddings

1レコード = 1 track。

- **`track_id`** (string) — 対応する track の識別子
- **`audio-laion_clap`** (list[double], dim=512) — CLAP モデルによる音声特徴量
- **`image-siglip2`** (list[double], dim=768) — SigLIP2 によるアルバムアート画像特徴量
- **`cf-bpr`** (list[double], dim=128) — BPR 協調フィルタリングで学習した track 潜在表現
- **`attributes-qwen3_embedding_0.6b`** (list[double], dim=1024) — track 属性テキストの LLM 埋め込み
- **`lyrics-qwen3_embedding_0.6b`** (list[double], dim=1024) — 歌詞テキストの LLM 埋め込み
- **`metadata-qwen3_embedding_0.6b`** (list[double], dim=1024) — メタデータテキストの LLM 埋め込み

---

### User Metadata

1レコード = 1 user。

- **`user_id`** (string) — ユーザーの一意識別子
- **`age`** (int) — 年齢
- **`age_group`** (string) — 年齢層（例: "25-34"）
- **`country_code`** (string) — ISO 国コード
- **`country_name`** (string) — 国名
- **`gender`** (string) — 性別

---

### User Embeddings

1レコード = 1 user。

- **`user_id`** (string) — ユーザーの一意識別子
- **`cf-bpr`** (list[double], dim=128) — BPR 協調フィルタリングで学習した user 潜在表現
  - `train` split: 学習済み表現あり（norm_median=0.0375）
  - `test_warm` split: 学習済み表現あり（norm_median=0.0651）
  - `test_cold` split: ゼロベクトル（協調フィルタリング情報なし）

---

## 補足: conversations の role の意味

Conversation の `role` フィールドが特殊で、推論パイプラインに大きく影響する:

- **`"user"`** — ユーザーの自然言語リクエスト。これが retrieval query の主要入力。
- **`"assistant"`** — システムの応答テキスト（学習時のみ利用可能、blind では生成対象）。
- **`"music"`** — その turn で推薦された track_id。dev/train では ground truth として recall/nDCG 計算に使い、blind では存在しない（予測対象）。
