# Evaluation Framework

RecSys Challenge 2026: Music-CRS では、推薦結果 (what) と説明応答 (how) の両面を評価する multi-dimensional framework が採用されています。本ドキュメントは公式の評価仕様を整理したものです。

## Composite Score

最終スコアは 4 つの指標を以下の重みで線形結合します。

$$
\text{Score} = 0.50 \times \text{nDCG@20} + 0.10 \times \text{CatalogDiversity} + 0.10 \times \text{LexicalDiversity} + 0.30 \times \text{LLM-Judge}
$$

| Dimension | Weight | What it measures | How it is computed | Role in evaluation |
|---|---|---|---|---|
| nDCG@20 | 0.50 | 推薦 track ランキングの品質。 | 予測 ranked list を ground-truth relevant item に対して評価。上位に正解があるほど高得点。 | 主要 recommendation metric。 |
| Catalog Diversity | 0.10 | カタログ全体をどれだけ広くカバーできているか。 | 全 prediction にわたる unique recommended track 数 / catalog 全体サイズ。 | Diversity 補助指標。 |
| Lexical Diversity | 0.10 | 生成テキストの語彙的多様性。 | Distinct-2 (生成 response 全体での unique bigrams / total bigrams)。 | Response 生成補助指標。 |
| LLM-as-a-Judge | 0.30 | 生成された explanation の質。 | Blind-set response を Gemini 系 LLM が自動評価。Personalization と Explanation Quality の 2 軸を text-only に評価する。評価 prompt は blind evaluation の整合性確保のため非公開。 | Blind-set 応答品質評価。 |

## 各 metric の定義

### nDCG@k (Normalized Discounted Cumulative Gain)

予測ランキングと理想ランキングを比較してランキング品質を測ります。

$$
\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}, \quad
\text{DCG@k} = \sum_{i=1}^{k} \frac{\mathbb{1}[\text{pred}_i \in \text{gold}]}{\log_2(i + 1)}
$$

- 各 conversation turn に対して ground-truth track はちょうど 1 件。
- 公式評価では nDCG@1, nDCG@10, nDCG@20 を報告し、**主要 metric は nDCG@20**。
- 結果は session・turn にわたって macro-average する。

### Catalog Diversity

$$
\text{CatalogDiversity} = \frac{|\text{unique recommended tracks}|}{|\text{catalog}|}
$$

人気アイテムに集中せず、カタログ全体を広く探索しているかを測ります。

### Lexical Diversity (Distinct-2)

$$
\text{LexicalDiversity} = \frac{|\text{unique bigrams across all responses}|}{|\text{total bigrams across all responses}|}
$$

多様で冗長でない自然言語生成を促す指標です。

### LLM-as-a-Judge

LLM が複数の評価軸 (Personalization / Explanation Quality) について **1–5 の整数スケール** でサンプル session を評価します。詳細な評価基準・prompt は非公開です。

Composite に組み込む前に、各次元を min=1 / max=5 として min-max 正規化して [0, 1] に揃えます。

$$
\text{score}_{\text{norm}} = \frac{\text{score} - 1}{5 - 1} = \frac{\text{score} - 1}{4}
$$

正規化後の各スコアを平均し、composite 内で 0.30 の重みを掛けます。

## Leaderboard Phases

| Phase | Dataset | Metrics |
|---|---|---|
| Blind A (interim) | `TalkPlayData-Challenge-Blind-A` | All metrics |
| Blind B (2026-06-23 開始 – 06-30 締切, Final) | `TalkPlayData-Challenge-Blind-B` | All metrics (Final, **leaderboard hidden**) |

> Blind B Phase の運用ルール（team registration 必須・**submission は team 単位で最大 3 回**・leaderboard は終了後まで hidden・final submission は手動選択・不正は失格）は AGENTS.md「Blind B Phase」節を参照。

## 実装・運用上のメモ

- nDCG@20 が重みの半分を占めるため、retrieval / ranking の改善が最も score への寄与が大きい。
- Catalog Diversity と Lexical Diversity は重み 0.10 ずつだが、極端に低いと composite が引きずられるので「nDCG を最大化しつつ最低限の多様性を保つ」設計が望ましい。
- LLM-as-a-Judge は Blind set でのみ評価される。Development set ではこの軸は確認できないため、応答品質は別途人手・自前 LLM 評価で担保する必要がある。
- 評価 prompt は非公開なので、judge の挙動をリバースエンジニアリングするのではなく、Personalization (ユーザー嗜好への沿い方) と Explanation Quality (推薦理由としての筋の通り方) を一般的に押さえる方針で改善する。

## 参考

- RecSys Challenge Website: <https://www.recsyschallenge.com/2026/>
- Challenge Website: <https://nlp4musa.github.io/music-crs-challenge/>
