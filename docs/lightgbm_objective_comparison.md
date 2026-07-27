# LightGBM objective: lambdarank / log_loss / rank_xendcg

## Summary

RecSys の reranking で LightGBM を使う場合、`lambdarank` と `rank_xendcg` は query / group 内の順位を直接扱う ranking objective で、`log_loss` は各 candidate を独立した binary classification として扱う pointwise objective です。

この repository の exp014d では、`log_loss` を実験名として扱い、LightGBM には `objective=binary`, `metric=binary_logloss` を渡しています。LightGBM の公式 objective 名に `log_loss` はなく、binary log loss は `binary` objective と `binary_logloss` metric で表します。

## Comparison

| 観点 | `lambdarank` | `rank_xendcg` | `log_loss` / `binary` |
|---|---|---|---|
| LightGBM objective | `lambdarank` | `rank_xendcg` | `binary` |
| LightGBM metric の代表例 | `ndcg` | `ndcg` | `binary_logloss` |
| 学習単位 | query / group 内の候補集合 | query / group 内の候補集合 | candidate row 単位 |
| 最適化の性質 | NDCG を意識した LambdaRank / LambdaMART 系 | XE_NDCG_MART 系の ranking objective | binary cross entropy |
| group 情報 | 必須 | 必須 | loss 自体には不要。ただし rerank 評価・top20 作成では group 内 sort が必要 |
| label | relevance label。今回のような 0/1 でも使える | relevance label。今回のような 0/1 でも使える | 0/1 label が前提 |
| 強い場面 | 評価指標が NDCG@k で、group 内の上位順位を直接合わせたい | `lambdarank` に近い目的で、より速い・滑らかな代替を試したい | 正例確率推定が安定し、ranking pair が sparse / noisy なとき |
| 弱い場面 | 正例なし group から有効な ranking signal が出ない。group 構成・label gain・truncation に敏感 | `lambdarank` と同様に group 依存。dataset によっては必ずしも改善しない | NDCG@20 を直接最適化しない。top position の入れ替えコストを loss が知らない |

## lambdarank

`lambdarank` は group 内の候補同士の相対順序を学習する ranking objective です。推薦の 1 session-turn を 1 query / group と見なし、その group 内で正例 track が負例 track より上に来るように木を学習します。

重要なのは、単なる pairwise 分類ではなく、順位を入れ替えたときの NDCG への影響を重みにした勾配を使う点です。そのため、上位 rank の誤りほど大きく扱いやすく、`nDCG@20` のような評価指標と相性が良いです。

Music-CRS での意味:

- `predicted_track_ids` の上位 20 件が評価対象なので、NDCG を意識した objective として自然。
- train group に正例がない場合、正例と負例の比較が作れないため、学習 signal がほぼありません。exp014b で positive-group filter を入れた理由はここです。
- `lambdarank_truncation_level` は、どの上位範囲に学習を集中させるかに関係します。NDCG@20 なら 20 より少し大きい値が候補になります。

## rank_xendcg

`rank_xendcg` は LightGBM の XE_NDCG_MART ranking objective です。公式 docs では、`lambdarank` と近い性能を出しつつより高速な objective と説明されています。

直感的には、group 内の relevance 分布と model score から作る順位分布を近づける listwise 寄りの objective です。`lambdarank` が pairwise swap の NDCG 差分を使うのに対して、`rank_xendcg` は NDCG を意識した cross-entropy 型の目的で順位を学習します。

Music-CRS での意味:

- `lambdarank` と同じく group 内 ranking を直接扱うため、`all_tasks_ndcg@20` の候補 objective になります。
- positive が sparse でも、pairwise swap に依存する `lambdarank` と違う勾配の出方になるため、相性差を見る価値があります。
- exp014d では `rank_xendcg` が `log_loss` より一貫して強かった一方、best の `s05 rank_xendcg` でも exp014b `lambdarank` best には届きませんでした。

## log_loss / binary

`log_loss` はこの repo での実験名で、LightGBM では `objective=binary` として学習します。各 candidate row に対して「この track が正例か」を 0/1 classification として予測し、binary log loss を最小化します。

ranking objective との最大の違いは、group 内の相対順位を loss が直接見ないことです。学習時には candidate A と candidate B のどちらを上に置くべきかではなく、各 candidate が個別に正例らしいかを学習します。推論時には、その score を group 内で降順 sort して top20 を作ります。

Music-CRS での意味:

- 正例確率に近い score が欲しい場合や、group 内 pair/listwise signal が noisy な場合の fallback として使いやすいです。
- 一方で、NDCG@20 の上位順位を直接最適化しないため、正例を top20 のどこに置くか、top1〜top5 をどれだけ重く見るかは loss に入りません。
- exp014d では `log_loss` は全 step で `rank_xendcg` を下回りました。

## Practical Guidance For This Repo

現在の exp014 系では、主指標を `all_tasks_ndcg@20` に置くため、第一候補は `lambdarank` のままでよいです。exp014d の結果でも、objective 変更だけでは exp014b `lambdarank` を超えませんでした。

使い分けは次の方針が妥当です。

| 状況 | 推奨 |
|---|---|
| NDCG@20 を直接伸ばしたい | `lambdarank` を第一候補にする |
| `lambdarank` が遅い、または sparse positive で勾配の出方を変えたい | `rank_xendcg` を比較する |
| 正例確率の calibration や pointwise な安定性を見たい | `binary` / `log_loss` を比較する |
| 正例なし group が train に多い | ranking objective ではなく、まず train group policy を直す |
| candidate source の noise が問題 | objective よりも source policy / feature-only 化 / negative sampling を優先して見直す |

## exp014d Result Note

exp014d は exp014b を base に、`rank_xendcg` と `log_loss` のみを s05〜s09 で比較しました。`lambdarank` は exp014b の既存結果を baseline として扱いました。

Best Dev results:

| Experiment | Step | Objective | all_tasks_ndcg@20 |
|---|---|---|---:|
| exp014b | s06 | `lambdarank` | 0.153337 |
| exp014d | s05 | `rank_xendcg` | 0.143842 |
| exp014d | s05 | `log_loss` / `binary` | 0.139038 |

結論として、exp014d の objective 比較では exp014b `lambdarank` を維持します。次に見るべきは objective 変更ではなく、candidate source の扱い、feature 設計、negative sampling、または response 固定後の別 rerank 構成です。

## References

- LightGBM Parameters: https://lightgbm.readthedocs.io/en/stable/Parameters.html
- LightGBM Query Data: https://lightgbm.readthedocs.io/en/stable/Parameters.html#query-data
- LightGBM Objective / Metric Parameters: https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective-parameters
