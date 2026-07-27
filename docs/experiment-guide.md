# 実験管理ガイド（Music-CRS）

共通モジュールと `mcrs/experiments/` 配下の experiment フォルダの境界が崩れ、exp014 のモジュールが
事実上の共有ライブラリ化したまま 1 実験フォルダに埋もれていた問題への、前進ルール。
**既存 exp000-016 は凍結し、ここに書く規約は新規・前進分にのみ適用する。**

## 1. コードの置き場所

| 種類 | 置き場所 |
|---|---|
| candidate 生成（source / query view） | `mcrs/candidates/`（`candidate_sources`, `query_views`） |
| LTR の dataset build / feature / 推論適用 | `mcrs/ltr/`（`data_utils`, `features`, `build_dataset`, `apply_lgbm`） |
| reranker（LightGBM LTR） | `mcrs/rerank_modules/`（現状 `lightgbm_ltr` のみ。embedding / fusion / cross-modal / LLM reranker は未使用のため削除済み） |
| retrieval backend（BM25 / BERT / metadata exact） | `mcrs/retrieval_modules/` |
| item / user DB、LM、system prompt | `mcrs/db_item/`, `mcrs/db_user/`, `mcrs/lm_modules/`, `mcrs/system_prompts/` |

`mcrs/ltr/` は exp014 由来の共有ライブラリ（exp014 base パイプライン）。LGBM 学習本体 `train_lgbm`
は monkeypatch 互換のため exp014 に凍結したまま残しており（§5）、`mcrs/ltr/` には含まれない。
B/B2 の modern 候補生成・特徴量は exp015 系 fork 側（§5）にあり、`mcrs/ltr/` ではない点に注意。

`expNNN_*/` フォルダが持つのは **hypothesis(README) + configs + results + 薄い pipeline / wrapper のみ**。
共有ロジック（候補生成・特徴量・dataset build）は experiment フォルダに書かず `mcrs/` 配下へ置く。

## 2. 他 expNNN フォルダから import しない

```python
# NG（legacy。shim で延命しているだけ。新規追加禁止）
from mcrs.experiments.exp014_lightgbm_ltr_reranker.features import LGBMFeatureBuilder
# OK
from mcrs.ltr.features import LGBMFeatureBuilder
from mcrs.candidates.candidate_sources import SOURCE_NAMES, WideCandidateBuilder
```

旧 exp014 パスは `mcrs.ltr.*` / `mcrs.candidates.*` への re-export shim になっており、過去 run の
再現性のために**凍結**している。新しい横断 import を増やさないこと（`git grep "from mcrs.experiments.exp0"`
が増えないのが健全な状態）。

## 3. 派生実験は injection で薄く作る（ABC / registry は使わない）

**目標形**: dataset build をカスタムする派生実験は、共有 `build_dataset` に builder / feature builder を
**注入**して、experiment 側は薄い wrapper だけにする。base class も Pydantic schema も registry も入れない
（既存の良い pattern を一般化する方針）。

**現状（重要・正確に把握すること）**:

- この injection hook（`feature_builder_cls=` / `builder_cls=`）が実装済みなのは **exp015c の
  `build_dataset.main` / `init_builders`** で、これは B/B2 の凍結 fork lineage（§5）。exp015d/e は 28 行の
  wrapper で exp015c の `main(args, feature_builder_cls=...)` を呼んでいる。
- 一方、本リファクタで切り出した **`mcrs.ltr.build_dataset.main(args)` は `args` だけを取り、まだ
  feature/candidate builder の injection hook を持たない**（exp014 base パイプラインの抽出のため）。
  CLI parser は `build_parser()` に切り出し済み（旧パス shim の `__main__` もこれを再利用する）だが、
  `mcrs.ltr.build_dataset.main(args, feature_builder_cls=...)` は今は動かない。

**新規実験が injection を使いたい場合の前進手順**:

1. `mcrs/ltr/build_dataset.py` の `main` / `init_builders` に **keyword-only かつ default が現挙動を保つ**
   引数（例 `feature_builder_cls: type[LGBMFeatureBuilder] = LGBMFeatureBuilder`）を追加する（additive）。
   追加前に `main` / `init_builders` の呼び出し元（exp014 pipeline / retrain script 等）を全て grep し、
   default で挙動不変なことを確認する。
2. experiment 側は薄い wrapper にする:

```python
# expNNN_*/build_dataset.py（exp015d の 28 行 wrapper と同じ形。hook 追加後に成立）
from mcrs.ltr import build_dataset as _base
from mcrs.experiments.expNNN_xxx.features import LGBMFeatureBuilder  # 派生 feature

def main(args):
    _base.main(args, feature_builder_cls=LGBMFeatureBuilder)
```

旧 exp014 パスの shim は import だけでなく `__main__`（`build_parser()` 委譲）も保持しているため、
frozen な build/apply スクリプトの `python mcrs/experiments/exp014_.../build_dataset.py ...` はそのまま動く。
新規コードからは `python -m mcrs.ltr.build_dataset` を使ってよい。

## 4. config layering（dev / blindA の重複を消す）

`mcrs/experiments/configs.py` は `_base:` キーをサポートする（後方互換。`_base` の無い config は不変）。
dev / blindA で本体が共通な新規 config は base を共有し、差分だけ override する。

```
expNNN_*/configs/_base_expNNN.yaml        # 共有本体
expNNN_*/configs/expNNN_x_devset.yaml     # _base: _base_expNNN.yaml + test_dataset_name: <devset>
expNNN_*/configs/expNNN_x_blindA.yaml     # _base: _base_expNNN.yaml + test_dataset_name: <blindA>
```

- `_base` は child の dir 基準で解決し、merge 前に除去される（推論側へは漏れない）。
- 共有 base file は `_` prefix にする（`find_experiment_config_path` の tid glob にマッチしない）。
- list は merge で置換される（追記ではない）。

## 5. 凍結 fork（統合しない）

以下は意図的に重複・分岐したまま凍結している。`mcrs/ltr/` などへ向け直さないこと。

- `mcrs/rerank_modules/lightgbm_ltr.py`(272 行) と `exp015_.../lightgbm_ltr.py`(242 行) — 分岐済み fork。
- `exp015_candidate_rules_ablation/{features,build_dataset}.py` — exp014 とは別系統（`LGBMFeatureBuilder`
  が `source_names` 引数）。exp015c/d/e はこちらを import する B/B2 の再現 lineage。
- `exp014_.../train_lgbm.py` — `exp015d_.../train_lowmem.py` が module-global を monkeypatch するため、
  `mcrs/ltr/` へ移動すると override が無音で no-op 化する。exp014 に凍結したまま残す。

## 6. results ledger（Current Best の機械化）

submit した variant ごとに `mcrs/experiments/results_ledger.jsonl` へ 1 行追記する
（1 行 = 1 JSON object。append-only）。列は experiment-plan.md の Current Best 表に対応:
`scope, exp, variant, split, ndcg, catalog_div, lexical_div, judge, composite, status, artifact, note`。
LB 未返却は数値 `null` + `status: "pending"`。status 語彙は
`gate | base | superseded | rejected | recovery_target | candidate | historical | pending`。
`scope` 列は experiment-plan.md の表ラベルをそのまま入れ、render 時の表示に使う。

表を更新するときは ledger に追記し、`python mcrs/experiments/render_ledger.py` の出力を
experiment-plan.md の Current Best 節へ貼り直す。**CV-LB 逆相関の根拠・Dev pass は必要条件のみ・
base vs gate 分離などの judgment prose は experiment-plan.md に手書きで残す**（ledger は数値表のみ機械化）。

## 7. 命名 / lifecycle

- experiment フォルダは `expNNN_short-slug` の連番（番号は project 全体で unique。subdir 横断で衝突させない）。
- 提出 slug は短く保つ（Codabench は submission ファイル名 64 字以上で error。ZIP root は `prediction.json`）。
- 採否は Blind A 提出で判断する（Dev 単独で cap / sampling を決めない。CV-LB 逆相関の前例あり）。
