# 再聴傾向 EDA サマリー

## 実行条件
- run scope: full run
- analyzed splits: train, dev_test
- metadata join: `input/track-metadata/all_tracks`
- missing metadata rows: 0
- output scope: `EDA/` only。`mcrs/experiments/` は更新していない。

## 主要結果
- 同一 session 内の exact track 再聴率は train=0.00%、dev_test=0.00%。
- 同一 user の過去 session 由来の exact track 再聴率は全 event 母数で train=4.70%、dev_test=3.50%。過去 session が存在する event に限ると train=10.82%、dev_test=7.00%。
- 同一 session 内の artist 継続率は train=56.88%、dev_test=40.29%。
- 同一 user の過去 session 由来の artist 継続率は train=13.15%、dev_test=12.64%。

## 解釈
- 会話内では同じ track をそのまま再提示する傾向は観測されず、同じ artist / album / tag 方向に寄せる傾向が強い。
- user の過去 session まで広げると exact track 再登場が観測されるため、再聴 feature は session-local より user-history 側で効く可能性がある。
- tag continuity は非常に広く当たりやすいため、単独の強い signal というより mood/genre continuity の補助指標として扱うのが安全。

## Actionability 上位
- train / combined / prior_artist@100: recall=0.5707, active_rate=0.9293
- train / session / prior_artist@100: recall=0.5548, active_rate=0.8750
- train / combined / prior_artist@50: recall=0.5070, active_rate=0.9293
- train / session / prior_artist@50: recall=0.4965, active_rate=0.8750
- train / combined / prior_album@100: recall=0.4590, active_rate=0.9293
- train / combined / prior_album@50: recall=0.4458, active_rate=0.9293
- train / session / prior_album@100: recall=0.4269, active_rate=0.8750
- train / session / prior_album@50: recall=0.4242, active_rate=0.8750

## 生成物
- `EDA/tables/relisten_rates_by_split.csv`
- `EDA/tables/relisten_rates_by_turn.csv`
- `EDA/tables/relisten_rates_by_goal.csv`
- `EDA/tables/relisten_rates_by_user_split.csv`
- `EDA/tables/relisten_actionability.csv`
- `EDA/figures/relisten_by_turn.png`
- `EDA/figures/relisten_by_goal.png`
- `EDA/figures/relisten_history_size.png`
