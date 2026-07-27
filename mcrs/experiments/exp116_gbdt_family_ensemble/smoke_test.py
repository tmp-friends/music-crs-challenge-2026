"""exp116 ensemble の小規模 smoke。先頭 N task で全 family 学習〜blend〜採点を通す。

3M 行の full 学習を回さずに、新規コード経路（XGBoost / CatBoost wrapper、group 内
z-score、family blend、score_to_predictions）の shape と挙動だけを高速に検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mcrs.experiments.exp022_continuity_lgbm.rank_source_lgbm import (
    load_dev_labels, load_records, ndcg, records_to_map, score_to_predictions,
)
from mcrs.experiments.exp027_wide_source_lgbm.rank_source_lgbm_mixed_topk import (
    build_features, feature_names, parse_mixed_source,
)
from mcrs.rerank_modules.multi_family import (
    blend_family_scores, catboost_params, fit_catboost_ranker, fit_xgb_ranker,
    groupwise_zscore, predict_catboost, predict_xgb, xgb_params,
)

D21 = "mcrs/experiments/exp021_candidate_fusion/results"
D23 = "mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100"
D24 = "mcrs/experiments/exp024_transition_memory/results/transition_memory_top100"
OUT = "mcrs/experiments/exp090_cross_session_source/results"
PRIMARY = "mcrs/experiments/exp015_candidate_rules_ablation/results/top500/B_exact_split/lgbm_B_ll_top500_dev_predictions.json"
SOURCES = [
    f"cont_aat={D21}/continuity_candidates/dev_tracks_exp021_cont_album_artist_tag.json",
    f"cont_aa={D21}/continuity_candidates/dev_tracks_exp021_cont_album_artist.json",
    f"cont_album={D21}/continuity_candidates/dev_tracks_exp021_cont_album.json",
    f"cont_artist={D21}/continuity_candidates/dev_tracks_exp021_cont_artist.json",
    f"qnm_rp={D21}/query_neighbor_memory/dev_tracks_exp021_qnm_recent_profile.json",
    f"qnm_hist={D21}/query_neighbor_memory/dev_tracks_exp021_qnm_history_music.json",
    f"qnm_cur={D21}/query_neighbor_memory/dev_tracks_exp021_qnm_current.json",
    f"ent_cur={D23}/dev_tracks_exp023_entity_current.json",
    f"ent_tight={D23}/dev_tracks_exp023_entity_current_tight.json",
    f"ent_r2={D23}/dev_tracks_exp023_entity_recent2.json",
    f"ent_r4={D23}/dev_tracks_exp023_entity_recent4.json",
    f"trans_r1={D24}/dev_tracks_exp024_transition_recent1.json",
    f"cross_session:50={OUT}/nopad/dev_tracks_cross_session.json",
]


def main() -> None:
    n = 400
    specs = [parse_mixed_source(s, default_topk=100) for s in SOURCES]
    names = feature_names(specs)
    print("n_features:", len(names))
    dev_keys, labels = load_dev_labels("talkpl-ai/TalkPlayData-Challenge-Dataset", "test")
    keys = dev_keys[:n]
    primary = records_to_map(load_records(Path(PRIMARY), topk=20))
    sources = {s.name: records_to_map(load_records(s.path, topk=s.topk, allow_short=True)) for s in specs}
    bundle = build_features(keys=keys, primary=primary, sources=sources,
                            source_specs=specs, labels=labels, exclude_map=None)
    print("rows:", bundle.x.shape, "groups:", len(bundle.groups), "positives:", bundle.candidate_positive_count)

    # z-score 健全性: 各 group の平均≈0
    xgbm = fit_xgb_ranker(bundle.x, bundle.y, bundle.groups,
                          params=xgb_params(seed=1, num_threads=8, device="cuda"),
                          num_boost_round=60, feature_name=names)
    xs = predict_xgb(xgbm, bundle.x, feature_name=names)
    cbm = fit_catboost_ranker(bundle.x, bundle.y, bundle.groups,
                              params={**catboost_params(seed=1, num_threads=8, task_type="GPU"), "iterations": 80},
                              feature_name=names)
    cs = predict_catboost(cbm, bundle.x)
    print("xgb score shape:", xs.shape, "catboost score shape:", cs.shape)

    z = groupwise_zscore(xs, bundle.groups)
    # group ごとの平均がほぼ 0 か確認
    off = np.cumsum([0] + bundle.groups)
    means = [z[off[i]:off[i+1]].mean() for i in range(len(bundle.groups)) if bundle.groups[i] > 1]
    print("zscore per-group mean abs max:", float(np.max(np.abs(means))) if means else 0.0)

    fam = {"xgb": xs, "catboost": cs}
    blended = blend_family_scores(fam, bundle.groups)
    preds = score_to_predictions(bundle, keys, primary, scores=blended, protect_top=0, topk=20)
    print("blend ndcg (smoke, tiny train):", ndcg(preds, keys, labels))
    # 単一 family の z-score は順位不変 → standalone ndcg == raw ndcg
    p_raw = score_to_predictions(bundle, keys, primary, scores=xs, protect_top=0, topk=20)
    p_z = score_to_predictions(bundle, keys, primary, scores=groupwise_zscore(xs, bundle.groups), protect_top=0, topk=20)
    print("xgb raw==zscore order preserved:", all(p_raw[k] == p_z[k] for k in keys))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
