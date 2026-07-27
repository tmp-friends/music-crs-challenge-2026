"""同一ユーザの「別セッションで提示済みの track」を exact-track 候補 source 化する（leak-safe）。

動機（exp073 調査）:
    within-session の既出 music は次 turn の正解と絶対一致しない（Dev 0/7000）が、**cross-session**
    では同一ユーザの別セッションで提示済みの track が次の正解になる確率が ~10%（Dev 実測）。exp058 の
    recall@20 が取りこぼした GT のうち、cross-session pool で回収可能なのは Dev で 305 件（+3.81pt）。
    現 no-dense pool は cross-session シグナルを未活用。これを exact-track の小cap source として足す。

設計（user 承認: exact-track のみでまず測る）:
    各予測タスク (session, turn) に対し、同一ユーザの **別セッション** で提示された track を候補化する。
    artist/album 展開は density-precision の罠（exp060 crash class）なので含めない（次段で feature 化検討）。
    track は「ユーザの別セッション群での出現セッション数」降順で並べ、cap する（高頻度＝再来しやすい）。

リーク禁止（重要）:
    - **leave-current-session-out**: cross-pool から現タスク自身のセッション(`s2==s`)を完全除外（学習時 leak 防止）。
    - **隠し最終 turn を使わない**: Blind A の各セッションは最終 turn が hidden（＝別セッションの正解）。
      別 Blind A セッションからは visible turn（最終 turn 未満）の music のみ採る。Dev は全 turn 公開なので全 music 可。
    - 時系列制限はしない: 本コンペはバッチ予測（全セッション同時提供）かつモデルに時系列特徴が無いため、
      未来セッション概念は deployment 上存在しない（user 指摘）。leak の硬制約は隠し最終 turn のみ。
    - future turn / conversation_goal / goal_progress_assessments は不使用。
"""

from __future__ import annotations

import argparse
import collections
import json
import os

from datasets import load_dataset


def session_music(convs, drop_final_turn: bool):
    """1 セッションの music track_id を出現順で返す。

    Args:
        convs: conversations（turn 列）。
        drop_final_turn: True なら最大 turn_number の music を除外（Blind の隠し最終 turn 対策）。

    Returns:
        track_id のリスト（出現順）。
    """
    musics = [
        (int(m.get("turn_number", 0)), str(m.get("content", "")))
        for m in convs
        if m.get("role") == "music"
    ]
    if not musics:
        return []
    if drop_final_turn:
        max_turn = max(t for t, _ in musics)
        musics = [(t, c) for t, c in musics if t < max_turn]
    return [c for _, c in musics]


def build_user_sessions(db, drop_final_turn: bool):
    """user_id -> {session_id -> [track_id,...]} を構築する。

    空 user_id（Blind B の cold start = user_id 未提供）は pool に入れない。
    入れてしまうと全 cold セッションが同一の ``""``/``"None"`` キーに集約され、cross_pool が
    「別ユーザの track」を cross-session 候補として混入させる（leak-safe でない noise）。除外すれば
    cold は cross-pool 無しになる（正しい挙動）。Dev / Blind A は user_id が常に非空（test_cold でも
    user_id 自体は存在）なので、この除外は no-op で従来出力と byte-identical を維持する。
    """
    user_sessions = collections.defaultdict(dict)
    for it in db:
        raw_uid = it["user_id"]
        # None・空文字・空白のみは「ユーザ不明」とみなし pool 対象外にする。
        if raw_uid is None or not str(raw_uid).strip():
            continue
        u = str(raw_uid); s = str(it["session_id"])
        user_sessions[u][s] = session_music(it["conversations"], drop_final_turn=drop_final_turn)
    return user_sessions


def cross_pool(user, current_session, *pools, cap):
    """同一ユーザの別セッション群から exact-track の cross-pool を作る（出現セッション数降順 cap）。

    Args:
        user: 対象 user_id。
        current_session: 除外する現タスクのセッション id（leave-current-session-out）。
        pools: user_id -> {session -> [tracks]} の dict を可変個（例: 別 Blind A 群 + Dev 群）。
        cap: 返す最大 track 数。

    Returns:
        cross-session track_id リスト（高頻度順、cap 済）。
    """
    freq = collections.Counter()
    order = {}
    idx = 0
    for pool in pools:
        for s2, tracks in pool.get(user, {}).items():
            if s2 == current_session:  # leave-current-session-out（学習 leak 防止）
                continue
            seen_in_sess = set()
            for tid in tracks:
                if not tid:
                    continue
                if tid not in seen_in_sess:
                    freq[tid] += 1          # 出現「セッション数」をカウント（同一セッション内重複は 1）
                    seen_in_sess.add(tid)
                if tid not in order:
                    order[tid] = idx; idx += 1
    # 出現セッション数降順 → 初出順で安定ソート。
    ranked = sorted(freq.keys(), key=lambda t: (-freq[t], order[t]))
    return ranked[:cap]


def popularity_order(item_db_name, pad_to):
    """catalog の popularity 降順 track_id を pad_to 件返す（sparse source の共有 padding 用）。

    ranker は source 行を `len==topk` 厳密に要求するため、真の cross tracks の後ろを共有 popularity で
    埋める（goal_artist 等の既存 sparse source と同一パターン）。真の cross tracks を先頭に置くので
    rank feature 1..N は真候補・N+1.. は popularity となり、padding は全行共有の bounded 集合に収まる。
    """
    from mcrs.db_item import MusicCatalogDB
    db = MusicCatalogDB(item_db_name, ["all_tracks"], ["track_name"])
    def pop(md):
        v = md.get("popularity")
        if isinstance(v, list):
            v = v[0] if v else 0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    ranked = sorted(db.metadata_dict.items(), key=lambda kv: -pop(kv[1]))
    return [tid for tid, _ in ranked[:pad_to]]


def emit_rows(db, pools_for_task, *, drop_final_turn_for_targets, cap, pad_to, pop_order):
    """各予測タスクに cross-pool 候補 row を作る（source JSON 形式）。

    Args:
        db: 対象データセット（Dev or Blind A）。
        pools_for_task: cross-pool 構築に使う pool の list（順に union）。
        drop_final_turn_for_targets: Blind なら True（最終 turn = 予測対象、それ自体は target で source には出さない）。
        cap: cross-pool cap。

    Returns:
        {session_id, user_id, turn_number, predicted_track_ids, predicted_response} の list。
    """
    rows = []
    for it in db:
        u = str(it["user_id"]); s = str(it["session_id"])
        convs = list(it["conversations"])
        if drop_final_turn_for_targets:
            # Blind: 予測対象は常に会話の最終 turn（最終 turn の music は hidden なので
            # music role からは取れない。turn-1 セッションも漏らさず 1 タスク出す）。
            target_turns = [int(convs[-1]["turn_number"])]
        else:
            # Dev: 全 music turn が予測対象（すべて公開）。
            musics = [int(m.get("turn_number", 0)) for m in convs if m.get("role") == "music"]
            if not musics:
                continue
            target_turns = sorted(set(musics))
        pool = cross_pool(u, s, *pools_for_task, cap=cap)
        if pop_order is None:
            # padding なし: 真の cross track のみ（no-cross 行は空）。ranker 側 --allow_short_sources 前提。
            padded = list(pool)
        else:
            # 真の cross tracks を先頭に置き、共有 popularity で pad_to まで埋める（dedup）。
            seen = set(pool)
            padded = list(pool)
            for tid in pop_order:
                if len(padded) >= pad_to:
                    break
                if tid not in seen:
                    padded.append(tid); seen.add(tid)
            padded = padded[:pad_to]
        for t in target_turns:
            rows.append({
                "session_id": s,
                "user_id": u,
                "turn_number": t,
                "predicted_track_ids": padded,  # cross-pool(真) + 共有popularity padding を固定長で
                "predicted_response": "",
                "n_cross": len(pool),  # 診断用: 真の cross tracks 数（padding 前）
            })
    return rows


def main(args):
    """Dev / Blind A の cross-session exact-track source JSON を書く。"""
    dev = load_dataset(args.dev_dataset_name, split=args.split)
    blind = load_dataset(args.blind_dataset_name, split=args.split)

    # Dev は全 turn 公開 → 別セッション pool は全 music。
    dev_pool = build_user_sessions(dev, drop_final_turn=False)
    # Blind A は各セッション最終 turn が hidden → 別セッション pool は visible music のみ。
    blind_pool = build_user_sessions(blind, drop_final_turn=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # sparse source を固定長にするための共有 popularity padding（leak なし: catalog popularity のみ）。
    # --no_pad 時は padding せず真の cross track のみ出力（ranker は --allow_short_sources で受ける）。
    pop_order = None if args.no_pad else popularity_order(args.item_db_name, args.pad_to)

    # Dev タスク: cross-pool は user の別 Dev セッションのみ（Dev の世界）。全 music turn が予測対象。
    dev_rows = emit_rows(dev, [dev_pool], drop_final_turn_for_targets=False, cap=args.cap, pad_to=args.pad_to, pop_order=pop_order)
    dev_path = os.path.join(args.output_dir, "dev_tracks_cross_session.json")
    json.dump(dev_rows, open(dev_path, "w", encoding="utf-8"), ensure_ascii=False)

    # Blind A タスク: cross-pool は user の別 Blind A セッション(visible) ∪ Dev セッション(全)。最終 turn のみ予測対象。
    blind_rows = emit_rows(blind, [blind_pool, dev_pool], drop_final_turn_for_targets=True, cap=args.cap, pad_to=args.pad_to, pop_order=pop_order)
    blind_path = os.path.join(args.output_dir, "blindA_tracks_cross_session.json")
    json.dump(blind_rows, open(blind_path, "w", encoding="utf-8"), ensure_ascii=False)

    # 簡易統計（n_cross = padding 前の真 cross tracks 数）。
    dev_nz = [r["n_cross"] for r in dev_rows if r["n_cross"]]
    blind_nz = [r["n_cross"] for r in blind_rows if r["n_cross"]]
    print(f"[cross-session] Dev rows {len(dev_rows)} (true-cross nonempty {len(dev_nz)}, avg {sum(dev_nz)/max(len(dev_nz),1):.1f}, pad_to {args.pad_to}) -> {dev_path}")
    print(f"[cross-session] Blind rows {len(blind_rows)} (true-cross nonempty {len(blind_nz)}, avg {sum(blind_nz)/max(len(blind_nz),1):.1f}, pad_to {args.pad_to}) -> {blind_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dev_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--blind_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    p.add_argument("--split", default="test")
    p.add_argument("--cap", type=int, default=50, help="cross-pool（真候補）の最大 track 数")
    p.add_argument("--pad_to", type=int, default=100, help="ranker の len==topk 要求を満たす固定長（popularity で pad）")
    p.add_argument("--no_pad", action="store_true", help="popularity padding せず真の cross track のみ出力（ranker は --allow_short_sources 必須）")
    p.add_argument("--item_db_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--output_dir", default="mcrs/experiments/exp090_cross_session_source/results")
    main(p.parse_args())
