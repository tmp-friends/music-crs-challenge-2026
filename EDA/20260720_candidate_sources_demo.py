"""train の実サンプル 1 session を使い、4 candidate source の入力→出力を具体的に再現する。"""
import ast
import re
from collections import Counter, defaultdict

from datasets import load_dataset


def first_of_liststr(value):
    """metadata の "['x']" 形式 list-string から先頭要素を取り出す。"""
    if value is None:
        return ""
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)) and parsed:
                return str(parsed[0])
            return ""
        except (ValueError, SyntaxError):
            return text
    return text


def liststr_all(value):
    if value is None:
        return []
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            return [text]
    return [text]


conv = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
meta = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")

id2m = {}
by_artist = defaultdict(list)
by_album = defaultdict(list)
for r in meta:
    tid = str(r["track_id"])
    m = {
        "track_name": str(r.get("track_name") or ""),
        "artist_name": first_of_liststr(r.get("artist_name")),
        "album_name": str(r.get("album_name") or ""),
        "tags": liststr_all(r.get("tag_list"))[:6],
        "popularity": r.get("popularity"),
    }
    id2m[tid] = m
    if m["artist_name"]:
        by_artist[m["artist_name"].lower()].append(tid)
    if m["album_name"]:
        by_album[m["album_name"].lower()].append(tid)


def tname(tid):
    m = id2m.get(tid)
    if not m:
        return f"<{tid[:8]} 不明>"
    return f"「{m['track_name']}」/ {m['artist_name']} (album: {m['album_name']})"


def turns_of(row):
    return [dict(t) for t in row["conversations"]]


row = conv[0]
turns = turns_of(row)
print("=" * 72)
print(f"DEMO train session: {row['session_id'][:13]}...  ({len(turns)} 個の turn 行)")
print("=" * 72)
for t in turns:
    role, c = t["role"], t["content"]
    if role == "music":
        print(f"  [t{t['turn_number']}] music    : {tname(c)}")
    else:
        print(f"  [t{t['turn_number']}] {role:9s}: {str(c)[:120]}")

music_seq = [t["content"] for t in turns if t["role"] == "music"]
gt = music_seq[-1]
hist = music_seq[:-1]
user_turns = [t for t in turns if t["role"] == "user"]
last_user = user_turns[-1]
print()
print(f"### ターゲット turn の GT      : {tname(gt)}")
print(f"### 観測済み music 履歴 {len(hist)} 件:")
for h in hist:
    print(f"      - {tname(h)}")
print(f"### 現在の user request: {str(last_user['content'])[:200]!r}")

# ---------------- 1. continuity ----------------
print()
print("=" * 72)
print("1) session-local continuity (cont_artist): 既出 music の artist を catalog 展開")
print("=" * 72)
seen_artists = []
for h in hist:
    a = id2m.get(h, {}).get("artist_name", "").lower()
    if a and a not in seen_artists:
        seen_artists.append(a)
print(f"  既出 artist: {seen_artists}")
cand = []
seen = set(hist)
for a in seen_artists:
    pool = sorted(
        by_artist[a],
        key=lambda t: (-(id2m[t].get("popularity") or 0), t),
    )
    for tid in pool:
        if tid not in seen:
            cand.append(tid)
            seen.add(tid)
print(f"  cont_artist 候補: {len(cand)} 件（同 artist catalog 全 track − within-session 既出）")
for tid in cand[:6]:
    mark = "   ← ★GT が候補に入った" if tid == gt else ""
    print(f"      - {tname(tid)}{mark}")
in_pool = gt in cand
rank = cand.index(gt) + 1 if in_pool else None
print(f"  → GT を含むか: {in_pool}" + (f"（popularity 順で rank {rank}）" if in_pool else ""))

# ---------------- 2. QNM ----------------
print()
print("=" * 72)
print("2) query-neighbor memory (qnm_cur): request 類似の train task の GT を伝播")
print("=" * 72)
STOP = {
    "the", "and", "you", "for", "that", "have", "with", "something", "anything",
    "want", "like", "can", "some", "more", "what", "this", "its", "your", "please",
    "play", "track", "song", "music", "artist", "artists", "but", "not", "really",
}


def toks(s):
    return set(re.findall(r"[a-z]{3,}", str(s).lower())) - STOP


q = toks(last_user["content"])
print(f"  query token(要約): {sorted(q)[:14]}")
scored = []
for i in range(1, len(conv)):
    ts2 = turns_of(conv[i])
    for j, t2 in enumerate(ts2):
        if t2["role"] != "user":
            continue
        ov = len(q & toks(t2["content"]))
        if ov >= 5:
            gt2 = next((x["content"] for x in ts2[j:] if x["role"] == "music"), None)
            if gt2:
                scored.append((ov, i, str(t2["content"]), gt2))
scored.sort(key=lambda x: -x[0])
for ov, i, txt, gt2 in scored[:3]:
    print(f"  [近傍 train task] overlap={ov} (session#{i})")
    print(f"      発話: {txt[:130]!r}")
    print(f"      → その task の GT を候補化: {tname(gt2)}")

# ---------------- 3. entity ----------------
print()
print("=" * 72)
print("3) entity candidates (ent_cur): user 発話中の catalog entity phrase を照合")
print("=" * 72)
low = str(last_user["content"]).lower()
hits = [a for a in by_artist if len(a) >= 5 and a in low]
print(f"  デモ session の現在発話から検出した artist entity: {hits}")
for a in hits:
    pool = by_artist[a]
    print(f"    \"{a}\" → catalog 同名 artist の全 track {len(pool)} 件を候補化（例）:")
    for tid in pool[:4]:
        mark = "   ← ★GT" if tid == gt else ""
        print(f"        - {tname(tid)}{mark}")

# ---------------- 4. transition ----------------
print()
print("=" * 72)
print("4) transition memory (trans_r1): train 全体の「直前 music → 次 GT」統計")
print("=" * 72)
last_seen = hist[-1]
last_artist = id2m.get(last_seen, {}).get("artist_name", "")
print(f"  直前 music entity: {tname(last_seen)}")
nxt_track = Counter()
nxt_artist = Counter()
for i in range(len(conv)):
    ts2 = turns_of(conv[i])
    ms = [t["content"] for t in ts2 if t["role"] == "music"]
    for a2, b2 in zip(ms, ms[1:]):
        if a2 == last_seen:
            nxt_track[b2] += 1
        if id2m.get(a2, {}).get("artist_name", "") == last_artist:
            nxt_artist[b2] += 1
print(f"  train 中で「この track の直後に来た GT」({sum(nxt_track.values())} 遷移) 上位:")
for tid, c in nxt_track.most_common(5):
    mark = "   ← ★GT" if tid == gt else ""
    print(f"      - {c:3d} 回: {tname(tid)}{mark}")
print(f"  train 中で「artist {last_artist} の直後に来た GT」({sum(nxt_artist.values())} 遷移) 上位:")
for tid, c in nxt_artist.most_common(5):
    mark = "   ← ★GT" if tid == gt else ""
    print(f"      - {c:3d} 回: {tname(tid)}{mark}")
