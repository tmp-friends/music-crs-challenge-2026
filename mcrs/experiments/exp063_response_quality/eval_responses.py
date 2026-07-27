"""response variant をローカル指標で比較する（ranking 非依存・ground truth 不要の軸のみ）。

judge は Blind A 提出でしか測れないが、lexical_diversity(Distinct-2) と apology/opening 重複は
ローカルで決定的に測れる。本スクリプトはその測定と、ranking が baseline と一致しているか
（=nDCG/catalog_div を壊していないか）の検査を行う。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter


def tokenize(text):
    """小文字化して語トークン列にする（Distinct-2 用の素朴な tokenizer）。"""
    return re.findall(r"\w+", text.lower())


def distinct2(texts):
    """全 response にわたる unique bigram / total bigram（公式 lexical_diversity の近似）。"""
    total = 0
    uniq = set()
    for t in texts:
        toks = tokenize(t)
        bi = list(zip(toks, toks[1:]))
        total += len(bi)
        uniq.update(bi)
    return (len(uniq) / total if total else 0.0), total


def opening_repeat(texts, n=5):
    """先頭 n-gram の重複度（opening 定型化 = Distinct-2 killer の指標）。"""
    openers = Counter(" ".join(tokenize(t)[:n]) for t in texts)
    repeated = sum(c for c in openers.values() if c > 1)
    return openers.most_common(6), repeated


def apology_count(texts):
    """謝罪/不一致を示す response 数（judge を自ら下げる失敗モード）。"""
    pat = re.compile(r"apolog|i'?m sorry|mismatch|doesn'?t match|does not match|isn'?t a |not a (good )?match|unfortunately")
    hits = [t for t in texts if pat.search(t.lower())]
    return len(hits), hits[:3]


def main(args):
    """variant JSON 群を読み、baseline と並べて指標表を出す。"""
    base = None
    if args.baseline:
        base = {(r["session_id"], r["turn_number"]): r for r in json.load(open(args.baseline))}

    print(f"{'variant':<34}{'n':>4}{'distinct2':>11}{'apology':>9}{'open_rep':>10}{'wlen':>7}{'rank_ok':>9}")
    for path in args.inputs:
        recs = json.load(open(path))
        resps = [r["predicted_response"] for r in recs]
        d2, _ = distinct2(resps)
        apo, _ = apology_count(resps)
        _, rep = opening_repeat(resps)
        wlen = sum(len(tokenize(t)) for t in resps) / len(resps)
        rank_ok = "-"
        empty = sum(1 for t in resps if not t.strip())
        if base is not None:
            ok = all(
                r["predicted_track_ids"] == base[(r["session_id"], r["turn_number"])]["predicted_track_ids"]
                for r in recs
                if (r["session_id"], r["turn_number"]) in base
            )
            rank_ok = "YES" if ok else "NO!"
        name = path.split("/")[-1].replace(".json", "")[:33]
        flag = f" empty={empty}" if empty else ""
        print(f"{name:<34}{len(recs):>4}{d2:>11.4f}{apo:>9}{rep:>10}{wlen:>7.0f}{rank_ok:>9}{flag}")

    if args.verbose and args.inputs:
        recs = json.load(open(args.inputs[-1]))
        print(f"\n=== opening 5-grams ({args.inputs[-1].split('/')[-1]}) ===")
        mc, _ = opening_repeat([r["predicted_response"] for r in recs])
        for k, v in mc:
            print(f"  {v:2d}x  {k}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="比較する prediction JSON 群")
    p.add_argument("--baseline", help="ranking 保全チェック用の基準 JSON（exp058）")
    p.add_argument("--verbose", action="store_true")
    main(p.parse_args())
