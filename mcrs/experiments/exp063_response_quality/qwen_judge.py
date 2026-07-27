"""cross-family judge: local Qwen で calibration set を採点し、Claude proxy の self-preference を点検する。

Claude が Claude 文を採点する proxy は self-preference で上方バイアスする（校正で実証済）。本スクリプトは
別 family の Qwen に同じ匿名応答を 1-5 採点させ、(a) 実 Gemini judge 既知 anchor との相関、(b) V-Claude が
anchor より上に出るか、を見る。Qwen が「Claude を特別扱いする理由」は無いので、Qwen も V-Claude を上位に
置くなら judge 改善は self-preference では説明できない（robust）。
"""

from __future__ import annotations

import argparse
import json
import re

import torch

from mcrs.lm_modules.llama import LLAMA_MODEL

JUDGE_SYS = (
    "You are a strict evaluator of music-recommendation chat replies. Rate the reply's OVERALL quality "
    "(how personalized to the user's request AND how clear/specific/convincing the explanation of why the "
    "track fits) on an INTEGER scale 1 to 5, where 1=generic/empty/hedging and 5=specific, confident, "
    "personalized. Reply with ONLY a single digit 1-5, nothing else."
)


def build_prompt(task, resp):
    """1 応答の採点 prompt（user_query + 応答 → 数字1つ）。"""
    hist = "\n".join(task.get("history", [])[-4:])
    return (
        f"User's request: {task['user_query']}\n"
        + (f"Recent context:\n{hist}\n" if hist else "")
        + f"\nAssistant reply to rate:\n{resp}\n\nScore (1-5):"
    )


def main(args):
    """calib_judge_input の全 (task,label) を Qwen で採点し、label 別平均を出す。"""
    data = json.load(open(args.input, encoding="utf-8"))
    lm = LLAMA_MODEL(args.lm_type, "cuda", "sdpa", torch.bfloat16)
    labels = list(data[0]["responses"].keys())

    sums = {lab: [] for lab in labels}
    # batch 化のため (label,task) を平らに展開
    sys_prompts, histories, recs, keys = [], [], [], []
    for t in data:
        for lab in labels:
            sys_prompts.append(JUDGE_SYS)
            histories.append([{"role": "user", "content": build_prompt(t, t["responses"][lab])}])
            recs.append("")  # recommend_item は使わない（応答テキストのみ採点）
            keys.append(lab)

    bs = args.batch_size
    outs = []
    for i in range(0, len(sys_prompts), bs):
        outs.extend(lm.batch_response_generation(
            sys_prompts[i:i+bs], histories[i:i+bs], recs[i:i+bs], max_new_tokens=8))
        print(f"[qjudge] {min(i+bs,len(sys_prompts))}/{len(sys_prompts)}", flush=True)

    for lab, o in zip(keys, outs):
        m = re.search(r"[1-5]", o)
        if m:
            sums[lab].append(int(m.group()))
    means = {lab: (sum(v)/len(v) if v else None) for lab, v in sums.items()}
    json.dump(means, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("=== Qwen cross-family judge means by label ===")
    for lab, mu in means.items():
        print(f"  {lab}: {mu}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="mcrs/experiments/exp063_response_quality/results/calib_judge_input.json")
    p.add_argument("--output", default="mcrs/experiments/exp063_response_quality/results/qwen_judge_means.json")
    p.add_argument("--lm_type", default="Qwen/Qwen3.5-4B")
    p.add_argument("--batch_size", type=int, default=14)
    main(p.parse_args())
