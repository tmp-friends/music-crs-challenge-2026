"""``results_ledger.jsonl`` から Current Best の markdown 表を生成する。

experiment-plan.md の「Current Best」表は手で同期され続けてきたが、数値表部分は機械化できる。
本スクリプトは ledger（append-only JSONL）を読み、experiment-plan.md と同一ヘッダの markdown 表を
**標準出力に print するだけ**で、prose（CV-LB 逆相関の根拠・base vs gate 分離などの judgment）は
experiment-plan.md に手書きのまま残す。表を更新したいときは ledger に 1 行追記し、本スクリプトの
出力を該当節へ貼り直す運用にする。

Usage:
    python mcrs/experiments/render_ledger.py [--ledger path/to/results_ledger.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# experiment-plan.md「Current Best」表と完全一致させるヘッダ。順序・文言を変えると貼り替え時に差分が出る。
HEADER = (
    "| Scope | Experiment | ndcg@20 | catalog_diversity | lexical_diversity "
    "| llm_judge_score | composite_score | Note |"
)
SEPARATOR = "|---|---|---:|---:|---:|---:|---:|---|"

DEFAULT_LEDGER = Path(__file__).resolve().parent / "results_ledger.jsonl"


def _fmt(value: Any) -> str:
    """数値セルを 4 桁固定小数で整形する。

    ``None``（LB 未返却の pending 行）は ``pending`` と表示し、空セルで誤魚を招かないようにする。

    Args:
        value: ndcg / catalog_div / lexical_div / judge / composite のいずれか。

    Returns:
        markdown セルへ埋める文字列。
    """
    if value is None:
        return "pending"
    return f"{float(value):.4f}"


def load_rows(ledger_path: Path) -> list[dict[str, Any]]:
    """JSONL を 1 行 1 オブジェクトとして読み込む（ファイル出現順を保持）。

    Args:
        ledger_path: ``results_ledger.jsonl`` のパス。

    Returns:
        ledger 行の dict リスト。表の行順は JSONL の並び順をそのまま使う。
    """
    rows: list[dict[str, Any]] = []
    with ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def render_table(rows: list[dict[str, Any]]) -> str:
    """ledger 行から Current Best の markdown 表文字列を組み立てる。

    前方互換のため各キーは ``.get`` で参照し、将来 ledger に列が増減しても落ちないようにする。

    Args:
        rows: :func:`load_rows` の戻り値。

    Returns:
        ヘッダ + 区切り + データ行を改行連結した markdown 表。
    """
    lines = [HEADER, SEPARATOR]
    for row in rows:
        experiment = f"{row.get('exp', '')} {row.get('variant', '')}".strip()
        lines.append(
            "| {scope} | {experiment} | {ndcg} | {catalog} | {lexical} "
            "| {judge} | {composite} | {note} |".format(
                scope=row.get("scope", ""),
                experiment=experiment,
                ndcg=_fmt(row.get("ndcg")),
                catalog=_fmt(row.get("catalog_div")),
                lexical=_fmt(row.get("lexical_div")),
                judge=_fmt(row.get("judge")),
                composite=_fmt(row.get("composite")),
                note=row.get("note", ""),
            )
        )
    return "\n".join(lines)


def main() -> None:
    """CLI エントリポイント。ledger を読み、Current Best 表を stdout へ出力する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="results_ledger.jsonl のパス（既定: 本スクリプトと同階層）",
    )
    args = parser.parse_args()
    rows = load_rows(args.ledger)
    print(render_table(rows))


if __name__ == "__main__":
    main()
