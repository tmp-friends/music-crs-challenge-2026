#!/usr/bin/env python3
"""Music-CRS の prediction JSON / submission ZIP を提出制約に照らして検証する。

このスクリプトは format validation を毎回手書きする代わりに使う、決定的で高速な
チェッカーである。提出が無効化されやすいポイント（必須フィールド欠落、top-20 件数、
track_id 重複、ZIP の root 構成、ファイル名 63 字制限）を機械的に検出し、違反を全件
列挙したうえで終了コードで通過/失敗を返す。

catalog 照合（全 track_id が catalog に存在するか）は track 一覧の読み込みが重いため
既定では行わず、`--catalog` を指定したときのみ実行する。

使い方:
    python validate_submission.py path/to/prediction.json
    python validate_submission.py path/to/submission.zip
    python validate_submission.py path/to/prediction.json --catalog track_ids.txt

終了コード:
    0  すべての制約を満たす。
    1  1 件以上の制約違反、または入力が読めない。
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

# Codabench はファイル名 64 字以上で submit error になるため、ZIP 名はこの上限以下にする。
MAX_FILENAME_LEN = 63
# 提出仕様で予測リストは上位 20 件。--expect-k で上書きできる。
DEFAULT_EXPECT_K = 20
# 各 prediction record に必須のフィールド。evaluator はこれらを前提に読む。
REQUIRED_FIELDS = (
    "session_id",
    "user_id",
    "turn_number",
    "predicted_track_ids",
    "predicted_response",
)


def load_catalog_ids(path: Path) -> set[str]:
    """catalog の track_id 一覧を読み込んで集合で返す。

    txt（1 行 1 id）と json（id の配列、または record 配列で `track_id` キーを持つ形）の
    両対応とする。照合用なので順序は保持しない。

    Args:
        path: track_id 一覧ファイル（.txt もしくは .json）。

    Returns:
        track_id の集合。

    Raises:
        ValueError: json の構造が解釈できない場合。
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                # record 配列なら track_id キーから抽出する。
                return {str(rec["track_id"]) for rec in data if "track_id" in rec}
            return {str(x) for x in data}
        raise ValueError(f"Unsupported catalog json structure: {path}")
    # txt 扱い: 空行を除いて 1 行 1 id。
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_predictions(json_path: Path) -> list[dict]:
    """prediction JSON（record の配列）を読み込む。

    Args:
        json_path: prediction.json のパス。

    Returns:
        prediction record の配列。

    Raises:
        ValueError: トップレベルが配列でない場合。
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"prediction JSON のトップレベルは配列である必要がある: {json_path}"
        )
    return data


def validate_records(
    records: list[dict], expect_k: int, catalog_ids: set[str] | None
) -> list[str]:
    """prediction record 群を検証し、違反メッセージの一覧を返す。

    違反があっても例外で止めず、可能な限り全件を集めてから返すことで、1 回の実行で
    複数の問題をまとめて把握できるようにする。

    Args:
        records: prediction record の配列。
        expect_k: `predicted_track_ids` に期待する件数。
        catalog_ids: catalog の track_id 集合。None の場合は catalog 照合を行わない。

    Returns:
        違反メッセージの一覧（空なら制約をすべて満たす）。
    """
    errors: list[str] = []
    if not records:
        errors.append("prediction が空である（record が 0 件）。")
        return errors

    # session-turn の重複は評価で順位が不定になるため検出する。
    seen_keys: set[tuple] = set()

    for i, rec in enumerate(records):
        loc = f"record[{i}]"
        if not isinstance(rec, dict):
            errors.append(f"{loc}: dict ではない。")
            continue

        for field in REQUIRED_FIELDS:
            if field not in rec:
                errors.append(f"{loc}: 必須フィールド '{field}' が無い。")

        key = (rec.get("session_id"), rec.get("turn_number"))
        if key in seen_keys:
            errors.append(
                f"{loc}: session_id/turn_number の重複 {key}（評価で順位が不定になる）。"
            )
        seen_keys.add(key)

        tracks = rec.get("predicted_track_ids")
        if not isinstance(tracks, list):
            errors.append(f"{loc}: predicted_track_ids がリストでない。")
        else:
            if len(tracks) != expect_k:
                errors.append(
                    f"{loc}: predicted_track_ids が {len(tracks)} 件（期待 {expect_k} 件）。"
                )
            if len(set(map(str, tracks))) != len(tracks):
                errors.append(f"{loc}: predicted_track_ids 内に重複 track_id がある。")
            if catalog_ids is not None:
                missing = [str(t) for t in tracks if str(t) not in catalog_ids]
                if missing:
                    sample = ", ".join(missing[:5])
                    errors.append(
                        f"{loc}: catalog に存在しない track_id {len(missing)} 件 (例: {sample})。"
                    )

        resp = rec.get("predicted_response")
        if isinstance(resp, str) and resp.strip() == "":
            errors.append(f"{loc}: predicted_response が空文字。")

    return errors


def validate_zip(zip_path: Path) -> tuple[list[str], list[dict] | None]:
    """submission ZIP の構造とファイル名長を検証し、内部の prediction.json を読み込む。

    ZIP root 直下に prediction.json が無い、もしくはファイル名が長すぎると submit error に
    なるため、record 検証の前にここで弾く。展開ファイルを残さないよう、prediction.json は
    ディスクへ書き出さずメモリ上で読む。

    Args:
        zip_path: submission ZIP のパス。

    Returns:
        (違反メッセージ一覧, prediction record の配列 or None)。
        root の prediction.json を読めた場合のみ第 2 要素に record 配列を返す。
    """
    errors: list[str] = []
    if len(zip_path.name) > MAX_FILENAME_LEN:
        errors.append(
            f"ZIP ファイル名が {len(zip_path.name)} 字（上限 {MAX_FILENAME_LEN} 字）: {zip_path.name}"
        )

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # root 直下（サブディレクトリを含まない）に prediction.json があるかを見る。
        if "prediction.json" not in names:
            nested = [n for n in names if n.endswith("prediction.json")]
            hint = f"（{nested[0]} に発見、root 直下へ移す必要がある）" if nested else ""
            errors.append(f"ZIP root 直下に prediction.json が無い{hint}。")
            return errors, None

        # 展開ファイルを残さないよう bytes で読んでパースする。
        raw = zf.read("prediction.json").decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            errors.append("ZIP 内 prediction.json のトップレベルが配列でない。")
            return errors, None
        return errors, data


def main() -> int:
    """CLI エントリポイント。検証結果を表示し、終了コードを返す。"""
    parser = argparse.ArgumentParser(
        description="Music-CRS prediction JSON / submission ZIP の提出制約チェッカー。"
    )
    parser.add_argument("path", type=Path, help="prediction.json または submission.zip")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="track_id 一覧（txt: 1行1id / json: id配列）。指定時のみ catalog 照合を行う。",
    )
    parser.add_argument(
        "--expect-k",
        type=int,
        default=DEFAULT_EXPECT_K,
        help=f"predicted_track_ids の期待件数（default: {DEFAULT_EXPECT_K}）。",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"[ERROR] 入力が見つからない: {args.path}", file=sys.stderr)
        return 1

    catalog_ids: set[str] | None = None
    if args.catalog is not None:
        if not args.catalog.exists():
            print(f"[ERROR] catalog が見つからない: {args.catalog}", file=sys.stderr)
            return 1
        catalog_ids = load_catalog_ids(args.catalog)
        print(f"[info] catalog 照合: {len(catalog_ids)} track_ids を読み込み。")

    errors: list[str] = []

    if args.path.suffix.lower() == ".zip":
        zip_errors, records = validate_zip(args.path)
        errors.extend(zip_errors)
        if records is None:
            _report(args.path, errors, n_records=None)
            return 1
    else:
        try:
            records = load_predictions(args.path)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"[ERROR] prediction を読めない: {exc}", file=sys.stderr)
            return 1

    errors.extend(validate_records(records, args.expect_k, catalog_ids))
    _report(args.path, errors, n_records=len(records))
    return 1 if errors else 0


def _report(path: Path, errors: list[str], n_records: int | None) -> None:
    """検証結果を標準出力へ整形して表示する。

    Args:
        path: 検証対象のパス。
        errors: 違反メッセージ一覧。
        n_records: 読み込めた record 件数（読めなかった場合 None）。
    """
    print(f"\n=== validate_submission: {path} ===")
    if n_records is not None:
        print(f"records: {n_records}")
    if not errors:
        print("OK: 提出制約をすべて満たす。")
        return
    print(f"NG: {len(errors)} 件の違反")
    for msg in errors:
        print(f"  - {msg}")


if __name__ == "__main__":
    sys.exit(main())