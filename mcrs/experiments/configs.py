from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


EXPERIMENTS_DIR = Path(__file__).resolve().parent


def find_experiment_config_path(tid: str) -> Path:
    """Resolve a task id to its experiment-local config path."""
    config_name = f"{tid}.yaml"
    matches = sorted(EXPERIMENTS_DIR.glob(f"*/configs/{config_name}"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find config '{config_name}' under mcrs/experiments/*/configs."
        )
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches)
        raise ValueError(f"Config '{config_name}' is ambiguous: {joined}")
    return matches[0]


def _load_with_base(path: Path) -> Any:
    """``_base`` キーを再帰的に解決しながら config を読み込む。

    config に ``_base: <相対パス>`` があれば、まず親 config を読み込んでから子で上書き
    merge する（child 優先・list は置換）。``_base`` が無い config は ``OmegaConf.load`` の
    結果をそのまま返すため、既存の凍結 config は挙動が一切変わらない（後方互換）。

    Args:
        path: 読み込む config ファイルの絶対パス。

    Returns:
        ``_base`` を解決・除去済みの OmegaConf config。
    """
    child = OmegaConf.load(path)
    # ``_base`` は merge 前に必ず pop して crs_kwargs へ漏らさない（推論側へ未知キーを渡さない）。
    base_ref = child.pop("_base", None)
    if base_ref is None:
        return child
    # base パスは CWD ではなく child の置かれた dir 基準で解決する（実行場所非依存にするため）。
    base_path = (path.parent / str(base_ref)).resolve()
    base = _load_with_base(base_path)  # base 自身が更に _base を持つ多段継承も許容する。
    return OmegaConf.merge(base, child)


def load_experiment_config(tid: str) -> tuple[Any, str]:
    config_path = find_experiment_config_path(tid)
    return _load_with_base(config_path), str(config_path)

