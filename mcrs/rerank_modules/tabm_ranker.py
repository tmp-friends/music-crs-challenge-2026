"""TabM（parameter-efficient MLP ensemble）を rerank の pointwise scorer として使う wrapper。

GBDT 3 種（LightGBM/XGBoost/CatBoost）は同一関数族なので、blend しても主に variance 低減
にしかならない（advisor 指摘）。TabM は MLP ベースで **唯一 GBDT と別関数族**であり、同じ
rank 特徴量から GBDT が取り逃す順序信号を拾える可能性がある「optional だが本命の probe」。

learning-to-rank の listwise loss でなく、**binary relevance を pos_weight 付き
BCEWithLogitsLoss で学習する pointwise scorer** として使う（候補プール・特徴量は GBDT と
完全共有）。出力スコアは family blend 側で group 内 z-score 正規化されるので、絶対スケールは
問わない。TabM は内部で k 本の MLP ensemble を並列学習し、推論時は k head の sigmoid 平均を
row スコアにする。

正例率が極端に低い（GT は group あたり高々 1 件、全体 ~0.15%）ため pos_weight=neg/pos を
既定にし、MLP が全 0 に潰れるのを防ぐ。特徴量は scale がまちまち（inv_rank∈[0,1] /
clipped_rank≲101 / turn_number など）なので train 統計で z-score 標準化してから入れる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TabMConfig:
    """TabM ranker の学習設定。"""

    n_blocks: int = 3
    d_block: int = 512
    dropout: float = 0.1
    k: int = 32
    arch_type: str = "tabm"
    epochs: int = 6
    batch_size: int = 16384
    lr: float = 2e-3
    weight_decay: float = 1e-4
    pos_weight: float | None = None  # None で neg/pos から自動算出


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """列ごとの平均と標準偏差を返す（std は 0 割回避のため下限 clamp）。

    Args:
        x: 特徴量行列（n_rows × n_features）。

    Returns:
        (mean, std)。いずれも長さ n_features。
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_tabm_ranker(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    config: TabMConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    """TabM を pointwise relevance scorer として学習する。

    Args:
        x: 特徴量行列（n_rows × n_features）float。
        y: binary relevance ラベル。
        seed: 乱数 seed（torch / numpy）。
        config: 学習設定。
        device: ``"cuda"`` か ``"cpu"``。

    Returns:
        ``{"model", "mean", "std", "device"}``。``predict_tabm`` に渡す。
    """
    import torch
    import tabm as tabm_lib

    torch.manual_seed(seed)
    np.random.seed(seed)

    mean, std = _standardize_fit(x)
    n_features = x.shape[1]
    # 全 train を GPU に常駐させる（3M×78 float32 ≈ 0.96GB、24GB に収まる）。CPU↔GPU 転送が
    # ボトルネックにならないようにする。
    xt = torch.from_numpy(((x - mean) / std).astype(np.float32)).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)

    pos = float(yt.sum().item())
    neg = float(yt.numel() - pos)
    pos_weight_value = config.pos_weight if config.pos_weight is not None else (neg / max(pos, 1.0))
    pos_weight = torch.tensor([pos_weight_value], device=device)

    model = tabm_lib.TabM(
        n_num_features=n_features,
        d_out=1,
        n_blocks=config.n_blocks,
        d_block=config.d_block,
        dropout=config.dropout,
        arch_type=config.arch_type,
        k=config.k,
        start_scaling_init="random-signs",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_rows = xt.shape[0]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.train()
    for _ in range(config.epochs):
        perm = torch.randperm(n_rows, generator=generator, device=device)
        for start in range(0, n_rows, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = xt[idx]
            # target を k head へ broadcast（TabM は (B, k, 1) を出す。全 head 同一 target で学習）。
            yb = yt[idx].view(-1, 1, 1)
            logits = model(x_num=xb)
            loss = loss_fn(logits, yb.expand_as(logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {"model": model, "mean": mean, "std": std, "device": device}


def predict_tabm(state: dict[str, Any], x: np.ndarray, *, batch_size: int = 65536) -> np.ndarray:
    """TabM state で row スコア（k head の sigmoid 平均）を返す。

    Args:
        state: ``fit_tabm_ranker`` の返り値。
        x: 採点する特徴量行列。
        batch_size: 推論 minibatch。

    Returns:
        row ごとの平均確率スコア配列。
    """
    import torch

    model = state["model"]
    device = state["device"]
    mean, std = state["mean"], state["std"]
    xs = ((x - mean) / std).astype(np.float32)
    out = np.empty(xs.shape[0], dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, xs.shape[0], batch_size):
            xb = torch.from_numpy(xs[start : start + batch_size]).to(device)
            logits = model(x_num=xb)  # (B, k, 1)
            prob = torch.sigmoid(logits).mean(dim=1).squeeze(-1)  # k head 平均
            out[start : start + batch_size] = prob.detach().cpu().numpy()
    return out
