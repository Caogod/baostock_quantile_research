"""实验公共工具：加载数据 + 特征对齐 + 时序 CV 评估。

供 experiments/ 下的独立实验脚本复用，与主线 train_model.py 保持口径一致：

- **数据来源**：明细/样本库路径不再硬编码（此前写死
  `backtest_detail_2025-11-12_2026-09-30.csv`，换区间就得改代码，而且该文件被别的
  回测覆盖后会静默用到过期数据）。统一走 core.datasets 解析：
  显式参数 > 环境变量 SQ_DETAIL_PATH / SQ_LIBRARY_PATH > 自动发现最完整的明细。
- **持仓周期**：明细同时含 holding_days=3/5/10 的样本，默认不混用——由
  `resolve_holdings()` 按各策略在 config/strategies.yaml 里声明的 holding 拆分
  （缺失则取该策略样本最多的周期），也可用 `holding=` 统一指定。
- **标签**：相对沪深300同期超额收益 > 0。
- **CV**：TimeSeriesSplit 5 折（时序，杜绝未来信息泄漏）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.datasets import (  # noqa: E402
    enabled_strategies,
    resolve_detail_path,
    resolve_holdings,
    resolve_library_path,
)
from core.features import precompute_features  # noqa: E402
from core.market import benchmark_returns_for_trades  # noqa: E402

__all__ = [
    "resolve_detail_path",
    "resolve_holdings",
    "resolve_library_path",
    "enabled_strategies",
    "load_dataset",
    "describe_dataset",
    "cv_evaluate",
    "summarize",
]


def load_dataset(
    detail_path=None,
    library_path=None,
    strategies=None,
    holding=None,
    required=None,
    verbose=True,
):
    """加载库 + 明细，对齐特征，返回带特征列与 win 标签的 DataFrame。

    参数：
        detail_path / library_path：为空时按 core.datasets 的优先级自动解析。
        strategies：逗号分隔的策略名，空/None = 全部。
        holding：持仓周期过滤。None（默认）不在此处过滤，由调用方用
                 `resolve_holdings()` 按策略分别筛选；给整数则只保留该周期的样本。
        required：自动发现明细时必须覆盖的策略名（避免选中不含目标策略的文件）。

    返回 df 包含列: code, entry_date, strategy, holding_days, return_pct, win,
    以及全部特征列。
    """
    detail_path = resolve_detail_path(detail_path, required=required, verbose=verbose)
    library_path = resolve_library_path(library_path, verbose=verbose)
    if verbose:
        print(f"[data] 明细: {detail_path}\n[data] 样本库: {library_path}")

    lib = pd.read_parquet(library_path).sort_values("date")
    detail = pd.read_csv(detail_path)
    if "holding_days" in detail.columns:
        detail["holding_days"] = pd.to_numeric(detail["holding_days"], errors="coerce")

    if strategies:
        wanted = {s.strip() for s in strategies.split(",") if s.strip()}
        if wanted:
            detail = detail[detail["strategy"].isin(wanted)].copy()

    if holding is not None and "holding_days" in detail.columns:
        before = len(detail)
        detail = detail[detail["holding_days"] == int(holding)].copy()
        if verbose:
            print(f"[data] 仅保留持仓 {holding} 日样本: {before} -> {len(detail)}")

    feat_all = precompute_features(lib)
    df = detail.merge(feat_all, left_on=["code", "entry_date"], right_on=["code", "date"], how="left")

    bench = benchmark_returns_for_trades(df) * 100.0
    df["bench_return_pct"] = bench.round(4)
    df["excess_return_pct"] = (df["return_pct"].astype(float) - df["bench_return_pct"]).round(4)
    df["win"] = (df["excess_return_pct"] > 0).astype(int)

    df = df.sort_values("entry_date").reset_index(drop=True)
    return df


def describe_dataset(df: pd.DataFrame) -> str:
    """一行摘要：样本量、策略、持仓周期分布。便于实验日志审计数据口径。"""
    if df.empty:
        return "数据集为空"
    parts = [f"样本 {len(df)}"]
    if "strategy" in df.columns:
        parts.append("策略 " + ", ".join(sorted(df["strategy"].astype(str).unique())))
    if "holding_days" in df.columns:
        hc = df["holding_days"].value_counts().sort_index()
        parts.append("持仓 " + ", ".join(f"{int(h)}日×{int(c)}" for h, c in hc.items()))
    return " | ".join(parts)


def cv_evaluate(model, X, y, tscv=None):
    """时序 CV 平均准确率（5 折 TimeSeriesSplit）。"""
    tscv = tscv or TimeSeriesSplit(n_splits=5)
    try:
        return float(cross_val_score(model, X, y, cv=tscv, scoring="accuracy").mean())
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] CV 失败: {exc}")
        return float("nan")


def summarize(df, features):
    """按策略统计样本量与 baseline（超额胜率）。返回 {strategy: (n, baseline)}。"""
    out = {}
    for strat, g in df.groupby("strategy"):
        y = g["win"].values
        base = max(float(y.mean()), 1 - float(y.mean())) if len(y) else 0.0
        out[strat] = (len(g), base)
    return out
