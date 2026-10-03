"""实验公共工具：加载数据 + 特征对齐 + 时序 CV 评估。

供 experiments/ 下的独立实验脚本复用，与主线 train_model.py 保持口径一致：
- 特征对齐：detail 按 (code, entry_date) 合并预计算特征
- 标签：相对沪深300同期超额收益 > 0
- CV：TimeSeriesSplit 5 折（时序，杜绝未来信息泄漏）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.features import precompute_features  # noqa: E402
from core.market import benchmark_returns_for_trades  # noqa: E402

DETAIL = PROJECT_ROOT / "backtest_output/backtest_detail_2025-11-12_2026-09-30.csv"
LIBRARY = PROJECT_ROOT / "data_library/combined_daily.parquet"


def load_dataset(detail_path=None, library_path=None, strategies=None):
    """加载库 + 明细，对齐特征，返回带特征列与 win 标签的 DataFrame。

    返回 df 包含列: code, entry_date, strategy, return_pct, win, 以及全部特征列。
    """
    detail_path = Path(detail_path or DETAIL)
    library_path = Path(library_path or LIBRARY)

    lib = pd.read_parquet(library_path).sort_values("date")
    detail = pd.read_csv(detail_path)

    if strategies:
        wanted = {s.strip() for s in strategies.split(",") if s.strip()}
        if wanted:
            detail = detail[detail["strategy"].isin(wanted)].copy()

    feat_all = precompute_features(lib)
    df = detail.merge(feat_all, left_on=["code", "entry_date"], right_on=["code", "date"], how="left")

    bench = benchmark_returns_for_trades(df) * 100.0
    df["bench_return_pct"] = bench.round(4)
    df["excess_return_pct"] = (df["return_pct"].astype(float) - df["bench_return_pct"]).round(4)
    df["win"] = (df["excess_return_pct"] > 0).astype(int)

    df = df.sort_values("entry_date").reset_index(drop=True)
    return df


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
