"""交易特征扩充（模块2）。

提供统一的特征计算，供"模型训练(train_model.py)"与"选股后二次过滤(win_model.py)"
共用，保证训练与预测的特征口径一致。

特征（18 个，均在入场日计算）：
    量比(5/20日)、log流通市值、MA5/10/20 差值、换手率、当日涨跌幅、
    5/10/20日动量、20日波动率、60日分位、距20日高低点、成交额比、RSI14、log价格。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import market as _market

# 特征列顺序（训练与预测必须一致）
# 18 个原始交易特征 + 6 个大盘市场特征（按 date 合并，作为胜率依据）
MARKET_FEATURES = _market.MARKET_FEATURES
FEATURES = [
    "volume_ratio_5", "volume_ratio_20", "mcap_log", "ma5_10", "ma5_20", "ma10_20",
    "turn", "pct_chg", "mom_5", "mom_10", "mom_20", "volatility_20",
    "pct_rank_60", "dist_high_20", "dist_low_20", "amount_ratio", "rsi_14", "close_log",
    *MARKET_FEATURES,  # mkt_ret_1/5/20, mkt_above_ma20/ma60, mkt_vol_20
]


def _merge_market(f: pd.DataFrame) -> pd.DataFrame:
    """按 date 合并市场特征（模块缓存自动加载）。缺失则用 0 填充（中性，向后兼容）。"""
    mkt = _market.get_market_features()
    if mkt is None or mkt.empty or "date" not in f.columns:
        for col in MARKET_FEATURES:
            f[col] = 0.0
        return f
    mkt = mkt.rename(columns={c: c for c in MARKET_FEATURES})
    f = f.merge(mkt[["date", *MARKET_FEATURES]], on="date", how="left")
    for col in MARKET_FEATURES:
        f[col] = f[col].fillna(0.0)
    return f


def compute_feature_series(hist: pd.DataFrame) -> pd.DataFrame:
    """对单只股票历史日 K 计算特征时间序列（与日期对齐）。hist 升序。"""
    close = hist["close"].astype(float)
    high = hist["high"].astype(float)
    low = hist["low"].astype(float)
    volume = hist["volume"].astype(float)
    amount = hist["amount"].astype(float)
    turn = hist["turn"].astype(float)
    pct = hist["pctChg"].astype(float) if "pctChg" in hist else close.pct_change() * 100

    f = pd.DataFrame({"date": hist["date"].values})

    # 量比（当日量 / 过去 N 日均量）
    f["volume_ratio_5"] = volume / volume.shift(1).rolling(5).mean()
    f["volume_ratio_20"] = volume / volume.shift(1).rolling(20).mean()

    # 流通市值 log（成交额×100/换手率）
    f["mcap_log"] = np.log(amount * 100.0 / turn.replace(0, np.nan))

    # MA 差值
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    f["ma5_10"] = (ma5 - ma10) / ma10
    f["ma5_20"] = (ma5 - ma20) / ma20
    f["ma10_20"] = (ma10 - ma20) / ma20

    f["turn"] = turn
    f["pct_chg"] = pct

    # 动量
    f["mom_5"] = close / close.shift(5) - 1
    f["mom_10"] = close / close.shift(10) - 1
    f["mom_20"] = close / close.shift(20) - 1

    # 波动率
    f["volatility_20"] = close.rolling(20).std() / close.rolling(20).mean()

    # 60 日分位
    f["pct_rank_60"] = close.rolling(60).rank(pct=True)

    # 距 20 日高低点
    f["dist_high_20"] = close / high.rolling(20).max() - 1
    f["dist_low_20"] = close / low.rolling(20).min() - 1

    # 成交额相对 20 日均额
    f["amount_ratio"] = amount / amount.rolling(20).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss)

    f["close_log"] = np.log(close)
    # 合并大盘市场特征（按 date；缓存缺失时填 0）
    f = _merge_market(f)
    return f


def precompute_features(library: pd.DataFrame) -> pd.DataFrame:
    """对整库（含 code/date 列）预计算特征矩阵，返回 (code, date, ...FEATURES)。"""
    frames = []
    for code, g in library.sort_values("date").groupby("code"):
        feat = compute_feature_series(g.reset_index(drop=True))
        feat.insert(1, "code", code)
        frames.append(feat)
    return pd.concat(frames, ignore_index=True)


def feature_at(library: pd.DataFrame, code: str, date: str) -> pd.DataFrame:
    """计算某只股票在指定日期的特征（单行 DataFrame）。"""
    hist = library[library["code"] == code]
    hist = hist[hist["date"] <= date].sort_values("date").reset_index(drop=True)
    if hist.empty:
        return pd.DataFrame(columns=FEATURES)
    return compute_feature_series(hist).iloc[[-1]]
