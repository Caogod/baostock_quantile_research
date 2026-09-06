"""大盘指数数据与市场特征（作为胜率依据）。

baostock 的 query_history_k_data_plus 支持指数代码（如 sh.000300 沪深300），
可拉取指数日 K 作为大盘代理。本模块：
1. 加载/缓存指数日 K（优先读 data_library/market_hs300.parquet；缺失则联网拉取）。
2. 计算市场特征（大盘涨跌幅/相对均线/regime/波动率），供 features.py 按日期合并。

设计要点：
- 市场特征对同一交易日全市场一致，按 date 合并到逐股特征即可。
- 采用模块级缓存：首次访问自动从 parquet 载入；训练与预测共用同一份，
  避免 train/predict 口径漂移。win_model 路径无需改调用签名即可自动获得市场特征。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_fetcher import DataFetcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_library"

# 沪深300 作为大盘代理（覆盖面广、代表性强；可换 sh.000016 上证50 / sz.399006 创业板指）
MARKET_INDEX = "sh.000300"
MARKET_PARQUET = DATA_DIR / "market_hs300.parquet"

# 市场特征列（与 features.FEATURES 中的市场段保持一致）
MARKET_FEATURES = [
    "mkt_ret_1", "mkt_ret_5", "mkt_ret_20",
    "mkt_above_ma20", "mkt_above_ma60", "mkt_vol_20",
]

# 模块级缓存：date -> 市场特征
_market_feat_cache: pd.DataFrame | None = None


def load_market_raw(
    library_dir: Path | str | None = None,
    fetcher: DataFetcher | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """加载指数日 K（升序）。

    优先读 parquet（离线）；若不存在且提供 fetcher，则联网拉取并落地。
    返回列：date, close, pctChg（数值化）。
    """
    parquet = Path(library_dir) / "market_hs300.parquet" if library_dir else MARKET_PARQUET

    if parquet.exists():
        df = pd.read_parquet(parquet)
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        for c in ("close", "pctChg"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    if fetcher is not None and fetcher._logged_in:
        # 默认拉取足够长区间以满足 60 日均线与 20 日波动率预热
        s = start or "2023-01-01"
        e = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        raw = fetcher.get_history(MARKET_INDEX, s, e, adjustflag="2",
                                  fields="date,code,close,pctChg")
        if not raw.empty:
            parquet.parent.mkdir(parents=True, exist_ok=True)
            raw.to_parquet(parquet, index=False)
        return raw.sort_values("date").reset_index(drop=True)

    return pd.DataFrame(columns=["date", "close", "pctChg"])


def compute_market_features(market_raw: pd.DataFrame) -> pd.DataFrame:
    """由指数日 K 计算市场特征（与日期对齐）。

    特征（均在当日可知，无前视）：
        mkt_ret_1       大盘当日涨跌幅
        mkt_ret_5       大盘 5 日收益
        mkt_ret_20      大盘 20 日收益
        mkt_above_ma20  大盘收盘相对 20 日均线（短期趋势）
        mkt_above_ma60  大盘收盘相对 60 日均线（中期 regime，风控核心）
        mkt_vol_20      大盘 20 日波动率
    """
    df = market_raw.copy()
    if df.empty or "close" not in df.columns:
        return pd.DataFrame(columns=["date", *MARKET_FEATURES])

    close = df["close"].astype(float)
    # 当日涨跌幅：优先用 baostock 的 pctChg，缺失则自行计算
    if "pctChg" in df.columns and df["pctChg"].notna().any():
        pct = df["pctChg"].astype(float)
    else:
        pct = close.pct_change() * 100.0

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    vol20_mean = close.rolling(20).mean()
    vol20_std = close.rolling(20).std()

    out = pd.DataFrame({"date": df["date"].astype(str).values})
    out["mkt_ret_1"] = (pct / 100.0).fillna(0.0).values
    out["mkt_ret_5"] = (close / close.shift(5) - 1).fillna(0.0).values
    out["mkt_ret_20"] = (close / close.shift(20) - 1).fillna(0.0).values
    out["mkt_above_ma20"] = ((close / ma20 - 1).fillna(0.0)).values
    out["mkt_above_ma60"] = ((close / ma60 - 1).fillna(0.0)).values
    out["mkt_vol_20"] = ((vol20_std / vol20_mean).fillna(0.0)).values
    return out


def get_market_features(
    library_dir: Path | str | None = None,
    fetcher: DataFetcher | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame | None:
    """获取市场特征表（含 date 列）。模块级缓存，避免重复计算。

    若无数据源（无 parquet 且未登录 fetcher），返回 None——
    上层 features.compute_feature_series 会用 0 填充（中性，保持向后兼容）。
    """
    global _market_feat_cache
    if _market_feat_cache is not None:
        return _market_feat_cache
    raw = load_market_raw(library_dir=library_dir, fetcher=fetcher, start=start, end=end)
    if raw.empty:
        return None
    _market_feat_cache = compute_market_features(raw)
    return _market_feat_cache


def reset_cache() -> None:
    """重置缓存（用于重新构建市场数据后）。"""
    global _market_feat_cache
    _market_feat_cache = None
