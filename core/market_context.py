"""策略上下文构建：市场指数行情 + 证券元数据（带本地缓存，避免每次回溯联网）。

某些策略需要"自身 K 线之外"的信息（如当日全市场涨跌、上市日期、板块归属）。
信号契约 signal(df) 只接收单只股票的日 K，因此这里把这类信息打包成 context，
在回溯/选股开始前一次性注入策略（调用 strategy.bind_context(ctx)）。

用法：
    from core.market_context import build_context
    ctx = build_context(index_code="399317", start=start, end=end)
    for _, strat in strategies:
        strat.bind_context(ctx)

ctx 内容：
    market_close : pd.Series，index=交易日字符串，值为指数收盘价
    market_down  : pd.Series[bool]，该交易日指数收跌（收盘 < 前一交易日收盘）
    meta         : pd.DataFrame，index=code，列 name / ipoDate / outDate / type / status
                   type: 1=股票 2=指数 3=其它 4=可转债 5=ETF

缓存目录：data_library/_ctx/（首次联网拉取，之后离线复用；--refresh 可强制刷新）
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .data_fetcher import DataFetcher


def _normalize_index_code(code: str) -> str:
    """把 '399317' 这类裸代码补全为 baostock 的 'sz.399317'。"""
    code = code.strip()
    if "." in code:
        return code
    if code.startswith(("399", "159", "150")):
        return f"sz.{code}"
    return f"sh.{code}"


def _read_csv(path: Path, dtype_cols: dict) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype=dtype_cols)
    except Exception:  # noqa: BLE001
        return None
    return None if df.empty else df


def build_context(
    index_code: str = "399317",
    start: str | None = None,
    end: str | None = None,
    cache_dir: str = "data_library",
    refresh: bool = False,
) -> dict:
    """构建并缓存策略上下文；获取失败时尽力返回已有缓存，不抛异常。"""
    ctx: dict = {"market_close": None, "market_down": None, "meta": None}
    cache = Path(cache_dir) / "_ctx"
    cache.mkdir(parents=True, exist_ok=True)

    # ---- 1. 市场指数日线 ----
    code = _normalize_index_code(index_code)
    need_start = start or "2015-01-01"
    if start:
        need_start = (datetime.strptime(start, "%Y-%m-%d")
                      - timedelta(days=45)).strftime("%Y-%m-%d")
    idx_path = cache / f"index_{code.replace('.', '_')}.csv"

    df = None
    if not refresh:
        cached = _read_csv(idx_path, {"date": str})
        if cached is not None and cached["date"].min() <= need_start and cached["date"].max() >= (end or ""):
            df = cached
            print(f"[ctx] 指数 {code} 命中缓存（{cached['date'].min()} ~ {cached['date'].max()}，{len(cached)} 行）")
    if df is None:
        try:
            fetcher = DataFetcher()
            fetcher.login()
            try:
                fresh = fetcher.get_index_history(code, "2015-01-01", end or datetime.now().strftime("%Y-%m-%d"))
            finally:
                fetcher.logout()
            old = _read_csv(idx_path, {"date": str})
            df = (pd.concat([old, fresh]).drop_duplicates(subset="date").sort_values("date")
                  if old is not None else fresh)
            df.to_csv(idx_path, index=False, encoding="utf-8-sig")
            print(f"[ctx] 指数 {code} 已下载并缓存 {len(df)} 行 -> {idx_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 指数 {code} 获取失败: {exc}")
            df = _read_csv(idx_path, {"date": str})
            if df is not None:
                print(f"[warn] 回退使用旧缓存（{df['date'].min()} ~ {df['date'].max()}）")

    if df is not None and not df.empty:
        s = pd.Series(df["close"].astype(float).values,
                      index=df["date"].astype(str).values).sort_index()
        ctx["market_close"] = s
        ctx["market_down"] = s < s.shift(1)
        print(f"[ctx] 市场下跌日占比: {ctx['market_down'].mean():.1%}")

    # ---- 2. 证券基本信息（名称 / 上市日期 / 类型）----
    meta_path = cache / "stock_basic.csv"
    meta = None
    if not refresh:
        meta = _read_csv(meta_path, {"code": str})
        if meta is not None:
            print(f"[ctx] 证券基本信息命中缓存 {len(meta)} 条")
    if meta is None:
        try:
            fetcher = DataFetcher()
            fetcher.login()
            try:
                meta = fetcher.get_stock_basic()
            finally:
                fetcher.logout()
            meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
            print(f"[ctx] 证券基本信息已下载并缓存 {len(meta)} 条")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 证券基本信息获取失败: {exc}")
            meta = _read_csv(meta_path, {"code": str})

    if meta is not None and not meta.empty:
        meta = meta.rename(columns={"code_name": "name"})
        ctx["meta"] = meta.set_index("code")

    return ctx
