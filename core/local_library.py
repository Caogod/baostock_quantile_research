"""本地回溯样本库读取器。

从 Parquet 文件读取已落地的历史日 K 数据，接口与 DataFetcher 兼容，
使 Backtester 可完全离线运行（无需 baostock / 网络）。

数据文件结构（长表）：
    code, date, open, high, low, close, preclose, volume, amount,
    turn, tradestatus, pctChg, isST, code_name
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_fetcher import _NUMERIC_COLS


class LocalDataLibrary:
    def __init__(self, parquet_path: str | Path, meta_parquet_path: str | Path | None = None) -> None:
        self.parquet_path = Path(parquet_path)
        self.daily = pd.read_parquet(self.parquet_path)
        # 统一列类型，按 code+date 排序，便于按代码过滤
        if "date" in self.daily.columns:
            self.daily["date"] = self.daily["date"].astype(str)
        for col in _NUMERIC_COLS:
            if col in self.daily.columns:
                self.daily[col] = pd.to_numeric(self.daily[col], errors="coerce")
        self.daily = self.daily.sort_values(["code", "date"]).reset_index(drop=True)

        self.meta: pd.DataFrame | None = None
        if meta_parquet_path and Path(meta_parquet_path).exists():
            self.meta = pd.read_parquet(meta_parquet_path)

        self.request_interval = 0.0  # 本地无网络，无节流

    # ------------------------------------------------------------------ #
    # 与 DataFetcher 对齐的无操作接口
    # ------------------------------------------------------------------ #
    def login(self) -> None:
        pass

    def logout(self) -> None:
        pass

    def relogin(self) -> None:
        pass

    def _throttle(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    def get_history(
        self,
        code: str,
        start: str,
        end: str,
        adjustflag: str = "2",
        fields: str | None = None,
    ) -> pd.DataFrame:
        """返回指定代码在 [start, end] 区间内的日 K（升序）。"""
        df = self.daily[self.daily["code"] == code]
        if df.empty:
            return df.reset_index(drop=True)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df.reset_index(drop=True)

    def get_all_stocks(self, day: str | None = None) -> pd.DataFrame:
        """返回库内全部证券列表（code / tradeStatus / code_name）。"""
        codes = sorted(self.daily["code"].unique().tolist())
        name_map: dict[str, Any] = {}
        if self.meta is not None and {"code", "code_name"}.issubset(self.meta.columns):
            name_map = dict(zip(self.meta["code"], self.meta["code_name"]))
        return pd.DataFrame({
            "code": codes,
            "tradeStatus": ["1"] * len(codes),
            "code_name": [name_map.get(c, "") for c in codes],
        })

    def get_trade_dates(self, start: str, end: str) -> list[str]:
        """返回库内 [start, end] 区间出现过的交易日（升序）。"""
        if self.daily.empty:
            return []
        dates = self.daily[(self.daily["date"] >= start) & (self.daily["date"] <= end)]["date"]
        return sorted(dates.unique().tolist())

    @property
    def date_range(self) -> tuple[str, str]:
        """返回库内数据的起止日期。"""
        if self.daily.empty:
            return ("", "")
        return (str(self.daily["date"].min()), str(self.daily["date"].max()))

    def describe(self) -> dict[str, Any]:
        lo, hi = self.date_range
        return {
            "parquet": str(self.parquet_path),
            "stocks": int(self.daily["code"].nunique()),
            "rows": int(len(self.daily)),
            "date_range": [lo, hi],
        }
