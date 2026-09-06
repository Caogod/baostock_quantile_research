"""baostock 数据获取封装。

统一登录/登出、股票列表、历史日 K（五价 + 量额 + 指标）、交易日历等能力，
全部返回 pandas.DataFrame，便于策略与回测复用。

健壮性说明：
- baostock 复用登录时建立的单条持久 socket；高频连续请求后服务端可能重置连接，
  表现为返回空结果或进程级异常。本模块对"空结果/异常"自动重连重试，
  并支持请求间隔限速，尽量降低服务端压力。
- 对于无法在 Python 内捕获的进程级崩溃，由上层（Selector/Backtester）的
  断点续跑机制兜底。
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from . import compat  # noqa: F401  必须先于 baostock 使用，补丁 pandas.append

import baostock as bs
import pandas as pd

# 日线"五价"字段（开盘/最高/最低/收盘/前收）+ 常用量价指标。
DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,"
    "volume,amount,turn,tradestatus,pctChg,isST"
)

_NUMERIC_COLS = [
    "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "pctChg",
]


class DataFetcher:
    """封装 baostock 的数据访问。使用前须 login，结束后 logout。"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        request_interval: float = 0.05,
    ) -> None:
        self._logged_in = False
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_interval = request_interval

    # ------------------------------------------------------------------ #
    def login(self) -> None:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock 登录失败: code={lg.error_code}, msg={lg.error_msg}")
        self._logged_in = True

    def logout(self) -> None:
        if self._logged_in:
            try:
                bs.logout()
            except Exception:
                pass
            self._logged_in = False

    def relogin(self) -> None:
        """断开并重建连接（用于连接被服务端重置后的恢复）。"""
        self.logout()
        self.login()

    def _throttle(self) -> None:
        if self.request_interval > 0:
            time.sleep(self.request_interval)

    # ------------------------------------------------------------------ #
    # 股票列表
    # ------------------------------------------------------------------ #
    def get_all_stocks(self, day: str) -> pd.DataFrame:
        """获取指定交易日的全部证券列表（带重试）。

        返回列: code / tradeStatus / code_name。
        """
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                rs = bs.query_all_stock(day=day)
                if rs.error_code == "0":
                    df = rs.get_data()
                    if df is not None and not df.empty:
                        return df
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
                self.relogin()
        raise RuntimeError(f"获取股票列表失败（重试 {self.max_retries} 次）: {last_err}")

    def get_hs300_stocks(self) -> pd.DataFrame:
        """获取沪深300指数成分股列表（带重试）。

        返回列: code / code_name。
        """
        return self.get_index_stocks("hs300")

    def get_index_stocks(self, name: str = "hs300") -> pd.DataFrame:
        """获取指定指数成分股列表（带重试）。

        name: hs300(沪深300) / zz500(中证500) / sz50(上证50)。
        返回列: code / code_name。
        """
        query_fn = {
            "hs300": bs.query_hs300_stocks,
            "zz500": bs.query_zz500_stocks,
            "sz50": bs.query_sz50_stocks,
        }.get(name.lower())
        if query_fn is None:
            raise ValueError(f"不支持的指数: {name}（可选 hs300/zz500/sz50）")

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                rs = query_fn()
                if rs.error_code == "0":
                    df = rs.get_data()
                    if df is not None and not df.empty:
                        return df
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
                self.relogin()
        raise RuntimeError(f"获取{name}成分股失败（重试 {self.max_retries} 次）: {last_err}")

    # ------------------------------------------------------------------ #
    # 历史日 K（五价）
    # ------------------------------------------------------------------ #
    def get_history(
        self,
        code: str,
        start: str,
        end: str,
        adjustflag: str = "2",
        fields: str = DAILY_FIELDS,
    ) -> pd.DataFrame:
        """获取单只股票的历史日 K 数据（升序），带重连重试。

        adjustflag: 1=后复权, 2=前复权, 3=不复权（默认前复权，保证除权后价格连续）。
        返回清洗后的 DataFrame，数值列已转为 float。
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                rs = bs.query_history_k_data_plus(
                    code, fields,
                    start_date=start, end_date=end,
                    frequency="d", adjustflag=adjustflag,
                )
                if rs.error_code == "0":
                    df = rs.get_data()
                    df = self._clean(df)
                    return df
            except Exception:  # noqa: BLE001
                pass
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
                self.relogin()
        # 全部失败后返回空（上层按无数据处理）
        return pd.DataFrame()

    def _clean(self, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # 仅保留有实际行情（成交量>0）的交易日记录
        if "volume" in df.columns:
            df = df[df["volume"] > 0].copy()
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # 交易日历
    # ------------------------------------------------------------------ #
    def get_trade_dates(self, start: str, end: str) -> list[str]:
        """返回 [start, end] 区间内的交易日列表（升序）。"""
        for _ in range(self.max_retries):
            try:
                rs = bs.query_trade_dates(start_date=start, end_date=end)
                if rs.error_code == "0":
                    df = rs.get_data()
                    if df is not None and not df.empty:
                        dates = df[df["is_trading_day"] == "1"]["calendar_date"].tolist()
                        return sorted(dates)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.retry_delay)
            self.relogin()
        return []

    def latest_trading_day(self, lookback_days: int = 15) -> str:
        """返回最近一个交易日（格式 YYYY-MM-DD）。"""
        today = datetime.now()
        start = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        dates = self.get_trade_dates(start, end)
        if not dates:
            raise RuntimeError("近 %d 天内未找到交易日" % lookback_days)
        return dates[-1]
