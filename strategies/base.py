"""策略基类与信号契约。

所有策略脚本需定义 `Strategy` 类并继承本基类，实现 `signal` 方法。

信号契约：
    signal(df) -> dict
        df: 升序排列的单只股票历史日 K DataFrame，至少含列
            date, open, high, low, close, preclose, volume, amount, pctChg, turn
        返回 dict，必须含：
            buy   : bool   是否入选（今日产生买入信号）
            score : float  打分（用于多策略排序，越大越靠前）
            reason: str    信号说明（可读性，写入输出）

可选性能契约（不影响信号语义，仅加速回溯）：
    prefilter_mask(close) -> np.ndarray | None
        close: 单只股票按日期升序的收盘价一维数组
        返回与 close 等长的 bool 掩码，掩码为 True 的位置才可能是信号日。
        这是「必要条件的超集」——掩码为 False 的交易日必须保证 signal()
        一定不产生买入信号，否则会漏掉信号。
        默认返回 None，表示不做预筛（逐日全量计算）。

    prefilter_mask_df(df) -> np.ndarray | None
        同上，但能拿到完整日 K（含 high/low/open 等），用于需要更多字段的预筛。
        引擎优先调用它；返回 None 时再退回 prefilter_mask(close)。

可选上下文契约（需要"自身 K 线之外"信息的策略使用）：
    bind_context(ctx: dict) -> None
        由回溯/选股入口在开跑前调用一次，ctx 由 core.market_context.build_context()
        生成，至少包含：
            market_close : pd.Series  指数收盘价（index=交易日字符串）
            market_down  : pd.Series[bool]  该日指数是否收跌
            meta         : pd.DataFrame（index=code）列 name / ipoDate / type / status
        默认空实现（不使用上下文）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_ratio(close: np.ndarray | pd.Series,
                     high: np.ndarray | pd.Series | None = None,
                     low: np.ndarray | pd.Series | None = None,
                     metric: str = "close") -> float:
    """计算一段价格序列的「极差 / 最小值」波动率指标。

    口径：
        close   : (close.max - close.min) / close.min      （默认，收盘价口径）
        highlow : (high.max  - low.min)   / low.min        （真实极差口径，需传 high/low）

    分母为 0 或序列为空时返回 float('inf')（表示波动无限大，天然不满足低波动阈值）。
    """
    c = np.asarray(close, dtype=float)
    if c.size == 0:
        return float("inf")
    if metric == "highlow":
        if high is None or low is None:
            raise ValueError("metric='highlow' 需要同时提供 high 与 low")
        h = np.asarray(high, dtype=float)
        lo = np.asarray(low, dtype=float)
        if h.size == 0 or lo.size == 0:
            return float("inf")
        denom = float(lo.min())
        if denom <= 0:
            return float("inf")
        return float((h.max() - denom) / denom)
    denom = float(c.min())
    if denom <= 0:
        return float("inf")
    return float((c.max() - denom) / denom)


class BaseStrategy:
    name = "base"
    description = ""

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}

    def signal(self, df: pd.DataFrame) -> dict:
        """在 df 最后一个交易日判断是否产生买入信号。"""
        raise NotImplementedError("策略必须实现 signal(df) 方法")

    def prefilter_mask(self, close: np.ndarray) -> np.ndarray | None:
        """可选的向量化预筛：返回候选日掩码（True=可能出信号）。

        必须满足「必要条件的超集」：掩码为 False 的日期 signal() 必然返回
        buy=False。默认 None（不预筛，逐日全量调用 signal()）。
        """
        return None

    def prefilter_mask_df(self, df: pd.DataFrame) -> np.ndarray | None:
        """可选的向量化预筛（完整日 K 版），引擎优先于 prefilter_mask(close)。

        用于预筛需要 high/low/open 等字段的策略；语义与契约同 prefilter_mask。
        默认 None（回退到 prefilter_mask(close)）。
        """
        return None

    def bind_context(self, ctx: dict | None) -> None:
        """接收外部上下文（市场指数 / 证券元数据），由入口在开跑前调用一次。

        默认空实现；需要上下文的策略覆写此方法并自行缓存所需字段。
        """
        return None

    def _no_signal(self, reason: str = "") -> dict:
        return {"buy": False, "score": 0.0, "reason": reason}
