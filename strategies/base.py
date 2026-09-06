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
"""
from __future__ import annotations

import pandas as pd


class BaseStrategy:
    name = "base"
    description = ""

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}

    def signal(self, df: pd.DataFrame) -> dict:
        """在 df 最后一个交易日判断是否产生买入信号。"""
        raise NotImplementedError("策略必须实现 signal(df) 方法")

    def _no_signal(self, reason: str = "") -> dict:
        return {"buy": False, "score": 0.0, "reason": reason}
