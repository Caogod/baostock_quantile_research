"""策略：ma_breakout 信号触发后延迟 shift 日入场。

复用 ma_breakout 的入选条件，但入场时点延后 shift 个交易日：
即"ma_breakout 在 shift 日前触发信号 → 今日入场"。
用于捕捉突破后的回踩/蓄势，避免追高。

参数：
    shift         : 5     信号触发后延迟入场的交易日数
    其余（ma_short/ma_mid/ma_long/converge_pct/breakout_pct/min_turn）
    透传给 ma_breakout，含义与其一致。
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy
from .ma_breakout import Strategy as MaBreakoutStrategy


class Strategy(BaseStrategy):
    name = "ma_breakout_shift5"
    description = "ma_breakout信号触发后延迟5日入场"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.shift = int(self.params.get("shift", 5))
        self._base = MaBreakoutStrategy(params)

    def signal(self, df: pd.DataFrame) -> dict:
        if len(df) < self.shift + 1:
            return self._no_signal(f"数据不足{self.shift + 1}日")

        # 用 shift 日前（不含最近 shift 日）的数据判断是否触发过信号
        sub = df.iloc[:-self.shift]
        base = self._base.signal(sub)
        if base.get("buy"):
            return {
                "buy": True,
                "score": float(base.get("score", 0.0)),
                "reason": (
                    f"ma_breakout于{self.shift}日前触发，延迟入场｜{base.get('reason', '')}"
                ),
            }
        return self._no_signal(f"ma_breakout未在{self.shift}日前触发")
