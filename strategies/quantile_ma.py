"""策略 2：在策略 1（分位区间）基础上，叠加均线条件。

依赖策略 1（strategies/quantile_range.py）：先满足"最新收盘价分位落在 25%~50%"
（基础筛选），再要求 MA(short) > MA(long)（默认 5 日均线 > 20 日均线）。

参数（分位参数透传给策略 1）：
    window        : 60    分位参考窗口
    quantile_low  : 0.25  分位下限
    quantile_high : 0.50  分位上限
    ma_short      : 5     短期均线
    ma_long       : 20    长期均线
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy
from .quantile_range import Strategy as QuantileStrategy


class Strategy(BaseStrategy):
    name = "quantile_ma"
    description = "分位区间(25%~50%) + 5日均线>20日均线"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        # 复用策略 1 的分位参数
        quantile_keys = ("window", "quantile_low", "quantile_high")
        qp = {k: params[k] for k in quantile_keys if k in params}
        self._quantile = QuantileStrategy(qp)
        self.ma_short = int(self.params.get("ma_short", 5))
        self.ma_long = int(self.params.get("ma_long", 20))

    def signal(self, df: pd.DataFrame) -> dict:
        base = self._quantile.signal(df)
        if not base.get("buy"):
            return self._no_signal(f"分位不满足：{base.get('reason', '')}")

        if len(df) < self.ma_long:
            return self._no_signal(f"均线数据不足{self.ma_long}日")

        close = df["close"].astype(float)
        ma_short = float(close.rolling(self.ma_short).mean().iloc[-1])
        ma_long = float(close.rolling(self.ma_long).mean().iloc[-1])
        if pd.isna(ma_short) or pd.isna(ma_long):
            return self._no_signal("均线数据不足")

        if ma_short > ma_long:
            gap = (ma_short / ma_long - 1) * 100
            score = round(float(base.get("score", 0.0)) + max(gap, 0.0), 4)
            return {
                "buy": True,
                "score": score,
                "reason": (
                    f"{base['reason']}；MA{self.ma_short}>MA{self.ma_long}"
                    f"({ma_short:.2f}>{ma_long:.2f})"
                ),
            }
        return self._no_signal(
            f"MA{self.ma_short}({ma_short:.2f})<=MA{self.ma_long}({ma_long:.2f})"
        )
