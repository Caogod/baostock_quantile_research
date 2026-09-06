"""示例策略 1：均线金叉 + 放量确认。

逻辑：短期均线上穿长期均线（金叉），且当日成交量放大到 20 日均量的指定倍数。
参数：short=5, long=20, vol_ratio=1.2
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class Strategy(BaseStrategy):
    name = "ma_golden_cross"
    description = "短期均线上穿长期均线（金叉）且放量"

    def signal(self, df: pd.DataFrame) -> dict:
        short = int(self.params.get("short", 5))
        long = int(self.params.get("long", 20))
        vol_ratio = float(self.params.get("vol_ratio", 1.2))

        if len(df) < long + 1:
            return self._no_signal("数据不足")

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        ma_short = close.rolling(short).mean()
        ma_long = close.rolling(long).mean()

        prev_short, prev_long = ma_short.iloc[-2], ma_long.iloc[-2]
        cur_short, cur_long = ma_short.iloc[-1], ma_long.iloc[-1]

        if prev_short <= prev_long and cur_short > cur_long:
            cur_vol = volume.iloc[-1]
            avg_vol = volume.rolling(20).mean().iloc[-1]
            if avg_vol > 0 and cur_vol >= avg_vol * vol_ratio:
                score = (cur_short / cur_long - 1) * 100 + (cur_vol / avg_vol - 1) * 10
                return {
                    "buy": True,
                    "score": round(float(score), 4),
                    "reason": f"{short}日线上穿{long}日线金叉, 量比{cur_vol / avg_vol:.2f}",
                }
            return self._no_signal("金叉但未放量")
        return self._no_signal("")
