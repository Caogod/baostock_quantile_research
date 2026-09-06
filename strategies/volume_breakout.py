"""示例策略 2：放量突破 N 日新高。

逻辑：收盘价创过去 N 日新高，且当日成交量放大到 N 日均量的指定倍数。
参数：n=20, vol_ratio=1.5
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class Strategy(BaseStrategy):
    name = "volume_breakout"
    description = "放量突破 N 日收盘新高"

    def signal(self, df: pd.DataFrame) -> dict:
        n = int(self.params.get("n", 20))
        vol_ratio = float(self.params.get("vol_ratio", 1.5))

        if len(df) < n + 2:
            return self._no_signal("数据不足")

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        cur_close = close.iloc[-1]
        prev_high = close.iloc[-n - 1:-1].max()

        if cur_close <= prev_high:
            return self._no_signal("")

        cur_vol = volume.iloc[-1]
        avg_vol = volume.iloc[-n - 1:-1].mean()
        if avg_vol > 0 and cur_vol >= avg_vol * vol_ratio:
            score = (cur_close / prev_high - 1) * 100 + (cur_vol / avg_vol - 1) * 10
            return {
                "buy": True,
                "score": round(float(score), 4),
                "reason": f"突破{n}日新高, 量比{cur_vol / avg_vol:.2f}",
            }
        return self._no_signal("创新高但未放量")
