"""策略 2：均线粘合 + 放量突破。

三个条件同时满足：
1. 10 日均线、5 日均线与 20 日基准均线的偏离均不超过 converge_pct（默认 3%）；
2. 最新收盘价较"前一日"的 5 日均线（MA5.shift(1)）上涨超过 breakout_pct（默认 5%）；
3. 换手率（turn，%）超过 min_turn（默认 10）。

参数：
    ma_short      : 5     短期均线
    ma_mid        : 10    中期均线
    ma_long       : 20    基准均线
    converge_pct  : 0.03  均线与基准的最大偏离比例
    breakout_pct  : 0.05  收盘较前一日 MA5 的最小涨幅
    min_turn      : 10.0  最低换手率（%）
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class Strategy(BaseStrategy):
    name = "ma_breakout"
    description = "均线粘合(±3%)+收盘破前一日MA5达5%+换手率>10%"

    def signal(self, df: pd.DataFrame) -> dict:
        ma_short = int(self.params.get("ma_short", 5))
        ma_mid = int(self.params.get("ma_mid", 10))
        ma_long = int(self.params.get("ma_long", 20))
        converge_pct = float(self.params.get("converge_pct", 0.03))
        breakout_pct = float(self.params.get("breakout_pct", 0.05))
        min_turn = float(self.params.get("min_turn", 10.0))

        if len(df) < ma_long:
            return self._no_signal(f"数据不足{ma_long}日")

        close = df["close"].astype(float)
        ma_short_series = close.rolling(ma_short).mean()
        ma_mid_v = float(close.rolling(ma_mid).mean().iloc[-1])
        ma_long_v = float(close.rolling(ma_long).mean().iloc[-1])
        if pd.isna(ma_long_v) or ma_long_v == 0:
            return self._no_signal("均线数据不足")

        # 条件 1：均线粘合（MA5、MA10 与 MA20 偏离均 <= converge_pct）
        ma_short_v = float(ma_short_series.iloc[-1])
        if abs(ma_mid_v / ma_long_v - 1) > converge_pct:
            return self._no_signal(
                f"MA{ma_mid}偏离MA{ma_long}{abs(ma_mid_v / ma_long_v - 1):.1%}>"
                f"{converge_pct:.0%}"
            )
        if abs(ma_short_v / ma_long_v - 1) > converge_pct:
            return self._no_signal(
                f"MA{ma_short}偏离MA{ma_long}{abs(ma_short_v / ma_long_v - 1):.1%}>"
                f"{converge_pct:.0%}"
            )

        # 条件 2：最新收盘较"前一日 MA5"（shift(1)）上涨超过 breakout_pct
        ma_short_prev = float(ma_short_series.shift(1).iloc[-1])
        if pd.isna(ma_short_prev) or ma_short_prev == 0:
            return self._no_signal("前一日均线数据不足")

        last_close = float(close.iloc[-1])
        gain = last_close / ma_short_prev - 1
        if gain <= breakout_pct:
            return self._no_signal(
                f"收盘较前一日MA{ma_short}涨幅{gain:.1%}未达{breakout_pct:.0%}"
            )

        # 条件 3：换手率超过 min_turn（%）
        turn = float(df["turn"].astype(float).iloc[-1]) if "turn" in df.columns else float("nan")
        if pd.isna(turn) or turn <= min_turn:
            return self._no_signal(f"换手率{turn:.2f}%未达{min_turn}%")

        score = round(gain * 100, 4)
        return {
            "buy": True,
            "score": score,
            "reason": (
                f"均线粘合(MA{ma_short}/{ma_mid}/MA{ma_long}偏离<{converge_pct:.0%})；"
                f"收盘{last_close:.2f}较前一日MA{ma_short}涨{gain:.1%}；换手{turn:.2f}%"
            ),
        }
