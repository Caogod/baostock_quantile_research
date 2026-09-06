"""策略 1（基础筛选）：60 日分位区间。

逻辑（对应需求描述的三步）：
1. 提取近 window 个交易日（默认 60 日）的收盘价，做分位排序；
2. 计算最新收盘价在该窗口内的分位数排名（percentile rank，0~1）；
3. 若最新价分位落在 [quantile_low, quantile_high]（默认 25%~50%）区间，则入选。

参数：
    window        : 60    分位参考窗口（交易日）
    quantile_low  : 0.25  分位下限
    quantile_high : 0.50  分位上限

说明：分位排名用 rolling(window).rank(pct=True) 计算，即"最新收盘价在近 60 日
价格分布中排在第百分之几"；25%~50% 表示处于历史区间中下段（回调后的中低位）。
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class Strategy(BaseStrategy):
    name = "quantile_range"
    description = "60日分位区间：最新收盘价分位落在25%~50%"

    def signal(self, df: pd.DataFrame) -> dict:
        window = int(self.params.get("window", 60))
        q_low = float(self.params.get("quantile_low", 0.25))
        q_high = float(self.params.get("quantile_high", 0.50))

        if len(df) < window:
            return self._no_signal(f"数据不足{window}日")

        close = df["close"].astype(float)
        # 每个交易日在近 window 日窗口内的分位排名（0~1）
        pct_rank = close.rolling(window).rank(pct=True)
        last_rank = float(pct_rank.iloc[-1])
        if pd.isna(last_rank):
            return self._no_signal("分位数据不足")

        if q_low <= last_rank <= q_high:
            return {
                "buy": True,
                "score": round(last_rank, 4),
                "reason": f"60日分位{last_rank:.1%}∈[{q_low:.0%},{q_high:.0%}]",
            }
        return self._no_signal(f"60日分位{last_rank:.1%}不在[{q_low:.0%},{q_high:.0%}]")
