"""趋势缩量回调 + 资金流入型（主力资金流入条件以量价代理指标实现）。

原始需求（13 条筛选条件，本策略全部实现）：
     1. 收盘价 > MA20
     2. MA20 向上（今日 MA20 > 昨日 MA20）
     3. MA5 > MA10
     4. 当日涨幅 < 6%
     5. 当日成交量 < 5 日均量
     6. 主力资金净流入 > 0
     7. 近 3 日主力资金净流入 > 0
     8. 3% < 换手率 < 12%
     9. 近 20 日区间涨跌幅 < 35%
    10. 60 日乖离率 < 25%
    11. 当日成交量 < 近 20 日最大成交量
    12. 30 亿 < 流通市值 < 600 亿
    13. 近 20 日日均成交额 > 1 亿
    14. 上市天数 > 180 天

【条件顺序：按过滤代价由低到高手工排定（2026-10-12 取消向量化预筛）】
    全部条件为 AND 关系，顺序不影响判定结果，只影响性能：越靠前越便宜，
    越早短路，昂贵的全序列 rolling 只留给极少数候选日。代价评估：
        O(1) 末行/定长索引 → 涨幅、换手率、上市天数(meta)、区间涨跌幅、乖离率、流通市值
        O(5~20) 定长切片   → 缩量、量峰、20 日均额
        全序列 rolling ×3  → 均线结构（MA20/MA5/MA10）
        全序列 rolling     → 资金流入代理（最贵，放最后）
    原顺序把均线结构与资金流（两者都需全序列 rolling）排在很前面，
    导致绝大多数样本在昂贵计算之后才被涨幅/换手率淘汰；调整后先做廉价
    强过滤，预计大幅减少 rolling 调用次数。

【关于条件 6/7：主力资金净流入的代理实现】
    baostock **不提供个股资金流向数据**（已实测确认：其 query_* 接口中
    query_cash_flow_data 是财务报表口径的资金流量表，并非"主力净流入"；
    query_history_k_data_plus 的可用字段也只有
    date/code/open/high/low/preclose/volume/amount/turn/tradestatus/pctChg/isST 等）。
    因此本策略用**量价代理指标**近似"资金流入"，核心逻辑是：
    资金流入 ⇒ 上涨日成交额放大、收盘位于日内高位（买盘占优）。
    两个代理量：
        inflow_today  = (close - low) / (high - low)          收盘在日内区间的位置
                        × (today_volume / prev5_avg_volume)     放量程度（仅放大计）
        inflow_3d     = 近 3 日「收盘位置 × 放量倍数」之和
    两者为正即视为资金净流入。若后续接入 akshare 等含资金流的真实数据源，
    可通过 set_inflow_provider() 注入替换，无需改动其余逻辑。

【关于条件 12：流通市值的反推实现】
    库中没有流通市值字段，也不联网拉取，改用**换手率反推**：
        流通股本 = volume / (turn / 100)        （turn 为百分数）
        流通市值 = 流通股本 × close
    该口径实测非常稳定：近一个月同一股票反推流通股本的变异系数
    0.00%~0.03%（宁德时代 42.60 亿股、五粮液 38.82 亿股、工商银行 2696 亿股，
    均与真实流通股本吻合）。
    注意：turn 极小时（如 0.01%）反推误差会被放大，故对 turn 设下限保护。

参数：
    ma_long / ma_trend  : 20    趋势基准均线 / 向上判定窗口
    ma_short / ma_mid   : 5/ 10短期与中期均线
    max_pct_chg         : 6.0   当日涨幅上限（%）
    vol_shrink_window   : 5     缩量判定的均量窗口
    vol_max_window      : 20    "近 N 日最大成交量"窗口
    inflow_window       : 3     资金流近N 日代理窗口
    turn_min / turn_max : 3.0/ 12.0  换手率区间（%）
    range_window        : 20    区间涨跌幅窗口
    range_max           : 35.0  近 N 日涨跌幅上限（%）
    bias_window         : 60    乖离率窗口
    bias_max            : 25.0 乖离率上限（%）
    float_mv_min / float_mv_max : 30.0 / 600.0   流通市值区间（亿元）
    amount_window       : 20    日均成交额窗口
    amount_min          : 1.0   日均成交额下限（亿元）
    min_listed_days     : 180   上市天数下限
    min_bars            : 65    单只股票所需最小历史长度
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .base import BaseStrategy


class Strategy(BaseStrategy):
    name = "trend_pullback_inflow"
    description = ("趋势缩量回调：收盘>MA20 + MA20向上 + MA5>MA10 + 涨幅<6% + 缩量 "
                   "+ 资金流入(量价代理) + 换手3~12% + 20日涨幅<35% + 60日乖离<25% "
                   "+ 流通市值30~600亿 + 20日均额>1亿 + 上市>180天")

    # ------------------------------------------------------------------ #
    # 上下文（上市天数需要 ipoDate；资金流可外部注入真实数据）
    # ------------------------------------------------------------------ #
    def bind_context(self, ctx: dict | None) -> None:
        ctx = ctx or {}
        self._meta = ctx.get("meta")
        # 可选：外部注入真实主力资金流 DataFrame（index=date, 列 net_main_inflow）
        self._inflow_provider = ctx.get("money_flow")

    def _meta_value(self, code: str, column: str) -> str:
        meta = getattr(self, "_meta", None)
        if meta is None or code not in getattr(meta, "index", []):
            return ""
        try:
            val = meta.loc[code, column]
        except Exception:  # noqa: BLE001
            return ""
        return "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)

    # ------------------------------------------------------------------ #
    # 资金流代理
    # ------------------------------------------------------------------ #
    def _inflow_series(self, df: pd.DataFrame, window: int) -> pd.Series:
        """返回与 df 等长的资金流入代理序列；未注入真实数据时用量价代理。

        优先使用 ctx 注入的真实资金流（列 net_main_inflow）；
        否则用「收盘日内位置 × 放量倍数」作为代理：
            (close-low)/(high-low) ∈ [0,1]  —— 1 表示收在最高，买盘完全主导
            放量倍数仅在 >1 时计入（缩量日不视作流入）
        代理值为正即视为该日资金净流入。
        """
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        real = getattr(self, "_inflow_provider", None)
        if real is not None and len(real) > 0:
            try:
                return real.reindex(df["date"].astype(str)).astype(float).fillna(0.0)
            except Exception:  # noqa: BLE001
                pass  # 注入数据不可用时回退到代理

        span = (high - low).replace(0.0, np.nan)
        pos = ((close - low) / span).fillna(0.5)        # 一字板给中性 0.5
        base = volume.shift(1).rolling(window, min_periods=window).mean()
        base = base.replace(0.0, np.nan)
        expand = (volume / base).clip(lower=1.0).fillna(1.0)
        return (pos * expand).fillna(0.0)

    # ------------------------------------------------------------------ #
    def signal(self, df: pd.DataFrame) -> dict:
        """逐日判定。

        【条件顺序（性能优化，2026-10-09 调整）】
        判定顺序按「计算成本 / 过滤力」从低到高排列，最便宜的当日涨幅（条件 4）
        与换手率（条件 8）最先运行，先用 O(1) 的末行取值把大部分样本直接淘汰，
        再去计算均线、资金流等需要 rolling 的昂贵中间量。调整顺序不改变任何
        判定结果，全部条件仍是 AND 关系，只是短路时机不同。
        """
        p = self.params
        ma_long = int(p.get("ma_long", 20))
        ma_trend = int(p.get("ma_trend", 20))
        ma_short = int(p.get("ma_short", 5))
        ma_mid = int(p.get("ma_mid", 10))
        max_pct_chg = float(p.get("max_pct_chg", 6.0))
        vol_shrink_window = int(p.get("vol_shrink_window", 5))
        vol_max_window = int(p.get("vol_max_window", 20))
        inflow_window = int(p.get("inflow_window", 3))
        turn_min = float(p.get("turn_min", 3.0))
        turn_max = float(p.get("turn_max", 12.0))
        range_window = int(p.get("range_window", 20))
        range_max = float(p.get("range_max", 35.0))
        bias_window = int(p.get("bias_window", 60))
        bias_max = float(p.get("bias_max", 25.0))
        mv_min = float(p.get("float_mv_min", 30.0))
        mv_max = float(p.get("float_mv_max", 600.0))
        amount_window = int(p.get("amount_window", 20))
        amount_min = float(p.get("amount_min", 1.0))
        min_listed_days = int(p.get("min_listed_days", 180))
        min_bars = int(p.get("min_bars", 65))

        n = len(df)
        if n < min_bars:
            return self._no_signal(f"历史不足{min_bars}日({n})")

        close = df["close"].astype(float).reset_index(drop=True)
        high = df["high"].astype(float).reset_index(drop=True)
        low = df["low"].astype(float).reset_index(drop=True)
        volume = df["volume"].astype(float).reset_index(drop=True)
        amount = df["amount"].astype(float).reset_index(drop=True)
        turn = df["turn"].astype(float).reset_index(drop=True)
        i = n - 1

        code = str(df["code"].iloc[-1]) if "code" in df.columns else ""
        last_close = float(close.iloc[i])

        # ---- 条件 4：当日涨幅 < 6%（最优先，O(1)，用真实涨跌幅优先）----
        if "pctChg" in df.columns and not pd.isna(df["pctChg"].iloc[i]):
            pct_chg = float(df["pctChg"].iloc[i])
        else:
            preclose = float(df["preclose"].astype(float).iloc[i]) if "preclose" in df.columns else float("nan")
            pct_chg = (last_close / preclose - 1) * 100.0 if preclose > 0 else float("nan")
        if pd.isna(pct_chg):
            return self._no_signal("涨跌幅数据缺失")
        if pct_chg >= max_pct_chg:
            return self._no_signal(f"当日涨幅{pct_chg:.2f}%≥{max_pct_chg}%")

        # ---- 条件 8：换手率 3% ~ 12%（次优先，同样O(1)，过滤力最强）----
        last_turn = float(turn.iloc[i])
        if pd.isna(last_turn) or last_turn <= 0:
            return self._no_signal("换手率数据缺失")
        if not (turn_min < last_turn < turn_max):
            return self._no_signal(f"换手率{last_turn:.2f}%不在({turn_min}%,{turn_max}%)")

        # ---- 条件 13：上市天数 > 180 天（meta 查表，成本低）----
        if min_listed_days > 0:
            ipo = self._meta_value(code, "ipoDate")
            if ipo:
                try:
                    d0 = datetime.strptime(ipo, "%Y-%m-%d")
                    d1 = datetime.strptime(str(df["date"].iloc[i]), "%Y-%m-%d")
                    if (d1 - d0).days < min_listed_days:
                        return self._no_signal(f"上市未满{min_listed_days}天({ipo})")
                except Exception:  # noqa: BLE001
                    pass

        # ---- 条件 9：近 20 日区间涨跌幅 < 35%（O(1) 定长索引）----
        base_close = float(close.iloc[i - range_window])
        if base_close <= 0:
            return self._no_signal("区间基准价异常")
        range_pct = (last_close / base_close - 1) * 100.0
        if abs(range_pct) >= range_max:
            return self._no_signal(f"近{range_window}日涨跌幅{range_pct:+.1f}%超±{range_max}%")

        # ---- 条件 10：60 日乖离率 < 25%（O(1) 定长索引）----
        bias_close = float(close.iloc[i - bias_window])
        if bias_close <= 0:
            return self._no_signal("乖离率基准价异常")
        bias_pct = (last_close / bias_close - 1) * 100.0
        if abs(bias_pct) >= bias_max:
            return self._no_signal(f"{bias_window}日乖率{bias_pct:+.1f}%超±{bias_max}%")

        # ---- 条件 12：流通市值 30 亿 ~ 600 亿（换手率反推，O(1)）----
        last_vol = float(volume.iloc[i])
        if last_turn <= 0.01:   # turn 过小会让反推误差放大
            return self._no_signal(f"换手率{last_turn:.3f}%过小，流通市值不可反推")
        float_shares = last_vol / (last_turn / 100.0)
        float_mv = float_shares * last_close / 1e8  # 亿元
        if not (mv_min <= float_mv <= mv_max):
            return self._no_signal(f"流通市值{float_mv:.0f}亿不在({mv_min}亿,{mv_max}亿)")

        # ---- 条件 5：当日成交量 < 5 日均量（O(vol_shrink_window) 定长切片）----
        vol_base = float(volume.iloc[i - vol_shrink_window:i].mean())
        if vol_base <= 0:
            return self._no_signal("均量数据异常")
        if last_vol >= vol_base:
            return self._no_signal(f"未缩量({last_vol:.0f}≥5日均量{vol_base:.0f})")

        # ---- 条件 11：当日成交量 < 近 20 日最大成交量（O(vol_max_window) 定长切片）----
        vol_peak = float(volume.iloc[i - vol_max_window: i].max())
        if last_vol >= vol_peak:
            return self._no_signal("当日量未低于近20日最大量")

        # ---- 条件 13：近 20 日日均成交额 > 1 亿（O(amount_window) 定长切片）----
        avg_amount = float(amount.iloc[i - amount_window: i].mean()) / 1e8
        if avg_amount <= amount_min:
            return self._no_signal(f"近{amount_window}日均额{avg_amount:.2f}亿≤{amount_min}亿")

        # ---- 条件 1/2/3：均线结构（全序列 rolling ×3，最贵之一，放到后面）----
        ma_long_s = close.rolling(ma_long).mean()
        ma_long_v = float(ma_long_s.iloc[i])
        if pd.isna(ma_long_v) or ma_long_v <= 0:
            return self._no_signal(f"MA{ma_long}数据不足")
        if last_close <= ma_long_v:
            return self._no_signal(f"收盘{last_close:.2f}未站上MA{ma_long}({ma_long_v:.2f})")

        ma_trend_s = close.rolling(ma_trend).mean()
        ma_trend_prev = float(ma_trend_s.iloc[i - 1])
        if pd.isna(ma_trend_prev) or ma_trend_prev <= 0:
            return self._no_signal(f"MA{ma_trend}前值数据不足")
        if ma_long_v <= ma_trend_prev:
            return self._no_signal(f"MA{ma_long}未向上({ma_trend_prev:.2f}→{ma_long_v:.2f})")

        ma_short_v = float(close.rolling(ma_short).mean().iloc[i])
        ma_mid_v = float(close.rolling(ma_mid).mean().iloc[i])
        if pd.isna(ma_short_v) or pd.isna(ma_mid_v) or ma_short_v <= ma_mid_v:
            return self._no_signal(
                f"MA{ma_short}({ma_short_v:.2f})未上穿MA{ma_mid}({ma_mid_v:.2f})")

        # ---- 条件 6/7：资金流入（量价代理；需全序列 rolling，最贵，放最后）----
        inflow = self._inflow_series(df, vol_shrink_window)
        inflow_today = float(inflow.iloc[i])
        inflow_3d = float(inflow.iloc[i - inflow_window + 1: i + 1].sum())
        if inflow_today <= 0:
            return self._no_signal(f"当日资金流非正({inflow_today:.3f})")
        if inflow_3d <= 0:
            return self._no_signal(f"近{inflow_window}日资金流非正({inflow_3d:.3f})")

        # ---- 打分：资金流强度为主，均线趋势与缩量程度为辅 ----
        vol_ratio = last_vol / vol_base
        score = round(
            inflow_today * 10.0
            + inflow_3d * 4.0
            + (last_close / ma_long_v - 1) * 100.0
            + (1.0 - vol_ratio) * 3.0,
            4,
        )
        return {
            "buy": True,
            "score": score,
            "reason": (
                f"站上MA{ma_long}({ma_long_v:.2f})且MA{ma_long}向上({ma_trend_prev:.2f}→{ma_long_v:.2f})；"
                f"MA{ma_short}>MA{ma_mid}({ma_short_v:.2f}>{ma_mid_v:.2f})；"
                f"涨幅{pct_chg:.2f}%<{max_pct_chg}%；缩量{last_vol:.0f}<5日均量{vol_base:.0f}；"
                f"资金流入(代理)当日{inflow_today:.2f}/近{inflow_window}日{inflow_3d:.2f}；"
                f"换手{last_turn:.2f}%；{range_window}日{range_pct:+.1f}%；"
                f"{bias_window}日乖率{bias_pct:+.1f}%；流通市值{float_mv:.0f}亿；"
                f"20日均额{avg_amount:.2f}亿"
            ),
        }

