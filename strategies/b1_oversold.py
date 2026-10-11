"""策略 B1 稳健超跌型 —— 同花顺远航版条件选股公式移植。

原公式（THS）：
    NOTST  := NOT(NAMELIKE('ST')) AND NOT(NAMELIKE('退'));
    ISAB   := CODELIKE('60') OR CODELIKE('00') OR CODELIKE('30');
    LISTED := BARSCOUNT(CLOSE) > 250;
    ZT     := IF(CODELIKE('30') OR CODELIKE('688'), 1.195, 1.095);
    LC     := REF(CLOSE, 1);
    RSI6   := SMA(MAX(CLOSE-LC,0),6,1) / SMA(ABS(CLOSE-LC),6,1) * 100;
    AMP    := (HIGH - LOW) / LC * 100;
    CD2    := CLOSE < LC AND REF(CLOSE,1) < REF(CLOSE,2);
    NOLU   := COUNT(REF(CLOSE,1)/REF(CLOSE,2) >= ZT, 20) = 0;
    MKTDN  := "399317$CLOSE" < REF("399317$CLOSE", 1);
    NOTYZ   := HIGH > LOW * 1.0001;
    NOTSEAL := CLOSE < LC * ZT - 0.005;
    RMAX60  := REF(HHV(ABS(CLOSE/LC - 1), 60), 1);
    NOT5    := NOT(RMAX60 >= 0.047 AND RMAX60 <= 0.053);
    CORE1  := RSI6 < 20 AND AMP > 8 AND CLOSE > 20;
    CORE2  := NOLU AND CD2 AND MKTDN;
    CANBUY := NOTYZ AND NOTSEAL AND NOT5 AND AMP > 1;
    XG: NOTST AND ISAB AND LISTED AND CORE1 AND CORE2 AND CANBUY;

【条件顺序：按过滤代价由低到高手工排定（2026-10-12 取消向量化预筛）】
    全部条件为 AND 关系，顺序不影响判定结果，只影响性能：越靠前越便宜、
    越早短路。代价评估与排定后的执行次序：
        1. 板块前缀            常量，逐股判定
        2. ST / 退市           常量（逐日 isST）
        3. 上市满 N 个交易日   meta 查表
        4. 振幅 > amp_min      O(1) 末行 high/low/last_lc
        5. 振幅 > amp_canbuy   O(1)，阈值更松
        6. 收盘价 > price_min  O(1)
        7. 连跌 2 日           O(1)
        8. 非一字板            O(1)
        9. 未封涨停            O(1)
       10. 指数下跌            O(1) 指数查表
       11. 近20日无涨停        O(20) 定长切片
       12. RMAX60 非5%限制股   O(60) 定长切片
       13. RSI6 < rsi_max      EWM 全序列（最贵）→ 放最后
    其中 RSI6 是原先最靠前的开销大头，改为最后执行后，只在其余条件全部通过的
    极少数候选日才触发 EWM 计算。

移植说明（与原公式的三处口径差异）：
1. ST 判定：优先用日线自带的 `isST` 字段（baostock 逐日标记，**无前视偏差**）；
   该字段缺失时回退为 code_name 含 ST/退（当前名称，存在前视偏差）。
2. 上市满一年：`BARSCOUNT(CLOSE) > 250` 表示已有 250 根 K 线。本地样本库只有
   250 根、无法据此判断，故改用证券元数据的 `ipoDate` 换算为日历天数
   （250 个交易日 ≈ 365/244×250 ≈ 374 个自然日）。
3. 市场环境 MKTDN：需要指数日线（默认 399317 国证A指），由 `bind_context`
   注入；入口未注入时该条件按"不满足"处理（可设 require_market_down=false 关闭）。
4. `CLOSE > 20` 用策略运行时的复权价（项目默认前复权，见 settings.adjustflag）。
5. THS `SMA(X,N,M) = (M*X + (N-M)*Y')/N`，等价于 alpha = M/N 的 EWM。

参数：
    rsi_period / rsi_max : 6 / 20    RSI 周期与超跌阈值
    amp_min    : 8.0     CORE1 当日振幅下限(%)
    amp_canbuy : 1.0     CANBUY 当日振幅下限(%)
    price_min  : 20.0    收盘价下限（<=0 不启用）
    lu_window  : 20      近 N 日无涨停的统计窗口
    rmax_window: 60      RMAX 统计窗口（用于排除 5% 涨跌幅限制的 ST 股）
    zt_main / zt_gem : 1.095 / 1.195   主板 / 创业板·科创板 涨停比例
    boards     : 60,00,30    允许的板块代码前缀（空表示不限）
    exclude_st : True    是否排除 ST / 退市
    min_listed_bars : 250   上市满 N 个交易日（0 不校验）
    require_market_down : True   是否要求当日全市场（指数）下跌
    market_index : 399317   市场环境所用指数
    min_bars   : 65     单只股票所需的最小历史长度
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .base import BaseStrategy


def _ths_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    """同花顺 SMA(X,N,M)：Y = (M*X + (N-M)*Y') / N，即 alpha=M/N 的 EWM。"""
    return series.ewm(alpha=m / n, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 6) -> pd.Series:
    """RSI = SMA(MAX(C-LC,0),N,1) / SMA(ABS(C-LC),N,1) * 100。"""
    diff = close - close.shift(1)
    gain = diff.clip(lower=0.0)
    absd = diff.abs()
    num = _ths_sma(gain, period, 1)
    den = _ths_sma(absd, period, 1)
    return num / den.replace(0.0, np.nan) * 100.0


class Strategy(BaseStrategy):
    name = "b1_oversold"
    description = ("B1 稳健超跌型：RSI6<20 + 当日振幅>8% + 连跌2日 + 近20日无涨停 "
                   "+ 当日全市场下跌 + 非一字板/未封板")

    # ------------------------------------------------------------------ #
    # 上下文
    # ------------------------------------------------------------------ #
    def bind_context(self, ctx: dict | None) -> None:
        ctx = ctx or {}
        self._market_down = ctx.get("market_down")
        self._market_close = ctx.get("market_close")
        self._meta = ctx.get("meta")
        if self.params.get("require_market_down", True) and self._market_down is None:
            print("[warn] b1_oversold: 缺少市场指数上下文，MKTDN 条件恒不满足（"
                  "可把 require_market_down 设为 false）")

    def _meta_value(self, code: str, column: str) -> str:
        meta = getattr(self, "_meta", None)
        if meta is None or code not in meta.index:
            return ""
        try:
            val = meta.loc[code, column]
        except Exception:  # noqa: BLE001
            return ""
        return "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)

    # ------------------------------------------------------------------ #
    def signal(self, df: pd.DataFrame) -> dict:
        p = self.params
        rsi_period = int(p.get("rsi_period", 6))
        rsi_max = float(p.get("rsi_max", 20.0))
        amp_min = float(p.get("amp_min", 8.0))
        amp_canbuy = float(p.get("amp_canbuy", 1.0))
        price_min = float(p.get("price_min", 20.0))
        lu_window = int(p.get("lu_window", 20))
        rmax_window = int(p.get("rmax_window", 60))
        zt_main = float(p.get("zt_main", 1.095))
        zt_gem = float(p.get("zt_gem", 1.195))
        boards = [b.strip() for b in str(p.get("boards", "60,00,30")).split(",") if b.strip()]
        exclude_st = bool(p.get("exclude_st", True))
        min_listed_bars = int(p.get("min_listed_bars", 250))
        require_market_down = bool(p.get("require_market_down", True))
        min_bars = int(p.get("min_bars", 65))

        n = len(df)
        if n < min_bars:
            return self._no_signal(f"历史不足{min_bars}日({n})")

        code = str(df["code"].iloc[-1]) if "code" in df.columns else ""
        num = code.split(".")[-1]

        # ---- 基础过滤：板块 ----
        if boards and not any(num.startswith(b) for b in boards):
            return self._no_signal(f"板块不符({num})")

        # ---- 基础过滤：ST / 退市（优先用逐日 isST，无前视偏差）----
        if exclude_st:
            if "isST" in df.columns:
                if str(df["isST"].iloc[-1]).strip() in {"1", "1.0", "True", "true"}:
                    return self._no_signal("ST/风险警示股")
            else:
                name = self._meta_value(code, "name")
                if ("ST" in name) or ("退" in name):
                    return self._no_signal(f"ST/退市({name})")

        # ---- 基础过滤：上市满 N 个交易日 ----
        if min_listed_bars > 0:
            ipo = self._meta_value(code, "ipoDate")
            if ipo:
                try:
                    d0 = datetime.strptime(ipo, "%Y-%m-%d")
                    d1 = datetime.strptime(str(df["date"].iloc[-1]), "%Y-%m-%d")
                    # 250 个交易日 ≈ 365/244 × 250 ≈ 374 个自然日
                    need_days = int(min_listed_bars * 365 / 244)
                    if (d1 - d0).days < need_days:
                        return self._no_signal(f"上市未满{min_listed_bars}个交易日({ipo})")
                except Exception:  # noqa: BLE001
                    pass

        close = df["close"].astype(float).reset_index(drop=True)
        high = df["high"].astype(float).reset_index(drop=True)
        low = df["low"].astype(float).reset_index(drop=True)
        i = n - 1

        zt = zt_gem if (num.startswith("30") or num.startswith("688")) else zt_main

        last_close = float(close.iloc[i])
        last_lc = float(close.iloc[i - 1])       # REF(CLOSE, 1)
        last_high = float(high.iloc[i])
        last_low = float(low.iloc[i])
        if last_lc <= 0:
            return self._no_signal("前收为0，振幅不可计算")

        # ============================================================== #
        # 条件顺序：按「过滤代价」由低到高手工排定（2026-10-12 取消向量化预筛）
        # 全部条件为 AND 关系，顺序不影响判定结果，只影响性能：越靠前越便宜，
        # 一旦不满足即提前 return，昂贵的计算只留给少数候选日。
        # 代价评估：
        #   O(1) 末行/相邻行取值  → 振幅、收盘价、连跌2日、一字板、未封板、指数查表
        #   O(20)/O(60) 定长切片  → 近20日无涨停、RMAX60
        #   EWM 全序列（最贵）    → RSI6，因此放最后，只在其余条件全通过后才计算
        # ============================================================== #

        # ---- 1) 振幅（O(1)，同时服务 CORE1 的 amp_min 与 CANBUY 的 amp_canbuy）----
        amp = (last_high - last_low) / last_lc * 100.0
        if amp <= amp_min:
            return self._no_signal(f"振幅{amp:.1f}%未达{amp_min}%")
        if amp <= amp_canbuy:
            return self._no_signal(f"振幅{amp:.1f}%未达{amp_canbuy}%（CANBUY）")

        # ---- 2) 收盘价下限（O(1)）----
        if price_min > 0 and last_close <= price_min:
            return self._no_signal(f"收盘{last_close:.2f}未达{price_min}")

        # ---- 3) 连跌 2 日（O(1)）----
        prev1 = float(close.iloc[i - 1])
        prev2 = float(close.iloc[i - 2])
        if not (last_close < prev1 and prev1 < prev2):
            return self._no_signal("未连跌2日")

        # ---- 4) 非一字板（O(1)）----
        notyz = last_high > last_low * 1.0001
        if not notyz:
            return self._no_signal("一字板")

        # ---- 5) 未封涨停（O(1)）----
        notseal = last_close < last_lc * zt - 0.005
        if not notseal:
            return self._no_signal("收盘已封涨停")

        # ---- 6) 当日全市场下跌（指数查表，O(1)）----
        if require_market_down:
            md = getattr(self, "_market_down", None)
            if md is None:
                return self._no_signal("缺少市场指数上下文")
            day = str(df["date"].iloc[i])
            try:
                mktdn_val = md.loc[day]
            except Exception:  # noqa: BLE001
                mktdn_val = None
            if mktdn_val is None or pd.isna(mktdn_val):
                return self._no_signal(f"{day} 无指数行情")
            mktdn = bool(mktdn_val)
            if not mktdn:
                return self._no_signal(f"{day} 指数未收跌")
        else:
            mktdn = True

        # ---- 7) 近 lu_window 日无涨停（O(20) 定长切片）----
        ratio = close / close.shift(1)           # ratio[j] = close[j]/close[j-1]
        win_lu = ratio.iloc[i - lu_window:i]     # REF(C,1)/REF(C,2) 的 20 个取值
        if bool((win_lu >= zt).any()):
            return self._no_signal(f"近{lu_window}日内出现涨停")

        # ---- 8) RMAX60：排除 5% 涨跌幅限制股（O(60) 定长切片）----
        rmax = float((ratio.iloc[i - rmax_window:i] - 1.0).abs().max())
        not5 = not (0.047 <= rmax <= 0.053)
        if not not5:
            return self._no_signal(f"疑为5%涨跌幅限制股(RMAX{rmax:.2%})")

        # ---- 9) RSI6 < rsi_max（最贵：EWM 全序列；放在最后，仅少数候选日执行）----
        rsi = float(_rsi(close, rsi_period).iloc[i])
        if pd.isna(rsi) or not (rsi < rsi_max):
            return self._no_signal(f"RSI{rsi_period}={rsi:.1f}未低于{rsi_max}")

        score = round(amp + (rsi_max - rsi), 4)
        return {
            "buy": True,
            "score": score,
            "reason": (
                f"RSI{rsi_period}={rsi:.1f}<{rsi_max}；振幅{amp:.1f}%>{amp_min}%；"
                f"收盘{last_close:.2f}>{price_min}；近{lu_window}日无涨停；连跌2日；"
                f"指数收跌；振幅比{rmax:.2%}（非5%限制股）"
            ),
        }
