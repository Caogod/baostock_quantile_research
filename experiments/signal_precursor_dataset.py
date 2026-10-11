"""信号锚点数据集构建：以策略信号为锚，抽取「信号前 N 天」的日线及派生特征。

用途
----
把策略的所有历史信号对齐到同一时间轴（锚点 = 信号日），回溯其**前 N 个交易日**
（默认 60）的日线数据与派生特征，产出可做前兆形态分析/机器学习的长表数据集。

与回测引擎的区别（刻意的）
------------------------
- **不施加冷却期**：回测里同一股票在持仓期内不重复记信号；此处要的是"策略在哪些
  日子真实触发过"，故逐日全量判定、全部留作锚点，避免锚点被人为稀疏化。
- **不做股票池过滤**：库内全部代码都参与扫描（回测的板块/ST 过滤由策略自身条件承担）。

产出
----
1. `precursor_<strategy>.csv` —— 长表，每行 = 一个信号 × 一个相对交易日偏移
   列：
     strategy, code, signal_date, signal_idx, offset, date,
     open, high, low, close, preclose, volume, amount, turn, pctChg, isST,
     ma5, ma10, ma20, ma60, vol_ma5, vol_ma20,
     ret1, ret5, dist_ma20, vol_ratio, close_norm
   offset ∈ [-window, -1]（-60 表示信号前第 60 个交易日，-1 表示信号前一日）；
   close_norm = close / 锚点日收盘 - 1（便于把所有信号叠在同一张图上比较形态）。
2. `outcomes_<strategy>.csv` —— 每行 = 一个信号的前向结局（用于盈亏归因）
   列：strategy, code, signal_date, score, reason,
       signal_close, signal_turn, signal_pct_chg,
       entry_date, entry_price, ret_3d, ret_5d, ret_10d, win_5d, index_ret_5d

用法
----
    python experiments/signal_precursor_dataset.py \
        --strategies consecutive_up5,trend_pullback_inflow \
        --library data_library/combined_daily.parquet \
        --window 60 --out research_output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.market_context import build_context            # noqa: E402
from core.strategy_loader import load_strategy_class, load_yaml  # noqa: E402

PRECURSOR_COLUMNS = [
    "strategy", "code", "signal_date", "signal_idx", "offset", "date",
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "pctChg", "isST", "tradestatus",
    "ma5", "ma10", "ma20", "ma60", "vol_ma5", "vol_ma20",
    "ret1", "ret5", "dist_ma20", "vol_ratio", "close_norm",
]
OUTCOME_COLUMNS = [
    "strategy", "code", "signal_date", "score", "reason",
    "signal_close", "signal_turn", "signal_pct_chg",
    "entry_date", "entry_price",
    "ret_3d", "ret_5d", "ret_10d", "win_5d", "index_ret_5d",
]


def _load_strategy(names: list[str], strategies_yaml: Path) -> list[tuple[str, object]]:
    cfg = load_yaml(strategies_yaml)
    by_name = {s.get("name"): s for s in cfg.get("strategies", [])}
    out: list[tuple[str, object]] = []
    for n in names:
        item = by_name.get(n)
        if item is None:
            raise SystemExit(f"[error] 配置中找不到策略 {n}")
        cls = load_strategy_class(item["script"])
        out.append((n, cls(item.get("params") or {})))
    return out


def _features_window(g: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """把 g.iloc[lo:hi+1] 转成带派生特征的行表（lo/hi 为窗口内位置）。"""
    c = g["close"].astype(float)
    v = g["volume"].astype(float)
    ma5 = c.rolling(5).mean()
    ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    vol_ma5 = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()

    def pick(s: pd.Series) -> np.ndarray:
        return s.to_numpy(dtype=float)[lo:hi + 1]

    close_arr = pick(c)
    anchor_close = float(c.iloc[hi + 1]) if hi + 1 < len(c) else float(c.iloc[-1])
    df = pd.DataFrame({
        "date": g["date"].astype(str).to_numpy()[lo:hi + 1],
        "open": g["open"].astype(float).to_numpy()[lo:hi + 1],
        "high": g["high"].astype(float).to_numpy()[lo:hi + 1],
        "low": g["low"].astype(float).to_numpy()[lo:hi + 1],
        "close": close_arr,
        "preclose": g["preclose"].astype(float).to_numpy()[lo:hi + 1],
        "volume": g["volume"].astype(float).to_numpy()[lo:hi + 1],
        "amount": g["amount"].astype(float).to_numpy()[lo:hi + 1],
        "turn": g["turn"].astype(float).to_numpy()[lo:hi + 1],
        "pctChg": g["pctChg"].astype(float).to_numpy()[lo:hi + 1],
        "isST": g["isST"].astype(str).to_numpy()[lo:hi + 1],
        "tradestatus": g["tradestatus"].astype(str).to_numpy()[lo:hi + 1],
        "ma5": pick(ma5), "ma10": pick(ma10),
        "ma20": pick(ma20), "ma60": pick(ma60),
        "vol_ma5": pick(vol_ma5), "vol_ma20": pick(vol_ma20),
        "ret1": pick(c.pct_change() * 100.0),
        "ret5": pick((c / c.shift(5) - 1.0) * 100.0),
        "dist_ma20": pick((c / ma20 - 1.0) * 100.0),
        "vol_ratio": pick(v / vol_ma5.replace(0.0, np.nan)),
    })
    df["close_norm"] = (df["close"] / anchor_close - 1.0) * 100.0
    # 相对信号日的交易日偏移：窗口末行是信号前一日 → -1，首行 → -window
    df["offset"] = np.arange(-len(df), 0)
    return df


def build_one(
    name: str,
    strat: object,
    daily: pd.DataFrame,
    window: int,
    commission: float,
    market_close: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre_rows: list[pd.DataFrame] = []
    out_rows: list[dict] = []

    for code, g in daily.groupby("code", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < window + 2:
            continue
        close = g["close"].astype(float).to_numpy()
        open_ = g["open"].astype(float).to_numpy()
        dates = g["date"].astype(str).to_numpy()

        for i in range(n):
            # 需要至少 window 天前置历史（窗口为 offset -window .. -1）
            if i < window:
                continue
            try:
                sig = strat.signal(g.iloc[: i + 1])
            except Exception:  # noqa: BLE001
                continue
            if not sig.get("buy"):
                continue

            # ---- 前 window 天（offset -window .. -1）----
            dfw = _features_window(g, i - window, i - 1)
            dfw.insert(0, "signal_idx", i)
            dfw.insert(0, "signal_date", dates[i])
            dfw.insert(0, "code", code)
            dfw.insert(0, "strategy", name)
            pre_rows.append(dfw)

            # ---- 前向结局（T+1 开盘入场，持有 h 个交易日收盘卖出）----
            def fwd(h: int) -> float:
                e, x = i + 1, i + 1 + h
                if e >= n or x >= n or open_[e] <= 0:
                    return float("nan")
                return (close[x] / open_[e] - 1.0 - 2 * commission) * 100.0

            idx_ret = float("nan")
            if market_close is not None:
                try:
                    d_e, d_x = dates[i + 1], dates[min(i + 1 + 5, n - 1)]
                    p_e, p_x = float(market_close.loc[d_e]), float(market_close.loc[d_x])
                    if p_e > 0:
                        idx_ret = (p_x / p_e - 1.0) * 100.0
                except Exception:  # noqa: BLE001
                    pass

            out_rows.append({
                "strategy": name,
                "code": code,
                "signal_date": dates[i],
                "score": sig.get("score"),
                "reason": sig.get("reason"),
                "signal_close": close[i],
                "signal_turn": float(g["turn"].astype(float).iloc[i]),
                "signal_pct_chg": float(g["pctChg"].astype(float).iloc[i]),
                "entry_date": dates[i + 1] if i + 1 < n else "",
                "entry_price": open_[i + 1] if i + 1 < n else float("nan"),
                "ret_3d": fwd(3),
                "ret_5d": fwd(5),
                "ret_10d": fwd(10),
                "index_ret_5d": idx_ret,
            })

    prec = (pd.concat(pre_rows, ignore_index=True) if pre_rows
            else pd.DataFrame(columns=PRECURSOR_COLUMNS))
    outc = (pd.DataFrame(out_rows) if out_rows
            else pd.DataFrame(columns=OUTCOME_COLUMNS))
    if not outc.empty:
        outc["win_5d"] = (outc["ret_5d"] > 0).astype("Int64")
    return prec[PRECURSOR_COLUMNS], outc[OUTCOME_COLUMNS]


def main() -> int:
    ap = argparse.ArgumentParser(description="构建策略信号前 N 天数据集")
    ap.add_argument("--strategies", required=True, help="策略名，逗号分隔")
    ap.add_argument("--library", default="data_library/combined_daily.parquet")
    ap.add_argument("--strategies-yaml", default="config/strategies.yaml")
    ap.add_argument("--window", type=int, default=60, help="信号前回溯交易日数")
    ap.add_argument("--commission", type=float, default=0.001)
    ap.add_argument("--out", default="research_output")
    args = ap.parse_args()

    names = [x.strip() for x in args.strategies.split(",") if x.strip()]
    strategies = _load_strategy(names, ROOT / args.strategies_yaml)

    print("[ctx] 构建市场上下文（指数 + 证券元数据）…")
    ctx = build_context(start="2025-11-12", end="2026-10-09")
    for _, s in strategies:
        s.bind_context(ctx)

    daily = pd.read_parquet(ROOT / args.library)
    daily["date"] = daily["date"].astype(str)
    daily = daily.sort_values(["code", "date"], kind="stable")
    print(f"[data] {args.library}: {daily['code'].nunique()} 只，{len(daily)} 行，"
          f"{daily['date'].min()} ~ {daily['date'].max()}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_p, all_o = [], []
    for name, strat in strategies:
        print(f"\n[scan] {name} 逐日扫描（不施加冷却期）…")
        prec, outc = build_one(name, strat, daily, args.window,
                               args.commission, ctx.get("market_close"))
        p_path = out_dir / f"precursor_{name}.csv"
        o_path = out_dir / f"outcomes_{name}.csv"
        prec.to_csv(p_path, index=False, encoding="utf-8-sig")
        outc.to_csv(o_path, index=False, encoding="utf-8-sig")
        print(f"  锚点 {len(outc)} 个，前兆长表 {len(prec)} 行 "
              f"({len(prec) / max(len(outc), 1):.1f} 行/锚点)")
        print(f"  -> {p_path.relative_to(ROOT)}")
        print(f"  -> {o_path.relative_to(ROOT)}")
        all_p.append(prec)
        all_o.append(outc)

    if len(all_p) > 1:
        pd.concat(all_p, ignore_index=True).to_csv(
            out_dir / "precursor_all.csv", index=False, encoding="utf-8-sig")
        pd.concat(all_o, ignore_index=True).to_csv(
            out_dir / "outcomes_all.csv", index=False, encoding="utf-8-sig")
        print(f"\n[合并] precursor_all.csv / outcomes_all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
