"""信号锚点数据集构建：以策略信号为锚，抽取「信号前 N 天」的日线及派生特征。

两种锚点模式（--anchor）
------------------------
- `signal`（默认）：锚点 = 策略信号日本身（序列**末端**）。
- `runstart`：锚点 = 该轮**连续上涨序列的首日**（即「大阳线起点」/ 起涨点）。
  做法：从信号日向前走，只要当日收盘高于前一日就继续前移，直到遇到第一根
  非上涨日，得到的首根阳线即起涨点。

为什么要有 runstart 模式
------------------------
`signal` 模式的前 60 天窗口会**包含上涨序列本身**（最后 5 根阳线就在窗口内），
所以它描述的是"爆拉之后"的状态，无法当作事前信号。
`runstart` 模式把锚点移到起涨点，前 60 天窗口就**完全落在起涨之前**，
此时窗口内的共同特征才是"能不能提前埋伏"的候选信号。

与回测引擎的区别（刻意的）
------------------------
- **不施加冷却期**：要的是"策略在哪些日子真实触发过"，全部留作锚点。
- **不做股票池过滤**：库内全部代码都参与扫描（过滤由策略自身条件承担）。

产出（`--tag` 非空时文件名追加 `_<tag>`）
----
1. `precursor_<strategy>[_<tag>].csv` —— 长表，每行 = 锚点 × 相对交易日偏移
     strategy, code, signal_date, origin_signal_date, run_len, signal_idx,
     offset, date, open, high, low, close, preclose, volume, amount,
     turn, pctChg, isST, tradestatus,
     ma5, ma10, ma20, ma60, vol_ma5, vol_ma20,
     ret1, ret5, dist_ma20, vol_ratio, close_norm, close_norm_origin
   - `signal_date`：**本次使用的锚点日**（runstart 模式下即起涨点）
   - `origin_signal_date`：策略原始信号日
   - `offset` ∈ [-window, -1]（-60 = 锚点前第 60 个交易日）
   - `close_norm` = close / 锚点日收盘 - 1（百分数）
   - `close_norm_origin` = close / 原始信号日收盘 - 1（便于看整段上涨）
2. `outcomes_<strategy>[_<tag>].csv` —— 每行 = 一个锚点的前向结局
     strategy, code, signal_date, origin_signal_date, run_len,
     anchor_close, anchor_turn, anchor_pct_chg, score, reason,
     entry_date, entry_price,
     ret_3d, ret_5d, ret_10d, ret_20d, win_5d, index_ret_5d,
     signal_close, surge_pct
   - 前向结局一律以**锚点日**为基准：锚点日次日开盘入场、持有 h 个交易日收盘卖出
   - `surge_pct`：从锚点收盘到原始信号日收盘的涨幅（本次上涨序列的幅度）

用法
----
    # 默认：锚点 = 信号日
    python experiments/signal_precursor_dataset.py --strategies consecutive_up5 \\
        --window 60 --out research_output

    # 锚点 = 起涨点（大阳线起点），文件名带 _runstart
    python experiments/signal_precursor_dataset.py --strategies consecutive_up5 \\
        --anchor runstart --window 60 --out research_output --tag runstart
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
    "strategy", "code", "signal_date", "origin_signal_date", "run_len",
    "signal_idx", "offset", "date",
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "pctChg", "isST", "tradestatus",
    "ma5", "ma10", "ma20", "ma60", "vol_ma5", "vol_ma20",
    "ret1", "ret5", "dist_ma20", "vol_ratio", "close_norm", "close_norm_origin",
]
OUTCOME_COLUMNS = [
    "strategy", "code", "signal_date", "origin_signal_date", "run_len",
    "anchor_close", "anchor_turn", "anchor_pct_chg", "score", "reason",
    "entry_date", "entry_price",
    "ret_3d", "ret_5d", "ret_10d", "ret_20d", "win_5d", "index_ret_5d",
    "signal_close", "surge_pct",
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


def run_start_index(close: np.ndarray, i: int) -> int:
    """返回以 i 结尾的连续上涨序列的**首日**索引（起涨点 / 大阳线起点）。

    当日收盘 > 前一日收盘 即视为"上涨日"。从 i 往前推，只要当日仍是上涨日就继续
    前移；循环退出时停在首根"非上涨日"上，**再 +1 才是上涨序列的首日**。

    注意这里有一个易错点：初始写法（直接返回循环结束时的 s）会把锚点落在
    「起涨前一日」上，表现为锚点日中位涨幅为负——那就不是"大阳线起点"了。
    另外对边界做了保护：若 close[i] 本身不是上涨日（理论上策略已保证不会），
    则直接返回 i，避免返回 i+1 这种越界锚点。
    """
    s = i
    while s >= 1 and close[s] > close[s - 1]:
        s -= 1
    # 循环退出时 s 为首根非上涨日（或 0），序列首日 = s + 1；再夹到 i 以内
    return min(s + 1, i)


def _features_window(g: pd.DataFrame, lo: int, hi: int,
                     anchor_close: float, origin_close: float) -> pd.DataFrame:
    """把 g.iloc[lo:hi] 转成带派生特征的行表（hi 不含）。"""
    c = g["close"].astype(float)
    v = g["volume"].astype(float)
    ma5, ma10 = c.rolling(5).mean(), c.rolling(10).mean()
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    vol_ma5, vol_ma20 = v.rolling(5).mean(), v.rolling(20).mean()

    def pick(s: pd.Series) -> np.ndarray:
        return s.to_numpy(dtype=float)[lo:hi]

    df = pd.DataFrame({
        "date": g["date"].astype(str).to_numpy()[lo:hi],
        "open": g["open"].astype(float).to_numpy()[lo:hi],
        "high": g["high"].astype(float).to_numpy()[lo:hi],
        "low": g["low"].astype(float).to_numpy()[lo:hi],
        "close": pick(c),
        "preclose": g["preclose"].astype(float).to_numpy()[lo:hi],
        "volume": g["volume"].astype(float).to_numpy()[lo:hi],
        "amount": g["amount"].astype(float).to_numpy()[lo:hi],
        "turn": g["turn"].astype(float).to_numpy()[lo:hi],
        "pctChg": g["pctChg"].astype(float).to_numpy()[lo:hi],
        "isST": g["isST"].astype(str).to_numpy()[lo:hi],
        "tradestatus": g["tradestatus"].astype(str).to_numpy()[lo:hi],
        "ma5": pick(ma5), "ma10": pick(ma10),
        "ma20": pick(ma20), "ma60": pick(ma60),
        "vol_ma5": pick(vol_ma5), "vol_ma20": pick(vol_ma20),
        "ret1": pick(c.pct_change() * 100.0),
        "ret5": pick((c / c.shift(5) - 1.0) * 100.0),
        "dist_ma20": pick((c / ma20 - 1.0) * 100.0),
        "vol_ratio": pick(v / vol_ma5.replace(0.0, np.nan)),
    })
    df["close_norm"] = (df["close"] / anchor_close - 1.0) * 100.0
    df["close_norm_origin"] = (df["close"] / origin_close - 1.0) * 100.0
    # 相对锚点的交易日偏移：末行是锚点前一日 → -1，首行 → -window
    df["offset"] = np.arange(-len(df), 0)
    return df


def build_one(
    name: str,
    strat: object,
    daily: pd.DataFrame,
    window: int,
    commission: float,
    market_close: pd.Series | None,
    anchor_mode: str = "signal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre_rows: list[pd.DataFrame] = []
    out_rows: list[dict] = []
    seen_anchor: set[tuple[str, str]] = set()

    for code, g in daily.groupby("code", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < window + 2:
            continue
        close = g["close"].astype(float).to_numpy()
        open_ = g["open"].astype(float).to_numpy()
        dates = g["date"].astype(str).to_numpy()

        for i in range(n):
            if i < window:
                continue
            try:
                sig = strat.signal(g.iloc[: i + 1])
            except Exception:  # noqa: BLE001
                continue
            if not sig.get("buy"):
                continue

            # ---- 决定锚点：信号日 或 起涨点 ----
            a = run_start_index(close, i) if anchor_mode == "runstart" else i
            # 去重：一轮长上涨序列会让策略在连续多日重复触发，但它们映射到**同一个**
            # 起涨点。不去重会让同一事件被重复计数（并放大后续统计），保留首次触发即可。
            key = (code, dates[a])
            if key in seen_anchor:
                continue
            seen_anchor.add(key)
            run_len = i - a + 1
            # 锚点前需要完整 window 天历史
            if a < window:
                continue

            anchor_close = float(close[a])
            origin_close = float(close[i])
            if anchor_close <= 0:
                continue

            dfw = _features_window(g, a - window, a, anchor_close, origin_close)
            dfw.insert(0, "run_len", run_len)
            dfw.insert(0, "origin_signal_date", dates[i])
            dfw.insert(0, "signal_date", dates[a])
            dfw.insert(0, "code", code)
            dfw.insert(0, "strategy", name)
            dfw["signal_idx"] = a
            pre_rows.append(dfw)

            # ---- 前向结局：一律以锚点为基准（锚点次日开盘入场）----
            def fwd(h: int) -> float:
                e, x = a + 1, a + 1 + h
                if e >= n or x >= n or open_[e] <= 0:
                    return float("nan")
                return (close[x] / open_[e] - 1.0 - 2 * commission) * 100.0

            idx_ret = float("nan")
            if market_close is not None and a + 1 < n:
                try:
                    d_e = dates[a + 1]
                    d_x = dates[min(a + 1 + 5, n - 1)]
                    p_e, p_x = float(market_close.loc[d_e]), float(market_close.loc[d_x])
                    if p_e > 0:
                        idx_ret = (p_x / p_e - 1.0) * 100.0
                except Exception:  # noqa: BLE001
                    pass

            out_rows.append({
                "strategy": name,
                "code": code,
                "signal_date": dates[a],
                "origin_signal_date": dates[i],
                "run_len": run_len,
                "anchor_close": anchor_close,
                "anchor_turn": float(g["turn"].astype(float).iloc[a]),
                "anchor_pct_chg": float(g["pctChg"].astype(float).iloc[a]),
                "score": sig.get("score"),
                "reason": sig.get("reason"),
                "entry_date": dates[a + 1] if a + 1 < n else "",
                "entry_price": open_[a + 1] if a + 1 < n else float("nan"),
                "ret_3d": fwd(3),
                "ret_5d": fwd(5),
                "ret_10d": fwd(10),
                "ret_20d": fwd(20),
                "index_ret_5d": idx_ret,
                "signal_close": origin_close,
                "surge_pct": (origin_close / anchor_close - 1.0) * 100.0,
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
    ap.add_argument("--window", type=int, default=60, help="锚点前回溯交易日数")
    ap.add_argument("--anchor", choices=["signal", "runstart"], default="signal",
                    help="signal=锚点为信号日；runstart=锚点为该轮上涨序列首日（大阳线起点）")
    ap.add_argument("--commission", type=float, default=0.001)
    ap.add_argument("--tag", default="", help="输出文件名后缀（非空时追加 _<tag>）")
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
    print(f"[mode] 锚点模式 = {args.anchor}，窗口 = {args.window} 个交易日")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    all_p, all_o = [], []
    for name, strat in strategies:
        print(f"\n[scan] {name} 逐日扫描（不施加冷却期）…")
        prec, outc = build_one(name, strat, daily, args.window,
                               args.commission, ctx.get("market_close"), args.anchor)
        p_path = out_dir / f"precursor_{name}{suffix}.csv"
        o_path = out_dir / f"outcomes_{name}{suffix}.csv"
        prec.to_csv(p_path, index=False, encoding="utf-8-sig")
        outc.to_csv(o_path, index=False, encoding="utf-8-sig")
        print(f"  锚点 {len(outc)} 个，前兆长表 {len(prec)} 行 "
              f"({len(prec) / max(len(outc), 1):.1f} 行/锚点)")
        if len(outc):
            print(f"  上涨序列长度 run_len: 中位 {outc['run_len'].median():.0f}，"
                  f"最长 {outc['run_len'].max():.0f}")
            print(f"  锚点日涨幅中位 {outc['anchor_pct_chg'].median():.2f}%，"
                  f"锚点→原信号涨幅中位 {outc['surge_pct'].median():.2f}%")
        print(f"  -> {p_path.relative_to(ROOT)}")
        print(f"  -> {o_path.relative_to(ROOT)}")
        all_p.append(prec)
        all_o.append(outc)

    if len(all_p) > 1:
        pd.concat(all_p, ignore_index=True).to_csv(
            out_dir / f"precursor_all{suffix}.csv", index=False, encoding="utf-8-sig")
        pd.concat(all_o, ignore_index=True).to_csv(
            out_dir / f"outcomes_all{suffix}.csv", index=False, encoding="utf-8-sig")
        print(f"\n[合并] precursor_all{suffix}.csv / outcomes_all{suffix}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
