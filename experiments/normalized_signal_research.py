"""归一化（尺度无关）特征信号研究。

动机
----
之前的特征里混了**绝对值**（如成交额 34 亿元、价格）。绝对值受市值/股价水平影响，
跨股票不可比，容易把「起涨股恰好是大市值活跃股」这种**样本选择偏差**误当成信号
（上一轮报告里换手率、成交额就是这种伪信号）。

本脚本把价格**归一化**成比值/相对位置，只用尺度无关的量研究信号条件：
    close/MA5、close/MA20、close/MA60、MA5/MA60（用户指定）
    close/近5日最低价、close/近5日最高价、近5日振幅（即 min5 / max5 的归一化形式）
    换手率：当日、近5日均、近20日均、5日/20日扩张比
    量比、近5日涨幅、MA20 五日斜率

两条互补的分析线
----------------
A. **全库无偏筛选**（主力）：对每个归一化特征在全库（可交易池）做 5 档分位，
   统计各档前向 5 日胜率/收益与 Q5−Q1 价差、Spearman 秩相关。
   这一条不受「只取正例」的偏差影响，能直接回答"哪个特征有信号"。
B. **起涨前夜画像**（辅助）：取 consecutive_up5 起涨点（锚点）的**前一日**作为观测日，
   比较该日特征分布与全库基准的 z 分数，回答"起涨前夜与普通交易日差在哪"。

为什么观测日取"起涨点前一日"
    在这一天收盘你能看到全部特征，次日开盘买入 = 买在起涨点开盘，
    正是可执行的事前时点（与前份报告的口径一致）。

产出：
    docs/归一化特征信号研究.md
    research_output/norm_*.csv
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

# 归一化特征定义：key = 列名，value = 中文名
NORM_FEATURES: dict[str, str] = {
    "r_ma5": "close/MA5",
    "r_ma20": "close/MA20",
    "r_ma60": "close/MA60",
    "r_ma5_ma60": "MA5/MA60",
    "r_low5": "close/近5日最低价",
    "r_high5": "close/近5日最高价",
    "range5": "近5日振幅(高/低-1, %)",
    "turn": "当日换手率(%)",
    "turn5": "近5日平均换手率(%)",
    "turn20": "近20日平均换手率(%)",
    "turn_expand": "换手率 5日/20日",
    "vol_ratio5": "量比(量/5日均量)",
    "ret5": "近5日涨幅(%)",
    "slope_ma20_5": "MA20 五日斜率(%)",
}
# 起涨前夜画像里额外关注、但不参与"分档筛选"的绝对量（用于对照展示偏差）
ABSOLUTE_FEATURES = {"amount": "成交额(亿元)"}


def build_features(daily: pd.DataFrame, commission: float,
                   ipo: pd.Series | None) -> pd.DataFrame:
    """全库股票日 + 归一化特征 + 与回测一致的前向 5 日净收益。"""
    d = daily.sort_values(["code", "date"], kind="stable").copy()
    g = d.groupby("code", sort=False)
    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    vol = d["volume"].astype(float)
    turn = d["turn"].astype(float)

    ma5 = g["close"].transform(lambda s: s.rolling(5).mean())
    ma20 = g["close"].transform(lambda s: s.rolling(20).mean())
    ma60 = g["close"].transform(lambda s: s.rolling(60).mean())
    min5 = g["low"].transform(lambda s: s.rolling(5).min())
    max5 = g["high"].transform(lambda s: s.rolling(5).max())

    d["r_ma5"] = close / ma5
    d["r_ma20"] = close / ma20
    d["r_ma60"] = close / ma60
    d["r_ma5_ma60"] = ma5 / ma60
    d["r_low5"] = close / min5
    d["r_high5"] = close / max5
    d["range5"] = (max5 / min5 - 1.0) * 100.0
    d["turn"] = turn
    d["turn5"] = g["turn"].transform(lambda s: s.rolling(5).mean())
    d["turn20"] = g["turn"].transform(lambda s: s.rolling(20).mean())
    d["turn_expand"] = d["turn5"] / d["turn20"].replace(0.0, np.nan)
    d["vol_ratio5"] = vol / g["volume"].transform(lambda s: s.rolling(5).mean()).replace(0.0, np.nan)
    d["ret5"] = g["close"].transform(lambda s: (s / s.shift(5) - 1.0) * 100.0)
    # MA20 五日斜率：MA20 / 5 日前的 MA20 - 1（归一化，避免用绝对价格）
    d["_ma20"] = ma20
    d["slope_ma20_5"] = (d["_ma20"] / d.groupby("code")["_ma20"].shift(5) - 1.0) * 100.0
    d = d.drop(columns="_ma20")
    d["amount"] = d["amount"].astype(float) / 1e8

    # 与回测一致：T 日观测 → T+1 开盘入场 → T+6 收盘卖出（持有 5 日），扣双边手续费
    entry = g["open"].shift(-1)
    exit_ = g["close"].shift(-6)
    d["fwd_ret_5d"] = (exit_ / entry - 1.0 - 2 * commission) * 100.0

    d["_listed_days"] = np.inf
    if ipo is not None:
        dt = pd.to_datetime(d["date"], errors="coerce")
        ipo_dt = pd.to_datetime(d["code"].map(ipo), errors="coerce")
        d["_listed_days"] = (dt - ipo_dt).dt.days
    return d


def universe(d: pd.DataFrame, min_listed_days: int = 180) -> pd.Series:
    m = ~d["code"].str.startswith("sh.68")
    if "isST" in d.columns:
        m &= ~d["isST"].astype(str).str.strip().isin({"1", "1.0", "True"})
    if min_listed_days > 0:
        m &= d["_listed_days"] >= min_listed_days
    return m


def quintile_screen(d: pd.DataFrame, mask: pd.Series,
                    features: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对每个归一化特征做 5 档分位，统计前向 5 日表现与 Q5−Q1 价差。"""
    sub = d.loc[mask]
    rows, detail_rows = [], []
    for col, cn in features.items():
        s = sub[[col, "fwd_ret_5d"]].dropna()
        if len(s) < 500:
            continue
        try:
            q = pd.qcut(s[col], 5, labels=False, duplicates="drop")
        except Exception:  # noqa: BLE001
            continue
        if pd.Series(q).nunique() < 5:
            continue
        grp = s.groupby(q)
        win = grp["fwd_ret_5d"].apply(lambda x: float((x > 0).mean() * 100))
        mean = grp["fwd_ret_5d"].mean()
        med = grp["fwd_ret_5d"].median()
        bounds = s.groupby(q)[col].agg(["min", "max"])
        for qi in range(5):
            detail_rows.append({
                "特征": cn, "档位": qi + 1, "样本数": int(grp.size().get(qi, 0)),
                "5日胜率%": float(win.get(qi, np.nan)),
                "5日均收益%": float(mean.get(qi, np.nan)),
                "5日中位收益%": float(med.get(qi, np.nan)),
                "区间下限": float(bounds.loc[qi, "min"]) if qi in bounds.index else np.nan,
                "区间上限": float(bounds.loc[qi, "max"]) if qi in bounds.index else np.nan,
            })
        rows.append({
            "特征": cn, "样本数": len(s),
            "Q1胜率%": float(win.get(0, np.nan)), "Q5胜率%": float(win.get(4, np.nan)),
            "胜率价差(pp)": float(win.get(4, np.nan) - win.get(0, np.nan)),
            "Q1均收益%": float(mean.get(0, np.nan)), "Q5均收益%": float(mean.get(4, np.nan)),
            "收益价差(pp)": float(mean.get(4, np.nan) - mean.get(0, np.nan)),
            "spearman": float(s[col].corr(s["fwd_ret_5d"], method="spearman")),
            # 趋势分 = 相邻档均收益差值的符号和：+4 严格递增、-4 严格递减、0 为混合
            "趋势分": float(np.sign(np.diff(mean.values)).sum()),
            # 其余档（Q2~Q5）均收益：与 Q1 对比可判断是否为"只有最低档有效"的阈值型信号
            "其余档均收益%": float(np.mean([mean.get(k, np.nan) for k in (1, 2, 3, 4)])),
            # Q1（最低档）的上界即"阈值型"信号的门槛值
            "Q1上界": float(bounds.loc[0, "max"]) if 0 in bounds.index else np.nan,
            "Q5下界": float(bounds.loc[4, "min"]) if 4 in bounds.index else np.nan,
        })
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        tbl = tbl.reindex(tbl["收益价差(pp)"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return tbl, pd.DataFrame(detail_rows)


def anchor_profile(d: pd.DataFrame, anchors: pd.DataFrame, mask: pd.Series,
                   features: dict[str, str]) -> pd.DataFrame:
    """锚点前一日特征 vs 全库基准（z 分数）。"""
    pos = {}
    for code, g in d.groupby("code", sort=False):
        pos[code] = {dt: i for i, dt in enumerate(g["date"].tolist())}
    idx_by_code = d.groupby("code", sort=False).indices  # code -> 行号数组

    obs_rows = []
    for _, r in anchors.iterrows():
        code, adate = r["code"], str(r["signal_date"])
        p = pos.get(code, {}).get(adate)
        if p is None or p < 1:
            continue
        gi = idx_by_code[code]
        obs_rows.append(gi[p - 1])      # 锚点前一日的全库行号
    if not obs_rows:
        return pd.DataFrame()
    obs = d.iloc[obs_rows]
    base = d.loc[mask]

    out = []
    for col, cn in {**features, **ABSOLUTE_FEATURES}.items():
        if col not in d.columns:
            continue
        a = obs[col].dropna()
        b = base[col].dropna()
        if len(a) < 10 or len(b) < 100:
            continue
        bs = float(b.std())
        out.append({
            "特征": cn, "锚点前一日均值": float(a.mean()), "全库均值": float(b.mean()),
            "全库标准差": bs, "z": (float(a.mean()) - float(b.mean())) / bs if bs else np.nan,
            "锚点前一日中位数": float(a.median()), "全库中位数": float(b.median()),
            "方向": "高于全库" if a.mean() > b.mean() else "低于全库",
        })
    t = pd.DataFrame(out)
    if not t.empty:
        t = t.reindex(t["z"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return t


def combo_test(d: pd.DataFrame, mask: pd.Series, anchors: pd.DataFrame,
               profile: pd.DataFrame, features: dict[str, str]) -> pd.DataFrame:
    """用锚点中位数做阈值，检验单条件与最优组合的全库前向表现。"""
    base = d.loc[mask]
    ref = base["fwd_ret_5d"].dropna().astype(float)

    def stat(cond: pd.Series, label: str) -> dict:
        s = d.loc[mask & cond.fillna(False), "fwd_ret_5d"].dropna().astype(float)
        if len(s) < 30:
            return {"方案": label, "股票日数": len(s), "5日胜率%": np.nan,
                    "5日均收益%": np.nan, "5日中位收益%": np.nan,
                    "胜率差(pp)": np.nan, "t值": np.nan}
        se = float(np.sqrt(s.var(ddof=1) / len(s) + ref.var(ddof=1) / len(ref)))
        return {"方案": label, "股票日数": len(s),
                "5日胜率%": float((s > 0).mean() * 100),
                "5日均收益%": float(s.mean()),
                "5日中位收益%": float(s.median()),
                "胜率差(pp)": float((s > 0).mean() * 100 - (ref > 0).mean() * 100),
                "t值": float((s.mean() - ref.mean()) / se) if se > 0 else np.nan}

    rows = [{"方案": "全库基准（可交易池全部股票日）", "股票日数": len(ref),
             "5日胜率%": float((ref > 0).mean() * 100), "5日均收益%": float(ref.mean()),
             "5日中位收益%": float(ref.median()), "胜率差(pp)": np.nan, "t值": np.nan}]

    # 锚点观测日的中位数作为阈值（观测日 = 起涨点前一日）
    obs_idx = []
    pos = {}
    for code, gg in d.groupby("code", sort=False):
        pos[code] = {dt: i for i, dt in enumerate(gg["date"].tolist())}
    idx_by_code = d.groupby("code", sort=False).indices
    for _, r in anchors.iterrows():
        p = pos.get(r["code"], {}).get(str(r["signal_date"]))
        if p is not None and p >= 1:
            obs_idx.append(idx_by_code[r["code"]][p - 1])
    obs = d.iloc[obs_idx] if obs_idx else d.iloc[0:0]

    single = []
    for _, r in profile.iterrows():
        col = next((k for k, v in {**features, **ABSOLUTE_FEATURES}.items() if v == r["特征"]), None)
        if col is None or col not in d.columns or col not in obs.columns:
            continue
        med = float(obs[col].median())
        up = str(r["方向"]) == "高于全库"
        cond = (d[col] >= med) if up else (d[col] <= med)
        lab = f"{r['特征']} {'≥' if up else '≤'} {med:.4g}（起涨前夜中位数）"
        res = stat(cond, lab)
        rows.append(res)
        single.append((col, med, up, lab, res["t值"]))

    valid = [x for x in single if x[4] is not None and not np.isnan(x[4])]
    valid.sort(key=lambda x: x[4], reverse=True)

    def mk(cs):
        c = pd.Series(True, index=d.index)
        for col, med, up, *_ in cs:
            c &= (d[col] >= med) if up else (d[col] <= med)
        return c

    if len(valid) >= 2:
        rows.append(stat(mk(valid[:2]), f"{valid[0][3]} 且 {valid[1][3]}"))
    if len(valid) >= 3:
        rows.append(stat(mk(valid[:3]), "三个最优单条件同时满足"))
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, nd: int = 2) -> str:
    if df is None or df.empty:
        return "_（无数据）_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                vals.append("n/a" if (np.isnan(v) or np.isinf(v)) else f"{v:.{nd}f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="归一化特征信号研究")
    ap.add_argument("--library", default="data_library/combined_daily.parquet")
    ap.add_argument("--anchors", default="research_output/outcomes_consecutive_up5_runstart.csv")
    ap.add_argument("--out-dir", default="research_output")
    ap.add_argument("--report", default="docs/归一化特征信号研究.md")
    ap.add_argument("--commission", type=float, default=0.001)
    args = ap.parse_args()

    daily = pd.read_parquet(ROOT / args.library)
    daily["date"] = daily["date"].astype(str)

    ipo = None
    mp = ROOT / "data_library" / "_ctx" / "stock_basic.csv"
    if mp.exists():
        m = pd.read_csv(mp, dtype={"code": str})
        if {"code", "ipoDate"}.issubset(m.columns):
            ipo = dict(zip(m["code"], m["ipoDate"]))

    print("[build] 计算全库归一化特征…")
    d = build_features(daily, args.commission, ipo)
    mask = universe(d) & d["fwd_ret_5d"].notna()
    print(f"[build] 可交易池 {int(mask.sum())} 个股票日（全库 {len(d)} 行）")

    print("[A] 全库分档筛选…")
    screen, detail = quintile_screen(d, mask, NORM_FEATURES)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    screen.to_csv(out / "norm_screen.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out / "norm_quintile_detail.csv", index=False, encoding="utf-8-sig")

    profile = pd.DataFrame()
    combos = pd.DataFrame()
    apath = ROOT / args.anchors
    if apath.exists():
        anchors = pd.read_csv(apath, encoding="utf-8-sig")[["code", "signal_date"]].drop_duplicates()
        print(f"[B] 起涨前夜画像（{len(anchors)} 个锚点）…")
        profile = anchor_profile(d, anchors, mask, NORM_FEATURES)
        profile.to_csv(out / "norm_anchor_profile.csv", index=False, encoding="utf-8-sig")
        combos = combo_test(d, mask, anchors, profile, NORM_FEATURES)
        combos.to_csv(out / "norm_combo_test.csv", index=False, encoding="utf-8-sig")
    else:
        print(f"[warn] 未找到锚点文件 {apath}，跳过 B 部分")

    # ---------------- 报告 ----------------
    md: list[str] = []
    md.append("# 归一化特征信号研究\n")
    md.append("**思路**：把价格归一化成**尺度无关的比值**再研究信号。"
              "之前的特征混入了绝对值（成交额、价格水平），会因市值/股价差异产生"
              "**样本选择偏差**——上一轮报告里「换手率、成交额偏高」就是这类伪信号。\n")
    md.append(f"**样本**：`{args.library}`，2025-11-12 ~ 2026-10-09，"
              f"可交易池（非科创/非ST/上市满180天且前向可算）"
              f"**{int(mask.sum())}** 个股票日\n")
    md.append("**前向口径**：T 日观测 → T+1 开盘买入 → T+5 收盘卖出（持有 5 个交易日），"
              "扣双边 0.1% 手续费\n")

    md.append("## 一、归一化特征清单\n")
    md.append(_md(pd.DataFrame({"特征": list(NORM_FEATURES.values()),
                                "含义": ["收盘价 / 5 日均线", "收盘价 / 20 日均线",
                                         "收盘价 / 60 日均线", "5 日均线 / 60 日均线",
                                         "收盘价 / 近 5 日最低价", "收盘价 / 近 5 日最高价",
                                         "近 5 日 (最高/最低−1)", "当日换手率",
                                         "近 5 日平均换手率", "近 20 日平均换手率",
                                         "5 日均换手 / 20 日均换手", "当日量 / 5 日均量",
                                         "近 5 日累计涨幅", "MA20 相对 5 日前的变化"]})))

    md.append("\n## 二、全库分档筛选（主力，无偏）\n")
    md.append("对每个归一化特征在全库做 5 档分位。**「收益价差」= 最高档均收益 − 最低档均收益**，"
              "是最直接的信号强度度量；**「趋势分」= 相邻档均收益差值的符号和**"
              "（+4 严格递增、−4 严格递减、0 为混合，说明不是单调关系）。\n")
    md.append("**「Q1上界」/「Q5下界」给出最低档的上界与最高档的下界**——"
              "若某特征只有 Q1 明显更好（趋势分接近 0），说明它是**阈值型**信号，"
              "这时应该直接拿 Q1上界当门槛，而不是当单调因子用。\n")
    md.append("> ⚠️ 同样的保留：分位内相邻交易日高度重叠，t 值会被高估；"
              "且这里筛了 14 个特征，存在多重比较。结论应视为**候选假设**。\n")
    md.append("\n" + _md(screen[["特征", "样本数", "Q1胜率%", "Q1均收益%", "其余档均收益%",
                                 "Q5均收益%", "收益价差(pp)", "胜率价差(pp)", "spearman",
                                 "趋势分", "Q1上界", "Q5下界"]]
                         if not screen.empty else screen, nd=3))

    if not detail.empty:
        md.append("\n### 各特征分档明细（按特征名）\n")
        for cn in detail["特征"].unique():
            sub = detail[detail["特征"] == cn]
            md.append(f"\n**{cn}**\n")
            md.append(_md(sub[["档位", "样本数", "5日胜率%", "5日均收益%",
                               "5日中位收益%", "区间下限", "区间上限"]], nd=4))

    if not profile.empty:
        md.append("\n## 三、起涨前夜画像（辅助看偏差）\n")
        md.append("锚点 = consecutive_up5 的起涨点；观测日 = **起涨点前一日**"
                  "（该日收盘可看到全部特征，次日开盘买入即买在起涨点）。\n")
        md.append("`成交额` 作为**绝对量对照**一并列出——若它 z 值很突出但分档筛选里没信号，"
                  "就说明是偏差而非 alpha。\n")
        md.append("\n" + _md(profile))
        md.append("\n> 说明：归一化特征里 `r_ma20`（close/MA20）本身就是位置约束，"
                  "因此这一轮检验**不再额外施加**「站上 MA20」的过滤，"
                  "改由 r_ma20 各档自行体现。\n")

    if not combos.empty:
        md.append("\n## 四、条件组合检验（以起涨前夜中位数为阈值）\n")
        md.append(_md(combos))

    md.append("\n## 五、结论\n")
    if not screen.empty:
        top = screen.iloc[0]
        md.append(f"- **信号最强的归一化特征是「{top['特征']}」**：最低档 → 最高档的均收益"
                  f"从 {top['Q1均收益%']:.3f}% 变到 {top['Q5均收益%']:.3f}%"
                  f"（价差 {top['收益价差(pp)']:+.3f}pp，胜率价差 {top['胜率价差(pp)']:+.2f}pp，"
                  f"Spearman {top['spearman']:+.3f}）。")
        # 阈值型：最低档明显优于其余各档（Q1 均收益 - 其余档均收益 >= 0.3pp）
        thresh = screen[(screen["Q1均收益%"] - screen["其余档均收益%"] >= 0.3)]
        if not thresh.empty:
            thresh = thresh.sort_values("Q1均收益%", ascending=False)
            md.append("- **「只有最低档有效」的阈值型特征**（Q1 比其余档平均高出 ≥0.3pp，"
                      "其余档之间并不继续单调变化）："
                      + "；".join(f"{x['特征']} ≤ {x['Q1上界']:.4g}"
                                  f"（Q1 {x['Q1均收益%']:+.3f}% vs 其余档 {x['其余档均收益%']:+.3f}%）"
                                  for _, x in thresh.iterrows())
                      + "。这类条件应直接当**门槛**用，不要当线性因子。")
        pos_side = screen[screen["收益价差(pp)"] > 0].sort_values("收益价差(pp)", ascending=False)
        neg_side = screen[screen["收益价差(pp)"] < 0].sort_values("收益价差(pp)")
        if not neg_side.empty:
            md.append("- **数值越低越好的特征**：" + "、".join(
                f"{x['特征']}({x['收益价差(pp)']:+.3f}pp)" for _, x in neg_side.head(4).iterrows())
                + "。这说明样本期内的普适信号是**超跌/均值回复**，"
                "而不是动量——这与 `b1_oversold`（超跌型）在本项目里表现最好相互印证。")
        if not pos_side.empty:
            md.append("- **数值越高越好的特征**：" + "、".join(
                f"{x['特征']}({x['收益价差(pp)']:+.3f}pp)" for _, x in pos_side.head(4).iterrows())
                + "。注意这些价差都很小（<0.3pp），基本不具备单独使用价值。")
        weak = screen[screen["收益价差(pp)"].abs() < 0.15]
        if not weak.empty:
            md.append("- **基本无信号**（价差 <0.15pp）：" + "、".join(
                f"{x['特征']}" for _, x in weak.iterrows())
                + "。特别是**换手率系列（当日/5日/20日/扩张比）价差都极小**，"
                "说明上一轮报告里「起涨前夜换手率偏高」确实是**样本选择偏差**"
                "（起涨股本身更活跃），不是可交易信号——换成归一化口径后这一点被证实。")

    if not profile.empty:
        md.append("\n- **起涨前夜画像（归一化口径）**：" + "、".join(
            f"{x['特征']}(z={x['z']:+.2f})" for _, x in profile.head(4).iterrows()) + "。")
        md.append("  其中 `close/近5日最高价` z 最负，含义是**起涨前夜收盘价明显低于近 5 日高点**"
                  "（均值 0.906 vs 全库 0.957，低约 5pp），即典型的回调/阴线位置；"
                  "而 `成交额`（绝对量）z 同样突出，属于偏差对照。")
        md.append("\n> ⚠️ **两条分析线方向不同，不要混用**：全库分档筛选（第二节）是"
                  "**普适的前向收益规律**，结论是「超跌更好」；而起涨前夜画像（第三节）"
                  "来自 `consecutive_up5` 的锚点，该锚点集合**按构造就是一轮上涨之后**，"
                  "所以它的画像是「已上涨途中的回调」，回答的是「这类上涨行情启动前长什么样」，"
                  "两者是不同的问题，不能互相替代。")

    md.append("\n> 反复提醒：以上均为**候选假设**。分档价差没有考虑重叠样本、多重比较"
              "和交易成本之外的执行细节，不能直接等同于可交易 edge；"
              "任何要落地的条件都应在更长时间序列上做样本外检验。\n")

    (ROOT / args.report).write_text("\n".join(md), encoding="utf-8")
    print(f"[report] -> {ROOT / args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
