"""训练数据集的路径解析与持仓周期对齐。

本模块集中处理两个此前被硬编码/被忽略的问题：

1. **明细路径硬编码**：train_model.py 与 experiments/exp_common.py 都把回溯明细写死为
   `backtest_output/backtest_detail_2025-11-12_2026-09-30.csv`，换一个回测区间就必须改代码；
   更麻烦的是该文件一旦被别的区间覆盖，训练会静默用到过期/不完整的数据。

2. **持仓周期混用**：回溯明细同时含 holding_days=3/5/10 三种持有期的交易，此前训练时
   全部混在一起，模型学到的 `win_prob` 既不是 3 日胜率也不是 5 日胜率，语义含混。
   这里提供按策略解析「该用哪个持仓周期」的统一口径。

路径解析优先级（明细 / 样本库相同）：
    1) 调用方显式传入的路径（CLI 参数）
    2) 环境变量 SQ_DETAIL_PATH / SQ_LIBRARY_PATH
    3) 自动发现（仅明细有；样本库回落到默认合并库）

自动发现的排序规则（**不是**简单取最大或最新，实测两者都会选错）：
    a. 若调用方给出 required（必须覆盖的策略名），先筛掉不包含这些策略的文件
       —— 否则会选中一个不含目标策略的大文件，该策略被静默跳过训练。
       实测坑：`backtest_detail_2025-11-12_2026-09-04.csv` 有 27863 行（最大），
       但只含 quantile_* / ma_* / volume_breakout，不含 b1_oversold；
       而 `..._2026-10-09.csv` 结束日最新，却只有 467 行（子集回测）。
    b. 在候选池内按「结束日期降序、行数降序」取第一份。

所有候选（含行数、结束日、覆盖策略、持仓周期）都会打印出来，选错时一眼可见；
要精确指定时用 --detail 或环境变量即可，无需改代码。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DETAIL_DIR = PROJECT_ROOT / "backtest_output"
DEFAULT_DETAIL_GLOB = "backtest_detail_*.csv"
DEFAULT_LIBRARY = PROJECT_ROOT / "data_library/combined_daily.parquet"
DEFAULT_STRATEGIES_YAML = PROJECT_ROOT / "config/strategies.yaml"

ENV_DETAIL = "SQ_DETAIL_PATH"
ENV_LIBRARY = "SQ_LIBRARY_PATH"

# backtest_detail_<start>_<end>.csv
_DETAIL_RE = re.compile(r"backtest_detail_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")


def _end_date(path: Path) -> str:
    m = _DETAIL_RE.search(path.name)
    return m.group(2) if m else ""


def _probe(path: Path) -> dict:
    """读取明细的关键元信息：行数、结束日、覆盖策略、覆盖的持仓周期。"""
    meta = {"path": path, "end_date": _end_date(path), "rows": 0,
            "strategies": [], "holdings": []}
    try:
        d = pd.read_csv(path, usecols=lambda c: c in ("strategy", "holding_days"))
    except Exception:  # noqa: BLE001  读不动就当空文件，不影响其他候选
        return meta
    meta["rows"] = len(d)
    if "strategy" in d.columns:
        meta["strategies"] = sorted(d["strategy"].dropna().astype(str).unique().tolist())
    if "holding_days" in d.columns:
        meta["holdings"] = sorted(
            pd.to_numeric(d["holding_days"], errors="coerce").dropna().astype(int).unique().tolist()
        )
    return meta


def list_detail_candidates(detail_dir: Path | str | None = None) -> list[dict]:
    """列出候选明细，按「结束日期降序、行数降序」排序。

    返回 [{"path", "end_date", "rows", "strategies", "holdings"}, ...]。
    """
    d = Path(detail_dir) if detail_dir else DETAIL_DIR
    if not d.is_dir():
        return []
    cands = [_probe(p) for p in sorted(d.glob(DEFAULT_DETAIL_GLOB))]
    cands.sort(key=lambda x: (x["end_date"], x["rows"]), reverse=True)
    return cands


def _print_candidates(cands: list[dict], chosen: Path | None, required: list[str] | None) -> None:
    print("[data] 明细候选（按结束日期降序）:")
    for c in cands:
        marks = []
        if c["path"] == chosen:
            marks.append("选中")
        if required:
            missing = sorted(set(required) - set(c["strategies"]))
            if missing:
                marks.append(f"缺策略{missing}")
        mark = f"  <- {', '.join(marks)}" if marks else ""
        print(f"[data]   {c['path'].name} | 行数 {c['rows']} | 结束 {c['end_date']} | "
              f"持仓 {c['holdings']} | 策略 {len(c['strategies'])} 个{mark}")
    if required:
        print(f"[data] 要求覆盖策略: {sorted(required)}")


def resolve_detail_path(
    explicit: str | Path | None = None,
    *,
    required: list[str] | None = None,
    verbose: bool = False,
) -> Path:
    """解析回溯明细路径。优先级：显式参数 > 环境变量 > 自动发现。

    required：必须覆盖的策略名列表。给出时，自动发现只在**包含全部这些策略**的
    候选里挑选；若没有任何候选满足，则忽略该约束并打印告警（不静默回退）。
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    env = os.environ.get(ENV_DETAIL)
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if verbose:
            print(f"[data] 明细来自环境变量 {ENV_DETAIL}: {p}")
        return p

    cands = list_detail_candidates()
    if not cands:
        if verbose:
            print(f"[data] {DETAIL_DIR} 下未找到 {DEFAULT_DETAIL_GLOB}")
        raise FileNotFoundError(
            f"未找到回溯明细（{DETAIL_DIR}/{DEFAULT_DETAIL_GLOB}）。"
            f"请先运行 backtest.py 生成明细，或用 --detail 显式指定。"
        )

    req = [s for s in (required or []) if s]
    pool = cands
    if req:
        covered = [c for c in cands if set(req) <= set(c["strategies"])]
        if covered:
            pool = covered
        elif verbose:
            print(f"[warn] 没有任何明细同时覆盖 {sorted(req)}，忽略覆盖约束后在全部候选中挑选")

    chosen = pool[0]["path"]
    if verbose:
        _print_candidates(cands, chosen, req)
    return chosen


def resolve_library_path(explicit: str | Path | None = None, *, verbose: bool = False) -> Path:
    """解析本地样本库路径。优先级：显式参数 > 环境变量 > 默认合并库。

    样本库不做自动发现：data_library/ 下可能同时存在多个指数库（zz500 / hs300），
    自动挑选有选错的风险，因此只保留显式覆盖，默认仍指向合并库。
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    env = os.environ.get(ENV_LIBRARY)
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if verbose:
            print(f"[data] 样本库来自环境变量 {ENV_LIBRARY}: {p}")
        return p
    return DEFAULT_LIBRARY


def load_strategies_config(config_path: str | Path | None = None) -> list[dict]:
    """读取 config/strategies.yaml 的策略条目列表。"""
    from core.strategy_loader import load_yaml

    path = Path(config_path) if config_path else DEFAULT_STRATEGIES_YAML
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        cfg = load_yaml(path)
    except Exception:  # noqa: BLE001  配置缺失不应阻断训练
        return []
    return list(cfg.get("strategies", []) or [])


def configured_holdings(config_path: str | Path | None = None) -> dict[str, int]:
    """读取各策略声明的持仓周期（params.holding），含已禁用策略。"""
    out: dict[str, int] = {}
    for item in load_strategies_config(config_path):
        name = item.get("name")
        if not name:
            continue
        try:
            out[str(name)] = int((item.get("params") or {}).get("holding"))
        except (TypeError, ValueError):
            continue
    return out


def enabled_strategies(config_path: str | Path | None = None) -> list[str]:
    """读取**已启用**的策略名（用于自动发现明细时的覆盖要求）。"""
    return [str(i["name"]) for i in load_strategies_config(config_path)
            if i.get("enabled", True) and i.get("name")]


def resolve_holdings(
    detail: pd.DataFrame,
    mode: str | int = "auto",
    *,
    config_path: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, int]:
    """决定每个策略训练/评估时使用哪个持仓周期，避免 3/5/10 混用。

    mode:
        "auto"（默认）：优先取 config/strategies.yaml 里该策略声明的 holding；
                        若该周期在明细中不存在，回落到该策略样本最多的周期。
        整数 N       ：所有策略统一用 holding_days = N。

    返回 {strategy: holding_days}（只包含明细中真实存在的策略）。
    明细无 holding_days 列时返回 {}（调用方按不拆分处理）。
    """
    if "holding_days" not in detail.columns or detail.empty:
        return {}

    strategies = detail["strategy"].astype(str)
    if str(mode).strip().lower() != "auto":
        want = int(mode)
        return {s: want for s in strategies.unique()}

    cfg = configured_holdings(config_path)
    out: dict[str, int] = {}
    for strat, g in detail.groupby(strategies):
        counts = g["holding_days"].value_counts()
        if counts.empty:
            continue
        declared = cfg.get(str(strat))
        if declared is not None and declared in set(int(x) for x in counts.index):
            out[str(strat)] = int(declared)
            src = "config"
        else:
            # 样本最多的周期；并列时取较小者，保证结果稳定
            out[str(strat)] = int(sorted(counts.index, key=lambda h: (-counts[h], h))[0])
            src = "数据多数"
        if verbose:
            print(f"[data] {strat}: 持仓周期 {out[str(strat)]}（来源: {src}）")
    return out
