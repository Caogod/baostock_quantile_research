"""胜率预测模型训练 + 因子重要度输出 + 模型留存（模块3）。

从本地样本库 + 回溯逐笔明细构建特征数据集，训练 决策树/随机森林/GBDT，
输出因子重要度排名，并留存 GBDT 模型供选股后二次过滤使用。

CV 口径（已修正）：
- 使用 TimeSeriesSplit 进行时序交叉验证，杜绝随机 KFold 的时序泄漏
  （"未来样本"进入训练集预测"过去样本"导致准确率虚高）。
- 数据按 entry_date 升序排列后分折。

极短期窗口训练（捕捉市场偏好变化）：
- --recent-days N：仅用最近 N 个交易日的回测明细训练。
  市场风格/偏好会随时间漂移（如大盘 vs 小盘、价值 vs 成长、
  量价关系强弱），用近期数据训练可让模型感知"当前市场偏好什么"，
  而非全历史的平均规律。适合短周期（5~10 日持仓）策略。
- 默认 0 = 全样本训练（与历史行为一致）。

用法：
    python train_model.py                                   # 默认 ma_breakout 家族子集，全样本
    python train_model.py --strategies ma_breakout,ma_breakout_shift5
    python train_model.py --recent-days 60                  # 仅用近 60 交易日训练（捕捉近期偏好）
    python train_model.py --strategies ""                   # 全策略
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.features import FEATURES, precompute_features
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent
DETAIL = "backtest_output/backtest_detail_2025-11-12_2026-09-04.csv"
LIBRARY = "data_library/zz500_daily.parquet"
MODEL_DIR = PROJECT_ROOT / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description="训练胜率预测模型并留存")
    parser.add_argument("--strategies", default="ma_breakout,ma_breakout_shift5",
                        help="训练子集（逗号分隔策略名，空串=全策略）")
    parser.add_argument("--detail", default=DETAIL, help="回溯逐笔明细 CSV")
    parser.add_argument("--library", default=LIBRARY, help="本地样本库 Parquet")
    parser.add_argument("--recent-days", type=int, default=0,
                        help="仅用最近 N 个交易日的明细训练（0=全样本）。"
                             "极短期窗口可捕捉市场偏好/风格漂移，适合短周期策略。")
    args = parser.parse_args()

    lib = pd.read_parquet(args.library).sort_values("date")
    detail = pd.read_csv(args.detail)

    # 子集过滤（经验：模型应与前置策略筛选出的子集同分布）
    wanted = {x.strip() for x in args.strategies.split(",") if x.strip()}
    if wanted:
        detail = detail[detail["strategy"].isin(wanted)].copy()
    print(f"训练子集交易数: {len(detail)}（策略 {sorted(wanted) or '全部'}）")

    # 极短期窗口训练：仅保留最近 N 个交易日的明细
    if args.recent_days > 0:
        detail["entry_date"] = detail["entry_date"].astype(str)
        all_dates = sorted(detail["entry_date"].unique())
        if len(all_dates) > args.recent_days:
            cutoff = all_dates[-args.recent_days]
            before = len(detail)
            detail = detail[detail["entry_date"] >= cutoff].copy()
            print(f"极短期窗口: 仅保留近 {args.recent_days} 交易日（{cutoff} 起），"
                  f"{before} -> {len(detail)} 笔")

    # 特征矩阵 + 对齐交易
    feat_all = precompute_features(lib)
    df = detail.merge(feat_all, left_on=["code", "entry_date"], right_on=["code", "date"], how="left")
    df["win"] = (df["return_pct"] > 0).astype(int)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"有效样本: {len(df)}，胜率 {df['win'].mean()*100:.1f}%")

    # 时序 CV 要求：按 entry_date 升序排列，避免时序泄漏
    df = df.sort_values("entry_date").reset_index(drop=True)
    X, y = df[FEATURES].values, df["win"].values
    baseline = max(y.mean(), 1 - y.mean())

    # TimeSeriesSplit：训练集始终在验证集之前（时序上），杜绝未来信息泄漏
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "决策树(DT)": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=0),
        "随机森林(RF)": RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=10, n_jobs=-1, random_state=0
        ),
        "GBDT": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0
        ),
    }

    print("\n=== 模型时序交叉验证准确率（TimeSeriesSplit 5折）===\n")
    importance = {}
    for name, model in models.items():
        acc = cross_val_score(model, X, y, cv=tscv, scoring="accuracy").mean()
        model.fit(X, y)
        importance[name] = model.feature_importances_
        print(f"  {name:<12} {acc*100:5.2f}%  (基线 {baseline*100:5.2f}%)")

    # 因子重要度输出
    rank = pd.DataFrame({"feature": FEATURES})
    for name, imp in importance.items():
        rank[name] = imp
    rank = rank.sort_values("随机森林(RF)", ascending=False).reset_index(drop=True)
    (PROJECT_ROOT / "output").mkdir(exist_ok=True)
    rank.to_csv("output/feature_importance.csv", index=False, encoding="utf-8-sig")
    print("\n=== 因子重要度排名（按 RF 降序）===\n")
    print(rank.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 留存 GBDT 模型 + 特征列表
    MODEL_DIR.mkdir(exist_ok=True)
    gbdt = models["GBDT"]
    gbdt.fit(X, y)
    model_path = MODEL_DIR / "win_model.joblib"
    joblib.dump(gbdt, model_path)
    (MODEL_DIR / "win_model.json").write_text(
        json.dumps(FEATURES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tag = f"，近{args.recent_days}日窗口" if args.recent_days > 0 else "，全样本"
    print(f"\n已留存模型: {model_path}（GBDT，特征数 {len(FEATURES)}{tag}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
