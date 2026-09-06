"""胜率预测模型（模块4）：加载已训练模型，对选股结果二次过滤。"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURES, feature_at

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WinModel:
    """加载训练好的胜率预测模型，并对选股结果打分/过滤。"""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型不存在: {self.model_path}（请先运行 train_model.py）")
        self.model = joblib.load(self.model_path)
        # 可选：特征列表（校验口径）
        meta = self.model_path.with_suffix(".json")
        self.features = FEATURES
        if meta.exists():
            try:
                self.features = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                pass

    def predict_prob(self, feat_df: pd.DataFrame) -> pd.Series:
        """对特征 DataFrame 预测盈利概率（0~1）。"""
        return pd.Series(self.model.predict_proba(feat_df[self.features].values)[:, 1])

    def filter_selection(
        self,
        selection_df: pd.DataFrame,
        library: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """对选股结果逐行计算胜率，返回含 win_prob 列、并按阈值过滤后的结果。

        selection_df 需含列: code, date（选股日期）。
        """
        if selection_df.empty:
            return selection_df.assign(win_prob=[])

        probs = []
        for _, row in selection_df.iterrows():
            feat = feature_at(library, row["code"], row["date"])
            if feat.empty or feat[self.features].isna().any().any():
                probs.append(float("nan"))
            else:
                probs.append(float(self.predict_prob(feat)[0]))

        out = selection_df.copy()
        out["win_prob"] = probs
        out = out.sort_values("win_prob", ascending=False).reset_index(drop=True)
        return out[out["win_prob"] >= threshold]


def apply_filter(
    out_path: str | Path,
    library_path: str | Path,
    model_path: str | Path = "models/win_model.joblib",
    threshold: float = 0.5,
) -> tuple[Path, int, int, pd.DataFrame]:
    """选股结果二次过滤（公共入口，供 main.py 与 scheduler.py 复用）。

    读取选股 CSV → 用本地样本库算特征 → 模型打分 → 过滤 → 写 *_filtered.csv。
    返回 (过滤后路径, 原始数量, 保留数量, 过滤结果 DataFrame)。
    """
    selection = pd.read_csv(out_path)
    if selection.empty:
        filtered = selection.assign(win_prob=pd.Series(dtype=float))
    else:
        lib_df = pd.read_parquet(library_path).sort_values("date")
        model = WinModel(model_path)
        filtered = model.filter_selection(selection, lib_df, threshold=threshold)

    filtered_path = Path(str(out_path).replace(".csv", "_filtered.csv"))
    filtered.to_csv(filtered_path, index=False, encoding="utf-8-sig")
    return filtered_path, len(selection), len(filtered), filtered
