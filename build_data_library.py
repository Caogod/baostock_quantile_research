"""构建本地回溯样本库。

下载指定指数成分股最近 N 个交易日的历史日 K（前复权），按股价过滤后，
以 Parquet（pandas 快速列式格式）落地到 data_library/。

用法：
    python build_data_library.py                          # 默认 中证500 + 200日 + 股价2~300元
    python build_data_library.py --index hs300 --days 200 --price-min 2 --price-max 300
    python build_data_library.py --index gem --days 200 --price-min 0 --price-max 0

指数（baostock 支持）: hs300(沪深300) / zz500(中证500) / sz50(上证50)。
股价过滤：以该股最新收盘价判断，超出 [price-min, price-max] 则剔除（0 表示不限）。

产物（按 index 命名）：
    data_library/<index>_daily.parquet   长表（code,date,OHLCV...）
    data_library/<index>_stocks.parquet  成分股元数据（code,code_name）
    data_library/<index>_meta.json       构建信息

支持断点续跑：中途中断重跑会自动跳过已下载的股票（检查点按 index+价格区间隔离）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from core.data_fetcher import DataFetcher

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_library"

INDEX_NAMES = {"hs300": "沪深300", "zz500": "中证500", "sz50": "上证50"}


def _in_price_range(df: pd.DataFrame, pmin: float, pmax: float) -> bool:
    """判断该股最新收盘价是否落在 [pmin, pmax]（0 表示不限）。"""
    if pmin <= 0 and pmax <= 0:
        return True
    close = float(df["close"].astype(float).iloc[-1])
    if pmin > 0 and close < pmin:
        return False
    if pmax > 0 and close > pmax:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="构建本地回溯样本库（Parquet）")
    parser.add_argument("--index", default="zz500", choices=list(INDEX_NAMES), help="指数")
    parser.add_argument("--days", type=int, default=200, help="交易日数量（默认200）")
    parser.add_argument("--price-min", type=float, default=2.0, help="股价下限（0不限）")
    parser.add_argument("--price-max", type=float, default=300.0, help="股价上限（0不限）")
    parser.add_argument("--request-interval", type=float, default=0.08, help="请求间隔（秒）")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily_parquet = DATA_DIR / f"{args.index}_daily.parquet"
    meta_parquet = DATA_DIR / f"{args.index}_stocks.parquet"
    meta_json = DATA_DIR / f"{args.index}_meta.json"

    # 断点续跑：检查点按 index + 价格区间 + 天数隔离，避免不同构建互相污染
    sig = hashlib.md5(
        f"{args.index}:{args.price_min}:{args.price_max}:{args.days}".encode("utf-8")
    ).hexdigest()[:8]
    ckpt_json = DATA_DIR / f"build_checkpoint_{args.index}_{sig}.json"
    partial_parquet = DATA_DIR / f"build_partial_{args.index}_{sig}.parquet"

    fetcher = DataFetcher(request_interval=args.request_interval)
    fetcher.login()
    try:
        # 1. 成分股
        constituents = fetcher.get_index_stocks(args.index)
        codes = constituents["code"].tolist()
        name_map = dict(zip(constituents["code"], constituents["code_name"]))
        print(f"{INDEX_NAMES[args.index]}成分股: {len(codes)} 只")

        # 2. 最近 N 个交易日区间
        today = datetime.now()
        trade_dates = fetcher.get_trade_dates(
            (today - timedelta(days=int(args.days * 1.8))).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
        )
        window = trade_dates[-args.days:]
        start, end = window[0], window[-1]
        print(f"交易日区间: {start} ~ {end}（{len(window)} 个交易日）")
        print(f"股价过滤: [{args.price_min if args.price_min > 0 else '-∞'}, "
              f"{args.price_max if args.price_max > 0 else '+∞'}] 元")

        # 3. 断点续跑
        done: set[str] = set()
        frames: list[pd.DataFrame] = []
        if ckpt_json.exists():
            try:
                done = set(json.loads(ckpt_json.read_text(encoding="utf-8")).get("done", []))
                if partial_parquet.exists():
                    frames = [pd.read_parquet(partial_parquet)]
                print(f"续跑: 已下载 {len(done)} 只")
            except Exception:
                done, frames = set(), []

        # 4. 逐只下载 + 股价过滤
        skipped_price = 0
        for i, code in enumerate(codes, 1):
            if code in done:
                continue
            df = fetcher.get_history(code, start, end, adjustflag="2")
            if not df.empty:
                if _in_price_range(df, args.price_min, args.price_max):
                    df = df.copy()
                    df["code_name"] = name_map.get(code, "")
                    frames.append(df)
                else:
                    skipped_price += 1
            done.add(code)
            fetcher._throttle()
            if i % 20 == 0 or i == len(codes):
                pd.concat(frames, ignore_index=True).to_parquet(partial_parquet, index=False)
                ckpt_json.write_text(
                    json.dumps({"done": sorted(done)}, ensure_ascii=False), encoding="utf-8"
                )
                print(f"  进度 {i}/{len(codes)}，纳入 {len(frames)} 只（价格剔除 {skipped_price}）")

        # 5. 最终落盘
        daily = pd.concat(frames, ignore_index=True)
        daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
        daily.to_parquet(daily_parquet, index=False)
        constituents.to_parquet(meta_parquet, index=False)

        meta = {
            "index": INDEX_NAMES[args.index],
            "index_code": args.index,
            "stocks": int(daily["code"].nunique()) if not daily.empty else 0,
            "constituents": int(len(codes)),
            "trading_days": int(len(window)),
            "date_range": [start, end],
            "price_range": [args.price_min, args.price_max],
            "price_filtered_out": skipped_price,
            "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": int(len(daily)),
            "format": "parquet",
        }
        meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        for _p in (ckpt_json, partial_parquet):
            try:
                _p.unlink(missing_ok=True)
            except OSError:
                pass
        print(f"完成: {daily_parquet}")
        print(f"  {meta['stocks']} 只股票 × 最多 {len(window)} 交易日 = {len(daily)} 行"
              f"（价格剔除 {skipped_price} 只）")
        return 0
    finally:
        fetcher.logout()


if __name__ == "__main__":
    raise SystemExit(main())
