"""构建本地回溯样本库。

下载指定指数成分股最近 N 个交易日的历史日 K（前复权），按股价过滤后，
以 Parquet（pandas 快速列式格式）落地到 data_library/。

用法：
    python build_data_library.py                          # 默认：更新 zz500 + hs300 两库并合并
    python build_data_library.py --index hs300            # 仅更新 hs300（不合并）
    python build_data_library.py --indexes zz500,hs300    # 更新指定多个指数并合并
    python build_data_library.py --index zz500 --days 200 --price-min 2 --price-max 300
    python build_data_library.py --combine                # 仅合并现有 zz500 + hs300 为统一库

指数（baostock 支持）: hs300(沪深300) / zz500(中证500) / sz50(上证50)。
股价过滤：以该股最新收盘价判断，超出 [price-min, price-max] 则剔除（0 表示不限）。

默认行为：不带 --index/--indexes 时，自动更新 COMBINE_ORDER（zz500 + hs300）里
的全部指数库，随后合并为 combined 统一库（交集以 zz500 优先）。

产物（按 index 命名）：
    data_library/<index>_daily.parquet   长表（code,date,OHLCV...）
    data_library/<index>_stocks.parquet  成分股元数据（code,code_name）
    data_library/<index>_meta.json       构建信息

合并库（--combine 时额外产出，交集以 zz500 优先）：
    data_library/combined_daily.parquet    zz500 + hs300 合并去重后的长表
    data_library/combined_stocks.parquet   成分股元数据并集
    data_library/combined_meta.json        合并信息

支持断点续跑：中途中断重跑会自动跳过已下载的股票（检查点按 index+价格区间隔离）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from core.data_fetcher import DataFetcher
from core.local_library import merge_libraries

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_library"

INDEX_NAMES = {"hs300": "沪深300", "zz500": "中证500", "sz50": "上证50"}

# 合并库的命名。combine 时按此顺序决定优先级：靠前者优先（交集以 zz500 优先）。
# 因此 zz500 必须排在 hs300 之前。
COMBINE_NAME = "combined"
COMBINE_ORDER = ["zz500", "hs300"]


def combine_libraries(daily_names: list[str] | None = None) -> dict:
    """把若干指数库合并为一个统一样本库。

    产物（data_library/）：
        combined_daily.parquet    合并去重后的长表（交集以 zz500 优先）
        combined_stocks.parquet   成分股元数据并集
        combined_meta.json        合并信息

    返回 meta dict。daily_names 缺省时用 COMBINE_ORDER（zz500 + hs300）。
    """
    names = daily_names or COMBINE_ORDER
    daily_paths = [DATA_DIR / f"{n}_daily.parquet" for n in names if (DATA_DIR / f"{n}_daily.parquet").exists()]
    if not daily_paths:
        raise RuntimeError(f"未找到可合并的库：{names}（请先 build 对应指数）")

    # 优先级：names 中靠前者优先（zz500 优先于 hs300）
    merged = merge_libraries(daily_paths, priority_paths=daily_paths)

    # 成分股元数据并集（code_name 也以优先级靠前者为准）
    stock_frames: list[pd.DataFrame] = []
    seen_codes: set[str] = set()
    for n in names:
        p = DATA_DIR / f"{n}_stocks.parquet"
        if not p.exists():
            continue
        st = pd.read_parquet(p)
        if st.empty or "code" not in st.columns:
            continue
        st = st[~st["code"].isin(seen_codes)].copy()
        seen_codes |= set(st["code"].tolist())
        stock_frames.append(st)
    stocks = (pd.concat(stock_frames, ignore_index=True)
              if stock_frames else pd.DataFrame())

    merged.to_parquet(DATA_DIR / f"{COMBINE_NAME}_daily.parquet", index=False)
    if not stocks.empty:
        stocks.to_parquet(DATA_DIR / f"{COMBINE_NAME}_stocks.parquet", index=False)

    lo = str(merged["date"].min()) if not merged.empty else ""
    hi = str(merged["date"].max()) if not merged.empty else ""
    meta = {
        "index": "合并样本库",
        "index_code": COMBINE_NAME,
        "sources": names,
        "priority": names,
        "stocks": int(merged["code"].nunique()) if not merged.empty else 0,
        "date_range": [lo, hi],
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": int(len(merged)),
        "format": "parquet",
    }
    (DATA_DIR / f"{COMBINE_NAME}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _auto_combine_if_ready() -> bool:
    """构建/更新完某指数库后，若合并所需的所有源库都已就绪，则自动合并一次。

    返回 True 表示已合并，False 表示源库不全（跳过）。
    """
    needed = [(DATA_DIR / f"{n}_daily.parquet").exists() for n in COMBINE_ORDER]
    if not all(needed):
        missing = [n for n, ok in zip(COMBINE_ORDER, needed) if not ok]
        print(f"[combine] 源库不全（缺 {missing}），跳过合并；待全部构建后可运行 "
              f"`python build_data_library.py --combine`")
        return False
    meta = combine_libraries(COMBINE_ORDER)
    print(f"[combine] 已合并生成统一库 {COMBINE_NAME}_daily.parquet："
          f"{meta['stocks']} 只 / {meta['rows']} 行")
    return True


def _latest_record_date(df: pd.DataFrame) -> str | None:
    """返回已落库数据中的最后一个日期（YYYY-MM-DD），不存在则返回 None。"""
    if df.empty or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().strftime("%Y-%m-%d")


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


def _build_one_index(
    index: str,
    days: int,
    price_min: float,
    price_max: float,
    request_interval: float,
    full: bool,
    fetcher: DataFetcher,
) -> int:
    """构建/增量更新单个指数库。返回 0 成功。复用同一 fetcher 连接。"""
    daily_parquet = DATA_DIR / f"{index}_daily.parquet"
    meta_parquet = DATA_DIR / f"{index}_stocks.parquet"
    meta_json = DATA_DIR / f"{index}_meta.json"

    # 断点续跑：检查点按 index + 价格区间 + 天数隔离，避免不同构建互相污染
    sig = hashlib.md5(
        f"{index}:{price_min}:{price_max}:{days}".encode("utf-8")
    ).hexdigest()[:8]
    ckpt_json = DATA_DIR / f"build_checkpoint_{index}_{sig}.json"
    partial_parquet = DATA_DIR / f"build_partial_{index}_{sig}.parquet"

    # 1. 成分股
    constituents = fetcher.get_index_stocks(index)
    codes = constituents["code"].tolist()
    name_map = dict(zip(constituents["code"], constituents["code_name"]))
    print(f"{INDEX_NAMES[index]}成分股: {len(codes)} 只")

    # 2. 读取已有本地库，若存在则以“最后一天为基点”增量补齐到最近交易日
    existing_daily = pd.read_parquet(daily_parquet) if daily_parquet.exists() else pd.DataFrame()
    if not full and not existing_daily.empty:
        last_date = _latest_record_date(existing_daily)
        if last_date is None:
            print(f"已存在 {daily_parquet}，但无有效日期列，改为重建全量数据")
        else:
            latest_trade = fetcher.latest_trading_day(lookback_days=30)
            if pd.Timestamp(last_date) >= pd.Timestamp(latest_trade):
                print(f"{INDEX_NAMES[index]}库已更新到最新交易日 {latest_trade}，无需补全")
                return 0

            backfill_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            backfill_end = latest_trade
            print(f"增量补全: {last_date} -> {backfill_end}（从 {backfill_start} 开始）")

            new_frames: list[pd.DataFrame] = []
            skipped_price = 0
            for code in codes:
                df = fetcher.get_history(code, backfill_start, backfill_end, adjustflag="2")
                if df.empty:
                    continue
                df = df[df["date"] > last_date].copy()
                if df.empty:
                    continue
                if _in_price_range(df, price_min, price_max):
                    df["code_name"] = name_map.get(code, "")
                    new_frames.append(df)
                else:
                    skipped_price += 1
                fetcher._throttle()

            if new_frames:
                merged = pd.concat([existing_daily, *new_frames], ignore_index=True)
                merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
                merged.to_parquet(daily_parquet, index=False)
            else:
                print("无新增行情数据需要补齐")

            constituents.to_parquet(meta_parquet, index=False)
            meta = {
                "index": INDEX_NAMES[index],
                "index_code": index,
                "stocks": int(merged["code"].nunique()) if "merged" in locals() else int(existing_daily["code"].nunique()),
                "constituents": int(len(codes)),
                "date_range": [
                    _latest_record_date(existing_daily) if "merged" not in locals() else pd.to_datetime(merged["date"]).min().strftime("%Y-%m-%d"),
                    latest_trade,
                ],
                "price_range": [price_min, price_max],
                "price_filtered_out": skipped_price,
                "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rows": int((merged if "merged" in locals() else existing_daily).shape[0]),
                "format": "parquet",
                "incremental_update": True,
            }
            meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"补全完成: {daily_parquet}，最新交易日 {latest_trade}")
            return 0

    # 3. 全量重建（原逻辑）
    today = datetime.now()
    trade_dates = fetcher.get_trade_dates(
        (today - timedelta(days=int(days * 1.8))).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
    )
    window = trade_dates[-days:]
    start, end = window[0], window[-1]
    print(f"交易日区间: {start} ~ {end}（{len(window)} 个交易日）")
    print(f"股价过滤: [{price_min if price_min > 0 else '-∞'}, "
          f"{price_max if price_max > 0 else '+∞'}] 元")

    # 4. 断点续跑
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

    # 5. 逐只下载 + 股价过滤
    skipped_price = 0
    for i, code in enumerate(codes, 1):
        if code in done:
            continue
        df = fetcher.get_history(code, start, end, adjustflag="2")
        if not df.empty:
            if _in_price_range(df, price_min, price_max):
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

    # 6. 最终落盘
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    daily.to_parquet(daily_parquet, index=False)
    constituents.to_parquet(meta_parquet, index=False)

    meta = {
        "index": INDEX_NAMES[index],
        "index_code": index,
        "stocks": int(daily["code"].nunique()) if not daily.empty else 0,
        "constituents": int(len(codes)),
        "trading_days": int(len(window)),
        "date_range": [start, end],
        "price_range": [price_min, price_max],
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


def main() -> int:
    parser = argparse.ArgumentParser(description="构建本地回溯样本库（Parquet）")
    parser.add_argument("--index", default=None,
                        help="构建单个指数（hs300/zz500/sz50）。缺省时构建默认组合并合并")
    parser.add_argument("--indexes", default=None,
                        help="构建多个指数（逗号分隔，如 zz500,hs300），构建完自动合并")
    parser.add_argument("--days", type=int, default=200, help="交易日数量（默认200）")
    parser.add_argument("--price-min", type=float, default=2.0, help="股价下限（0不限）")
    parser.add_argument("--price-max", type=float, default=300.0, help="股价上限（0不限）")
    parser.add_argument("--request-interval", type=float, default=0.08, help="请求间隔（秒）")
    parser.add_argument("--full", action="store_true", help="重建全部历史数据（忽略现有库）")
    parser.add_argument("--no-combine", action="store_true",
                        help="构建完成后不自动合并（仅对多指数/默认流程生效）")
    parser.add_argument("--combine", action="store_true",
                        help="仅合并现有 zz500 + hs300 库为 combined 统一库（不做下载）")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 仅合并：把当前 zz500 / hs300 库合并为一个统一库后退出。
    if args.combine:
        meta = combine_libraries(COMBINE_ORDER)
        print(f"合并完成: {DATA_DIR / (COMBINE_NAME + '_daily.parquet')}")
        print(f"  {meta['stocks']} 只股票（{meta['sources']} 并集，交集以 {meta['priority'][0]} 优先）"
              f"，{meta['rows']} 行，范围 {meta['date_range']}")
        return 0

    # 确定要构建的指数列表
    if args.indexes:
        indexes = [x.strip() for x in args.indexes.split(",") if x.strip()]
    elif args.index:
        indexes = [args.index]
    else:
        # 默认：更新 zz500 + hs300 两库并合并
        indexes = list(COMBINE_ORDER)

    # 校验指数名
    for idx in indexes:
        if idx not in INDEX_NAMES:
            print(f"[error] 不支持的指数: {idx}（可选 {list(INDEX_NAMES)}）", file=sys.stderr)
            return 1

    do_combine = (len(indexes) > 1) and not args.no_combine
    print(f"待构建指数: {indexes}" + ("（构建后自动合并）" if do_combine else ""))

    fetcher = DataFetcher(request_interval=args.request_interval)
    fetcher.login()
    try:
        for idx in indexes:
            _build_one_index(idx, args.days, args.price_min, args.price_max,
                             args.request_interval, args.full, fetcher)

        if do_combine:
            _auto_combine_if_ready()
        return 0
    finally:
        fetcher.logout()


if __name__ == "__main__":
    raise SystemExit(main())
