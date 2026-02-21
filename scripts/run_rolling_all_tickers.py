#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批次執行所有 Tickers 的 Rolling HGB Walk-Forward 訓練腳本
提供多進程並行執行，並將個別 Ticker 的結果獨立存放在輸出目錄中。
"""

import os
import sys
import argparse
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 依賴共用模組，以利參數物件裝配
try:
    from scripts.train_rolling_hgb import run_rolling_training
except ImportError as e:
    print(f"❌ 無法載入 train_rolling_hgb: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="批次執行所有 Tickers 的 Rolling 評估")
    
    # 目標 Tickers 與輸出目錄
    parser.add_argument('--tickers', nargs='+', 
                        default=['GOOGL', 'NVDA', 'MSFT', 'AMZN', 'META', 'AVGO', 'NFLX', 'AAPL', 'TSLA', 'PLTR', 'TSM'],
                        help="欲批次執行的股票代碼列表")
    parser.add_argument('--output-dir', type=str, default='output_rolling_all', help="批次根目錄")
    
    # 執行控制
    parser.add_argument('--max-workers', type=int, default=1, 
                        help="並行處理的最大進程數 (預設: 1)，若為 1 則循序執行")
    parser.add_argument('--dry-run', action='store_true', help="僅列印預計處理之股票與參數，不實際觸發")
    
    # 傳遞給 train_rolling_hgb 的通用參數
    parser.add_argument('--window-years', type=int, default=3, help="訓練窗格長度 (年)")
    parser.add_argument('--target-days', type=int, default=120, help="目標天數")
    parser.add_argument('--target-return', type=float, default=0.20, help="目標報酬門檻")
    parser.add_argument('--use-regime-features', action='store_true', help="包含大盤 Regime 特徵")
    parser.add_argument('--reversal-gap-margin', type=float, default=0.10, help="反轉警告門檻差距")
    parser.add_argument('--val-years', nargs='+', type=str, help="指定驗證年度 (例如: 2018 2019 2020 ...)")
    parser.add_argument('--seed', type=int, default=42, help="亂數種子")
    parser.add_argument('--no-cache', action='store_true', help="不使用特徵快取")
    
    return parser.parse_args()


def process_single_ticker(ticker, base_args, root_output_dir):
    """
    對單一 Ticker 重建 args namespace 並呼叫核心訓練函數。
    """
    ticker_out_dir = os.path.join(root_output_dir, ticker)
    if not base_args.dry_run:
        os.makedirs(ticker_out_dir, exist_ok=True)
    
    # 建立偽 Args 物件模擬 argparse
    class DummyArgs:
        pass
    
    d_args = DummyArgs()
    d_args.tickers = [ticker]
    d_args.output_dir = ticker_out_dir
    d_args.target_days = base_args.target_days
    d_args.target_return = base_args.target_return
    d_args.window_years = base_args.window_years
    d_args.val_years = base_args.val_years
    d_args.start_year = None
    d_args.end_year = None
    d_args.model = 'hgb'
    d_args.seed = base_args.seed
    d_args.balance_train = 'none' # 預設固定 None，要求實作平穩
    d_args.use_regime_features = base_args.use_regime_features
    d_args.reversal_gap_margin = base_args.reversal_gap_margin
    d_args.reversal_use_top10 = 'true' # 預設固定 True 以供 V2 Reversal 雙保險
    d_args.no_cache = base_args.no_cache
    d_args.dry_run = base_args.dry_run
    
    if base_args.dry_run:
        print(f"[DRY-RUN] 會執行 Ticker: {ticker} => Output: {ticker_out_dir}")
        return ticker, True, None
        
    try:
        # 呼叫已經模組化好的 run_rolling_training (內部會建 rolling_summary.csv 等)
        # 注意: 如果開啟並行, train_rolling_hgb 內部的列印訊息可能會互相交錯
        summary_list = run_rolling_training(d_args)
        return ticker, True, summary_list
    except Exception as e:
        return ticker, False, str(e)


def main():
    args = parse_args()
    
    print("====================================================================")
    print("🚀 Batch Rolling Evaluation - All Tickers")
    print("====================================================================")
    print(f"🎯 Target Tickers : {', '.join(args.tickers)}")
    print(f"📦 Output Dir     : {args.output_dir}")
    print(f"⚙️ Workers        : {args.max_workers}")
    print(f"⚙️ Window Years   : {args.window_years}")
    print(f"⚙️ Target Definition: {args.target_days}d >= {args.target_return*100:g}%")
    print(f"⚙️ Regime Active  : {args.use_regime_features}")
    print(f"📅 Val Years Filt : {args.val_years if args.val_years else 'Auto-detect'}")
    print("====================================================================\n")
    
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
        
    start_time = time.time()
    
    results = {}
    
    if args.max_workers <= 1:
        # Sequential Execution
        for tk in args.tickers:
            print(f">>> 啟動 Rolling 執行緒: {tk} <<<")
            tk_ret, is_success, msg = process_single_ticker(tk, args, args.output_dir)
            results[tk] = (is_success, msg)
            if not is_success:
                print(f"❌ {tk} 執行失敗: {msg}")
    else:
        # Parallel Execution
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_single_ticker, tk, args, args.output_dir): tk 
                for tk in args.tickers
            }
            
            for future in as_completed(futures):
                tk = futures[future]
                try:
                    tk_ret, is_success, msg = future.result()
                    results[tk] = (is_success, msg)
                    status_emoji = "✅" if is_success else "❌"
                    print(f"[{status_emoji}] 處理完成: {tk}", "" if is_success else f"- {msg}")
                except Exception as exc:
                    results[tk] = (False, str(exc))
                    print(f"[❌] 執行緒崩潰: {tk} 產生異常 {exc}")

    if not args.dry_run:
        passed = sum(1 for status, msg in results.values() if status)
        failed = len(results) - passed
        print("\n====================================================================")
        print(f"🎉 批次執行完畢 | 耗時: {time.time() - start_time:.2f} 秒")
        print(f"✔️ 成功: {passed} 檔 | ❌ 失敗: {failed} 檔")
        print("====================================================================")
        print("下一步建議: 呼叫 scripts/summarize_all_tickers.py 來進行年度總表彙整。")


if __name__ == "__main__":
    main()
