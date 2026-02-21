#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Rolling HGB Grid Search Wrapper
================================================================================
自動跑多個 window_years (例如 3, 5, 7) 並彙整總表的腳本。
這支腳本為 scripts/train_rolling_hgb.py 的擴編 wrapper。
================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy

# 將專案根目錄加到 sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from scripts.train_rolling_hgb import run_rolling_training
except ImportError:
    print("❌ 找不到 scripts.train_rolling_hgb 模組，請確定您在專案根目錄下執行。")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Rolling HGB Window Years Grid Search")
    
    parser.add_argument('--tickers', nargs='+', default=['GOOGL'], help="目標股票 (預設: GOOGL)")
    parser.add_argument('--output-dir', type=str, default='output_rolling_grid', help="輸出根目錄")
    parser.add_argument('--window-years-list', nargs='+', type=int, default=[3, 5, 7], help="要尋找的多組 window years (例如 3 5 7)")
    
    # 以下全數跟 train_rolling_hgb 共通
    parser.add_argument('--target-days', type=int, default=120, help="目標預測天數 (預設: 120)")
    parser.add_argument('--target-return', type=float, default=0.20, help="目標報酬率門檻 (預設: 0.20)")
    parser.add_argument('--val-years', nargs='+', type=int, help="限定的驗證年度")
    parser.add_argument('--start-year', type=int, default=None, help="限定的起始驗證年度")
    parser.add_argument('--end-year', type=int, default=None, help="限定的結束驗證年度")
    
    parser.add_argument('--model', type=str, default='hgb', choices=['hgb'], help="目前僅支援 HGB")
    parser.add_argument('--seed', type=int, default=42, help="亂數種子")
    parser.add_argument('--balance-train', type=str, default='none', 
                        choices=['none', 'undersample_50_50', 'class_weight_balanced'])
    
    parser.add_argument('--no-cache', action='store_true', help="強制重新計算特徵不使用快取")
    parser.add_argument('--dry-run', action='store_true', help="僅輸出將要跑的設定與路徑")
    
    return parser.parse_args()


def calculate_grid_metrics(master_summary, w):
    """
    從多個年度的 master_summary 計算單一個 window_years 的全局總指標
    """
    if not master_summary:
        return {'window_years': w, 'n_years_evaluated': 0, 'error': 'No data'}
        
    df = pd.DataFrame(master_summary)
    
    n_years = len(df)
    
    reversal_df = df[df['reversal_warning'] == True]
    reversal_count = len(reversal_df)
    reversal_list = ",".join(map(str, reversal_df['val_year'].tolist()))
    
    # 若有 None值，需轉為 NaN 計算
    roc_auc_series = pd.to_numeric(df['roc_auc'], errors='coerce').dropna()
    pr_auc_series = pd.to_numeric(df['pr_auc'], errors='coerce').dropna()
    
    mean_roc = roc_auc_series.mean() if not roc_auc_series.empty else None
    med_roc  = roc_auc_series.median() if not roc_auc_series.empty else None
    mean_pr  = pr_auc_series.mean() if not pr_auc_series.empty else None
    med_pr   = pr_auc_series.median() if not pr_auc_series.empty else None
    
    mean_hit_p = df['top5_hit_proba'].mean()
    med_hit_p  = df['top5_hit_proba'].median()
    mean_hit_inv = df['top5_hit_invproba'].mean()
    med_hit_inv  = df['top5_hit_invproba'].median()
    
    # 找尋最差表現年份
    worst_roc_row = df.loc[roc_auc_series.idxmin()] if not roc_auc_series.empty else None
    worst_year_roc = f"{int(worst_roc_row['val_year'])} ({worst_roc_row['roc_auc']:.3f})" if worst_roc_row is not None else "N/A"
    
    df['top5_gap'] = df['top5_hit_proba'] - df['top5_hit_invproba']
    worst_gap_row = df.loc[df['top5_gap'].idxmin()]
    worst_year_gap = f"{int(worst_gap_row['val_year'])} ({worst_gap_row['top5_gap']:.1%})"
    
    # 近五年表現 (2017 之後)
    df_recent = df[df['val_year'] >= 2017]
    rec_roc = pd.to_numeric(df_recent['roc_auc'], errors='coerce').dropna()
    mean_roc_recent = rec_roc.mean() if not rec_roc.empty else None
    rev_count_recent = len(df_recent[df_recent['reversal_warning'] == True])
    
    return {
        'window_years': w,
        'n_years_evaluated': n_years,
        'reversal_year_count': reversal_count,
        'reversal_years_list': reversal_list,
        'mean_roc_auc': mean_roc,
        'median_roc_auc': med_roc,
        'mean_pr_auc': mean_pr,
        'median_pr_auc': med_pr,
        'mean_top5_hit_proba': mean_hit_p,
        'median_top5_hit_proba': med_hit_p,
        'mean_top5_hit_invproba': mean_hit_inv,
        'median_top5_hit_invproba': med_hit_inv,
        'worst_year_by_roc_auc': worst_year_roc,
        'worst_year_by_top5_gap': worst_year_gap,
        'mean_roc_auc_2017plus': mean_roc_recent,
        'reversal_count_2017plus': rev_count_recent
    }


def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ticker_str = args.tickers[0] if len(args.tickers) == 1 else "MULTI"
    run_name = f"{ticker_str}_{args.target_days}d{int(args.target_return*100)}pct_{timestamp}"
    root_output_dir = os.path.join(args.output_dir, run_name)
    windows_dir = os.path.join(root_output_dir, "windows")
    
    print(f"============================================================")
    print(f"🎯 Rolling HGB Grid Search")
    print(f"============================================================")
    print(f"Tickers       : {args.tickers}")
    print(f"Window Years  : {args.window_years_list}")
    print(f"Target        : {args.target_days} Days / {args.target_return*100}%")
    print(f"Output Root   : {root_output_dir}")
    print(f"============================================================\n")
    
    if not args.dry_run:
        os.makedirs(windows_dir, exist_ok=True)
        # 儲存 Grid Config
        with open(os.path.join(root_output_dir, "grid_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)
            
    grid_summary_data = []
    yearly_long_data = []
    
    for w in args.window_years_list:
        print(f"\n{'#'*80}\n🌀 啟動 Grid Search: Window_Years = {w}\n{'#'*80}")
        
        # 準備傳遞給 train_rolling_hgb 的參數Namespace
        sub_args = deepcopy(args)
        sub_args.window_years = w
        sub_args.output_dir = os.path.join(windows_dir, f"w{w}")
        
        if args.dry_run:
            print(f"[Dry-Run] 將執行 Window: {w} 寫出至 {sub_args.output_dir}")
            continue
            
        print(f"輸出目錄設定至: {sub_args.output_dir}")
        master_summary = run_rolling_training(sub_args)
        
        # 彙整總表與轉換 Long Format
        stats = calculate_grid_metrics(master_summary, w)
        grid_summary_data.append(stats)
        
        for row in master_summary:
            yearly_long_data.append({
                'window_years': w,
                'year': row['val_year'],
                'roc_auc': row['roc_auc'],
                'pr_auc': row['pr_auc'],
                'top5_hit_proba': row['top5_hit_proba'],
                'top5_hit_invproba': row['top5_hit_invproba'],
                'reversal_warning': row['reversal_warning'],
                'n_val': row['val_n'],
                'val_pos_rate': row['val_pos_rate']
            })
            
    if not args.dry_run and grid_summary_data:
        print(f"\n{'='*80}\n✅ Grid Search 全部完成，正在寫出彙總報告...")
        
        df_grid = pd.DataFrame(grid_summary_data)
        csv_path = os.path.join(root_output_dir, "grid_summary.csv")
        json_path = os.path.join(root_output_dir, "grid_summary.json")
        df_grid.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(grid_summary_data, f, indent=4, ensure_ascii=False)
            
        long_path = os.path.join(root_output_dir, "grid_yearly_long.csv")
        df_long = pd.DataFrame(yearly_long_data)
        df_long.to_csv(long_path, index=False)
        
        print(f"📊 主彙整表 (Grid Summary)         : {csv_path}")
        print(f"📊 年度明細長表 (Yearly Long Data) : {long_path}")
        
        print("\n【網格搜尋結果概覽】")
        display_cols = ['window_years', 'n_years_evaluated', 'reversal_year_count', 'mean_roc_auc', 'worst_year_by_top5_gap']
        print(df_grid[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
