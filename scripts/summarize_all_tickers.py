#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批次滾動結果彙整工具 (Summarize All Tickers)
讀取 run_rolling_all_tickers 產出的各股票目錄下之 rolling_summary.csv
並針對指定的年份區段計算：Mean/Median AUC、Top10 Hit、Worst Gap 及 Reversal 警報次數，
最後輸出一份總表與 JSON 可讓使用者一眼比較哪檔股票最具動能預測穩定性。
"""

import os
import argparse
import pandas as pd
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="彙整批次 Rolling 的所有股票年度成效表")
    parser.add_argument('--input-dir', type=str, default='output_rolling_all', 
                        help="包含各 Ticker 目錄的輸入根目錄。預設: output_rolling_all")
    parser.add_argument('--output-dir', type=str, default='output_rolling_all', 
                        help="總表輸出的存放目錄。預設同 input-dir")
    parser.add_argument('--years-from', type=int, default=2017, help="彙整起始年份 (預設: 2017)")
    parser.add_argument('--years-to', type=int, default=2030, help="彙整結束年份 (預設: 2030)")
    parser.add_argument('--topk', type=int, default=10, help="參考的分位數。目前 rolling 自帶 top5 / top10")
    parser.add_argument('--reversal-gap-margin', type=float, default=0.10, 
                        help="反轉計算條件之 gap 容忍。若 summary 中無 reversal_warning 才重算")
    parser.add_argument('--sort-by', type=str, default='mean_top10_hit_proba', 
                        choices=['reversal_year_count_v2', 'mean_top10_hit_proba', 'mean_roc_auc', 'worst_top10_gap'],
                        help="總表輸出的預設排序欄位")
    return parser.parse_args()


def safe_mean(series):
    return float(series.mean()) if not series.empty else np.nan

def safe_median(series):
    return float(series.median()) if not series.empty else np.nan

def safe_min(series):
    return float(series.min()) if not series.empty else np.nan


def main():
    args = parse_args()
    print("====================================================================")
    print("📊 Summarizing All Tickers Rolling Performance")
    print("====================================================================")
    
    if not os.path.exists(args.input_dir):
        print(f"❌ 找不到輸入目錄: {args.input_dir}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    tickers_results = []
    skipped_tickers = []
    
    # 掃描輸入目錄底下所有的子目錄 (理想情況: 每個名稱都是一個 Ticker)
    for ticker_item in sorted(os.listdir(args.input_dir)):
        ticker_dir = os.path.join(args.input_dir, ticker_item)
        
        # 排除非目錄物件
        if not os.path.isdir(ticker_dir):
            continue
            
        csv_path = os.path.join(ticker_dir, "rolling_summary.csv")
        if not os.path.exists(csv_path):
            # 不是 Ticker 目錄，可能只是外層的檔案
            continue
            
        ticker = ticker_item
        df = pd.read_csv(csv_path)
        
        if df.empty:
            skipped_tickers.append({"ticker": ticker, "reason": "Empty CSV dataset"})
            continue
            
        # 篩選要求的年份區間
        df['val_year'] = df['val_year'].astype(int)
        mask = (df['val_year'] >= args.years_from) & (df['val_year'] <= args.years_to)
        df_filt = df[mask].copy()
        
        if df_filt.empty:
            skipped_tickers.append({
                "ticker": ticker, 
                "reason": f"No data in year range {args.years_from}-{args.years_to}"
            })
            continue
            
        n_years = len(df_filt)
        
        # 動態判定有無內建 V2 reversal 欄位
        if 'reversal_warning' in df_filt.columns:
            has_reversals = df_filt['reversal_warning'] == True
        else:
            # Fallback 舊版的重算 (假設仍有 top10_gap 與 roc_auc)
            is_gap_fail = (df_filt['top10_gap'] <= -args.reversal_gap_margin)
            is_roc_fail = (df_filt['roc_auc'] < 0.5)
            has_reversals = is_gap_fail | is_roc_fail
            
        rev_count = has_reversals.sum()
        rev_years = df_filt.loc[has_reversals, 'val_year'].astype(str).tolist()
        
        # 尋找 ROC AUC 最差的一年
        worst_auc_idx = df_filt['roc_auc'].idxmin() if 'roc_auc' in df_filt.columns else None
        worst_roc_val = df_filt.loc[worst_auc_idx, 'roc_auc'] if worst_auc_idx is not None else np.nan
        worst_roc_yr = df_filt.loc[worst_auc_idx, 'val_year'] if worst_auc_idx is not None else np.nan
        
        # 尋找 Top10 Gap 最差的一年 (最小)
        gap_col = f"top{args.topk}_gap"
        prob_col = f"top{args.topk}_hit_proba"
        top5_gap_col = "top5_gap"
        top5_prob_col = "top5_hit_proba"
        
        worst_gap_val, worst_gap_yr = np.nan, np.nan
        if gap_col in df_filt.columns:
            worst_gap_idx = df_filt[gap_col].idxmin()
            worst_gap_val = df_filt.loc[worst_gap_idx, gap_col]
            worst_gap_yr = df_filt.loc[worst_gap_idx, 'val_year']
            
        # 打包 Single Ticker Aggregate Metrics
        metrics = {
            "ticker": ticker,
            "n_years_evaluated": n_years,
            "mean_roc_auc": safe_mean(df_filt.get('roc_auc', pd.Series(dtype=float))),
            "median_roc_auc": safe_median(df_filt.get('roc_auc', pd.Series(dtype=float))),
            "mean_pr_auc": safe_mean(df_filt.get('pr_auc', pd.Series(dtype=float))),
            "median_pr_auc": safe_median(df_filt.get('pr_auc', pd.Series(dtype=float))),
            f"mean_{prob_col}": safe_mean(df_filt.get(prob_col, pd.Series(dtype=float))),
            f"median_{prob_col}": safe_median(df_filt.get(prob_col, pd.Series(dtype=float))),
            f"mean_{gap_col}": safe_mean(df_filt.get(gap_col, pd.Series(dtype=float))),
            f"worst_{gap_col}": float(worst_gap_val) if pd.notna(worst_gap_val) else np.nan,
            "reversal_year_count_v2": int(rev_count),
            "reversal_years_list_v2": ",".join(rev_years) if rev_years else "None",
            "worst_year_by_roc_auc": f"{worst_roc_yr} ({worst_roc_val:.3f})" if pd.notna(worst_roc_yr) else "N/A",
            f"worst_year_by_{gap_col}": f"{worst_gap_yr} ({worst_gap_val:.3f})" if pd.notna(worst_gap_yr) else "N/A",
        }
        
        # 若有 Top5 欄位一併匯出可選資訊
        if top5_prob_col in df_filt.columns:
             metrics[f"mean_{top5_prob_col}"] = safe_mean(df_filt[top5_prob_col])
        if top5_gap_col in df_filt.columns:
             metrics[f"mean_{top5_gap_col}"] = safe_mean(df_filt[top5_gap_col])
             
        tickers_results.append(metrics)
        print(f"  ✅ 處理完成: {ticker} (彙整 {n_years} 筆年度紀錄)")

    print("-" * 68)
    
    if len(tickers_results) == 0:
        print("⚠️ 未找到任何成功的 Ticker 總結可以輸出。請確保已經先跑過 run_rolling_all_tickers.py。")
        return
        
    # 組合 DataFrame 與排序
    df_out = pd.DataFrame(tickers_results)
    
    # 決定排列順序 (reversal_count 升序，其他可能為降序較好)
    ascending = True if args.sort_by == 'reversal_year_count_v2' else False
    if args.sort_by in df_out.columns:
        df_out = df_out.sort_values(by=[args.sort_by, 'ticker'], ascending=[ascending, True])
    
    # 準備 JSON Output 的包裝
    final_output = {
        "summary_params": {
            "years_from": args.years_from,
            "years_to": args.years_to,
            "topk_used": args.topk,
            "sort_by": args.sort_by
        },
        "skipped_tickers": skipped_tickers,
        "tickers_data": df_out.to_dict(orient="records")
    }
    
    csv_path = os.path.join(args.output_dir, "all_tickers_summary.csv")
    json_path = os.path.join(args.output_dir, "all_tickers_summary.json")
    
    df_out.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"🎉 成功輸出跨 Ticker 大表比對!")
    print(f"👉 JSON Path: {json_path}")
    print(f"👉 CSV Path:  {csv_path}")
    
    # 終端印出一份縮減版的重點表格
    cols_to_print = ['ticker', 'n_years_evaluated', f'mean_{prob_col}', f'worst_{gap_col}', 
                     'reversal_year_count_v2', 'reversal_years_list_v2']
    valid_cols = [c for c in cols_to_print if c in df_out.columns]
    print("\n🎯 [快速預覽]")
    print(df_out[valid_cols].to_string(index=False))
    print("\n====================================================================")


if __name__ == "__main__":
    main()
