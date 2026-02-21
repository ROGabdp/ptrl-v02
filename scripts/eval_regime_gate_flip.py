#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Regime Gate Flip Evaluation
================================================================================
這個腳本設計用來「離線」評估簡單的 Regime Gate 是否能有效翻轉反向指標年份的預測。
不需要重新訓練模型，直接讀取現有的 rolling HGB val_predictions.csv，
並結合從 Benchmark 計算的 Gate 規則 (A, B, C, D) 進行 score flip。
================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.eval.gate_utils import compute_gate_features, apply_regime_gates
try:
    from train_us_tech_buy_agent import fetch_all_stock_data, BENCHMARK
except ImportError:
    print("❌ 找不到 train_us_tech_buy_agent 模組，請確定您在專案根目錄下執行。")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Regime Gate Flip 離線評估腳本")
    parser.add_argument('--ticker', type=str, default='GOOGL', help="目標股票 (預設: GOOGL)")
    parser.add_argument('--pred-dir', type=str, help="[向前相容] 指向單一 window year 的 rolling 輸出目錄")
    parser.add_argument('--pred-dirs', type=str, nargs='+', help="直接指定多個 rolling 輸出目錄 (例如: windows/w3 windows/w5)")
    parser.add_argument('--base-dir', type=str, help="基礎目錄 (例如 output_rolling_grid/RUN_NAME/windows)")
    parser.add_argument('--windows', type=str, nargs='+', help="搭配 base-dir 使用，指定要評估的 window_years (例如 3 5 7)")
    parser.add_argument('--topk-pct', type=float, default=5.0, help="評估 Top K% 的 Hitachi Rate (預設: 5.0)")
    parser.add_argument('--output-dir', type=str, default='output_gate_eval', help="輸出目錄")
    parser.add_argument('--no-cache', action='store_true', help="強制重新擷取歷史資料")
    return parser.parse_args()


def get_topk_hit_rate(df, score_col, target_col='y_true', k_pct=0.05):
    """根據指定的 score 欄位排序，取出前 k%，計算 y_true 的平均 (Hit Rate)"""
    n_samples = len(df)
    k = max(1, int(n_samples * k_pct))
    
    # 按照 score 降序排
    df_sorted = df.sort_values(by=score_col, ascending=False)
    top_k_y = df_sorted.head(k)[target_col]
    return float(top_k_y.mean())


def main():
    args = parse_args()
    
    pred_dirs_dict = {}
    if args.base_dir and args.windows:
        for w in args.windows:
            label = f"w{w}" if not str(w).startswith("w") else str(w)
            pred_dirs_dict[label] = os.path.join(args.base_dir, label)
    elif args.pred_dirs:
        for d in args.pred_dirs:
            label = os.path.basename(os.path.normpath(d))
            pred_dirs_dict[label] = d
    elif args.pred_dir:
        label = os.path.basename(os.path.normpath(args.pred_dir))
        pred_dirs_dict[label] = args.pred_dir
    else:
        print("❌ 請提供 --pred-dirs, 或者 --base-dir 加上 --windows")
        sys.exit(1)

    # 1. 取得 Benchmark 歷史資料並計算 Gate 狀態
    print(f"📦 正在獲取 Benchmark ({BENCHMARK}) 最新資料以計算 Regime Gates...")
    all_data = fetch_all_stock_data()
    benchmark_df = all_data.get(BENCHMARK)
    if benchmark_df is None:
         print(f"❌ 無法取得 Benchmark ({BENCHMARK}) 資料")
         sys.exit(1)
         
    df_bmk_features = compute_gate_features(benchmark_df)
    df_gates = apply_regime_gates(df_bmk_features)
    df_gates['date'] = pd.to_datetime(df_gates['date']).dt.strftime('%Y-%m-%d')
    
    k_pct = args.topk_pct / 100.0
    gate_names = ['Gate_A', 'Gate_B', 'Gate_C', 'Gate_D']
    master_summary_long = []

    print(f"\n🚀 開始進行 Gate Flip 評估 (Top {args.topk_pct}%)")
    print(f"目標 Windows: {list(pred_dirs_dict.keys())}")

    # 2. 掃描各個 window 目錄
    for window_label, pred_dir in pred_dirs_dict.items():
        search_path = os.path.join(pred_dir, f"{args.ticker}_*")
        year_dirs = sorted(glob(search_path))
        
        if not year_dirs:
            print(f"  ⚠️ [{window_label}] 找不到任何預測資料，跳過。路徑: {pred_dir}")
            continue
            
        print(f"\n🌀 處理 Window: {window_label} | 找到 {len(year_dirs)} 個驗證年份")
        
        for y_dir in year_dirs:
            val_csv = os.path.join(y_dir, "val_predictions.csv")
            param_json = os.path.join(y_dir, "params.json")
            
            if not os.path.exists(val_csv):
                continue
                
            df_pred = pd.read_csv(val_csv)
            if len(df_pred) == 0:
                continue
                
            with open(param_json, 'r', encoding='utf-8') as f:
                 params = json.load(f)
                 val_y = params.get('val_year')
                 val_n = params.get('val_samples')
                 val_pos = params.get('val_pos_rate')
                 
            df_pred['date_str'] = pd.to_datetime(df_pred['date']).dt.strftime('%Y-%m-%d')
            
            df_merged = pd.merge(df_pred, df_gates, left_on='date_str', right_on='date', how='inner')
            if len(df_merged) == 0:
                print(f"  ⚠️ {window_label} - {val_y} 無法與大盤日期對齊，跳過。")
                continue
                
            hit_proba = get_topk_hit_rate(df_merged, 'y_proba', 'y_true', k_pct)
            df_merged['inv_proba'] = 1.0 - df_merged['y_proba']
            hit_invproba = get_topk_hit_rate(df_merged, 'inv_proba', 'y_true', k_pct)
            
            reversal_warning_orig = hit_invproba > hit_proba
            
            row_data = {
                'window_years': window_label,
                'year': val_y,
                'n_val': val_n,
                'pos_rate': val_pos,
                'topk_hit_proba': hit_proba,
                'topk_hit_invproba': hit_invproba,
                'reversal_warning_orig': reversal_warning_orig
            }
            
            for g in gate_names:
                 g_score_col = f'score_flip_{g}'
                 g_inv_score_col = f'inv_score_flip_{g}'
                 
                 # 翻轉邏輯
                 df_merged[g_score_col] = np.where(df_merged[g] == 'normal', 
                                                   df_merged['y_proba'], 
                                                   1.0 - df_merged['y_proba'])
                 df_merged[g_inv_score_col] = 1.0 - df_merged[g_score_col]
                 
                 hit_flip = get_topk_hit_rate(df_merged, g_score_col, 'y_true', k_pct)
                 hit_inv_flip = get_topk_hit_rate(df_merged, g_inv_score_col, 'y_true', k_pct)
                 
                 row_data[f'topk_hit_{g}'] = hit_flip
                 row_data[f'improv_{g}'] = hit_flip - hit_proba
                 row_data[f'flip_ratio_{g}'] = (df_merged[g] == 'reversal').mean()
                 row_data[f'reversal_after_{g}'] = hit_inv_flip > hit_flip
                 
            master_summary_long.append(row_data)

    if not master_summary_long:
        print("\n❌ 沒有完成任何年份的彙整計算。")
        sys.exit(0)
        
    df_long = pd.DataFrame(master_summary_long)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 輸出長表 (By Window Year)
    out_csv_long = os.path.join(args.output_dir, f"gate_eval_summary_{args.ticker}_by_window_year.csv")
    df_long.to_csv(out_csv_long, index=False)
    
    # 3. 建立 Aggregated 彙整表
    agg_data = []
    for w_label, df_w in df_long.groupby('window_years'):
        agg_row = {'window_years': w_label}
        agg_row['n_years_eval'] = len(df_w)
        agg_row['mean_topk_hit_proba'] = df_w['topk_hit_proba'].mean()
        agg_row['reversal_year_count_before'] = df_w['reversal_warning_orig'].sum()
        
        for g in gate_names:
            agg_row[f'mean_topk_hit_{g}'] = df_w[f'topk_hit_{g}'].mean()
            agg_row[f'reversal_year_count_after_{g}'] = df_w[f'reversal_after_{g}'].sum()
            agg_row[f'worst_year_drop_{g}'] = df_w[f'improv_{g}'].min()
            
        agg_data.append(agg_row)
        
    df_agg = pd.DataFrame(agg_data)
    out_csv_agg = os.path.join(args.output_dir, f"gate_eval_summary_{args.ticker}_window_agg.csv")
    df_agg.to_csv(out_csv_agg, index=False)
    
    print(f"\n{'='*80}\n✅ Regime Gate Flip 跨 Windows 離線評估完成！\n{'='*80}")
    print(f"📂 詳細年度長表已輸出至: {out_csv_long}")
    print(f"📂 Windows 綜合比較表:  {out_csv_agg}")
    
    print("\n📊 各 Window 彙整概覽 (Gate_C 為例):")
    for _, row in df_agg.iterrows():
        w = row['window_years']
        pct_b = row['mean_topk_hit_proba'] * 100
        pct_a = row['mean_topk_hit_Gate_C'] * 100
        rev_b = row['reversal_year_count_before']
        rev_a = row['reversal_year_count_after_Gate_C']
        drop = row['worst_year_drop_Gate_C'] * 100
        print(f"  [{w}] 勝率: {pct_b:.1f}% -> {pct_a:.1f}% | 反轉年數: {rev_b} -> {rev_a} | 最慘負改善: {drop:+.1f}%")


if __name__ == '__main__':
    main()
