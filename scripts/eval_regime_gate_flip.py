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
    parser.add_argument('--pred-dir', type=str, required=True, 
                        help="指向特定 window year 的 rolling 輸出目錄 (例如 output_rolling_grid/RUN_NAME/windows/w3)")
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
    
    # 1. 取得 Benchmark 歷史資料並計算 Gate 狀態
    print(f"📦 正在獲取 Benchmark ({BENCHMARK}) 最新資料以計算 Regime Gates...")
    all_data = fetch_all_stock_data()
    benchmark_df = all_data.get(BENCHMARK)
    if benchmark_df is None:
         print(f"❌ 無法取得 Benchmark ({BENCHMARK}) 資料")
         sys.exit(1)
         
    df_bmk_features = compute_gate_features(benchmark_df)
    df_gates = apply_regime_gates(df_bmk_features)
    # 日期正規化，為了後續跟 val_predictions merge
    df_gates['date'] = pd.to_datetime(df_gates['date']).dt.strftime('%Y-%m-%d')
    
    # 2. 掃描 `--pred-dir` 下的個別年份子目錄
    search_path = os.path.join(args.pred_dir, f"{args.ticker}_*")
    year_dirs = sorted(glob(search_path))
    
    if not year_dirs:
        print(f"❌ 找不到任何年份的預測資料。檢查目錄: {search_path}")
        sys.exit(1)
        
    print(f"🔍 找到 {len(year_dirs)} 個驗證年份，開始進行 Gate Flip 測試 (Top {args.topk_pct}%)")
    
    k_pct = args.topk_pct / 100.0
    gate_names = ['Gate_A', 'Gate_B', 'Gate_C', 'Gate_D']
    master_summary = []
    
    for y_dir in year_dirs:
        val_csv = os.path.join(y_dir, "val_predictions.csv")
        metrics_json = os.path.join(y_dir, "metrics.json")
        param_json = os.path.join(y_dir, "params.json")
        
        if not os.path.exists(val_csv):
            continue
            
        df_pred = pd.read_csv(val_csv)
        if len(df_pred) == 0:
            continue
            
        # 讀取原本存入的資訊 (主要是年分與 roc-auc 用來參考)
        with open(param_json, 'r', encoding='utf-8') as f:
             params = json.load(f)
             val_y = params.get('val_year')
             val_n = params.get('val_samples')
             val_pos = params.get('val_pos_rate')
             
        # Normalize date
        df_pred['date_str'] = pd.to_datetime(df_pred['date']).dt.strftime('%Y-%m-%d')
        
        # Merge gates into predictions
        df_merged = pd.merge(df_pred, df_gates, left_on='date_str', right_on='date', how='inner')
        if len(df_merged) == 0:
            print(f"⚠️ {val_y} 找不到任何符合基準日期的 Gate 數據，跳過。")
            continue
            
        # 計算 Baseline
        hit_proba = get_topk_hit_rate(df_merged, 'y_proba', 'y_true', k_pct)
        df_merged['inv_proba'] = 1.0 - df_merged['y_proba']
        hit_invproba = get_topk_hit_rate(df_merged, 'inv_proba', 'y_true', k_pct)
        
        # 若 inv_proba 的命中率比正向高出任何一點 (或高過 10%) 就代表原始預測出現反轉
        reversal_warning_orig = hit_invproba > hit_proba
        
        # 針對每一種 Gate 進行 Score Flip
        row_data = {
            'year': val_y,
            'n_val': val_n,
            'pos_rate': val_pos,
            'topk_hit_proba': hit_proba,
            'topk_hit_invproba': hit_invproba,
            'reversal_warning_orig': reversal_warning_orig
        }
        
        # 動態計算各種 Gate
        for g in gate_names:
             # 如果狀態是 normal，保持原本機率；如果是 reversal 就 1 - y_proba
             g_score_col = f'score_flip_{g}'
             df_merged[g_score_col] = np.where(df_merged[g] == 'normal', 
                                               df_merged['y_proba'], 
                                               1.0 - df_merged['y_proba'])
             
             hit_flip = get_topk_hit_rate(df_merged, g_score_col, 'y_true', k_pct)
             row_data[f'topk_hit_{g}'] = hit_flip
             
             # 算算這一年這個 Gate "救回" 多少 hit rate
             # 相對於原本如果單純信任 proba 的改變幅度
             row_data[f'improv_{g}'] = hit_flip - hit_proba
             
             # 該Gate發動翻轉的日數比例
             flip_ratio = (df_merged[g] == 'reversal').mean()
             row_data[f'flip_ratio_{g}'] = flip_ratio
             
        master_summary.append(row_data)

    if not master_summary:
        print("❌ 沒有完成任何年份的彙整計算。")
        sys.exit(0)
        
    df_sum = pd.DataFrame(master_summary)
    
    # 建立輸出結果目錄
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"gate_eval_summary_{args.ticker}.csv")
    df_sum.to_csv(out_csv, index=False)
    
    print(f"\n{'='*80}\n✅ Regime Gate Flip 離線評估完成！\n{'='*80}")
    
    for g in gate_names:
        # 計算是否改善 reversal year 數量
        total_rev_years = df_sum['reversal_warning_orig'].sum()
        
        # 計算 Gate 修正後，這一年還是不是「反過來做會更好」
        # 理論上如果 Gate 很準，翻轉過後，你再去 inv 它一定會變差，表示當下方向是對的。
        # 所以我們看 "如果用 flip score，再去 inv 它一次，會不會更好？"
        # 若依舊更好，代表 Gate 沒把顛倒修正過來 (或是濫殺無辜導致新的顛倒)
        # 這裡從簡：看平均提升勝率
        avg_hit_orig_proba = df_sum['topk_hit_proba'].mean()
        avg_hit_gate = df_sum[f'topk_hit_{g}'].mean()
        avg_improv = avg_hit_gate - avg_hit_orig_proba
        
        print(f"🔸 【{g}】 測試結果：")
        print(f"    - 全部年度平均 Top {args.topk_pct}% 命中率: 原本 {avg_hit_orig_proba:.1%} -> 變成 {avg_hit_gate:.1%} ({avg_improv*100:+.1f}%)")
        
        # 觀察 2019/2023 兩個魔咒年份
        for prob_y in [2019, 2021, 2022, 2023]:
            if prob_y in df_sum['year'].values:
                y_row = df_sum[df_sum['year'] == prob_y].iloc[0]
                orig_h = y_row['topk_hit_proba']
                gate_h = y_row[f'topk_hit_{g}']
                imp = y_row[f'improv_{g}']
                print(f"    - {prob_y} 表現: {orig_h:.1%} -> {gate_h:.1%} ({imp*100:+.1f}%) | 翻轉天數佔比: {y_row[f'flip_ratio_{g}']:.1%}")
        print("-" * 50)
        
    print(f"\n📂 完整彙整表已輸出至: {out_csv}")


if __name__ == "__main__":
    main()
