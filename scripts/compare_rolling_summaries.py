#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import json

def parse_args():
    parser = argparse.ArgumentParser(description="比較 Baseline (w3 無特徵) vs Regime (w3 加特徵) 的逐年 Rolling Summary 差異")
    parser.add_argument('--baseline', type=str, required=True, help="Baseline 的 rolling_summary.csv 路徑")
    parser.add_argument('--regime', type=str, required=True, help="Regime (加特徵) 的 rolling_summary.csv 路徑")
    parser.add_argument('--output-dir', type=str, default='output_compare', help="差分報表輸出目錄")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 確保輸出目錄或檔案路徑存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.baseline) or not os.path.exists(args.regime):
         print("❌ 找不到來源的 summary CSV 檔案。")
         return 
         
    df_base = pd.read_csv(args.baseline)
    df_regime = pd.read_csv(args.regime)
    
    # Inner Join on 'val_year'
    df_base.set_index('val_year', inplace=True)
    df_regime.set_index('val_year', inplace=True)
    
    common_years = df_base.index.intersection(df_regime.index)
    
    missing_in_base = df_regime.index.difference(df_base.index)
    missing_in_regime = df_base.index.difference(df_regime.index)
    
    if len(missing_in_base) > 0:
        print(f"⚠️ Baseline 缺少的年份: {missing_in_base.tolist()}")
    if len(missing_in_regime) > 0:
        print(f"⚠️ Regime 缺少的年份: {missing_in_regime.tolist()}")
        
    df_diff = pd.DataFrame(index=common_years)
    df_diff.index.name = 'year'
    
    # 計算並產出逐年比較
    # 取需要的指標，需要容錯 (可能舊版沒有 top10)
    for metric in ['roc_auc', 'pr_auc', 'top5_hit_proba', 'top5_hit_invproba', 'top5_gap', 
                   'top10_hit_proba', 'top10_hit_invproba', 'top10_gap', 'reversal_warning']:
        
        col_base = f"{metric}_baseline"
        col_regime = f"{metric}_regime"
        col_delta = f"delta_{metric}"
        
        if metric in df_base.columns and metric in df_regime.columns:
            df_diff[col_base] = df_base[metric]
            df_diff[col_regime] = df_regime[metric]
            
            # Bool不相減
            if df_base[metric].dtype == bool or metric == 'reversal_warning':
                pass # 僅並列
            else:
                df_diff[col_delta] = df_regime[metric] - df_base[metric]
    
    # 特別標註關注年
    focus_years = [2019, 2021, 2022, 2023]
    df_diff['note'] = df_diff.index.map(lambda y: 'FOCUS' if y in focus_years else '')
    
    # 儲存
    out_csv = os.path.join(args.output_dir, "yearly_diff.csv")
    df_diff.reset_index().to_csv(out_csv, index=False)
    print(f"✅ Yearly Difference 已儲存至 {out_csv}")
    
    # --- 聚合統計 (從 2017 開始) ---
    df_base_17 = df_base.loc[df_base.index >= 2017]
    df_regime_17 = df_regime.loc[df_regime.index >= 2017]
    
    agg = {
        "analysis_period": "2017 - Max",
        "baseline_years_n": len(df_base_17),
        "regime_years_n": len(df_regime_17),
        
        "roc_auc_mean": {
             "baseline": df_base_17['roc_auc'].mean(),
             "regime": df_regime_17['roc_auc'].mean()
        },
        "roc_auc_median": {
             "baseline": df_base_17['roc_auc'].median(),
             "regime": df_regime_17['roc_auc'].median()
        },
        "top5_hit_proba_mean": {
             "baseline": df_base_17['top5_hit_proba'].mean(),
             "regime": df_regime_17['top5_hit_proba'].mean()
        },
        "reversal_year_count": {
             "baseline": int(df_base_17['reversal_warning'].sum()) if 'reversal_warning' in df_base_17 else 0,
             "regime": int(df_regime_17['reversal_warning'].sum()) if 'reversal_warning' in df_regime_17 else 0
        },
        "worst_year_by_roc_auc": {
             "baseline": int(df_base_17['roc_auc'].idxmin()),
             "regime": int(df_regime_17['roc_auc'].idxmin())
        }
    }
    
    if 'top5_gap' in df_base_17.columns and 'top5_gap' in df_regime_17.columns:
        agg['worst_year_by_top5_gap'] = {
             "baseline": int(df_base_17['top5_gap'].idxmin()),
             "regime": int(df_regime_17['top5_gap'].idxmin())
        }
    
    if 'top10_hit_proba' in df_base_17.columns and 'top10_hit_proba' in df_regime_17.columns:
        agg['top10_hit_proba_mean'] = {
             "baseline": df_base_17['top10_hit_proba'].mean(),
             "regime": df_regime_17['top10_hit_proba'].mean()
        }
        
    out_json = os.path.join(args.output_dir, "aggregate_compare.json")
    with open(out_json, "w", encoding='utf-8') as f:
        json.dump(agg, f, indent=4)
    print(f"✅ Aggregate Summary 已儲存至 {out_json}")
    
if __name__ == "__main__":
    main()
