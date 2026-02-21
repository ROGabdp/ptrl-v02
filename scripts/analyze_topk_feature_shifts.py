#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Feature Regime Shift Diagnostics
================================================================================
針對特定的 ticker 與推論結果 (val_predictions.csv)，自動切出每年的
「模型最有信心會漲 (Top K% by p)」與「模型最沒信心會漲 (Top K% by 1-p)」子集。
接著，計算這些極端樣本群各自在特徵分佈上的統計數據，並且比對差異 (Feature Shift)。
這有助於診斷為何某幾年模型的預測方向發生了倒反 (ROC-AUC < 0.5)。

使用範例:
python scripts/analyze_topk_feature_shifts.py --val-predictions output_sklearn/run_hgb_120d_20260221_083838/val_predictions.csv --ticker GOOGL --topk-pct 5 --output-dir output_analysis
================================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import warnings

# 將專案根目錄加到 sys.path，以便 import 共用模組
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from train_us_tech_buy_agent import fetch_all_stock_data, calculate_features, FEATURE_COLS, BENCHMARK
except ImportError as e:
    print(f"❌ 無法從 train_us_tech_buy_agent.py 載入共用邏輯: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Feature Shifts for False Positives / Regime Changes")
    parser.add_argument("--val-predictions", type=str, required=True, 
                        help="必填：包含推論結果的 val_predictions.csv 檔案路徑")
    parser.add_argument("--ticker", type=str, required=True, 
                        help="必填：要診斷的股票代碼 (e.g. GOOGL)")
    parser.add_argument("--topk-pct", type=float, default=5.0, 
                        help="Top K 百分比閾值，預設 5 (代表 5%)")
    parser.add_argument("--output-dir", type=str, default="output_analysis", 
                        help="分析報告的輸出目錄")
    parser.add_argument("--years", nargs="*", type=int, 
                        help="欲分析的年份清單 (預設: 自動偵測 csv 包含的所有年份)")
    parser.add_argument("--no-cache", action="store_true", 
                        help="關閉特徵快取讀取，強制重新運算")
    parser.add_argument("--seed", type=int, default=42, 
                        help="排序發生平手時的隨機種子")
    return parser.parse_args()


def compute_feature_stats(df_subset, features):
    """計算指定特徵在給定子集內的統計摘要"""
    if len(df_subset) == 0:
        return pd.DataFrame(columns=['feature', 'mean', 'median', 'std', 'p10', 'p90', 'n'])
        
    stats = []
    for f in features:
        if f not in df_subset.columns:
            continue
        vals = df_subset[f].dropna()
        n = len(vals)
        if n == 0:
            stats.append({'feature': f, 'mean': np.nan, 'median': np.nan, 'std': np.nan, 'p10': np.nan, 'p90': np.nan, 'n': 0})
        else:
            stats.append({
                'feature': f,
                'mean': vals.mean(),
                'median': vals.median(),
                'std': vals.std(),
                'p10': vals.quantile(0.10),
                'p90': vals.quantile(0.90),
                'n': n
            })
    return pd.DataFrame(stats)


def main():
    args = parse_args()
    
    # 防止 pandas print 被省略
    pd.set_option('display.max_rows', 100)
    
    print("============================================================")
    print("🔍 Feature Regime Shift Analytics")
    print("============================================================")
    print(f"  Ticker       : {args.ticker}")
    print(f"  Predictions  : {args.val_predictions}")
    print(f"  Top K%       : {args.topk_pct}%")
    print(f"  Output Dir   : {args.output_dir}")
    print("============================================================")
    
    # 1. 讀取 Predictions CSV
    if not os.path.exists(args.val_predictions):
        print(f"❌ 找不到預測檔: {args.val_predictions}")
        sys.exit(1)
        
    df_pred_all = pd.read_csv(args.val_predictions)
    
    # 驗證必備欄位
    req_cols = ['date', 'ticker', 'y_true', 'y_proba']
    if not all(c in df_pred_all.columns for c in req_cols):
        print(f"❌ val_predictions 缺少必備欄位，請確認包含: {req_cols}")
        print(f"目前欄位: {df_pred_all.columns.tolist()}")
        sys.exit(1)
        
    # 2. 篩選與日期轉換
    df_pred = df_pred_all[df_pred_all['ticker'] == args.ticker].copy()
    if len(df_pred) == 0:
        print(f"❌ 在預測檔中找不到 Ticker: {args.ticker} 的相關紀錄。")
        sys.exit(1)
        
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    df_pred['year'] = df_pred['date'].dt.year
    df_pred['inv_proba'] = 1.0 - df_pred['y_proba'] # 反向分數
    
    years_to_analyze = args.years if args.years else sorted(df_pred['year'].unique())
    print(f"👉 涵蓋的目標年份: {years_to_analyze}")
    
    # 3. 獲取並建構 Feature DataFrame
    print(f"📥 正在產生 {args.ticker} 的特徵歷史序列...")
    all_raw_data = fetch_all_stock_data()
    benchmark_df = all_raw_data.get(BENCHMARK)
    if benchmark_df is None:
        print(f"❌ 無法載入基準指數 {BENCHMARK}。")
        sys.exit(1)
        
    raw_df = all_raw_data.get(args.ticker)
    if raw_df is None:
        print(f"❌ 無法載入原始標的資料: {args.ticker}。")
        sys.exit(1)
        
    df_features = calculate_features(raw_df, benchmark_df, ticker=args.ticker, use_cache=not args.no_cache)
    
    # 讓 Feature DF 的 index 轉成 regular column 取名 date 並調整格式，便於 merge
    df_features = df_features.reset_index()
    df_features.rename(columns={'Date': 'date'}, inplace=True)
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    # 4. Inner Join
    # 檢查是否會有大量漏切的情形
    join_test = pd.merge(df_pred, df_features[['date'] + FEATURE_COLS], on='date', how='left')
    missing_mask = join_test[FEATURE_COLS[0]].isna()
    if missing_mask.any():
        missing_count = missing_mask.sum()
        missing_pct = missing_count / len(join_test) * 100
        print(f"⚠️ 警告: Join 之後發現有 {missing_count} 筆 ({missing_pct:.2f}%) 的特徵為空！")
        print("  可能原因是 calculate_features 在最新推論資料上的 NaN 被濾掉，或者時間完全脫鉤。")
        print("  前 5 筆遺失日期:")
        print(join_test[missing_mask].head(5)[['date', 'y_proba']])
    
    # 執行乾淨的 Inner Join (去除 feature 空值)
    df_merged = pd.merge(df_pred, df_features[['date'] + FEATURE_COLS], on='date', how='inner')
    print(f"✅ 對齊完成，共有 {len(df_merged)} 筆可用特徵樣本。")
    
    # 5. 建立輸出根目錄
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = f"shift_{args.ticker}_top{int(args.topk_pct)}_{run_ts}"
    out_dir = os.path.join(args.output_dir, sub_dir)
    per_year_dir = os.path.join(out_dir, "per_year")
    os.makedirs(per_year_dir, exist_ok=True)
    
    # 隨機數確保排序一致性
    rng = np.random.default_rng(args.seed)
    
    # 整體彙總
    summary_data = {
        "ticker": args.ticker,
        "predictions_source": args.val_predictions,
        "topk_pct": args.topk_pct,
        "total_merged_samples": len(df_merged),
        "yearly_performance": {},
        "top_flipped_features_per_year": {}
    }
    
    # =========================================================
    # 6. 開始分年掃描
    # =========================================================
    for yr in years_to_analyze:
        df_yr = df_merged[df_merged['year'] == yr].copy()
        
        if len(df_yr) < 30:
            print(f"⚠️ [Year {yr}] 樣本數只有 {len(df_yr)} 不足 30 筆，統計可能沒有代表性。")
            if len(df_yr) == 0:
                continue
        
        # 決定 K 筆數
        k_sz = max(1, int(len(df_yr) * args.topk_pct / 100.0))
        
        # 為了避免機率完全相同導致次序亂跳，加入微小的 noise 來斷 tie
        noise = rng.uniform(0, 1e-9, size=len(df_yr))
        df_yr['tie_breaker_p'] = df_yr['y_proba'] + noise
        df_yr['tie_breaker_inv_p'] = df_yr['inv_proba'] + noise
        
        # A組: 分數最高 Top K%
        df_top_A = df_yr.nlargest(k_sz, 'tie_breaker_p')
        # B組: 反向分數最高 (最不看好) Top K%
        df_top_B = df_yr.nlargest(k_sz, 'tie_breaker_inv_p')
        
        # 計算命中率 (Precision@K)
        prec_A = df_top_A['y_true'].mean()
        prec_B = df_top_B['y_true'].mean()
        
        summary_data["yearly_performance"][str(yr)] = {
            "total_samples": int(len(df_yr)),
            "group_size_k": int(k_sz),
            "baseline_pos_rate": float(df_yr['y_true'].mean()),
            "GroupA_HighConf_PrecAtK": float(prec_A),
            "GroupB_LowConf_PrecAtK": float(prec_B),
            "warning_reversal": bool(prec_B > prec_A) # 低分群反而更會漲！
        }
        
        # 產出 A / B 的統計特徵表
        stats_A = compute_feature_stats(df_top_A, FEATURE_COLS)
        stats_B = compute_feature_stats(df_top_B, FEATURE_COLS)
        
        stats_A.to_csv(os.path.join(per_year_dir, f"{yr}_topk_by_proba_stats.csv"), index=False)
        stats_B.to_csv(os.path.join(per_year_dir, f"{yr}_topk_by_invproba_stats.csv"), index=False)
        
        # 產出該年 top-k date 清單，便於視覺化回測
        pd.concat([
            df_top_A[['date', 'y_true', 'y_proba']].assign(Group='HighConf_A'),
            df_top_B[['date', 'y_true', 'y_proba']].assign(Group='LowConf_B')
        ]).to_csv(os.path.join(per_year_dir, f"{yr}_topk_dates.csv"), index=False)
        
        
        # =========================================================
        # 計算特徵差異度量
        # =========================================================
        if not stats_A.empty and not stats_B.empty:
            diff_merged = pd.merge(
                stats_A[['feature', 'median', 'mean', 'std']], 
                stats_B[['feature', 'median', 'mean', 'std']], 
                on='feature', suffixes=('_A', '_B')
            )
            
            diff_merged['median_diff'] = diff_merged['median_A'] - diff_merged['median_B']
            
            # Pooled STD Approximation for Standardized Diff
            # (n1-1)*s1^2 + (n2-1)*s2^2 / (n1+n2-2)
            var_A = diff_merged['std_A'] ** 2
            var_B = diff_merged['std_B'] ** 2
            pooled_std = np.sqrt((var_A + var_B) / 2.0) + 1e-9  # 防止除以 0
            
            diff_merged['standardized_diff'] = (diff_merged['mean_A'] - diff_merged['mean_B']) / pooled_std
            
            # 排序：以標準化差異的絕對值排序，尋找「A組與B組看法截然不同」的顛倒特徵
            diff_merged['abs_std_diff'] = diff_merged['standardized_diff'].abs()
            diff_merged = diff_merged.sort_values(by='abs_std_diff', ascending=False)
            diff_merged = diff_merged.drop(columns=['abs_std_diff'])
            
            diff_merged.to_csv(os.path.join(per_year_dir, f"{yr}_feature_diff_A_vs_B.csv"), index=False)
            
            # 把前 20 個潛在翻轉特徵名稱寫入 Summary
            top_N = min(20, len(diff_merged))
            top_features_yr = diff_merged.head(top_N)[['feature', 'standardized_diff', 'median_diff']].to_dict(orient='records')
            summary_data["top_flipped_features_per_year"][str(yr)] = top_features_yr
            
            # 印出簡單報表
            print(f"\n[{yr}] N={len(df_yr)}, k={k_sz} ({args.topk_pct}%)")
            print(f"   Baseline Pos% : {df_yr['y_true'].mean()*100:5.2f}%")
            print(f"   Group A Prec@k: {prec_A*100:5.2f}%  (Top Proba)")
            print(f"   Group B Prec@k: {prec_B*100:5.2f}%  (Low Proba)")
            if prec_B > prec_A:
                print("   ⚠️ 發現反向預測 (分數越低越容易中)！")
                
            print("   ► Top 3 差異最大的特徵:")
            for row in diff_merged.head(3).itertuples():
                print(f"      - {row.feature:<20} | StdDiff: {row.standardized_diff:>7.3f} | MedianDiff: {row.median_diff:>7.3f}")

    # =========================================================
    # 7. 匯出整體 Summary
    # =========================================================
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
        
    print("\n============================================================")
    print(f"✅ 診斷報告產生完畢，請查閱: {out_dir}")
    print("============================================================")

if __name__ == "__main__":
    main()
