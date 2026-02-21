#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# 將專案根目錄加到 sys.path，以便 import 共用模組
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.train.sklearn_utils import get_positive_proba, apply_class_balancing, get_model, calc_metrics
from src.features.regime_features import compute_regime_features, REGIME_COLS

try:
    from train_us_tech_buy_agent import fetch_all_stock_data, calculate_features, FEATURE_COLS, BENCHMARK
except ImportError:
    print("❌ 找不到 train_us_tech_buy_agent 模組，請確定您在專案根目錄下執行。")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Rolling HGB Walk-Forward 訓練腳本 (Regime Shift 防禦起手式)")
    
    # 目標與輸出
    parser.add_argument('--tickers', nargs='+', default=['GOOGL'], help="目標股票 (預設: GOOGL)")
    parser.add_argument('--output-dir', type=str, default='output_rolling_hgb', help="輸出根目錄")
    
    # 目標定義參數
    parser.add_argument('--target-days', type=int, default=120, help="目標預測天數 (預設: 120)")
    parser.add_argument('--target-return', type=float, default=0.20, help="目標報酬率門檻 (預設: 0.20)")
    
    # 訓練/驗證邊界與時間窗設定
    parser.add_argument('--window-years', type=int, default=5, help="Train window 的長度(以年為單位) (預設: 5)")
    parser.add_argument('--val-years', nargs='+', type=int, 
                        help="欲驗證的年度，若未給定則會自動掃描可用的所有年份。範例: --val-years 2018 2019 2020")
    parser.add_argument('--start-year', type=int, default=None, help="與 --end-year 搭配用於範圍設定")
    parser.add_argument('--end-year', type=int, default=None, help="與 --start-year 搭配用於範圍設定")
    
    # 模型超參數與行為
    parser.add_argument('--model', type=str, default='hgb', choices=['hgb'], help="目前實作專注於 HGB")
    parser.add_argument('--seed', type=int, default=42, help="亂數種子")
    parser.add_argument('--balance-train', type=str, default='none', 
                        choices=['none', 'undersample_50_50', 'class_weight_balanced'],
                        help="Train Set 的平衡策略，Val Set 一律不平衡以反映真實分佈")
    parser.add_argument('--use-regime-features', action='store_true', 
                        help="是否要合併 Benchmark Regime Features (例如 MA200, HV20) 一起丟給模型評估")
    
    # Reversal 判定防呆
    parser.add_argument('--reversal-gap-margin', type=float, default=0.10, 
                        help="定義差距 (Hit Proba - Inv Proba) 小於負多少時發出警告 (預設: 0.10)")
    parser.add_argument('--reversal-use-top10', type=str, default='true', choices=['true', 'false'],
                        help="是否合併納入 Top 10% 樣本進行反向雙重確認 (預設: true)")
    
    # 工具控制
    parser.add_argument('--no-cache', action='store_true', help="強制重新計算特徵不使用快取")
    parser.add_argument('--dry-run', action='store_true', help="僅輸出設定與切分的邊界與樣本數，不進行訓練")
    
    return parser.parse_args()


def get_sanity_reversal_metrics(y_true, y_proba, margin_threshold=0.10, use_top10=True):
    """
    計算並判斷是否有反向（Regime Shift 到連低分群都比高分群準）的問題。
    回傳 top5 與 top10 的精度、gap，以及基於 gap_margin 發布的 warning。
    """
    n_samples = len(y_true)
    k5 = max(1, int(n_samples * 0.05))
    k10 = max(1, int(n_samples * 0.10))
    
    sort_idx_proba = np.argsort(y_proba)[::-1]
    inv_proba = 1.0 - y_proba
    sort_idx_inv = np.argsort(inv_proba)[::-1]
    
    def _calc_hit_rate(sort_idx, k):
        top_k_y_true = y_true.iloc[sort_idx[:k]] if isinstance(y_true, pd.Series) else y_true[sort_idx[:k]]
        return float(np.mean(top_k_y_true))
        
    # Top 5%
    top5_proba_hr = _calc_hit_rate(sort_idx_proba, k5)
    top5_inv_hr = _calc_hit_rate(sort_idx_inv, k5)
    gap5 = top5_proba_hr - top5_inv_hr
    warn5 = (gap5 <= -margin_threshold)
    
    # Top 10%
    top10_proba_hr = _calc_hit_rate(sort_idx_proba, k10)
    top10_inv_hr = _calc_hit_rate(sort_idx_inv, k10)
    gap10 = top10_proba_hr - top10_inv_hr
    warn10 = (gap10 <= -margin_threshold)
    
    final_warning = warn5 or warn10 if use_top10 else warn5

    return {
        'top5_n': k5,
        'top5_hit_proba': top5_proba_hr,
        'top5_hit_invproba': top5_inv_hr,
        'top5_gap': float(gap5),
        'top10_n': k10,
        'top10_hit_proba': top10_proba_hr,
        'top10_hit_invproba': top10_inv_hr,
        'top10_gap': float(gap10),
        'reversal_warning_top5': warn5,
        'reversal_warning_top10': warn10,
        'reversal_warning': final_warning
    }


def prepare_dataset_for_ticker(ticker, target_days, target_return, use_cache):
    """取得資料並根據 Target 動態建立 y 標籤，然後回傳完整清理過的 DataFrame"""
    print(f"\n📦 正在準備 {ticker} 的資料集並計算特徵...")
    all_raw_data = fetch_all_stock_data()
    
    if ticker not in all_raw_data:
        raise ValueError(f"Ticker {ticker} 無法取得數據")
        
    raw_df = all_raw_data[ticker]
    benchmark_df = all_raw_data.get(BENCHMARK)
    
    df_features = calculate_features(raw_df, benchmark_df, ticker=ticker, use_cache=use_cache)
    
    target_col = f'Next_{target_days}d_Max'
    if target_col not in df_features.columns:
        raise ValueError(f"特徵欄位中找不到 {target_col}，請確認 calculate_features 的支援。")
    
    # 建立標籤
    df_dataset = df_features.dropna(subset=FEATURE_COLS + [target_col]).copy()
    df_dataset['y'] = (df_dataset[target_col] >= target_return).astype(int)
    
    # 將 datetime index 變為欄位方便操作並確保其叫做 'date'
    df_dataset = df_dataset.reset_index()
    if 'Date' in df_dataset.columns:
        df_dataset.rename(columns={'Date': 'date'}, inplace=True)
    elif 'index' in df_dataset.columns:
        df_dataset.rename(columns={'index': 'date'}, inplace=True)
        
    df_dataset['date_str'] = pd.to_datetime(df_dataset['date']).dt.strftime('%Y-%m-%d')
    df_dataset['ticker'] = ticker
    
    return df_dataset, benchmark_df


def extract_val_years(df_dataset, args):
    """決定需要跑驗證的年份清單"""
    if args.val_years is not None:
        return sorted([int(y) for y in args.val_years])
    
    df_dataset['year'] = df_dataset['date'].dt.year
    available_years = sorted([int(y) for y in df_dataset['year'].unique()])
    
    # 假設最短 window 為 N，那可以被 evaluate 的第一年至少要大於 min_year + N
    min_year = available_years[0]
    first_viable_val_year = min_year + args.window_years
    
    val_years_candidates = [y for y in available_years if y >= first_viable_val_year]
    
    if args.start_year:
        val_years_candidates = [y for y in val_years_candidates if y >= args.start_year]
    if args.end_year:
        val_years_candidates = [y for y in val_years_candidates if y <= args.end_year]
        
    return val_years_candidates


def run_rolling_training(args):
    """
    執行 Walk-Forward 滾動訓練的核心邏輯。
    可由原本 CLI 的 main() 或外部 wrapper (如 run_rolling_grid.py) 傳入 args 呼叫。
    回傳值：
      master_summary (list of dict): 收錄所有年度、所有 ticker 的執行統計指標。
    """
    if hasattr(args, 'seed') and args.seed is not None:
        np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_hgb_{args.target_days}d_{timestamp}"
    root_output_dir = args.output_dir
    
    if getattr(args, 'dry_run', False):
        pass
    else:
        os.makedirs(root_output_dir, exist_ok=True)
        print(f"📁 建立輸出目錄: {root_output_dir}")
        
    master_summary = []
    use_cache = not getattr(args, 'no_cache', False)
    
    for ticker in args.tickers:
        print(f"\n{'='*80}\n🚀 打開 Walk-Forward 引擎: Ticker = {ticker}\n{'='*80}")
        try:
            df_full, benchmark_df = prepare_dataset_for_ticker(ticker, args.target_days, args.target_return, use_cache)
        except Exception as e:
            print(f"❌ 初始化 {ticker} 資料失敗: {e}")
            continue
            
        # 處理 Regime Features 整合
        active_feature_cols = FEATURE_COLS.copy()
        if getattr(args, 'use_regime_features', False):
            print("🧲 啟動 Regime Features (HGB 自研防禦), 準備結合大盤特徵...")
            df_regime = compute_regime_features(benchmark_df)
            
            # 建立 Date Str 以供 Merge
            if 'date_str' not in df_full.columns:
                df_full['date_str'] = pd.to_datetime(df_full['date']).dt.strftime('%Y-%m-%d')
                
            # 將 df_regime (已經有 date string) Merge 起來
            df_full = pd.merge(df_full, df_regime, left_on='date_str', right_on='date', how='inner', suffixes=('', '_regime'))
            # 重新 Dropna 保障新特徵沒有洞 (大盤最前面會有歷史長度的洞)
            df_full = df_full.dropna(subset=REGIME_COLS).copy()
            active_feature_cols += REGIME_COLS
            print(f"   => 合併完成, X 變數從 {len(FEATURE_COLS)} 增長為 {len(active_feature_cols)} 個。")
            
        val_years = extract_val_years(df_full, args)
        if not val_years:
            print(f"⚠️ {ticker} 找不到符合 window size 要求的可用年度資料，跳過。")
            continue
            
        print(f"📅 預計執行 Rolling 的年度 (Val Years): {val_years}")
        
        for val_y in val_years:
            print(f"\n--- ⏳ Epoch: 驗證年度 {val_y} ---")
            
            # 定義 Requested range
            req_train_start = f"{val_y - args.window_years}-01-01"
            req_train_end = f"{val_y - 1}-12-31"
            req_val_start = f"{val_y}-01-01"
            req_val_end = f"{val_y}-12-31"
            
            # 實際取得切割
            df_train_raw = df_full[(df_full['date'] >= req_train_start) & (df_full['date'] <= req_train_end)].copy()
            df_val = df_full[(df_full['date'] >= req_val_start) & (df_full['date'] <= req_val_end)].copy()
            
            # 檢查筆數
            if len(df_train_raw) == 0:
                print(f"  ⚠️ Train Set 筆數為 0，跳過此年度。")
                continue
            if len(df_val) == 0:
                print(f"  ⚠️ Val Set 筆數為 0，跳過此年度。")
                continue
                
            # 取得 Actual Boundary (dropna 之後的邊界，確保紀錄不失真)
            actual_tr_min = df_train_raw['date'].min().strftime('%Y-%m-%d')
            actual_tr_max = df_train_raw['date'].max().strftime('%Y-%m-%d')
            actual_va_min = df_val['date'].min().strftime('%Y-%m-%d')
            actual_va_max = df_val['date'].max().strftime('%Y-%m-%d')
            
            tr_n, va_n = len(df_train_raw), len(df_val)
            tr_pos_r = df_train_raw['y'].mean()
            va_pos_r = df_val['y'].mean()
            
            print(f"  [Requested Train] {req_train_start} ~ {req_train_end}")
            print(f"  [Actual    Train] {actual_tr_min} ~ {actual_tr_max} | N: {tr_n} | Pos%: {tr_pos_r*100:.2f}%")
            print(f"  [Requested Val  ] {req_val_start} ~ {req_val_end}")
            print(f"  [Actual    Val  ] {actual_va_min} ~ {actual_va_max} | N: {va_n} | Pos%: {va_pos_r*100:.2f}%")
            
            # 如果年份不足（真實有資料的期間跟要求差太多，例如 req train 要 5 年但實際資料只有半年）
            # 不一定要 skip 但提醒我們可能不準確
            date_span_days = (df_train_raw['date'].max() - df_train_raw['date'].min()).days
            if date_span_days < (args.window_years * 365) * 0.7:
                 print(f"  ⚠️ 注意: 實際 Train Window 覆蓋天數 ({date_span_days}) 遠小於設定的 {args.window_years} 年。")
            
            if args.dry_run:
                continue
                
            y_train_raw = df_train_raw['y']
            
            # 在 Train Set 實施 class balancing (Val Set 絕對不可動)
            df_train_bal = apply_class_balancing(df_train_raw, args.balance_train, args.seed)
            X_train = df_train_bal[active_feature_cols]
            y_train = df_train_bal['y']
            
            X_val = df_val[active_feature_cols]
            y_val = df_val['y']
            
            # --- 訓練與推論 ---
            model, use_sample_weight = get_model(args.model, args.balance_train, args.seed)
            print("  [Train] 正在訓練模型 (Random State 固定)...")
            
            if use_sample_weight:
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_train)
                weight_arr = compute_class_weight('balanced', classes=classes, y=y_train)
                weight_dict = dict(zip(classes, weight_arr))
                sw = np.array([weight_dict[c] for c in y_train])
                model.fit(X_train, y_train, sample_weight=sw)
                used_balancing_method = 'sample_weight (simulated class_weight)'
            else:
                model.fit(X_train, y_train)
                used_balancing_method = 'built-in class_weight / None'
                
            y_proba_val, clz_list, pos_idx = get_positive_proba(model, X_val, positive_label=1)
            y_pred_val = model.predict(X_val)
            
            # --- 指標計算與 Sanity Check ---
            metrics = calc_metrics(y_val, y_proba_val, y_pred_val, prefix="Yearly")
            
            # Mean Proba
            mask_pos = (y_val == 1)
            mask_neg = (y_val == 0)
            mean_pos_proba = y_proba_val[mask_pos].mean() if mask_pos.sum() > 0 else 0.0
            mean_neg_proba = y_proba_val[mask_neg].mean() if mask_neg.sum() > 0 else 0.0
            
            # Top-K Reversal Check
            rev_stats = get_sanity_reversal_metrics(
                y_val, y_proba_val, 
                margin_threshold=args.reversal_gap_margin, 
                use_top10=(args.reversal_use_top10 == 'true')
            )
            
            # Combo reversal warning (roc < 0.5 is also strictly bad)
            is_roc_fail = (metrics.get('ROC-AUC') is not None) and (metrics['ROC-AUC'] < 0.5)
            final_reversal_warning = rev_stats['reversal_warning'] or is_roc_fail
            
            print(f"  [Metric] {val_y} ROC-AUC: {metrics.get('ROC-AUC', 'N/A')}")
            print(f"  [Metric] Top5% Hit Rate by Proba: {rev_stats['top5_hit_proba']*100:.1f}%")
            print(f"  [Metric] Top5% Hit Rate by Inv. : {rev_stats['top5_hit_invproba']*100:.1f}%")
            print(f"  [Metric] Top5% Gap              : {rev_stats['top5_gap']*100:.1f}%")
            if args.reversal_use_top10 == 'true':
                 print(f"  [Metric] Top10% Hit Rate by Pro: {rev_stats['top10_hit_proba']*100:.1f}% | Gap: {rev_stats['top10_gap']*100:.1f}%")
            
            if final_reversal_warning:
                print(f"  🚨⚠️ [REVERSAL OCURRED IN {val_y}] 觸發反向警告機制！")
                
            metrics['Sanity Check'] = {
                'reversal_rule_version': 'v2',
                'reversal_gap_margin': args.reversal_gap_margin,
                'reversal_check_top10': args.reversal_use_top10,
                'mean_pos_proba': float(mean_pos_proba),
                'mean_neg_proba': float(mean_neg_proba),
                'reversal_warning_final': final_reversal_warning,
                **rev_stats
            }
            
            # --- 建立單份結果的 Params dict ---
            epoch_params = {
                'ticker': ticker,
                'run_name': run_name,
                'target_definition': f"Next_{args.target_days}d_Max >= {args.target_return}",
                'val_year': val_y,
                'requested_train_range': [req_train_start, req_train_end],
                'actual_train_range': [actual_tr_min, actual_tr_max],
                'requested_val_range': [req_val_start, req_val_end],
                'actual_val_range': [actual_va_min, actual_va_max],
                'train_samples': tr_n,
                'train_pos_rate': float(tr_pos_r),
                'val_samples': va_n,
                'val_pos_rate': float(va_pos_r),
                'window_years': args.window_years,
                'seed': args.seed,
                'model_class': type(model).__name__,
                'model_params': model.get_params(),
                'balance_train': args.balance_train,
                'used_balancing_method': used_balancing_method,
                'use_regime_features': getattr(args, 'use_regime_features', False),
                'regime_cols': REGIME_COLS if getattr(args, 'use_regime_features', False) else [],
                'reversal_rule_version': 'v2',
                'reversal_gap_margin': args.reversal_gap_margin,
                'reversal_check_top10': args.reversal_use_top10
            }
            
            # --- 收集 Regime Summary (當年市場狀況) ---
            regime_dict = {}
            if getattr(args, 'use_regime_features', False):
                # 統計該年度 (Val Set) 中，這些大盤特徵的表現概況，用來關聯是否造成模型崩壞
                regime_dict = {
                    'regime_above_ma200_rate': df_val['REGIME_BM_ABOVE_MA200'].mean(),
                    'regime_hv20_mean': df_val['REGIME_BM_HV20'].mean(),
                    'regime_hv20_p50': df_val['REGIME_BM_HV20'].median(),
                    'regime_hv20_p90': df_val['REGIME_BM_HV20'].quantile(0.90),
                    'regime_hv20_pctl_mean': df_val['REGIME_BM_HV20_PCTL'].mean(),
                    'regime_hv20_pctl_p50': df_val['REGIME_BM_HV20_PCTL'].median(),
                    'regime_ret_120_mean': df_val['REGIME_BM_RET_120'].mean(),
                    'regime_ret_60_mean': df_val['REGIME_BM_RET_60'].mean(),
                }
            
            # 準備 Master 這一行的 Data
            row = {
                'ticker': ticker,
                'val_year': val_y,
                'window': args.window_years,
                'train_n': tr_n,
                'val_n': va_n,
                'val_pos_rate': va_pos_r,
                'roc_auc': metrics.get('ROC-AUC', None),
                'pr_auc': metrics.get('PR-AUC', None),
                'precision@5%': metrics.get('Precision@5%', None),
                'precision@10%': metrics.get('Precision@10%', None),
                'th0.5_precision': metrics.get('Threshold Sweep', {}).get('Threshold=0.5', {}).get('Precision', None),
                'th0.5_recall': metrics.get('Threshold Sweep', {}).get('Threshold=0.5', {}).get('Recall', None),
                'th0.5_f1': metrics.get('Threshold Sweep', {}).get('Threshold=0.5', {}).get('F1', None),
                'top5_n': rev_stats['top5_n'],
                'top5_hit_proba': rev_stats['top5_hit_proba'],
                'top5_hit_invproba': rev_stats['top5_hit_invproba'],
                'top5_gap': rev_stats['top5_gap'],
                'top10_n': rev_stats['top10_n'],
                'top10_hit_proba': rev_stats['top10_hit_proba'],
                'top10_hit_invproba': rev_stats['top10_hit_invproba'],
                'top10_gap': rev_stats['top10_gap'],
                'mean_pos_proba': mean_pos_proba,
                'mean_neg_proba': mean_neg_proba,
                'reversal_warning_top5': rev_stats['reversal_warning_top5'],
                'reversal_warning_top10': rev_stats['reversal_warning_top10'],
                'reversal_warning': final_reversal_warning
            }
            # 如果有啟動 Regime，就把那些統計指標塞入 Master
            row.update(regime_dict)
            
            master_summary.append(row)
            
            # --- Output 到年份獨立資料夾 ---
            year_dir = os.path.join(root_output_dir, f"{ticker}_{val_y}")
            os.makedirs(year_dir, exist_ok=True)
            
            with open(os.path.join(year_dir, "params.json"), "w", encoding="utf-8") as f:
                json.dump(epoch_params, f, indent=4, ensure_ascii=False)
                
            with open(os.path.join(year_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False)
                
            # Prediction CSV
            df_out = df_val[['date', 'ticker']].copy()
            df_out['y_true'] = y_val.values
            df_out['y_proba'] = y_proba_val
            df_out['y_pred'] = y_pred_val
            df_out.to_csv(os.path.join(year_dir, "val_predictions.csv"), index=False)
            
    # 全局總結與寫檔
    if not getattr(args, 'dry_run', False) and master_summary:
        print(f"\n{'='*80}\n✅ 所有 Rolling Epochs 測試完畢，整理結果...")
        df_summary = pd.DataFrame(master_summary)
        csv_path = os.path.join(root_output_dir, "rolling_summary.csv")
        json_path = os.path.join(root_output_dir, "rolling_summary.json")
        
        df_summary.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(master_summary, f, indent=4, ensure_ascii=False)
            
        print(f"📊 年度總結報告已寫出：{csv_path}")
        
    return master_summary


def main():
    """原本作為獨立 CLI 時的進入點"""
    args = parse_args()
    
    # 單純執行腳本時，自動補上目標目錄的一層 timestamp 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{args.model}_{args.target_days}d_{timestamp}"
    args.output_dir = os.path.join(args.output_dir, run_name)
    
    master_summary = run_rolling_training(args)
    if not args.dry_run and master_summary:
        df_summary = pd.DataFrame(master_summary)
        print(df_summary[['val_year', 'val_n', 'val_pos_rate', 'roc_auc', 'precision@5%', 
                          'top5_hit_invproba', 'reversal_warning']].to_string(index=False))


if __name__ == "__main__":
    main()
