#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
US Tech Stock - Daily Train & Predictor (Single Ticker Approach)
================================================================================
每天依據最新抓取的市場資料為每個股票「獨立」建構滾動特寫模型，並依據該標的今日分數
在其最近 252 交易日（歷史分位數）間的相對強度，與今日大盤風控指標結合，進行評級與佈局判斷。
================================================================================
"""

import os
import sys
import argparse
import joblib
import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from scipy.stats import percentileofscore

warnings.filterwarnings('ignore', category=UserWarning)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.train.sklearn_utils import get_positive_proba
from src.features.regime_features import compute_regime_features, REGIME_COLS

try:
    from train_us_tech_buy_agent import fetch_all_stock_data, calculate_features, FEATURE_COLS, BENCHMARK
except ImportError as e:
    print(f"❌ 無法從 train_us_tech_buy_agent.py 載入共用邏輯: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict 'Today's Buy Decision' single ticker approach")
    
    # 執行與檔案模式
    parser.add_argument("--model-path", type=str, default=None, 
                        help="選填：提供已預先訓練好之模型檔案(路徑需含 {ticker} 佔位符)。若無，則進行當日獨立訓練。")
    parser.add_argument("--output-dir", type=str, default="output_daily", help="當日模型與結果儲存根目錄")
    
    # 推薦與目標參數
    parser.add_argument("--tickers", nargs="+", 
                        default=["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "TSLA", "NFLX", "PLTR"],
                        help="要預測的目標股票列表")
    parser.add_argument("--target-days", type=int, default=120, help="預測未來的交易天數 (預設 120)")
    parser.add_argument("--target-return", type=float, default=0.20, help="目標最高價漲幅門檻 (預設 0.20)")
    
    # 模型推斷設定
    parser.add_argument("--window-years", type=int, default=3, help="訓練窗格大小 (預設 3 年)")
    parser.add_argument("--lookback-years", type=int, default=8, help="下載資料所需推前年數供 MA/特徵暖機用 (預設 8 年)")
    parser.add_argument("--pct-lookback-days", type=int, default=252, help="分位數計算推斷的樣本天數 (預設 252 交易日)")
    parser.add_argument("--topk-threshold-pct", type=float, default=0.90, help="正常市場下作為買入標準之歷史分位數 (預設 0.90 -> Top 10%)")
    parser.add_argument("--risk-threshold-pct", type=float, default=0.95, help="高風險市場下嚴格化之分位數門檻 (預設 0.95 -> Top 5%)")
    parser.add_argument("--use-regime-features", type=str, default="true", choices=["true", "false"])
    
    parser.add_argument("--force-retrain", action="store_true", help="強制重新訓練新模型忽略當日快取")
    parser.add_argument("--no-cache", action="store_true", help="強制重新擷取/計算盤後特徵不讀取歷史暫存檔")
    return parser.parse_args()


def load_model_and_predict(model_path, model_type, X_input):
    if model_type == "ppo":
        from stable_baselines3 import PPO
        import torch
        model = PPO.load(model_path, device="cpu")
        with torch.no_grad():
            obs_tensor = torch.tensor(X_input, dtype=torch.float32, device="cpu")
            distribution = model.policy.get_distribution(obs_tensor)
            proba = distribution.distribution.probs[:, 1].cpu().numpy()[0]
        return proba
        
    elif model_type in ["sklearn", "daily"]:
        model = joblib.load(model_path)
        y_proba, _, _ = get_positive_proba(model, pd.DataFrame(X_input), positive_label=1)
        # Handle single vs batch predictions
        if len(y_proba) == 1:
            return float(y_proba[0])
        else:
            return y_proba
    else:
        raise ValueError(f"不認得的模型格式: {model_type}")

def evaluate_regime_risk(benchmark_df):
    """判斷市場是否處於高風險狀態 (Proxy 風控機制)"""
    regime_df = compute_regime_features(benchmark_df)
    if len(regime_df) == 0: return "NORMAL", False
    
    latest_regime = regime_df.iloc[-1]
    
    bm_above_ma200 = latest_regime.get('REGIME_BM_ABOVE_MA200', 1.0)
    hv20_pctl = latest_regime.get('REGIME_BM_HV20_PCTL', 0.0)
    ret_120 = latest_regime.get('REGIME_BM_RET_120', 0.0)
    
    is_risk = False
    reasons = []
    
    if bm_above_ma200 == 0 and hv20_pctl > 0.8:
        is_risk = True
        reasons.append("MA200 Below & HV20_Pctl > 0.8")
        
    if ret_120 < 0 and hv20_pctl > 0.8:
        is_risk = True
        reasons.append("120d Return < 0 & HV20_Pctl > 0.8")
        
    if is_risk:
        return f"HIGH_RISK ({'|'.join(reasons)})", True
    return "NORMAL", False


def main():
    args = parse_args()
    
    print("====================================================================")
    print("🚀 US Tech Stock - Daily Train & Predictor (Single Ticker Rank)")
    print("====================================================================")
    
    today_str = datetime.today().strftime("%Y%m%d")
    output_daily_dir = os.path.join(args.output_dir, today_str)
    os.makedirs(output_daily_dir, exist_ok=True)
    
    # 1. 抓取資料
    start_date = (datetime.today() - timedelta(days=args.lookback_years*365)).strftime("%Y-%m-%d")
    print(f"📥 正在從 Yahoo Finance 獲取/更新最新股價 (自 {start_date} 起)...")
    try:
         all_data = fetch_all_stock_data(start_date=start_date)
         benchmark_df = all_data.get(BENCHMARK)
         if benchmark_df is None: raise ValueError(f"無法載入基準 {BENCHMARK}")
    except Exception as e:
         print(f"❌ {e}")
         sys.exit(1)
         
    use_regime = (args.use_regime_features == "true")
    active_cols = FEATURE_COLS + (REGIME_COLS if use_regime else [])
    target_col = f"Next_{args.target_days}d_Max"
    
    # 2. Proxy Risk 計算
    risk_status_text, is_high_risk = evaluate_regime_risk(benchmark_df)
    regime_df = compute_regime_features(benchmark_df) if use_regime else None
    
    # 決定今天用的 Threshold
    active_threshold_pct = args.risk_threshold_pct if is_high_risk else args.topk_threshold_pct
    
    print(f"  [風控狀態] {risk_status_text} | 預計使用門檻: 分位數 >= {active_threshold_pct*100:g}%")
    
    results = [] # output row dictionaries
    run_summary = {
        "run_date": today_str,
        "target_days": args.target_days,
        "target_return": args.target_return,
        "window_years": args.window_years,
        "tickers": args.tickers,
        "global_risk_state": "High Risk" if is_high_risk else "Normal",
        "ticker_summaries": {}
    }
    
    # 3. 逐檔開始 Train & Predict 流程
    for tk in args.tickers:
        raw_df = all_data.get(tk)
        if raw_df is None or len(raw_df) == 0:
            print(f"⚠️ {tk}: 無法取得足夠報價，跳過。")
            continue
            
        print(f"\n⚙️ 處理股票 [{tk}] ...")
        
        # A) 特徵裝配
        feat_df = calculate_features(raw_df, benchmark_df, ticker=tk, use_cache=not args.no_cache)
        if target_col not in feat_df.columns:
            print(f"  {tk}: 未找到指定的 Target Col {target_col}，跳過。")
            continue
            
        feat_df = feat_df.reset_index()
        if 'Date' in feat_df.columns: feat_df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'index' in feat_df.columns: feat_df.rename(columns={'index': 'date'}, inplace=True)
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        
        if use_regime:
            feat_df['date_str'] = feat_df['date'].dt.strftime('%Y-%m-%d')
            feat_df = pd.merge(feat_df, regime_df, left_on='date_str', right_on='date', how='inner', suffixes=('', '_regime'))
        
        # B) 模型路徑與訓練
        ticker_model_dir = os.path.join(output_daily_dir, tk)
        os.makedirs(ticker_model_dir, exist_ok=True)
        
        is_legacy_mode = (args.model_path is not None)
        if is_legacy_mode:
             model_path = args.model_path.replace("{ticker}", tk)
             model_type = "ppo" if ".zip" in model_path else "sklearn"
        else:
             model_path = os.path.join(ticker_model_dir, "model.joblib")
             model_type = "daily"
             
        # C) 資料切片 [Today - 3y, Today] (給 Daily Train 用，Legacy 也要算出實際範圍以供記錄)
        train_end = feat_df['date'].max()
        train_start = train_end - pd.DateOffset(years=args.window_years)
        
        mask_train = (feat_df['date'] >= train_start) & (feat_df['date'] <= train_end)
        train_slice = feat_df[mask_train].dropna(subset=active_cols + [target_col]).copy()
        
        n_train = len(train_slice)
        if n_train == 0:
             print(f"  {tk}: 資料因 NA 或過短而清空，無法建立模型預測。")
             run_summary["ticker_summaries"][tk] = {"status": "Error: Insufficient Data"}
             continue
             
        train_slice['y'] = (train_slice[target_col] >= args.target_return).astype(int)
        pos_rate = train_slice['y'].mean()
        
        if not is_legacy_mode:
            if os.path.exists(model_path) and not args.force_retrain:
                print(f"  [{tk}] 模型已快取，省略訓練。")
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                model = HistGradientBoostingClassifier(random_state=42)
                model.fit(train_slice[active_cols], train_slice['y'])
                joblib.dump(model, model_path)
                print(f"  [{tk}] 單檔模型訓練完畢 (Train size: {n_train}, Pos Rate: {pos_rate*100:.2f}%)")
        
        # D) 推論今天 p_today
        # 今天是包含在 feat_df 最後一筆 (因為 target_col NaNs 也被算進 calculate_features)
        # 必須手動取 feat_df 最後一筆並確保 active_cols 無 NaN
        latest_feat = feat_df.iloc[-1:].copy()
        latest_date_str = latest_feat['date'].iloc[0].strftime("%Y-%m-%d")
        
        if latest_feat[active_cols].isnull().any().any():
             print(f"  [{tk}] 最新一筆資料({latest_date_str})特徵存在空值，退出。")
             # 或許部分 regime 還未更新所以最後一天空值，安全起見我們取 feat_df.dropna(subset=active_cols).iloc[-1:] 
             # 但這最符合使用者所認知的「今天(或最新一筆有效日)」之條件
             latest_feat = feat_df.dropna(subset=active_cols).iloc[-1:]
             if len(latest_feat) == 0: continue
             latest_date_str = latest_feat['date'].iloc[0].strftime("%Y-%m-%d")
             
        X_today = pd.DataFrame(latest_feat[active_cols])
        if model_type == 'ppo': X_today = X_today.values.astype(np.float32)
        p_today = load_model_and_predict(model_path, model_type, X_today)
        
        # E) 計算歷史分位數 p_history (pct_lookback_days)
        # 取有效特徵的歷史資料
        valid_history_df = feat_df.dropna(subset=active_cols).copy()
        # 切過去 pct_lookback_days 筆 (不含今天自己，或者含也可以，不影響大局)
        base_lookback_df = valid_history_df.iloc[-(args.pct_lookback_days+1):-1]
        
        if len(base_lookback_df) < (args.pct_lookback_days // 2): 
             # Fallback
             pct_rank_today = np.nan
             print(f"  [{tk}] 歷史可用紀錄 {len(base_lookback_df)} 天過短，無法計算可靠的 Percentile (>50% required)。")
        else:
             X_hist = pd.DataFrame(base_lookback_df[active_cols])
             if model_type == 'ppo': X_hist = X_hist.values.astype(np.float32)
             p_hist_array = load_model_and_predict(model_path, model_type, X_hist)
             # scipy percentileofscore [0, 100]
             pct_rank_today = percentileofscore(p_hist_array, p_today) / 100.0
             
        # F) 決策判斷
        action = "WATCH"
        position_scale = 0.0
        
        if np.isnan(pct_rank_today):
             action = "WATCH_INSUFFICIENT_DATA"
        elif pct_rank_today >= active_threshold_pct:
             if is_high_risk:
                 action = "BUY_REDUCED"
                 position_scale = 0.5
             else:
                 action = "BUY"
                 position_scale = 1.0
        else:
             if is_high_risk:
                 action = "SKIP_RISK"
        
        print(f"  [{tk}] P({args.target_days}): {p_today*100:.2f}% | PctRank(252d): {pct_rank_today*100 if not np.isnan(pct_rank_today) else np.nan:.1f}% => {action}")
        
        # 紀錄檔保存
        results.append({
             "date": latest_date_str,
             "ticker": tk,
             "p_today": float(p_today),
             "pct_rank_today": float(pct_rank_today),
             "action": action,
             "position_scale": float(position_scale),
             "is_high_risk": is_high_risk,
             "threshold_pct_used": float(active_threshold_pct),
             "train_start_requested": train_start.strftime('%Y-%m-%d'),
             "train_end_requested": train_end.strftime('%Y-%m-%d'),
             "train_start_actual": train_slice['date'].min().strftime('%Y-%m-%d') if len(train_slice) > 0 else "N/A",
             "train_end_actual": train_slice['date'].max().strftime('%Y-%m-%d') if len(train_slice) > 0 else "N/A",
             "n_train": n_train,
             "pos_rate_train": float(pos_rate)
        })
        
        run_summary["ticker_summaries"][tk] = {
             "model_path": model_path,
             "n_train": n_train,
             "pos_rate": pos_rate,
             "valid_history_days": len(base_lookback_df)
        }

    # 4. CSV 與 JSON 寫檔
    csv_path = os.path.join(output_daily_dir, "predictions.csv")
    json_path = os.path.join(output_daily_dir, "run_summary.json")
    
    if len(results) > 0:
        df_out = pd.DataFrame(results)
        df_out.to_csv(csv_path, index=False)
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(run_summary, f, indent=4)
        
    # --- 報表輸出 ---
    print("\n📊 今日推論結果 (Single Ticker Approach)")
    print("-" * 88)
    print(f"{'Ticker':<8} | {'Latest Date':<12} | {'Score(p)':<10} | {'PctRank':<8} | {'Act Thresh':<10} | {'Action':<15} | {'Pos Scale'}")
    print("-" * 88)
    for r in results:
        pct_str = f"{r['pct_rank_today']*100:.1f}%" if not np.isnan(r['pct_rank_today']) else "N/A"
        print(f"{r['ticker']:<8} | {r['date']:<12} | {r['p_today']*100:6.2f}%    | {pct_str:<8} | >={r['threshold_pct_used']*100:g}%     | {r['action']:<15} | x{r['position_scale']}")
        
    print("-" * 88)
    print(f"📝 報告輸出完成於: {output_daily_dir}")
    print(f"✅ predictions.csv 與 run_summary.json 已更新檔案")
    print("====================================================================\n")

if __name__ == "__main__":
    main()
