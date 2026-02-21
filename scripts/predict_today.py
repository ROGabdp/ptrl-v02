#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
US Tech Stock - Daily Train & Predictor
================================================================================
這個腳本用來幫助您每天載入最新的股票資料，進行自動化當日建模 (Daily Train) 
並推斷「今日最新的收盤數值」是否滿足未來漲幅的買點特徵，並產出 Top K 推薦清單。

如果提供 --model-path，則會退回傳統模式，直接使用提前預先訓練好的模型進行推論。
================================================================================
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

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
    parser = argparse.ArgumentParser(description="Predict 'Today's Buy Decision' for US Tech Stocks")
    
    # 執行模式 (Daily Train 或是 傳統讀取模式)
    parser.add_argument("--model-path", type=str, default=None, 
                        help="選填：提供已訓練模型路徑 (.zip/.joblib)。若未提供，則啟動 Daily Train 模式自動建構當日模型。")
    
    # 目標設定
    parser.add_argument("--tickers", nargs="+", 
                        default=["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "TSLA", "NFLX", "PLTR"],
                        help="要預測的目標股票列表 (預設 10 檔)")
    parser.add_argument("--target-days", type=int, default=120, help="預測未來的交易天數 (預設 120)")
    parser.add_argument("--target-return", type=float, default=0.20, help="目標最高價漲幅門檻 (預設 0.20)")
    
    # Daily Train 參數
    parser.add_argument("--window-years", type=int, default=3, help="Daily Train 抓取的歷史訓練窗格大小 (預設 3 年)")
    parser.add_argument("--use-regime-features", type=str, default="true", choices=["true", "false"], 
                        help="是否掛載大盤 Regime Features 一併訓練/預測 (預設 true)")
    parser.add_argument("--force-retrain", action="store_true", help="強制重新訓練新模型，即使今日已存在快取")
    parser.add_argument("--output-dir", type=str, default="output_daily", help="當日模型儲存根目錄")
    parser.add_argument("--no-cache", action="store_true", help="強制重新計算特徵而不是讀取昨天快取")
    
    # 決策輸出參數
    parser.add_argument("--topk-pct", type=int, default=10, help="Top K 輸出的百分比 (預設 10%)")
    parser.add_argument("--topk-n", type=int, default=None, help="絕對數值的 Top K，若提供則優先於 pct")
    parser.add_argument("--threshold", type=float, default=0.5, help="(傳統模式用) 決定買進的正類機率閾值 (預設 0.5)")
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
        # 用 pandas 塞進去避免 warn (若有 col names)
        # 若傳進來是 np array 的 X_input 也會直接對付
        y_proba, _, _ = get_positive_proba(model, pd.DataFrame(X_input), positive_label=1)
        return float(y_proba[0])
        
    else:
        raise ValueError(f"不認得的模型格式: {model_type}")

def fetch_and_prepare_daily_data(args):
    # 下載推往前推 8 年的資料以保留足夠 Buffer (MA240 + window_years)
    start_date = (datetime.today() - timedelta(days=8*365)).strftime("%Y-%m-%d")
    print(f"📥 正在從 Yahoo Finance 獲取/更新最新股價 (自 {start_date} 起)...")
    
    all_data = fetch_all_stock_data(start_date=start_date)
    benchmark_df = all_data.get(BENCHMARK)
    
    if benchmark_df is None:
        raise ValueError(f"❌ 無法載入基準指數 {BENCHMARK} 的資料。")
        
    return all_data, benchmark_df

def train_daily_model(args, all_data, benchmark_df):
    """將各 Tickers 的特徵串接在一起，建立一份當日統整模型"""
    today_str = datetime.today().strftime("%Y%m%d")
    daily_model_dir = os.path.join(args.output_dir, today_str)
    os.makedirs(daily_model_dir, exist_ok=True)
    
    model_save_path = os.path.join(daily_model_dir, "model.joblib")
    
    use_regime = (args.use_regime_features == "true")
    active_cols = FEATURE_COLS + (REGIME_COLS if use_regime else [])
    
    if os.path.exists(model_save_path) and not args.force_retrain:
        print(f"♻️ 發現今日快取模型，直接載入: {model_save_path}")
        return model_save_path, active_cols, ("Loaded from cache", "Loaded from cache")
        
    print(f"⚙️ 準備 Daily Train 訓練資料集 (Window: {args.window_years} years)...")
    
    train_dfs = []
    regime_df = compute_regime_features(benchmark_df) if use_regime else None
    
    # 決定訓練切割邊界: 確保 y label 不漏看未來
    # 取全部股票最新的一天作為 T
    latest_date_overall = None
    for tk in args.tickers:
        if tk in all_data and len(all_data[tk]) > 0:
            last_dt = all_data[tk].index[-1] if isinstance(all_data[tk].index, pd.DatetimeIndex) else pd.to_datetime(all_data[tk]['Date']).max()
            if latest_date_overall is None or last_dt > latest_date_overall:
                latest_date_overall = last_dt
                
    if latest_date_overall is None:
        latest_date_overall = pd.to_datetime(datetime.today())
        
    train_end = latest_date_overall
    train_start = train_end - pd.DateOffset(years=args.window_years)
    
    train_start_str = train_start.strftime("%Y-%m-%d")
    train_end_str = train_end.strftime("%Y-%m-%d")
    print(f"  [Train Window Range] {train_start_str} ~ {train_end_str}")
    
    target_col = f"Next_{args.target_days}d_Max"
    
    for tk in args.tickers:
        raw_df = all_data.get(tk)
        if raw_df is None or len(raw_df) == 0: continue
            
        feat_df = calculate_features(raw_df, benchmark_df, ticker=tk, use_cache=not args.no_cache)
        if target_col not in feat_df.columns:
            continue
            
        feat_df = feat_df.reset_index()
        if 'Date' in feat_df.columns:
            feat_df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'index' in feat_df.columns:
            feat_df.rename(columns={'index': 'date'}, inplace=True)
            
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        
        # Merge regime
        if use_regime:
            feat_df['date_str'] = feat_df['date'].dt.strftime('%Y-%m-%d')
            # regime_df 的 date 也是字串
            feat_df = pd.merge(feat_df, regime_df, left_on='date_str', right_on='date', how='inner', suffixes=('', '_regime'))
            
        # 切割訓練集
        mask = (feat_df['date'] >= train_start) & (feat_df['date'] <= train_end)
        train_slice = feat_df[mask].copy()
        train_slice = train_slice.dropna(subset=active_cols + [target_col])
        
        train_slice['y'] = (train_slice[target_col] >= args.target_return).astype(int)
        train_dfs.append(train_slice)
        
    if not train_dfs:
        print("❌ 找不到任何有效的訓練資料，請檢查區間或 Tickers 設定")
        sys.exit(1)
        
    df_train_pooled = pd.concat(train_dfs, ignore_index=True)
    X_train = df_train_pooled[active_cols]
    y_train = df_train_pooled['y']
    
    print(f"🧠 進行 HistGradientBoosting 模型集訓 (N={len(X_train)}, Pos Rate={y_train.mean()*100:.2f}%) ...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_save_path)
    print(f"✅ 當日模型儲存完畢: {model_save_path}")
    
    return model_save_path, active_cols, (train_start_str, train_end_str)

def evaluate_regime_risk(benchmark_df):
    """判斷市場是否處於高風險狀態 (V2 Proxy 風控機制)"""
    regime_df = compute_regime_features(benchmark_df)
    if len(regime_df) == 0: return "NORMAL", False
    
    latest_regime = regime_df.iloc[-1]
    
    bm_above_ma200 = latest_regime.get('REGIME_BM_ABOVE_MA200', 1.0)
    hv20_pctl = latest_regime.get('REGIME_BM_HV20_PCTL', 0.0)
    ret_120 = latest_regime.get('REGIME_BM_RET_120', 0.0)
    
    is_risk = False
    reason = []
    
    if bm_above_ma200 == 0 and hv20_pctl > 0.8:
        is_risk = True
        reason.append(" MA200 Below & HV20 Pctl > 80% ")
        
    if ret_120 < 0 and hv20_pctl > 0.8:
        is_risk = True
        reason.append(" 120d Return < 0 & HV20 Pctl > 80% ")
        
    if is_risk:
        return f"HIGH RISK (Proxy: {'|'.join(reason)})", True
    return "NORMAL", False
    

def main():
    args = parse_args()
    
    is_daily_train = (args.model_path is None)
    
    print("====================================================================")
    print("🚀 US Tech Stock - Daily Train & Predictor")
    print("====================================================================")
    print(f"  Mode        : {'Daily Train & Predict' if is_daily_train else 'Legacy Predict (Loaded Model)'}")
    print(f"  Target      : Next_{args.target_days}d_Max >= {args.target_return*100:g}%")
    print(f"  Tickers     : {', '.join(args.tickers)}")
    if not is_daily_train:
        print(f"  Threshold   : {args.threshold} (Legacy Mode)")
    print("====================================================================\n")
    
    try:
         all_data, benchmark_df = fetch_and_prepare_daily_data(args)
    except Exception as e:
         print(f"❌ {e}")
         sys.exit(1)
         
    # --- 模型處理與前置作業 ---
    active_cols = FEATURE_COLS
    train_range = ("N/A", "N/A")
    use_regime_features = False
    
    if is_daily_train:
         use_regime_features = (args.use_regime_features == "true")
         model_path, active_cols, train_range = train_daily_model(args, all_data, benchmark_df)
         model_type = "daily"
    else:
         multi_model = "{ticker}" in args.model_path
         model_ext = ".zip" if ".zip" in args.model_path else ".joblib"
         model_type = "ppo" if model_ext == ".zip" else "sklearn"
         model_path = args.model_path
         
    # --- Regime 風險推斷 ---
    risk_status_text, is_high_risk = evaluate_regime_risk(benchmark_df)
    
    
    # --- 逐 Ticker 萃取今日特徵與推論 ---
    results = [] # (ticker, latest_date, proba, warning_text)
    
    regime_df = compute_regime_features(benchmark_df) if use_regime_features else None
    
    for ticker in args.tickers:
        raw_df = all_data.get(ticker)
        if raw_df is None or len(raw_df) == 0: continue
        
        cur_model_path = model_path.replace("{ticker}", ticker) if not is_daily_train and "{ticker}" in model_path else model_path
        if not os.path.exists(cur_model_path):
             results.append((ticker, "N/A", -1.0, "No Model"))
             continue
             
        try:
             feat_df = calculate_features(raw_df, benchmark_df, ticker=ticker, use_cache=not args.no_cache)
             latest_feat = feat_df.iloc[-1:].copy()
             
             if 'Date' in latest_feat.columns:
                 latest_date = latest_feat['Date'].iloc[0].strftime("%Y-%m-%d")
             elif latest_feat.index.name == 'Date' or isinstance(latest_feat.index, pd.DatetimeIndex):
                 latest_date = latest_feat.index[0].strftime("%Y-%m-%d")
             else:
                 latest_date = "Unknown"
                 
             if use_regime_features:
                 # 取回 regime 最後一筆 (因為是大盤，不一定對齊，取對應日期)
                 matching_regime = regime_df.loc[regime_df['date'] == latest_date]
                 if matching_regime.empty:
                      # 假如對不到日期，退而求其次抓大盤最後一筆
                      latest_regime_row = regime_df.iloc[-1]
                 else:
                      latest_regime_row = matching_regime.iloc[0]
                      
                 for c in REGIME_COLS:
                      latest_feat[c] = latest_regime_row[c]
                      
             # 取值預測，將 DataFrame 包裝塞入以保留 pandas column 名字
             X_input = pd.DataFrame([latest_feat[active_cols].iloc[0]], columns=active_cols)
             if model_type == "ppo": X_input = X_input.values.astype(np.float32)
             
             proba = load_model_and_predict(cur_model_path, model_type, X_input)
             results.append((ticker, latest_date, proba, ""))
             
        except Exception as e:
             results.append((ticker, "N/A", -1.0, f"Error: {str(e)[:15]}"))
             
             
    # --- 決策邏輯 (Top K) ---
    if is_daily_train:
         # 計算名額
         total_valid = len([r for r in results if r[2] >= 0])
         if args.topk_n is not None:
              base_k = args.topk_n
         else:
              base_k = max(1, int(total_valid * (args.topk_pct / 100.0)))
              
         final_k = base_k
         if is_high_risk:
              final_k = max(0, base_k // 2)
              
         # 排序
         results.sort(key=lambda x: x[2], reverse=True)
         
         final_rows = []
         rank = 1
         for tk, dt, pb, warn in results:
             if pb < 0:
                 action = warn
             else:
                 if is_high_risk and final_k == 0:
                      action = "SKIP_RISK 🛑"
                 elif rank <= final_k:
                      action = "BUY_TOPK 🟢"
                 elif rank <= base_k and is_high_risk:
                      action = "DOWNGRADE_RISK ⚠️"
                 else:
                      action = "WATCHLIST ⚪"
                 rank += 1
             final_rows.append((tk, dt, pb, action))
             
    else:
         # Legacy Threshold mode
         final_rows = []
         for tk, dt, pb, warn in results:
             if pb < 0: action = warn
             else:
                 action = "BUY 🟢" if pb >= args.threshold else "WAIT ⚪"
             final_rows.append((tk, dt, pb, action))
             

    # --- 報表輸出 ---
    print("\n📊 今日推論結果 (Prediction for Latest Close)")
    print("-" * 75)
    header_prob = f"Score P({args.target_days})"
    print(f"{'Rank':<5} | {'Ticker':<8} | {'Latest Date':<12} | {header_prob:<14} | {'Action':<15}")
    print("-" * 75)
    
    r_idx = 1
    for tk, dt, pb, act in final_rows:
        pb_str = "N/A" if (pb < 0 or np.isnan(pb)) else f"{pb*100:6.2f}%"
        rank_str = f"#{r_idx}" if pb >= 0 else "-"
        print(f"{rank_str:<5} | {tk:<8} | {dt:<12} | {pb_str:<14} | {act:<15}")
        if pb >= 0: r_idx += 1
        
    print("-" * 75)
    print("📝 【報表總結】")
    if is_daily_train:
         print(f"  [模型窗格] {train_range[0]} ~ {train_range[1]} (3y Pooled HGB) {'含 Regime 特徵' if use_regime_features else ''}")
         print(f"  [風控狀態] {risk_status_text}")
         
         topk_desc = f"{args.topk_n} 檔" if args.topk_n else f"{args.topk_pct}%"
         if is_high_risk:
              print(f"  [出手策略] 原目標 Top {topk_desc} (降槓桿縮倉: 取 {final_k} 檔)")
         else:
              print(f"  [出手策略] 取 Top {topk_desc} (發放名額: {final_k} 檔)")
         print(f"  [快取路徑] {model_path}")
    else:
         print(f"  [模型路徑] {args.model_path}")
         print(f"  [評估門檻] Threshold = {args.threshold}")
    print("====================================================================\n")

if __name__ == "__main__":
    main()
