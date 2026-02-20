#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
US Tech Stock - Daily Buy Agent Predictor
================================================================================
這個腳本用來幫助您每天載入最新的股票資料，並使用訓練好的模型（支援 PPO 與 Sklearn）
來推斷「今日最新的收盤數值」是否滿足未來 20 天漲幅 >= 10% 的買點特徵。

使用方式:
python scripts/predict_today.py --model-path output_sklearn/run_hgb_123/model.joblib
python scripts/predict_today.py --model-path models_v5/finetuned/{ticker}/best/best_model.zip
================================================================================
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import warnings

# 解決一些 torch 載入可能產生的警告
warnings.filterwarnings('ignore', category=UserWarning)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from train_us_tech_buy_agent import fetch_all_stock_data, calculate_features, FEATURE_COLS, BENCHMARK
except ImportError as e:
    print(f"❌ 無法從 train_us_tech_buy_agent.py 載入共用邏輯: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict 'Today's Buy Decision' for US Tech Stocks")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="必填：模型檔案路徑。支援 .zip (PPO) 或 .joblib (Sklearn)。允許多模型 {ticker} 動態變數。")
    parser.add_argument("--tickers", nargs="+", 
                        default=["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "TSLA", "NFLX", "PLTR"],
                        help="要預測的目標股票列表 (預設 10 檔)")
    parser.add_argument("--target-days", type=int, default=20, help="預測未來的交易天數 (對應模型訓練設定)")
    parser.add_argument("--target-return", type=float, default=0.10, help="目標最高價漲幅門檻 (對應模型訓練設定)")
    parser.add_argument("--threshold", type=float, default=0.5, help="決定買進的正類機率閾值 (預設 0.5)")
    parser.add_argument("--no-cache", action="store_true", help="強制重新計算特徵而不是讀取昨天快取")
    return parser.parse_args()


def load_model_and_predict(model_path, model_type, X_input):
    """根據模型種類 (PPO or Sklearn) 載入並預測機率"""
    if model_type == "ppo":
        from stable_baselines3 import PPO
        import torch
        model = PPO.load(model_path, device="cpu")
        with torch.no_grad():
            obs_tensor = torch.tensor(X_input, dtype=torch.float32, device="cpu")
            distribution = model.policy.get_distribution(obs_tensor)
            proba = distribution.distribution.probs[:, 1].cpu().numpy()[0]
        return proba
        
    elif model_type == "sklearn":
        model = joblib.load(model_path)
        # sklearn predict_proba 輸出為 (n_samples, n_classes)，取 positive class [1]
        proba = model.predict_proba(X_input)[0][1]
        return float(proba)
        
    else:
        raise ValueError(f"不認得的模型格式: {model_type}")


def main():
    args = parse_args()
    
    print("====================================================================")
    print("🚀 US Tech Stock - Daily Buy Predictor")
    print("====================================================================")
    
    # 判斷輸入模型是哪種系統 
    multi_model = "{ticker}" in args.model_path
    if not multi_model and not os.path.exists(args.model_path):
        print(f"❌ 找不到模型: {args.model_path}")
        sys.exit(1)
        
    model_ext = ".zip" if ".zip" in args.model_path else ".joblib"
    model_type = "ppo" if model_ext == ".zip" else "sklearn"
    
    print(f"  System Type : {model_type.upper()} ({model_ext})")
    print(f"  Model Path  : {args.model_path}")
    print(f"  Target      : Next_{args.target_days}d_Max >= {args.target_return*100:g}%")
    print(f"  Tickers     : {', '.join(args.tickers)}")
    print(f"  Threshold   : {args.threshold}")
    print("====================================================================\n")
    
    # 1. 下載並讀取最新資料
    print("📥 正在從 Yahoo Finance 獲取/更新最新股價...")
    # 只需擷取最近 5 年內資料足以計算全部特徵與暖機
    all_data = fetch_all_stock_data(start_date="2020-01-01")
    benchmark_df = all_data.get(BENCHMARK)
    
    if benchmark_df is None:
        print(f"❌ 無法載入基準指數 {BENCHMARK} 的資料。")
        sys.exit(1)
        
    results = []
    
    # 2. 為每檔目標股票推斷最新買點
    for ticker in args.tickers:
        latest_date = "N/A"
        proba = np.nan
        status = "-"
        
        # 確認資料存在
        raw_df = all_data.get(ticker)
        if raw_df is None or len(raw_df) == 0:
            status = "No Data"
            results.append((ticker, latest_date, proba, status))
            continue
            
        # 確認模型存在
        cur_model_path = args.model_path.replace("{ticker}", ticker) if multi_model else args.model_path
        if not os.path.exists(cur_model_path):
            status = "No Model"
            results.append((ticker, latest_date, proba, status))
            continue
            
        try:
            # 計算特徵 (包含最新一筆還沒有 Next_20d_Max 真實標籤的資料)
            features_df = calculate_features(raw_df, benchmark_df, ticker=ticker, use_cache=not args.no_cache)
            
            # 從 features Dataframe 取出最後一筆
            latest_feat = features_df.iloc[-1:]
            latest_date = latest_feat.index[0].strftime("%Y-%m-%d")
            
            # 準備輸入 Matrix (1, N_features)
            X_input = latest_feat[FEATURE_COLS].values
            if model_type == "ppo":
                 X_input = X_input.astype(np.float32)
                 
            # 進行機率推論
            proba = load_model_and_predict(cur_model_path, model_type, X_input)
            
            decision = "BUY 🟢" if proba >= args.threshold else "WAIT ⚪"
            status = decision
            
        except Exception as e:
             status = f"Error: {str(e)[:15]}.."
             
        results.append((ticker, latest_date, proba, status))
        
    # 3. 列印最終報表
    print("\n📊 今日推論結果 (Prediction for Latest Close)")
    print("-" * 65)
    header_prob = f"P({args.target_days}d>={args.target_return*100:g}%)"
    print(f"{'Ticker':<8} | {'Latest Date':<12} | {header_prob:<14} | {'Decision':<15}")
    print("-" * 65)
    
    buy_count = 0
    for tk, dt, pb, st in results:
        if isinstance(pb, str) or np.isnan(pb):
             pb_str = "N/A"
        else:
             pb_str = f"{pb*100:6.2f}%"
             if pb >= args.threshold: buy_count += 1
             
        print(f"{tk:<8} | {dt:<12} | {pb_str:<14} | {st:<15}")
    print("-" * 65)
    print(f"🎯 總計 ({header_prob}) 符合買進門檻 ({args.threshold}): {buy_count} 檔")
    print("====================================================================\n")

if __name__ == "__main__":
    main()
