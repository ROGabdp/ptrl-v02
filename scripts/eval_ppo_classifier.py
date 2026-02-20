#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
US Tech Stock - PPO Classifier Offline Evaluation Script
================================================================================
直接載入已訓練好的 PPO Buy Agent (best_model.zip 等)，對 Validation 區間進行離線推論。
不重新訓練、不觸發 learn()、不修改分佈，以全量真實 Validation 進行評估。

輸出格式與 Metrics 100% 與 sklearn 分類器對齊，以便直接比較兩者。
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

from stable_baselines3 import PPO
import torch

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix)

# 解決一些 torch 載入可能產生的警告
warnings.filterwarnings('ignore', category=UserWarning)

# 將專案根目錄加到 sys.path，以便 import 共用模組
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from train_us_tech_buy_agent import (fetch_all_stock_data, calculate_features, 
                                         FEATURE_COLS, VAL_RANGE, BENCHMARK)
except ImportError as e:
    print(f"❌ 無法從 train_us_tech_buy_agent.py 載入共用邏輯: {e}")
    print("請確保腳本有放置在正確的根目錄下層 scripts 資料夾內。")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO Buy Agent (Next 20d Max >= 10%) Offline")
    
    # 必填模型參數
    parser.add_argument("--model-path", type=str, required=True, 
                        help="必填：PPO 儲存的模型壓縮檔路徑 (例如: models_v5/ppo_buy_base_us_tech.zip)")
                        
    # 資料參數
    parser.add_argument("--tickers", nargs="+", 
                        default=["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "TSLA", "NFLX", "PLTR"],
                        help="目標股票代碼列表")
    parser.add_argument("--val-start-date", type=str, help="驗證區間起始日 (未提供則用 VAL_RANGE)")
    parser.add_argument("--val-end-date", type=str, help="驗證區間結束日 (未提供則用 VAL_RANGE)")
    
    # 評估參數
    parser.add_argument("--threshold", type=float, default=0.5, help="用來決定 y_pred 的正類閾值 (預設 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (不影響抽樣，僅固定排序)")
    
    # 環境與輸出參數
    parser.add_argument("--output-dir", default=os.path.join(ROOT_DIR, "output_eval_ppo"), 
                        help="評估結果與指標輸出根目錄")
    parser.add_argument("--no-cache", action="store_true", help="關閉特徵快取")
    parser.add_argument("--dry-run", action="store_true", 
                        help="只檢查資料與維度，印出驗證總數後結束 (不載入模型推論)")
    
    return parser.parse_args()


def prepare_validation_data(args, val_range):
    """
    載入並處理資料，輸出 Validation 驗證集全量真實分佈
    """
    all_raw_data = fetch_all_stock_data()
    benchmark_df = all_raw_data.get(BENCHMARK)
    if benchmark_df is None:
        raise ValueError(f"無法載入 benchmark {BENCHMARK} 的資料。")
        
    use_cache = not args.no_cache
    val_dfs = []
    
    print("\n🔍 正在生成/載入特徵與擷取 Validation 區段...")
    val_start, val_end = val_range
    
    for ticker in args.tickers:
        if ticker not in all_raw_data:
            print(f"  ⚠️ {ticker} 原始資料不存在，跳過。")
            continue
            
        df_raw = all_raw_data[ticker]
        df_features = calculate_features(df_raw, benchmark_df, ticker=ticker, use_cache=use_cache)
        
        # 1. 確保目標欄位存在並過濾 NaN
        if 'Next_20d_Max' not in df_features.columns:
            print(f"  ⚠️ {ticker} 無 Next_20d_Max 欄位，跳過。")
            continue
        df_features = df_features.dropna(subset=['Next_20d_Max'])
        
        # 2. 驗證集時間切分
        val_mask = (df_features.index >= pd.Timestamp(val_start)) & (df_features.index <= pd.Timestamp(val_end))
        df_val_tick = df_features[val_mask]
        
        if len(df_val_tick) == 0:
            print(f"  ⚠️ {ticker} 於驗證區間 ({val_start} ~ {val_end}) 內無任何有效樣本，跳過。")
            continue
            
        # 3. 補充標記
        df_val_tick.loc[:, 'ticker'] = ticker
        df_val_tick.loc[:, 'date'] = df_val_tick.index.strftime('%Y-%m-%d')
        df_val_tick.loc[:, 'y'] = (df_val_tick['Next_20d_Max'] >= 0.10).astype(int)
        
        val_dfs.append(df_val_tick)
        
    df_val = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    return df_val


def print_val_stats(df_val, tickers):
    """印出 Validation 資料集統計資訊"""
    print("\n📊 Validation 資料與類別比例統計")
    print("-" * 45)
    print(f"{'Ticker':<8} | {'Val (N)':<10} | {'Val Pos%':<10}")
    print("-" * 45)
    
    for tk in tickers:
        d_va = df_val[df_val['ticker'] == tk]
        va_len = len(d_va)
        if va_len > 0:
            va_pos = d_va['y'].mean()
            print(f"{tk:<8} | {va_len:<10} | {va_pos*100:6.2f}%")
        
    print("-" * 45)
    tot_va_len = len(df_val)
    tot_va_pos = df_val['y'].mean() if tot_va_len > 0 else 0
    print(f"{'TOTAL':<8} | {tot_va_len:<10} | {tot_va_pos*100:6.2f}%")
    print("-" * 45)


def get_ppo_probabilities(model, X_val_np):
    """
    從 PPO model 獲得預測的正類機率 P(action=1|x)
    
    PPO 屬於 Actor-Critic，連續推論時需動用 Policy 的 get_distribution 方法
    而不使用 env step 來規避重新連動的問題
    """
    print("🧠 正在進行 PPO Offline Inference (no_grad)...")
    device = model.device
    
    # 將 Numpy Array 轉成 PyTorch Tensor，同時送往同裝置
    obs_tensor = torch.tensor(X_val_np, dtype=torch.float32, device=device)
    
    # 關閉梯度進行快速運算
    with torch.no_grad():
        # 對連續的 observation 獲得其 Categorical 分佈
        distribution = model.policy.get_distribution(obs_tensor)
        # 取得每一個 sample 在 action=0 跟 action=1 上的 softmax 機率分佈
        # .probs 維度為 (batch_size, 2)
        probs = distribution.distribution.probs
        
        # 提取 Action=1 (BUY) 的機率
        y_proba = probs[:, 1].cpu().numpy()
        
    return y_proba


def calc_metrics(y_true, y_proba, threshold, prefix="Overall"):
    """計算並回傳驗證集的各種指標 (統一與 Sklearn 版產出格式對齊)"""
    metrics = {}
    y_pred = (y_proba >= threshold).astype(int)
    
    # 避免 y_true 全 0 或全 1 導致 auc 失敗
    has_mixed_classes = len(np.unique(y_true)) > 1
    
    metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['Precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['Recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['F1'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_proba)) if has_mixed_classes else None
    metrics['PR-AUC'] = float(average_precision_score(y_true, y_proba)) if has_mixed_classes else None
    
    metrics['Confusion Matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Precision@k (Top 1%, 5%, 10%)
    sort_idx = np.argsort(y_proba)[::-1]
    sorted_y_true = np.array(y_true)[sort_idx]
    
    for k_pct in [0.01, 0.05, 0.10]:
        k = max(1, int(len(y_true) * k_pct))
        top_k_y_true = sorted_y_true[:k]
        metrics[f'Precision@{int(k_pct*100)}%'] = float(np.mean(top_k_y_true))
        
    # Threshold sweep
    metrics['Threshold Sweep'] = {}
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred_th = (y_proba >= th).astype(int)
        metrics['Threshold Sweep'][f'Threshold={th}'] = {
            'Precision': float(precision_score(y_true, y_pred_th, zero_division=0)),
            'Recall': float(recall_score(y_true, y_pred_th, zero_division=0)),
            'F1': float(f1_score(y_true, y_pred_th, zero_division=0))
        }
        
    return metrics


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    # 決定切割區段
    val_range = (
        args.val_start_date if args.val_start_date else VAL_RANGE[0],
        args.val_end_date if args.val_end_date else VAL_RANGE[1]
    )
    
    print("=" * 60)
    print("🚀 PPO Classifier Offline Evaluation (Validation Only)")
    print("=" * 60)
    print(f"  PPO Model   : {args.model_path}")
    print(f"  Target      : Next_20d_Max >= 10%")
    print(f"  Tickers     : {', '.join(args.tickers)}")
    print(f"  Val Range   : {val_range}")
    print(f"  Threshold   : {args.threshold}")
    print(f"  Dry Run     : {args.dry_run}")
    print("=" * 60)
    
    if not os.path.exists(args.model_path):
        print(f"❌ 找不到指定的模型路徑: {args.model_path}")
        sys.exit(1)
    
    # 1. 準備資料
    df_val = prepare_validation_data(args, val_range)
    
    if len(df_val) == 0:
        print("❌ 驗證資料集為空，請檢查日期與資料狀態。")
        sys.exit(1)
        
    # 印出數據分佈
    print_val_stats(df_val, args.tickers)
    
    if args.dry_run:
        print("\n✅ Dry-Run 模式結束。")
        sys.exit(0)
        
    # 2. 準備特徵陣列 
    X_val = df_val[FEATURE_COLS].values.astype(np.float32)
    y_val = df_val['y'].values
    
    # 3. 載入模型 (不使用 custom_objects，只倚靠基底 model_path 中記載的網路架構即可推論)
    print("\n📦 載入 PPO 模型...")
    try:
        # 強制指定 device="cpu" 防止 device map error
        model_ppo = PPO.load(args.model_path, device="cpu")
    except Exception as e:
        print(f"❌ PPO 模型載入失敗: {e}")
        sys.exit(1)
        
    # 4. 推論提取機率
    y_proba_val = get_ppo_probabilities(model_ppo, X_val)
    
    # 5. 計算指標
    print("📈 正在計算指標陣列...")
    metrics = calc_metrics(y_val, y_proba_val, threshold=args.threshold, prefix="Pooled Overall")
    
    # 6. 輸出儲存
    # 由 model 檔名當作資料夾前綴
    base_name = os.path.basename(args.model_path).replace(".zip", "")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"eval_ppo_{base_name}_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    # (a) Params Json
    params = {
        "cli_args": vars(args),
        "actual_val_range": val_range,
        "val_samples": len(df_val),
        "val_pos_ratio": float(np.mean(y_val)),
        "eval_model": os.path.abspath(args.model_path),
        "used_threshold": args.threshold
    }
    with open(os.path.join(run_dir, "eval_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
        
    # (b) Metrics Json
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    # (c) Prediction CSV
    df_val_export = df_val[['date', 'ticker']].copy()
    df_val_export['y_true'] = y_val
    df_val_export['y_proba'] = y_proba_val
    df_val_export['y_pred'] = (y_proba_val >= args.threshold).astype(int)
    df_val_export.to_csv(os.path.join(run_dir, "val_predictions.csv"), index=False)
    
    print("\n✅ 推論與評估完成！")
    print("-" * 60)
    print(f"  [Validation Metrics (Pooled / Micro) @ T={args.threshold}]")
    print(f"  Accuracy : {metrics['Accuracy']:.4f}")
    print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}" if metrics['ROC-AUC'] else "  ROC-AUC  : N/A")
    print(f"  PR-AUC   : {metrics['PR-AUC']:.4f}" if metrics['PR-AUC'] else "  PR-AUC   : N/A")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall   : {metrics['Recall']:.4f}")
    print(f"  F1-Score : {metrics['F1']:.4f}")
    print(f"  Precision@5%: {metrics.get('Precision@5%', 0):.4f}")
    print(f"\n📂 結果已儲存於: {run_dir}")

if __name__ == "__main__":
    main()
