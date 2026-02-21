#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
US Tech Stock - Sklearn Binary Classifier Training Script
================================================================================
用於訓練二分分類模型，預測未來 20 個交易日內是否上漲超過 10%。
將重用 train_us_tech_buy_agent.py 內的 fetch_all_stock_data 與 calculate_features。

支援模型: RandomForest, AdaBoost, HistGradientBoosting
================================================================================
"""

import os
import sys
import json
import joblib
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance

# 將專案根目錄加到 sys.path，以便 import 共用模組
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.train.sklearn_utils import get_positive_proba, apply_class_balancing, get_model, calc_metrics, get_feature_importances

try:
    from train_us_tech_buy_agent import (fetch_all_stock_data, calculate_features, 
                                         FEATURE_COLS, TRAIN_RANGES, VAL_RANGE, BENCHMARK)
except ImportError as e:
    print(f"❌ 無法從 train_us_tech_buy_agent.py 載入共用邏輯: {e}")
    print("請確保腳本有放置在正確的根目錄下層 scripts 資料夾內。")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sklearn binary classifier for Buy Agent (Next 20d Max >= 10%)")
    
    # 資料參數
    parser.add_argument("--tickers", nargs="+", 
                        default=["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "TSLA", "NFLX", "PLTR", "TSM"],
                        help="目標股票代碼列表")
    parser.add_argument("--train-ranges", nargs="*", 
                        help="訓練資料區間。格式: YYYY-MM-DD:YYYY-MM-DD (可提供多段，空白分隔)")
    parser.add_argument("--val-start-date", type=str, help="驗證區間起始日")
    parser.add_argument("--val-end-date", type=str, help="驗證區間結束日")
    
    # 訓練參數
    parser.add_argument("--model", choices=["rf", "adaboost", "hgb"], default="rf",
                        help="選擇訓練模型種類")
    parser.add_argument("--target-days", type=int, default=20, help="預測未來的交易天數 (e.g. 20, 60, 120)")
    parser.add_argument("--target-return", type=float, default=0.10, help="目標最高價漲幅門檻 (e.g. 0.10, 0.20)")
    parser.add_argument("--balance-train", choices=["none", "undersample_50_50", "class_weight_balanced"], 
                        default="none", help="類別不平衡處理方式 (僅作用於訓練集)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    
    # 環境與輸出參數
    parser.add_argument("--output-dir", default=os.path.join(ROOT_DIR, "output_sklearn"), 
                        help="模型與指標輸出目錄")
    parser.add_argument("--no-cache", action="store_true", help="關閉特徵快取")
    parser.add_argument("--dry-run", action="store_true", 
                        help="只檢查資料與切分，不進行實際訓練")
    
    return parser.parse_args()


def get_positive_proba(model, X, positive_label=1) -> tuple:
    """
    取得預測為 positive_label (預設為 1) 的機率陣列。
    避免依賴 hard-coded 的 [:, 1]，改由 model.classes_ 動態尋找。
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"模型 {type(model).__name__} 不支援 predict_proba()")
        
    proba_all = model.predict_proba(X)
    classes = list(model.classes_)
    
    if positive_label not in classes:
        raise ValueError(f"標籤 {positive_label} 不存在於 model.classes_ {classes} 中。")
        
    pos_idx = classes.index(positive_label)
    return proba_all[:, pos_idx], classes, pos_idx


def parse_date_ranges(train_ranges_arg):
    if not train_ranges_arg:
        return TRAIN_RANGES
    parsed = []
    for r in train_ranges_arg:
        parts = r.split(':')
        if len(parts) == 2:
            parsed.append((parts[0], parts[1]))
        else:
            raise ValueError(f"訓練區間格式錯誤: {r} (應為 YYYY-MM-DD:YYYY-MM-DD)")
    return parsed


def prepare_data(args, train_ranges, val_range):
    """
    載入並處理資料，輸出訓練集與驗證集
    """
    all_raw_data = fetch_all_stock_data()
    benchmark_df = all_raw_data.get(BENCHMARK)
    if benchmark_df is None:
        raise ValueError(f"無法載入 benchmark {BENCHMARK} 的資料。")
        
    use_cache = not args.no_cache
    target_col = f"Next_{args.target_days}d_Max"
    
    train_dfs = []
    val_dfs = []
    
    print(f"\n🔍 正在生成/載入特徵... (目標: {target_col} >= {args.target_return*100:g}%)")
    for ticker in args.tickers:
        if ticker not in all_raw_data:
            print(f"  ⚠️ 找不到 {ticker} 原始資料，跳過。")
            continue
            
        df_raw = all_raw_data[ticker]
        df_features = calculate_features(df_raw, benchmark_df, ticker=ticker, use_cache=use_cache)
        
        # 1. 確保目標欄位存在並過濾 NaN
        if target_col not in df_features.columns:
            print(f"  ⚠️ 找不到特徵欄位 {target_col}，請確定 calculate_features 已支援該天數。")
            continue
        df_features = df_features.dropna(subset=[target_col])
        
        # 2. 加入 date 與 ticker
        df_features['ticker'] = ticker
        df_features['date'] = df_features.index.strftime('%Y-%m-%d')
        
        # 3. 建立標籤 y
        df_features['y'] = (df_features[target_col] >= args.target_return).astype(int)
        
        # 4. 時間切分 (Walk-forward Split)
        # 訓練集
        train_mask = pd.Series(False, index=df_features.index)
        for start, end in train_ranges:
            train_mask |= (df_features.index >= pd.Timestamp(start)) & (df_features.index <= pd.Timestamp(end))
        df_train_tick = df_features[train_mask]
        
        # 驗證集
        val_start, val_end = val_range
        val_mask = (df_features.index >= pd.Timestamp(val_start)) & (df_features.index <= pd.Timestamp(val_end))
        df_val_tick = df_features[val_mask]
        
        train_dfs.append(df_train_tick)
        val_dfs.append(df_val_tick)
        
    df_train = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    df_val = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    
    return df_train, df_val


def print_data_stats(df_train, df_val, tickers):
    """印出資料集統計資訊"""
    print("\n📊 資料切分與類別比例統計")
    print("-" * 60)
    print(f"{'Ticker':<8} | {'Train (N)':<10} | {'Train Pos%':<10} | {'Val (N)':<10} | {'Val Pos%':<10}")
    print("-" * 60)
    
    for tk in tickers:
        d_tr = df_train[df_train['ticker'] == tk]
        d_va = df_val[df_val['ticker'] == tk]
        tr_len = len(d_tr)
        va_len = len(d_va)
        tr_pos = d_tr['y'].mean() if tr_len > 0 else 0
        va_pos = d_va['y'].mean() if va_len > 0 else 0
        print(f"{tk:<8} | {tr_len:<10} | {tr_pos*100:6.2f}%    | {va_len:<10} | {va_pos*100:6.2f}%")
        
    print("-" * 60)
    tot_tr_len = len(df_train)
    tot_va_len = len(df_val)
    tot_tr_pos = df_train['y'].mean() if tot_tr_len > 0 else 0
    tot_va_pos = df_val['y'].mean() if tot_va_len > 0 else 0
    print(f"{'TOTAL':<8} | {tot_tr_len:<10} | {tot_tr_pos*100:6.2f}%    | {tot_va_len:<10} | {tot_va_pos*100:6.2f}%")
    print("-" * 60)


def main():
    args = parse_args()
    
    # 決定切割區段
    train_ranges = parse_date_ranges(args.train_ranges)
    val_range = (
        args.val_start_date if args.val_start_date else VAL_RANGE[0],
        args.val_end_date if args.val_end_date else VAL_RANGE[1]
    )
    
    print("=" * 60)
    print("🚀 Sklearn Binary Classifier Training")
    print("=" * 60)
    print(f"  Model       : {args.model}")
    print(f"  Target      : Next_{args.target_days}d_Max >= {args.target_return*100:g}%")
    print(f"  Tickers     : {', '.join(args.tickers)}")
    print(f"  Train Ranges: {train_ranges}")
    print(f"  Val Range   : {val_range}")
    print(f"  Balance Mode: {args.balance_train}")
    print(f"  Dry Run     : {args.dry_run}")
    print("=" * 60)
    
    # 1. 準備資料
    df_train, df_val = prepare_data(args, train_ranges, val_range)
    
    if len(df_train) == 0 or len(df_val) == 0:
        print("❌ 訓練或驗證資料集為空，請檢查日期與資料下載狀態。")
        sys.exit(1)
        
    # 印出預設統計分布
    print_data_stats(df_train, df_val, args.tickers)
    
    # 如果是 Dry-run 就直接結束
    if args.dry_run:
        print("\n✅ Dry-Run 模式結束。")
        sys.exit(0)
    
    # 2. 類別平衡 (僅在訓練階段處理)
    df_train_b = apply_class_balancing(df_train, args.balance_train, args.seed)
    
    X_train = df_train_b[FEATURE_COLS]
    y_train = df_train_b['y']
    
    X_val = df_val[FEATURE_COLS]
    y_val = df_val['y']
    
    # 3. 準備模型
    model, needs_sample_weight = get_model(args.model, args.balance_train, args.seed)
    
    sample_weight = None
    if needs_sample_weight:
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        w_dict = dict(zip(np.unique(y_train), cw))
        sample_weight = np.array([w_dict[y] for y in y_train])
        
    print(f"\n⚙️  開始訓練 {args.model.upper()} ...")
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
        
    # 4. 預測與計算指標
    print("📈 正在對 Validation Subset 進行評估...")
    try:
        y_proba_val, clz_list, pos_idx = get_positive_proba(model, X_val, positive_label=1)
    except Exception as e:
        print(f"❌ 取得預測機率失敗: {e}")
        sys.exit(1)
        
    y_pred_val = model.predict(X_val)
    
    # 計算正負樣本的平均機率 (Sanity Check)
    mask_pos = (y_val == 1)
    mask_neg = (y_val == 0)
    mean_pos_proba = y_proba_val[mask_pos].mean() if mask_pos.sum() > 0 else 0.0
    mean_neg_proba = y_proba_val[mask_neg].mean() if mask_neg.sum() > 0 else 0.0
    
    proba_direction_warning = False
    if mean_pos_proba < mean_neg_proba:
        print(f"  ⚠️ [WARNING] 正樣本的平均預測機率 ({mean_pos_proba:.4f}) 小於 負樣本 ({mean_neg_proba:.4f})！")
        print("     這可能暗示分類器的學習結果方向相反，或正類被錯誤對應。ROC-AUC 可能 < 0.5。")
        proba_direction_warning = True
    
    metrics = calc_metrics(y_val, y_proba_val, y_pred_val, prefix="Pooled Overall")
    metrics['Sanity Check'] = {
        'mean_pos_proba': float(mean_pos_proba),
        'mean_neg_proba': float(mean_neg_proba),
        'proba_direction_warning': proba_direction_warning
    }
    
    # 計算 Feature Importances
    importances = get_feature_importances(model, args.model, X_val, y_val, FEATURE_COLS)
    metrics['Feature Importances'] = importances
    
    # 5. 輸出儲存
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{args.model}_{args.target_days}d_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    # (a) Model joblib
    joblib.dump(model, os.path.join(run_dir, "model.joblib"))
    
    # (b) Params Json
    params = {
        "cli_args": vars(args),
        "target_definition": f"Next_{args.target_days}d_Max >= {args.target_return}",
        "actual_train_ranges": train_ranges,
        "actual_val_range": val_range,
        "train_samples_raw": len(df_train),
        "train_samples_balanced": len(df_train_b),
        "val_samples": len(df_val),
        "impl_details": {
            "balance_application": "sample_weight passed to fit" if sample_weight is not None else ("class_weight arg passed" if args.balance_train == "class_weight_balanced" else args.balance_train),
            "model_classes": [int(c) for c in clz_list],
            "positive_class_index": int(pos_idx)
        }
    }
    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
        
    # (c) Metrics Json
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    # (d) Prediction CSV
    df_val_export = df_val[['date', 'ticker']].copy()
    df_val_export['y_true'] = y_val
    df_val_export['y_proba'] = y_proba_val
    df_val_export['y_pred'] = y_pred_val
    df_val_export.to_csv(os.path.join(run_dir, "val_predictions.csv"), index=False)
    
    print("\n✅ 訓練完成！")
    print("-" * 60)
    print(f"  [Validation Metrics (Pooled / Micro)]")
    print(f"  Accuracy : {metrics['Accuracy']:.4f}")
    print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}" if metrics['ROC-AUC'] else "  ROC-AUC  : N/A")
    print(f"  PR-AUC   : {metrics['PR-AUC']:.4f}" if metrics['PR-AUC'] else "  PR-AUC   : N/A")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall   : {metrics['Recall']:.4f}")
    print(f"  F1-Score : {metrics['F1']:.4f}")
    print(f"\n📂 結果已儲存於: {run_dir}")

if __name__ == "__main__":
    main()
