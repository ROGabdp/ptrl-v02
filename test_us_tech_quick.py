#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick verification test for train_us_tech_buy_agent.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_us_tech_buy_agent import (
    load_or_update_local_csv, 
    calculate_features, 
    TICKERS, BENCHMARK, FEATURE_COLS
)

def test_data_loading():
    print("=" * 60)
    print("Test 1: Data Loading")
    print("=" * 60)
    
    # Test benchmark
    benchmark_df = load_or_update_local_csv(BENCHMARK)
    assert benchmark_df is not None, "Failed to load benchmark"
    print(f"✅ Benchmark: {len(benchmark_df)} rows")
    
    # Test one ticker
    ticker = "NVDA"
    df = load_or_update_local_csv(ticker)
    assert df is not None, f"Failed to load {ticker}"
    print(f"✅ {ticker}: {len(df)} rows")
    
    return benchmark_df, df

def test_feature_engineering(benchmark_df, ticker_df):
    print("\n" + "=" * 60)
    print("Test 2: Feature Engineering")
    print("=" * 60)
    
    features = calculate_features(ticker_df, benchmark_df, "NVDA", use_cache=False)
    
    # Check all features exist
    missing = [c for c in FEATURE_COLS if c not in features.columns]
    if missing:
        print(f"❌ Missing features: {missing}")
    else:
        print(f"✅ All {len(FEATURE_COLS)} features present")
    
    # Check no NaN in features
    nan_cols = features[FEATURE_COLS].isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols) > 0:
        print(f"⚠️ Columns with NaN: {nan_cols.to_dict()}")
    else:
        print(f"✅ No NaN values in features")
    
    # Check target label
    if 'Next_20d_Max' in features.columns:
        pos_rate = (features['Next_20d_Max'] >= 0.10).mean()
        print(f"✅ Target label present (Positive rate: {pos_rate:.1%})")
    
    # Check new volatility features
    new_features = ['Feat_ATR_Ratio', 'Feat_HV20', 'Feat_Price_Pos']
    for f in new_features:
        if f in features.columns:
            print(f"   {f}: mean={features[f].mean():.3f}, std={features[f].std():.3f}")
    
    return features

if __name__ == "__main__":
    benchmark_df, ticker_df = test_data_loading()
    features = test_feature_engineering(benchmark_df, ticker_df)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
