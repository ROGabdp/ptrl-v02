#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenerate best params from existing sensitivity results
"""
import pandas as pd
import os

# Load existing results
df = pd.read_csv('sensitivity_results/sensitivity_analysis_results.csv')
TARGET_TICKERS = ['PLTR', 'NVDA', 'TSLA', 'NFLX']

print('=' * 80)
print('參數敏感度分析 - 最佳參數建議')
print('=' * 80)

# Baseline
baseline = df[(df['hard_stop'] == -0.08) & (df['callback_base'] == 0.08) & (df['callback_high'] == 0.11)]
print('\n【原始參數 (-8%/-8%/-11%)】')
for _, row in baseline.iterrows():
    print(f"  {row['ticker']:>6}: Return={row['total_return']*100:>6.1f}% | MDD={row['mdd']*100:>6.1f}% | Sharpe={row['sharpe']:>5.2f}")

print('\n【各股票最佳參數 (Sharpe 最高)】')
recommendations = []
for ticker in TARGET_TICKERS:
    ticker_df = df[df['ticker'] == ticker]
    best = ticker_df.nlargest(1, 'sharpe').iloc[0]
    baseline_row = baseline[baseline['ticker'] == ticker]
    mdd_imp = best['mdd'] - baseline_row.iloc[0]['mdd'] if len(baseline_row) > 0 else 0
    
    print(f"\n  {ticker}:")
    print(f"    Params: Stop={best['hard_stop']:.0%} | CB_Base={best['callback_base']:.0%} | CB_High={best['callback_high']:.0%}")
    print(f"    Return: {best['total_return']*100:.1f}% | MDD: {best['mdd']*100:.1f}% | Sharpe: {best['sharpe']:.2f}")
    print(f"    MDD 改善: {mdd_imp*100:+.1f}%")
    
    recommendations.append({
        'Ticker': ticker,
        'Hard_Stop': f"{best['hard_stop']:.0%}",
        'Callback_Base': f"{best['callback_base']:.0%}",
        'Callback_High': f"{best['callback_high']:.0%}",
        'Total_Return': f"{best['total_return']*100:.1f}%",
        'MDD': f"{best['mdd']*100:.1f}%",
        'Sharpe': f"{best['sharpe']:.2f}",
        'MDD_Improvement': f"{mdd_imp*100:+.1f}%"
    })

# Save
rec_df = pd.DataFrame(recommendations)
rec_df.to_csv('sensitivity_results/sensitivity_best_params.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 已更新: sensitivity_results/sensitivity_best_params.csv")
print('=' * 80)
