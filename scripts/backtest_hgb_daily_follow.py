#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
HGB 通用跟單回測腳本 (Walk-Forward Daily Backtrader)
================================================================================
特點：
- 動態讀取 configs/rolling_profiles.json 覆寫單檔股票策略
- 防前視 (No Lookahead) 的滾動訓練 (支援 Daily/Monthly/Quarterly/Yearly 重訓)
- 原汁原味還原 predict_today.py 的 pct_rank 與 proxy risk 判定
- 完整的資金注入與停損停利模組
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import joblib
from scipy.stats import percentileofscore
from sklearn.ensemble import HistGradientBoostingClassifier

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 導入特徵計算與工具
try:
    from train_us_tech_buy_agent import (
        load_or_update_local_csv,
        calculate_features,
        BENCHMARK, FEATURE_COLS
    )
    from src.features.regime_features import compute_regime_features, REGIME_COLS
    from src.features.regime_features_stock import compute_stock_regime_features, STOCK_REGIME_COLS
    from src.train.sklearn_utils import get_positive_proba
except ImportError:
    # 支援腳本於子目錄或根目錄執行
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_us_tech_buy_agent import load_or_update_local_csv, calculate_features, BENCHMARK, FEATURE_COLS
    from src.features.regime_features import compute_regime_features, REGIME_COLS
    from src.features.regime_features_stock import compute_stock_regime_features, STOCK_REGIME_COLS
    from src.train.sklearn_utils import get_positive_proba


# =============================================================================
# 輔助函式 (風控與 Profile)
# =============================================================================
def get_ticker_profile(tk, profiles):
    if not profiles:
        return {}, "cli_args"
    if tk in profiles:
        return profiles[tk], tk
    if "default" in profiles:
        return profiles["default"], "default"
    return {}, "cli_args"

def evaluate_regime_risk_from_row(row: pd.Series) -> bool:
    above_ma = row.get("BM_ABOVE_MA200", 1)
    ret_120 = row.get("BM_RET_120", 0)
    hv20_pctl = row.get("BM_HV20_PCTL", 0)
    
    is_high_risk = False
    if above_ma == 0 and hv20_pctl > 0.8:
        is_high_risk = True
    elif ret_120 < 0 and hv20_pctl > 0.8:
        is_high_risk = True
    return is_high_risk

def is_retrain_needed(freq: str, current_date: pd.Timestamp, last_train_date: pd.Timestamp) -> bool:
    if last_train_date is None:
        return True
    if freq == 'daily':
        return True
    elif freq == 'monthly':
        return current_date.month != last_train_date.month
    elif freq == 'quarterly':
        return (current_date.month - 1) // 3 != (last_train_date.month - 1) // 3
    elif freq == 'yearly':
        return current_date.year != last_train_date.year
    return False

# =============================================================================
# 交易回測引擎 (與 backtest_nvda 對齊)
# =============================================================================
class HGBDailyFollowBacktester:
    def __init__(self, ticker, args):
        self.ticker = ticker
        self.yearly_injection = 2400
        self.capital_pool = 0.0
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.injection_log = []
        
        # 參數
        self.threshold_normal = args.threshold_normal
        self.threshold_risk = args.threshold_risk
        self.buy_ratio_normal = args.buy_ratio_normal
        self.buy_ratio_risk = args.buy_ratio_risk
        
        # 停損停利 (給定預設值)
        self.hard_stop_pct = -0.08
        self.trailing_activation = 0.20
        self.high_profit_thr = 0.25
        self.callback_base = 0.08
        self.callback_high = 0.17
        
    def run(self, signals_df: pd.DataFrame, start_date: str, end_date: str) -> dict:
        df = signals_df[
            (signals_df['date'] >= pd.Timestamp(start_date)) & 
            (signals_df['date'] <= pd.Timestamp(end_date))
        ].copy()
        
        if len(df) == 0:
            return None
            
        dates = df['date'].tolist()
        closes = df['price'].values
        
        self.capital_pool = self.yearly_injection
        self.injection_log.append({'date': dates[0], 'amount': self.yearly_injection, 'type': 'initial'})
        
        current_year = dates[0].year
        year_first_day_processed = {current_year: True}
        
        for i in tqdm(range(len(df)), desc=f"  Trading {self.ticker}", leave=False):
            date = dates[i]
            price = closes[i]
            row = df.iloc[i]
            
            # 年度資金注入
            if date.year != current_year:
                current_year = date.year
                if current_year not in year_first_day_processed:
                    self.capital_pool += self.yearly_injection
                    self.injection_log.append({'date': date, 'amount': self.yearly_injection, 'type': 'yearly'})
                    year_first_day_processed[current_year] = True
                    
            # 計算淨值
            position_value = sum(p['shares'] * price for p in self.positions)
            current_value = self.capital_pool + position_value
            self.equity_curve.append({
                'date': date, 'value': current_value, 'capital_pool': self.capital_pool, 'position_value': position_value
            })
            
            # 持倉管理檢查
            positions_to_remove = []
            for idx, pos in enumerate(self.positions):
                buy_price = pos['buy_price']
                current_return = price / buy_price - 1
                highest_return = pos['highest_price'] / buy_price - 1
                drawdown_from_high = (pos['highest_price'] - price) / pos['highest_price']
                
                if price > pos['highest_price']:
                    pos['highest_price'] = price
                    
                sell_reason = None
                if current_return <= self.hard_stop_pct:
                    sell_reason = "Hard Stop"
                elif highest_return >= self.trailing_activation:
                    cb_limit = self.callback_high if highest_return >= self.high_profit_thr else self.callback_base
                    if drawdown_from_high >= cb_limit:
                        sell_reason = "Trailing Stop"
                        
                if sell_reason:
                    sell_value = pos['shares'] * price
                    profit = sell_value - pos['cost']
                    self.capital_pool += sell_value
                    
                    self.trades.append({
                        'buy_date': pos['buy_date'],
                        'buy_price': buy_price,
                        'sell_date': date,
                        'sell_price': price,
                        'shares': pos['shares'],
                        'cost': pos['cost'],
                        'sell_value': sell_value,
                        'return': current_return,
                        'profit': profit,
                        'hold_days': (date - pos['buy_date']).days,
                        'exit_reason': sell_reason,
                        'confidence': pos['confidence']
                    })
                    positions_to_remove.append(idx)
                    
            for idx in sorted(positions_to_remove, reverse=True):
                self.positions.pop(idx)
                
            # 判斷買入
            action = row['action']
            buy_ratio = row['buy_ratio']
            
            if action == 'BUY' and buy_ratio > 0 and self.capital_pool > 0:
                invest_amount = self.capital_pool * buy_ratio
                if invest_amount >= price:
                    shares = invest_amount / price
                    cost = shares * price
                    self.capital_pool -= cost
                    self.positions.append({
                        'shares': shares,
                        'buy_price': price,
                        'buy_date': date,
                        'cost': cost,
                        'highest_price': price,
                        'confidence': row['p_t']
                    })
                    
        return self._generate_results(df)
        
    def _generate_results(self, df):
        if not self.equity_curve:
            return None
            
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        total_injected = sum(l['amount'] for l in self.injection_log)
        
        final_price = df.iloc[-1]['price']
        position_value = sum(p['shares'] * final_price for p in self.positions)
        final_value = self.capital_pool + position_value
        total_return = (final_value - total_injected) / total_injected
        
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.0
        cagr = (final_value / total_injected) ** (1 / years) - 1 if years > 0 else 0
        
        daily_ret = equity_df['value'].pct_change().dropna()
        sharpe = (daily_ret.mean() * 252 - 0.02) / (daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 0 and daily_ret.std() > 0 else 0
        
        roll_max = equity_df['value'].cummax()
        mdd = ((equity_df['value'] - roll_max) / roll_max).min()
        
        win_rate = sum(1 for t in self.trades if t['return'] > 0) / len(self.trades) if self.trades else 0
        
        return {
            'ticker': self.ticker,
            'total_injected': total_injected,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'mdd': mdd,
            'trades': len(self.trades),
            'win_rate': win_rate,
            'equity_df': equity_df,
            'trades_list': self.trades
        }

# =============================================================================
# 主程式
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-path", type=str, default="configs/rolling_profiles.json")
    parser.add_argument("--tickers", type=str, nargs='+', default=[])
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--retrain-freq", type=str, default="monthly", choices=["daily", "monthly", "quarterly", "yearly"])
    parser.add_argument("--threshold-normal", type=float, default=0.90)
    parser.add_argument("--threshold-risk", type=float, default=0.95)
    parser.add_argument("--buy-ratio-normal", type=float, default=0.25)
    parser.add_argument("--buy-ratio-risk", type=float, default=0.15)
    parser.add_argument("--risk-mode", type=str, default="proxy_regime")
    parser.add_argument("--output-dir", type=str, default="backtest_results_hgb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    profiles = {}
    if os.path.exists(args.profiles_path):
        with open(args.profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
    tickers = args.tickers
    if not tickers:
        tickers = [k for k in profiles.keys() if k != "default"]
        if not tickers:
            tickers = ["NVDA", "MSFT", "AAPL"]
            
    # Load Benchmark globally
    start_buffer_date = (pd.Timestamp(args.start) - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    print("📥 Loading benchmark data...")
    benchmark_df = load_or_update_local_csv(BENCHMARK)
    benchmark_df['date'] = pd.to_datetime(benchmark_df['Date'] if 'Date' in benchmark_df.columns else benchmark_df.index)
    
    regime_df = compute_regime_features(benchmark_df)
    regime_df['date_str'] = pd.to_datetime(regime_df['date']).dt.strftime('%Y-%m-%d')
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_summaries = []
    
    for tk in tickers:
        print(f"\n========================================")
        print(f"🚀 Processing {tk}...")
        print(f"========================================")
        tk_dir = os.path.join(args.output_dir, tk, f"run_{run_timestamp}")
        os.makedirs(tk_dir, exist_ok=True)
        
        # Load profile
        profile, profile_name = get_ticker_profile(tk, profiles)
        target_days = profile.get("target_days", profiles.get("default", {}).get("target_days", 120))
        target_return = profile.get("target_return", profiles.get("default", {}).get("target_return", 0.20))
        window_years = profile.get("window_years", profiles.get("default", {}).get("window_years", 3))
        regime_profile = profile.get("regime_profile", profiles.get("default", {}).get("regime_profile", "bm_only"))
        hgb_preset = profile.get("hgb_reg_preset", profiles.get("default", {}).get("hgb_reg_preset", "default"))
        
        target_col = f"Next_{target_days}d_Max"
        
        active_cols = FEATURE_COLS.copy()
        if regime_profile in ["bm_only", "bm_plus_stock"]:
            active_cols += REGIME_COLS
        if regime_profile == "bm_plus_stock":
            active_cols += STOCK_REGIME_COLS
            
        # Write params
        params_dict = {
            "profile_name": profile_name, "target_days": target_days, "target_return": target_return,
            "window_years": window_years, "regime_profile": regime_profile, "hgb_preset": hgb_preset,
            "retrain_freq": args.retrain_freq, "threshold_normal": args.threshold_normal,
            "threshold_risk": args.threshold_risk, "buy_ratio_normal": args.buy_ratio_normal,
            "buy_ratio_risk": args.buy_ratio_risk, "seed": args.seed
        }
        with open(os.path.join(tk_dir, "params.json"), "w") as f:
            json.dump(params_dict, f, indent=4)
            
        # Load Ticker Data
        raw_df = load_or_update_local_csv(tk)
        if raw_df is None or len(raw_df) == 0:
            print(f"⚠️ {tk} data missing, skip.")
            continue
            
        feat_df = calculate_features(raw_df, benchmark_df, ticker=tk, use_cache=not args.no_cache)
        if target_col not in feat_df.columns:
            print(f"⚠️ Target col {target_col} missing in {tk}, skip.")
            continue
            
        feat_df = feat_df.reset_index()
        if 'Date' in feat_df.columns: feat_df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'index' in feat_df.columns: feat_df.rename(columns={'index': 'date'}, inplace=True)
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        
        if regime_profile in ["bm_only", "bm_plus_stock"]:
            feat_df['date_str'] = feat_df['date'].dt.strftime('%Y-%m-%d')
            feat_df = pd.merge(feat_df, regime_df, left_on='date_str', right_on='date_str', how='left', suffixes=('', '_reg'))
            
            if regime_profile == "bm_plus_stock":
                df_stock_regime = compute_stock_regime_features(raw_df, benchmark_df)
                df_stock_regime['date_str'] = pd.to_datetime(df_stock_regime['date']).dt.strftime('%Y-%m-%d')
                feat_df = pd.merge(feat_df, df_stock_regime, left_on='date_str', right_on='date_str', how='left', suffixes=('', '_stk'))
                
        # Fill NA safely from merge dropping if needed (handled by train_slice dropping later)
        
        # Test slice
        test_df = feat_df[
            (feat_df['date'] >= pd.Timestamp(args.start)) & 
            (feat_df['date'] <= pd.Timestamp(args.end))
        ].copy().sort_values('date').reset_index(drop=True)
        
        if len(test_df) == 0:
            print(f"⚠️ No data in test period for {tk}.")
            continue
            
        signals = []
        current_model = None
        last_train_date = None
        model_version_id = 0
        
        print("  Generating daily signals (Walk-forward)...")
        for i, row in tqdm(test_df.iterrows(), total=len(test_df), leave=False):
            t_date = row['date']
            
            if is_retrain_needed(args.retrain_freq, t_date, last_train_date):
                train_start = t_date - pd.Timedelta(days=window_years * 365)
                train_slice = feat_df[(feat_df['date'] >= train_start) & (feat_df['date'] < t_date)].copy()
                train_slice = train_slice.dropna(subset=active_cols + [target_col])
                
                if len(train_slice) > 50: # min samples
                    y_train = (train_slice[target_col] >= target_return).astype(int)
                    current_model = HistGradientBoostingClassifier(random_state=args.seed)
                    if hgb_preset == 'regularized':
                        current_model.set_params(min_samples_leaf=50, max_depth=3, l2_regularization=0.1)
                    current_model.fit(train_slice[active_cols], y_train)
                    last_train_date = t_date
                    model_version_id += 1
                else:
                    # Not enough data to retrain, keep using bad/old model or skip
                    if current_model is None:
                        continue # can't do anything yet
                        
            if current_model is None:
                continue
                
            # Predict today
            x_today = pd.DataFrame([row])[active_cols].fillna(0) # quick fix if any na
            p_today = float(get_positive_proba(current_model, x_today)[0][0])
            
            # Predict history past 252 days
            past_252_start = t_date - pd.Timedelta(days=400) # buffer for trading days
            hist_slice = feat_df[(feat_df['date'] >= past_252_start) & (feat_df['date'] < t_date)]
            hist_slice = hist_slice.dropna(subset=active_cols).tail(252)
            
            if len(hist_slice) > 0:
                hist_scores = get_positive_proba(current_model, hist_slice[active_cols])[0]
                pct_rank_today = percentileofscore(hist_scores, p_today) / 100.0
            else:
                pct_rank_today = 0.5 # default mid if no history
                
            # Risk evaluate
            is_high_risk = evaluate_regime_risk_from_row(row)
            act_thresh = args.threshold_risk if is_high_risk else args.threshold_normal
            
            if pct_rank_today >= act_thresh:
                action = 'BUY'
                buy_ratio = args.buy_ratio_risk if is_high_risk else args.buy_ratio_normal
            else:
                action = 'SKIP_RISK' if is_high_risk else 'WATCH'
                buy_ratio = 0.0
                
            signals.append({
                'date': t_date,
                'price': row['Close'],
                'p_t': p_today,
                'pct_rank_t': pct_rank_today,
                'is_high_risk': is_high_risk,
                'action': action,
                'buy_ratio': buy_ratio,
                'model_version_id': model_version_id
            })
            
        signals_df = pd.DataFrame(signals)
        if len(signals_df) == 0:
            print(f"⚠️  No valid signals generated for {tk}. Skipping backtest.")
            continue
            
        signals_df.to_csv(os.path.join(tk_dir, "daily_signals.csv"), index=False)
        
        # Run Backtest
        print("  Running backtester engine...")
        backtester = HGBDailyFollowBacktester(tk, args)
        res = backtester.run(signals_df, args.start, args.end)
        
        if res:
            res['equity_df'].to_csv(os.path.join(tk_dir, "equity_curve.csv"))
            pd.DataFrame(res['trades_list']).to_csv(os.path.join(tk_dir, "trades.csv"), index=False)
            
            with open(os.path.join(tk_dir, "summary.json"), "w") as f:
                json.dump({k: v for k, v in res.items() if k not in ['equity_df', 'trades_list']}, f, indent=4)
                
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(res['equity_df'].index, res['equity_df']['value'], label=f"HGB Follow ({res['total_return']:.1%})")
            ax.set_title(f"{tk} HGB Walk-Forward Daily Follow ({args.start} ~ {args.end})")
            ax.legend()
            plt.savefig(os.path.join(tk_dir, "equity_curve.png"), dpi=100)
            plt.close()
            
            all_summaries.append({
                "ticker": tk, "profile": profile_name, "trades": res['trades'],
                "cagr": res['cagr'], "sharpe": res['sharpe'], "mdd": res['mdd'], "win_rate": res['win_rate']
            })
            print(f"✅  {tk} done: CAGR={res['cagr']:.1%}, Sharpe={res['sharpe']:.2f}")

    if all_summaries:
        agg_df = pd.DataFrame(all_summaries)
        agg_path = os.path.join(args.output_dir, f"all_tickers_summary_{run_timestamp}.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"\n🎉 All tests completed. Summary saved to {agg_path}")

if __name__ == "__main__":
    main()
