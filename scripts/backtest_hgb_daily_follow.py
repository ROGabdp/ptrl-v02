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

def resolve_bm_value(row: pd.Series, candidates: list) -> tuple:
    """
    從 row 中按優先序對每個候選欄位名嘗試取得非 NaN 的數字。
    回傳 (value, source_key, source_name)。
    source_name 必表2018\u5f0f之一: 'REGIME', 'REGIME_reg', 'BM', 'BM_reg', 'DEFAULT_MISSING'
    """
    for col in candidates:
        if col in row.index:
            val = row[col]
            if not pd.isna(val):
                # 判斷來源
                if col.startswith("REGIME_BM_") and col.endswith("_reg"):
                    src = "REGIME_reg"
                elif col.startswith("REGIME_BM_"):
                    src = "REGIME"
                elif col.startswith("BM_") and col.endswith("_reg"):
                    src = "BM_reg"
                elif col.startswith("BM_"):
                    src = "BM"
                else:
                    src = "UNKNOWN"
                return float(val), col, src
    return np.nan, None, "DEFAULT_MISSING"

def evaluate_regime_risk_from_row(row: pd.Series) -> tuple:
    """
    使用 resolve_bm_value 安全解析大盤風控指標。
    回傳：(is_high_risk, risk_reason, bm_above_ma200, bm_ret_120, bm_hv20_pctl, bm_source)
    bm_source = 三個指標各自的讀取來源，用 '|' 隔開，方便 debug。
    严禁使用無 BM_ 前綴的欄位（如 RET_120），避免誤用單股特徵進行判斷。
    """
    CANDIDATES = {
        "above_ma200": [
            "REGIME_BM_ABOVE_MA200",
            "REGIME_BM_ABOVE_MA200_reg",
            "BM_ABOVE_MA200",
            "BM_ABOVE_MA200_reg",
        ],
        "ret_120": [
            "REGIME_BM_RET_120",
            "REGIME_BM_RET_120_reg",
            "BM_RET_120",
            "BM_RET_120_reg",
        ],
        "hv20_pctl": [
            "REGIME_BM_HV20_PCTL",
            "REGIME_BM_HV20_PCTL_reg",
            "BM_HV20_PCTL",
            "BM_HV20_PCTL_reg",
        ],
    }
    
    above_ma, above_ma_key, above_ma_src = resolve_bm_value(row, CANDIDATES["above_ma200"])
    ret_120, ret_120_key, ret_120_src = resolve_bm_value(row, CANDIDATES["ret_120"])
    hv20_pctl, hv20_pctl_key, hv20_pctl_src = resolve_bm_value(row, CANDIDATES["hv20_pctl"])
    
    bm_source = f"{above_ma_src}|{ret_120_src}|{hv20_pctl_src}"
    
    # 三者任一為 NaN 就前視為缺失
    missing_keys = [k for k, v in [(above_ma_key, above_ma), (ret_120_key, ret_120), (hv20_pctl_key, hv20_pctl)] if np.isnan(v)]
    if missing_keys:
        missing_str = ",".join([k if k else "(not_found)" for k in missing_keys])
        return False, f"MISSING_BM_FEATURES:{missing_str}", np.nan, np.nan, np.nan, bm_source
    
    is_high_risk = False
    risk_reason = ""
    
    if above_ma == 0 and hv20_pctl > 0.8:
        is_high_risk = True
        risk_reason = "MA200_below_and_HVhigh"
    elif ret_120 < 0 and hv20_pctl > 0.8:
        is_high_risk = True
        risk_reason = "RET120_neg_and_HVhigh"
        
    return is_high_risk, risk_reason, above_ma, ret_120, hv20_pctl, bm_source

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
        self.buy_signals = []
        self.sell_signals = []
        
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
                    self.sell_signals.append((date, price, sell_reason))
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
                    self.buy_signals.append((date, price, 'BUY', row['p_t'], buy_ratio))
                    
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
            'trades_list': self.trades,
            'injection_log': self.injection_log,
            'buy_signals': self.buy_signals,
            'sell_signals': self.sell_signals
        }

# =============================================================================
# 繪圖與 Benchmark
# =============================================================================
def calculate_nasdaq_benchmark(benchmark_df: pd.DataFrame, start_date: str, end_date: str, yearly_injection: float = 2400) -> dict:
    """計算 Nasdaq 買入持有的基準 (同等年度資金注入)"""
    test_df = benchmark_df[
        (benchmark_df.index >= pd.Timestamp(start_date)) &
        (benchmark_df.index <= pd.Timestamp(end_date))
    ].copy()
    
    if len(test_df) == 0:
        return None
    
    dates = test_df.index.tolist()
    closes = test_df['Close'].values
    
    total_shares = 0.0
    total_invested = 0.0
    equity_curve = []
    
    current_year = dates[0].year
    year_first_day_processed = {current_year: True}
    
    if len(closes) > 0:
        initial_price = closes[0]
        initial_shares = yearly_injection / initial_price
        total_shares += initial_shares
        total_invested += yearly_injection
    
    for i, (date, price) in enumerate(zip(dates, closes)):
        if date.year != current_year:
            current_year = date.year
            if current_year not in year_first_day_processed:
                new_shares = yearly_injection / price
                total_shares += new_shares
                total_invested += yearly_injection
                year_first_day_processed[current_year] = True
        
        current_value = total_shares * price
        equity_curve.append({'date': date, 'value': current_value})
    
    if not equity_curve:
        return None
        
    equity_df = pd.DataFrame(equity_curve)
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df.set_index('date', inplace=True)
    
    final_value = total_shares * closes[-1]
    total_return = (final_value - total_invested) / total_invested
    return {'equity_df': equity_df, 'total_return': total_return}

def plot_equity_curve(result: dict, benchmark: dict, output_dir: str, ticker: str, start_date: str, end_date: str):
    """繪製具有 Benchmark、買賣點標記與資金注入線的淨值曲線圖"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    equity_df = result['equity_df']
    ax.plot(equity_df.index, equity_df['value'], 
            label=f"{ticker} HGB Follow ({result['total_return']:.0%})",
            linewidth=2, color='#4CAF50')
    
    if benchmark:
        bench_equity = benchmark['equity_df']
        ax.plot(bench_equity.index, bench_equity['value'],
                label=f"Nasdaq B&H ({benchmark['total_return']:.0%})",
                linewidth=2, linestyle='--', color='gray')
    
    # 資金注入標記
    for log in result['injection_log']:
        ax.axvline(x=log['date'], color='blue', linestyle=':', alpha=0.5)
        
    # 買入訊號
    if result.get('buy_signals'):
        buy_dates = [s[0] for s in result['buy_signals']]
        buy_values = [equity_df.loc[d, 'value'] if d in equity_df.index else None for d in buy_dates]
        valid = [(d, v) for d, v in zip(buy_dates, buy_values) if v is not None]
        if valid:
            dates, values = zip(*valid)
            ax.scatter(dates, values, marker='^', color='green', s=80, zorder=5, label='Buy')
            
    # 賣出訊號
    if result.get('sell_signals'):
        sell_dates = [s[0] for s in result['sell_signals']]
        sell_values = [equity_df.loc[d, 'value'] if d in equity_df.index else None for d in sell_dates]
        valid = [(d, v) for d, v in zip(sell_dates, sell_values) if v is not None]
        if valid:
            dates, values = zip(*valid)
            ax.scatter(dates, values, marker='v', color='red', s=80, zorder=5, label='Sell')
            
    ax.axhline(y=result['total_injected'], color='black', linestyle=':', alpha=0.3, 
               label=f"Total Injected (${result['total_injected']:,.0f})")
               
    ax.set_title(f'{ticker} HGB Walk-Forward Daily Follow ({start_date} ~ {end_date})', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

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
        
        # 為了讓風控 evaluate_regime_risk_from_row 總是能動，強制合併大盤 regime_df
        feat_df['date_str'] = feat_df['date'].dt.strftime('%Y-%m-%d')
        feat_df = pd.merge(feat_df, regime_df, left_on='date_str', right_on='date_str', how='left', suffixes=('', '_reg'))
        
        # 若有要求個股 regime，才合併 df_stock_regime
        if regime_profile == "bm_plus_stock":
            df_stock_regime = compute_stock_regime_features(raw_df, benchmark_df)
            df_stock_regime['date_str'] = pd.to_datetime(df_stock_regime['date']).dt.strftime('%Y-%m-%d')
            feat_df = pd.merge(feat_df, df_stock_regime, left_on='date_str', right_on='date_str', how='left', suffixes=('', '_stk'))
                
        # 欄位正規化對齊 (Strategy A: auto-fix _reg suffixes & aliases)
        # 本專案的真實特徵全名其實帶有 REGIME_ 前綴
        target_bms = ["REGIME_BM_ABOVE_MA200", "REGIME_BM_RET_120", "REGIME_BM_HV20_PCTL"]
        for bm_key in target_bms:
            if bm_key not in feat_df.columns:
                if f"{bm_key}_reg" in feat_df.columns:
                    feat_df[bm_key] = feat_df[f"{bm_key}_reg"]
                elif bm_key.replace("REGIME_", "") in feat_df.columns:
                    feat_df[bm_key] = feat_df[bm_key.replace("REGIME_", "")]
        
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
            x_today = pd.DataFrame([row])[active_cols]
            if x_today.isnull().any().any():
                signals.append({
                    'date': t_date,
                    'price': row['Close'],
                    'p_t': float('nan'),
                    'pct_rank_t': float('nan'),
                    'is_high_risk': False,
                    'action': 'SKIP_NO_FEATURES',
                    'buy_ratio': 0.0,
                    'model_version_id': model_version_id,
                    'hist_n': 0,
                    'retrain_date': last_train_date.strftime('%Y-%m-%d') if last_train_date else None
                })
                continue
                
            p_res = get_positive_proba(current_model, x_today)
            p_today = float((p_res[0] if isinstance(p_res, tuple) else p_res)[0])
            
            # Predict history past 252 days
            past_252_start = t_date - pd.Timedelta(days=400) # buffer for trading days
            hist_slice = feat_df[(feat_df['date'] >= past_252_start) & (feat_df['date'] < t_date)]
            hist_slice = hist_slice.dropna(subset=active_cols).tail(252)
            hist_n = len(hist_slice)
            
            if hist_n > 0:
                h_res = get_positive_proba(current_model, hist_slice[active_cols])
                hist_scores = h_res[0] if isinstance(h_res, tuple) else h_res
                pct_rank_today = percentileofscore(hist_scores, p_today) / 100.0
            else:
                pct_rank_today = 0.5 # default mid if no history
                
            # Risk evaluate (完全委託 evaluate_regime_risk_from_row，不做外部覆蓋)
            is_high_risk, risk_reason, bm_above_ma, bm_ret_120, bm_hv20_pctl, bm_src = evaluate_regime_risk_from_row(row)
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
                'risk_reason': risk_reason,
                'bm_source': bm_src,
                'bm_above_ma200': bm_above_ma,
                'bm_ret_120': bm_ret_120,
                'bm_hv20_pctl': bm_hv20_pctl,
                'act_thresh_used': act_thresh,
                'action': action,
                'buy_ratio': buy_ratio,
                'model_version_id': model_version_id,
                'hist_n': hist_n,
                'retrain_date': last_train_date.strftime('%Y-%m-%d') if last_train_date else None
            })
            
        signals_df = pd.DataFrame(signals)
        if len(signals_df) == 0:
            print(f"⚠️  No valid signals generated for {tk}. Skipping backtest.")
            continue
            
        high_risk_days = signals_df['is_high_risk'].sum()
        reasons_dist = signals_df[signals_df['risk_reason'] != '']['risk_reason'].value_counts().to_dict()
        bm_src_dist = signals_df['bm_source'].value_counts().to_dict() if 'bm_source' in signals_df else {}
        missing_count = signals_df['risk_reason'].str.startswith('MISSING_BM_FEATURES').sum()
        missing_pct = missing_count / len(signals_df) if len(signals_df) > 0 else 0
        
        print(f"  [Risk Logic Check] High Risk Days: {high_risk_days} / {len(signals_df)}")
        if reasons_dist:
            print(f"  [Risk Logic Check] Reasons: {reasons_dist}")
        print(f"  [Risk Logic Check] bm_source dist: {bm_src_dist}")
        print(f"  [Risk Logic Check] MISSING_BM_FEATURES: {missing_count} ({missing_pct:.1%})")
            
        if high_risk_days == 0 or missing_pct > 0.10:
            print(f"  ⚠️ WARNING: is_high_risk=0 or MISSING rate >{missing_pct:.1%}! Check BM Feats:")
            missing_check = ["REGIME_BM_ABOVE_MA200", "REGIME_BM_RET_120", "REGIME_BM_HV20_PCTL"]
            for mc in missing_check:
                if mc in test_df.columns:
                    print(f"      - {mc} isnull mean: {test_df[mc].isnull().mean():.2%}, unique: {test_df[mc].nunique()}")
                else:
                    print(f"      - {mc} NOT FOUND in test_df fields!")
                    
        signals_df.to_csv(os.path.join(tk_dir, "daily_signals.csv"), index=False)
        
        # Run Backtest
        print("  Running backtester engine...")
        backtester = HGBDailyFollowBacktester(tk, args)
        res = backtester.run(signals_df, args.start, args.end)
        
        if res:
            res['equity_df'].to_csv(os.path.join(tk_dir, "equity_curve.csv"))
            pd.DataFrame(res['trades_list']).to_csv(os.path.join(tk_dir, "trades.csv"), index=False)
            
            with open(os.path.join(tk_dir, "summary.json"), "w") as f:
                json.dump({k: v for k, v in res.items() if k not in ['equity_df', 'trades_list', 'injection_log', 'buy_signals', 'sell_signals']}, f, indent=4)
                
            # 計算 Benchmark
            bench_df_idx = benchmark_df.copy()
            if 'date' in bench_df_idx.columns:
                bench_df_idx.set_index('date', inplace=True)
            bench_res = calculate_nasdaq_benchmark(bench_df_idx, args.start, args.end)
            
            # 繪製美化的曲線圖
            plot_equity_curve(res, bench_res, tk_dir, tk, args.start, args.end)
            
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
