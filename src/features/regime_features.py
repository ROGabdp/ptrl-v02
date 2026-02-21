import numpy as np
import pandas as pd

REGIME_COLS = [
    'REGIME_BM_ABOVE_MA200',
    'REGIME_BM_MA200_SLOPE',
    'REGIME_BM_HV20',
    'REGIME_BM_HV20_PCTL',
    'REGIME_BM_RET_60',
    'REGIME_BM_RET_120'
]

def compute_regime_features(benchmark_df):
    """
    計算 Benchmark 大盤的 Regime Features (當下可得的 proxy metrics)，供機器學習模型感知市場狀態。
    注意：此處設計不得洩漏未來與個股走勢，純從大盤資料萃取。
    """
    df = benchmark_df.copy()
    
    # 確保按照日期遞增排序
    if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        
    prices = df['Close']
    
    # === 1. MA200 & Above Flag ===
    ma200 = prices.rolling(window=200, min_periods=100).mean()
    # 0 = Below, 1 = Above
    df['REGIME_BM_ABOVE_MA200'] = np.where(prices > ma200, 1.0, 0.0)
    
    # MA200 因早期 100 天為 NaN，會讓這段期間都是 0，我們保持原本的 NaN 以便後續 Dropna
    df.loc[ma200.isna(), 'REGIME_BM_ABOVE_MA200'] = np.nan
    
    # === 2. MA200 Slope (20日變動差值比例) ===
    # 這裡計算 t 日與 t-20 日的 MA200 差值 (%) 作為趨勢的斜率 proxy
    df['REGIME_BM_MA200_SLOPE'] = (ma200 / ma200.shift(20)) - 1.0
    
    # === 3. HV20 (Historical Volatility) ===
    # 20日對數報酬標準差年化
    log_ret = np.log(prices / prices.shift(1))
    hv20 = log_ret.rolling(window=20, min_periods=16).std() * np.sqrt(252)
    df['REGIME_BM_HV20'] = hv20
    
    # === 4. HV20 Percentile Rank (過去 3 年, 約 756 天) ===
    # 用 Pandas 原生的 rolling rank 計算 percentile
    # 若資料不足 252 天 (1年) 則保留 NaN (避免過於早期的分位數意義不清楚)
    df['REGIME_BM_HV20_PCTL'] = hv20.rolling(window=756, min_periods=252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
    )
    
    # === 5. 中長期大盤報酬 (Momentum) ===
    df['REGIME_BM_RET_60'] = (prices / prices.shift(60)) - 1.0
    df['REGIME_BM_RET_120'] = (prices / prices.shift(120)) - 1.0
    
    # 整理日期與輸出欄位
    df_out = df[REGIME_COLS].copy()
    df_out['date'] = pd.to_datetime(df_out.index).strftime('%Y-%m-%d')
    
    # 確保沒有 inf
    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df_out.dropna()
