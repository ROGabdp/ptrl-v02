import numpy as np
import pandas as pd

STOCK_REGIME_COLS = [
    'REGIME_STOCK_HV20',
    'REGIME_STOCK_HV20_PCTL',
    'REGIME_STOCK_RS120',
    'REGIME_EXTREME_DIST_MA240_FLAG'
]

def compute_stock_regime_features(df_stock, df_bm):
    """
    計算個股專屬的 Regime Features (例如 TSM 自己與相對於大盤的狀態)。
    包含:
      1. HV20 (近20日歷史波動率)
      2. HV20_PCTL (近3年 HV20 分位數)
      3. RS120 (近120日起源於大盤的相對報酬)
      4. EXTREME_DIST_MA240_FLAG (距離 MA240 突破近3年 P90 的極端標記)
    """
    df_out = pd.DataFrame(index=df_stock.index)
    
    if 'Date' in df_stock.columns:
        date_series = pd.to_datetime(df_stock['Date'])
        df_out['date'] = date_series.dt.strftime('%Y-%m-%d')
    elif 'date' in df_stock.columns:
        date_series = pd.to_datetime(df_stock['date'])
        df_out['date'] = date_series.dt.strftime('%Y-%m-%d')
    else:
        date_series = pd.to_datetime(df_stock.index)
        df_out['date'] = date_series.strftime('%Y-%m-%d')

    close = df_stock['Close']
    
    # 1. REGIME_STOCK_HV20
    log_ret = np.log(close / close.shift(1))
    hv20 = log_ret.rolling(window=20).std() * np.sqrt(252)
    df_out['REGIME_STOCK_HV20'] = hv20
    
    # 2. REGIME_STOCK_HV20_PCTL (756 days rolling rank)
    df_out['REGIME_STOCK_HV20_PCTL'] = hv20.rolling(window=756).apply(
        lambda x: (x <= x[-1]).sum() / len(x) if len(x) > 0 else np.nan, 
        raw=True
    )
    
    # 3. REGIME_STOCK_RS120
    # Align benchmark close to stock dates
    df_bm_aligned = pd.DataFrame({'bm_close': df_bm['Close']})
    if 'Date' in df_bm.columns:
        df_bm_aligned['date'] = pd.to_datetime(df_bm['Date']).dt.strftime('%Y-%m-%d')
    else:
        df_bm_aligned['date'] = pd.to_datetime(df_bm.index).strftime('%Y-%m-%d')
        
    merged = pd.merge(df_out[['date']], df_bm_aligned, on='date', how='left')
    bm_close = merged['bm_close'].values
    
    ret_stock_120 = close / close.shift(120) - 1
    
    # Avoid zero division or NA issues on bm_shift
    bm_close_series = pd.Series(bm_close, index=close.index)
    ret_bm_120 = bm_close_series / bm_close_series.shift(120) - 1
    
    df_out['REGIME_STOCK_RS120'] = ret_stock_120 - ret_bm_120
    
    # 4. REGIME_EXTREME_DIST_MA240_FLAG
    ma240 = close.rolling(window=240).mean()
    dist_ma240 = (close - ma240) / ma240
    abs_dist = dist_ma240.abs()
    
    p90_abs_dist = abs_dist.rolling(window=756).quantile(0.90)
    
    # 先預設 0，然後把大於 p90 的設為 1，但保留開頭 NaN 讓後續能 dropna
    is_extreme = (abs_dist > p90_abs_dist).astype(float)
    is_extreme.loc[p90_abs_dist.isna()] = np.nan
    df_out['REGIME_EXTREME_DIST_MA240_FLAG'] = is_extreme
    
    return df_out
