import numpy as np
import pandas as pd

def compute_gate_features(benchmark_df):
    """
    根據 benchmark_df (raw ohlcv)，計算出供 Gate 判斷的特徵。
    需要:
      - trend_down: Close < MA200
      - high_vol: HV20 > median(HV20)
      - negative_mom: Return_120d < 0
    """
    df = benchmark_df.copy()
    
    # 算 MA200
    df['MA200'] = df['Close'].rolling(window=200, min_periods=100).mean()
    
    # 計算 Historical Volatility 20 (HV20)
    # 取 log return 算出 20 日標準差再年化 (* sqrt(252))
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HV20'] = df['log_ret'].rolling(window=20, min_periods=10).std() * np.sqrt(252)
    
    # 計算 Return 120d
    df['Ret_120d'] = (df['Close'] / df['Close'].shift(120)) - 1.0
    
    # 計算中位數 (作為 threshold) - 若為了避免 future leak，可以改 rolling median
    # 但為「最小投入評估」，我們直接拿目前資料區間的全局或 2年 rolling median 都可以。
    # 這裡採用嚴謹點的 252 日 rolling median 避免未來資料
    df['HV20_median'] = df['HV20'].rolling(window=252, min_periods=120).median()
    # 如果初期的資料不夠，可以用 expanding median
    df['HV20_median'] = df['HV20_median'].fillna(df['HV20'].expanding().median())
    
    # Boolean 旗標 (Reversal 的條件)
    df['trend_down'] = df['Close'] < df['MA200']
    df['high_vol']   = df['HV20'] > df['HV20_median']
    df['negative_mom'] = df['Ret_120d'] < 0
    
    # 正規化 date 為字串 (如果它原本是 datetime index 或欄位)
    if 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    else:
        # 直接拿 index 當 date
        df['date'] = pd.to_datetime(df.index)
        
    return df[['date', 'Close', 'MA200', 'HV20', 'HV20_median', 'Ret_120d', 
               'trend_down', 'high_vol', 'negative_mom']].dropna()

def apply_regime_gates(df_features):
    """
    給定包含特徵旗標的 DataFrame，產生對應的四種 Gate 的判定結果。
    回傳 'normal' 或 'reversal'。
    """
    df = df_features.copy()
    
    # Gate A: Trend
    # trend_down = reversal 
    df['Gate_A'] = np.where(df['trend_down'], 'reversal', 'normal')
    
    # Gate B: Volatility
    # high_vol = reversal
    df['Gate_B'] = np.where(df['high_vol'], 'reversal', 'normal')
    
    # Gate C: Momentum
    # negative_mom = reversal
    df['Gate_C'] = np.where(df['negative_mom'], 'reversal', 'normal')
    
    # Gate D: Combo
    # if (trend_down AND high_vol) OR (negative_mom AND high_vol)
    combo_reversal = (df['trend_down'] & df['high_vol']) | (df['negative_mom'] & df['high_vol'])
    df['Gate_D'] = np.where(combo_reversal, 'reversal', 'normal')
    
    return df
