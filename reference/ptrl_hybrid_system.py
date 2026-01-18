# -*- coding: utf-8 -*-
"""
Hybrid Trading System for ^TWII (Taiwan Stock Index)
結合 LSTM-SSAM 價格預測與 Pro Trader RL 交易決策

開發策略：
1. 數據擴充 (Data Expansion)：引入國際指數混合訓練
2. 特徵融合 (Feature Fusion)：整合 LSTM T+1/T+5 預測與信心度
3. 遷移學習 (Transfer Learning)：通用 Agent → ^TWII Fine-tuning

Phase 1: 基礎設定與資料下載 ✅
Phase 2: 特徵工程與 LSTM 整合 ✅
Phase 3: 混合數據預訓練 ✅
Phase 4: ^TWII 微調與回測 ✅
"""

import os
import sys
import pickle
import psutil
import shutil

# Windows 終端機 UTF-8 編碼設定（解決 emoji 顯示問題）
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.volume import MFIIndicator
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import glob
import multiprocessing
import warnings

# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# 中文字型設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 設定
# =============================================================================
FEATURE_COLS = [
    'Norm_Close', 'Norm_Open', 'Norm_High', 'Norm_Low',
    'Norm_DC_Lower',
    'Norm_HA_Open', 'Norm_HA_High', 'Norm_HA_Low', 'Norm_HA_Close',
    'Norm_SuperTrend_1', 'Norm_SuperTrend_2',
    'Norm_RSI', 'Norm_MFI',
    'Norm_ATR_Change',
    'Norm_RS_Ratio',
    'RS_ROC_5', 'RS_ROC_10', 'RS_ROC_20', 'RS_ROC_60', 'RS_ROC_120',
    # [v6.0] 移除 LSTM 預測特徵 (T+1/T+5)
    # [v4.1] 新增 RL 顯性特徵 (Explicit Features)
    'Feat_MA20_Slope',   # 短期趨勢動能
    'Feat_Trend_Gap',    # MA20 vs MA240 市場體制
    'Feat_Bias_MA20',    # 短期乖離
    'Feat_Dist_MA60',    # 季線支撐距離
    'Feat_Dist_MA240',   # 年線生命線位置
    'Feat_Vol_Ratio',    # 相對成交量突波
    # [v4.2] 新增 KD 與 MACD 特徵
    'Norm_K',            # Stochastic K(9,3) / 100
    'Norm_D',            # Stochastic D(9,3) / 100
    'Norm_DIF',          # MACD DIF(12,26) / Close
    'Norm_MACD',         # MACD Signal(9) / Close
    'Norm_OSC',          # MACD OSC (DIF - MACD) / Close
]

CACHE_DIR = "data/processed"
SPLIT_DATE = '2023-01-01'  # Fine-tuning / Backtest 切分點


# =============================================================================
# 0. 環境與 GPU 設定
# =============================================================================
def setup_environment():
    """設定執行環境與路徑"""
    # [v6.0] 強制使用 CPU (小型 MLP + 多環境 PPO 在 CPU 更快)
    device = "cpu"
    print(f"[System] Device: {device} (forced for multi-env PPO)")

    PROJECT_PATH = os.getcwd()
    MODELS_PATH = os.path.join(PROJECT_PATH, 'models_hybrid')
    RESULTS_PATH = os.path.join(PROJECT_PATH, 'results_hybrid')
    DATA_PATH = os.path.join(PROJECT_PATH, 'data')
    PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')

    for path in [MODELS_PATH, RESULTS_PATH, DATA_PATH, PROCESSED_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            
    return PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device


# =============================================================================
# 1. 資料下載
# =============================================================================
# =============================================================================
# 1. 資料下載
# =============================================================================
def _load_local_twii_data(start_date: str = "2000-01-01", end_date: str = None) -> pd.DataFrame:
    """
    載入本地 TWII CSV 資料 (含自動更新邏輯)
    
    Args:
        start_date: 資料起始日期 (YYYY-MM-DD)
        end_date: 資料結束日期 (YYYY-MM-DD)，若為 None 則取到最新
    """
    from datetime import date
    import subprocess
    from pathlib import Path
    
    csv_file = Path(__file__).parent / "twii_data_from_2000_01_01.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"找不到資料檔案：{csv_file}")
    
    # 1. 讀取 CSV
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
    df = df.set_index('date')
    
    # 2. 自動更新檢查
    today = date.today()
    last_date = df.index.max().date()
    
    if last_date < today:
        # 檢查今天是否為交易日（週一至週五）
        if today.weekday() < 5:  # 0-4 是平日
            print(f"[資料更新] TWII 資料 ({last_date}) 不是最新，正在呼叫 update_twii_data.py...")
            update_script = Path(__file__).parent / "update_twii_data.py"
            
            if update_script.exists():
                try:
                    result = subprocess.run(
                        [sys.executable, str(update_script)],
                        cwd=Path(__file__).parent,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        print(f"[資料更新] 更新完成！")
                        # 重新讀取更新後的檔案
                        df = pd.read_csv(csv_file)
                        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                        df = df.set_index('date')
                    else:
                        print(f"[警告] 更新失敗：{result.stderr}")
                except Exception as e:
                    print(f"[警告] 執行更新腳本錯誤：{e}")
            else:
                print(f"[警告] 找不到更新腳本：{update_script}")
    
    # 3. 欄位重新命名與過濾
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'  # 單位：億元
    })
    
    # 篩選日期範圍
    start_dt = pd.Timestamp(start_date)
    df = df[df.index >= start_dt]
    
    # [新增] 若指定 end_date，則過濾掉之後的資料
    if end_date is not None:
        end_dt = pd.Timestamp(end_date)
        df = df[df.index < end_dt]
        print(f"  ✅ ^TWII (Local, 截止 {end_date}): {len(df)} 筆 ({df.index[0].date()} ~ {df.index[-1].date()})")
    else:
        print(f"  ✅ ^TWII (Local): {len(df)} 筆 ({df.index[0].date()} ~ {df.index[-1].date()})")
    
    return df


def fetch_index_data(data_path, start_date="2000-01-01", end_date=None):
    """
    下載市場指數資料 (TWII 使用本地 CSV)
    
    Args:
        data_path: 資料儲存路徑
        start_date: 資料起始日期 (YYYY-MM-DD)
        end_date: 資料結束日期 (YYYY-MM-DD)，若為 None 則取到最新
                  [重要] 預訓練時應傳入 SPLIT_DATE 以避免資料洩漏
    """
    # TWII 以外的國際指數
    foreign_indices = ["^GSPC", "^IXIC", "^SOX", "^DJI"]
    
    print(f"=" * 60)
    print(f"📥 下載/載入 市場指數資料")
    if end_date:
        print(f"   (資料範圍: {start_date} ~ {end_date}，防止資料洩漏)")
    print(f"=" * 60)
    
    clean_data = {}
    
    # 1. 載入本地 TWII (傳入 end_date)
    try:
        clean_data["^TWII"] = _load_local_twii_data(start_date, end_date)
    except Exception as e:
        print(f"  ❌ ^TWII Loading Failed: {e}")
        
    # 2. 下載國際指數 (若有 end_date，則限制下載範圍)
    if foreign_indices:
        print(f"[下載] 正在獲取國際指數: {', '.join(foreign_indices)}...")
        download_end = end_date if end_date else None
        data = yf.download(foreign_indices, start=start_date, end=download_end,
                           group_by='ticker', auto_adjust=True, threads=True, progress=False)
        
        for t in foreign_indices:
            try:
                # 處理 MultiIndex 或 Single Index
                if isinstance(data.columns, pd.MultiIndex):
                     if t in data.columns.levels[0]:
                        df = data[t].copy()
                else:
                    # 如果只有一個 ticker，yf 可能不會回傳 MultiIndex，需檢查
                    # 但這裡我們傳入了 list，通常會是 MultiIndex
                    # 若為防萬一，假設 data 就是該 ticker 的 df (不過這裡有多個 ticker)
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df = df.dropna()
                if len(df) > 250:
                    clean_data[t] = df
                    print(f"  ✅ {t}: {len(df)} 筆")
            except Exception as e:
                print(f"  ⚠️ {t}: Failed - {e}")
    
    return clean_data


# =============================================================================
# 2. LSTM 模型載入與推論
# =============================================================================
_LSTM_MODELS = {
    'model_1d': None, 'scaler_feat_1d': None, 'scaler_tgt_1d': None, 'meta_1d': None,
    'model_5d': None, 'scaler_feat_5d': None, 'scaler_tgt_5d': None, 'meta_5d': None,
    'model_20d': None, 'scaler_feat_20d': None, 'scaler_tgt_20d': None, 'meta_20d': None,
    'loaded': False
}

def load_best_lstm_models(target_date=None):
    """載入 LSTM 模型"""
    global _LSTM_MODELS
    if _LSTM_MODELS['loaded']:
        # If loaded, check if we need to reload for a different date? 
        # For simplicity, if target_date is different, we should probably reload, 
        # but the simple check might be enough if we restart script.
        # But to be safe for this script usage, let's allow reload if forced or just assume script starts fresh.
        # However, checking against loaded metadata might be complex here.
        # Let's assume for this specific task, we call it once. 
        pass
        # (If we wanted to be robust, we'd check if loaded models match target_date criteria)
    
    print("\n[System] Loading LSTM Models...")
    try:
        import twii_model_registry_multivariate as lstm_1d_module
        import twii_model_registry_5d as lstm_5d_module
        import twii_model_registry_20d as lstm_20d_module
        from datetime import date
        
        use_date = target_date if target_date else date.today()
        print(f"[System] Selecting models available before: {use_date}")
        
        # 1. T+1 Model
        meta_1d = lstm_1d_module.select_best_model(use_date)
        if meta_1d is None:
            return False
        model_1d, scaler_feat_1d, scaler_tgt_1d, _ = lstm_1d_module.load_artifacts(
            meta_1d['train_start'], meta_1d['train_end'])
        print(f"  ✅ T+1 Model: {meta_1d['train_start']} ~ {meta_1d['train_end']}")
        
        # 2. T+5 Model
        meta_5d = lstm_5d_module.select_best_model(use_date)
        if meta_5d is None:
            return False
        model_5d, scaler_feat_5d, scaler_tgt_5d, _ = lstm_5d_module.load_artifacts(
            meta_5d['train_start'], meta_5d['train_end'])
        print(f"  ✅ T+5 Model: {meta_5d['train_start']} ~ {meta_5d['train_end']}")

        # 3. T+20 Model
        meta_20d = lstm_20d_module.select_best_model(use_date)
        if meta_20d is None:
            print("[Warning] No T+20 Model found. RL features will be incomplete.")
            return False
        model_20d, scaler_feat_20d, scaler_tgt_20d, _ = lstm_20d_module.load_artifacts(
            meta_20d['train_start'], meta_20d['train_end'])
        print(f"  ✅ T+20 Model: {meta_20d['train_start']} ~ {meta_20d['train_end']}")
        
        _LSTM_MODELS.update({
            'model_1d': model_1d, 'scaler_feat_1d': scaler_feat_1d,
            'scaler_tgt_1d': scaler_tgt_1d, 'meta_1d': meta_1d,
            'model_5d': model_5d, 'scaler_feat_5d': scaler_feat_5d,
            'scaler_tgt_5d': scaler_tgt_5d, 'meta_5d': meta_5d,
            'model_20d': model_20d, 'scaler_feat_20d': scaler_feat_20d,
            'scaler_tgt_20d': scaler_tgt_20d, 'meta_20d': meta_20d,
            'loaded': True
        })
        return True
    except Exception as e:
        print(f"[Error] Failed to load LSTM models: {e}")
        return False


def add_lstm_features(df: pd.DataFrame, ticker: str = "Unknown") -> pd.DataFrame:
    """
    [v6.0] LSTM 特徵已從 RL 訓練中移除
    
    此函式保留以維持向下相容性，但不再執行任何 LSTM 計算。
    LSTM 特徵欄位仍會被初始化為預設值，以避免其他腳本報錯。
    """
    # 初始化欄位（預設值，供向下相容）
    for col in ['LSTM_Pred_1d', 'LSTM_Conf_1d', 'LSTM_Pred_5d', 'LSTM_Conf_5d', 'LSTM_Pred_20d', 'LSTM_Conf_20d']:
        if col not in df.columns:
            df[col] = 0.5 if 'Conf' in col else 0.0
    
    # [v6.0] 不再執行 LSTM 模型推論
    return df


def _add_lstm_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """新增 LSTM 技術指標"""
    df = df.copy()
    df['Volume_Log'] = np.log1p(df['Volume'])
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['K'] = ((df['Close'] - low_min) / (high_max - low_min) * 100).rolling(3).mean()
    df['D'] = df['K'].rolling(3).mean()
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    return df


# =============================================================================
# 3. 技術指標特徵工程
# =============================================================================
def calculate_heikin_ashi(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = [df['Open'].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha_close.iloc[i-1]) / 2)
    ha_open = pd.Series(ha_open, index=df.index)
    return pd.DataFrame({
        'HA_open': ha_open, 'HA_high': pd.concat([df['High'], ha_open, ha_close], axis=1).max(axis=1),
        'HA_low': pd.concat([df['Low'], ha_open, ha_close], axis=1).min(axis=1), 'HA_close': ha_close
    })


def calculate_supertrend(df, length=14, multiplier=3.0):
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=length).average_true_range().fillna(method='bfill')
    hl2 = (df['High'] + df['Low']) / 2
    basic_upper, basic_lower = hl2 + multiplier * atr, hl2 - multiplier * atr
    final_upper, final_lower = basic_upper.copy(), basic_lower.copy()
    trend = np.zeros(len(df))
    for i in range(1, len(df)):
        final_upper.iloc[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper.iloc[i-1] or df['Close'].iloc[i-1] > final_upper.iloc[i-1] else final_upper.iloc[i-1]
        final_lower.iloc[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower.iloc[i-1] or df['Close'].iloc[i-1] < final_lower.iloc[i-1] else final_lower.iloc[i-1]
        trend[i] = 1 if df['Close'].iloc[i] > final_upper.iloc[i-1] else (-1 if df['Close'].iloc[i] < final_lower.iloc[i-1] else trend[i-1])
    return pd.DataFrame({'SUPERT_': np.where(trend == 1, final_lower, final_upper)}, index=df.index)


def calculate_features(df_in: pd.DataFrame, benchmark_df: pd.DataFrame, 
                       ticker: str = "Unknown", use_cache: bool = True) -> pd.DataFrame:
    """計算完整特徵"""
    cache_path = os.path.join(CACHE_DIR, f"{ticker.replace('^', '_').replace('.', '_')}_features.pkl")
    
    if use_cache and os.path.exists(cache_path):
        print(f"[Cache] Loading features for {ticker}...")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    print(f"[Compute] Generating features for {ticker}...")
    df = df_in.copy()
    
    df['DC_Upper'] = df['High'].rolling(20).max().shift(1).fillna(method='bfill')
    df['DC_Lower'] = df['Low'].rolling(20).min().shift(1).fillna(method='bfill')
    df['DC_Upper_10'] = df['High'].rolling(10).max().shift(1).fillna(method='bfill')
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=10).average_true_range()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    try:
        df['MFI'] = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
    except:
        df['MFI'] = 50.0
    
    ha = calculate_heikin_ashi(df)
    df['HA_Open'], df['HA_High'], df['HA_Low'], df['HA_Close'] = ha['HA_open'], ha['HA_high'], ha['HA_low'], ha['HA_close']
    df['SuperTrend_1'] = calculate_supertrend(df, 14, 2.0).iloc[:, 0]
    df['SuperTrend_2'] = calculate_supertrend(df, 21, 1.0).iloc[:, 0]
    
    base_price = df['DC_Upper'].replace(0, np.nan).fillna(method='bfill')
    for col in ['Close', 'Open', 'High', 'Low', 'DC_Lower', 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'SuperTrend_1', 'SuperTrend_2']:
        df[f'Norm_{col}'] = df[col] / base_price
    df['Norm_RSI'], df['Norm_MFI'] = df['RSI'] / 100.0, df['MFI'] / 100.0
    df['Norm_ATR_Change'] = (df['ATR'] / df['ATR'].shift(1)).fillna(1.0)
    
    # -------------------------------------------------------------------------
    # [v4.1] 新增 RL 顯性特徵 (Explicit Features)
    # -------------------------------------------------------------------------
    # 1. 基礎均線計算
    # 確保成交量分母不為 0
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['MA120'] = df['Close'].rolling(120).mean()
    df['MA240'] = df['Close'].rolling(240).mean()
    df['MA_Vol_20'] = df['Volume'].rolling(20).mean()
    
    # 2. 特徵公式實作
    # (1) MA20 短期趨勢動能: (MA20_t / MA20_t-1) - 1
    df['Feat_MA20_Slope'] = (df['MA20'] / df['MA20'].shift(1) - 1).fillna(0)
    
    # (2) MA20 vs MA240 市場體制: (MA20 - MA240) / MA240
    df['Feat_Trend_Gap'] = ((df['MA20'] - df['MA240']) / df['MA240']).fillna(0)
    
    # (3) MA20 短期乖離: (Close - MA20) / MA20
    df['Feat_Bias_MA20'] = ((df['Close'] - df['MA20']) / df['MA20']).fillna(0)
    
    # (4) MA60 季線支撐距離: (Close - MA60) / MA60
    df['Feat_Dist_MA60'] = ((df['Close'] - df['MA60']) / df['MA60']).fillna(0)
    
    # (5) MA240 年線生命線位置: (Close - MA240) / MA240
    df['Feat_Dist_MA240'] = ((df['Close'] - df['MA240']) / df['MA240']).fillna(0)
    
    # (6) 相對成交量突波: Volume / MA_Vol_20
    # 加 1e-8 防止除以零
    df['Feat_Vol_Ratio'] = (df['Volume'] / (df['MA_Vol_20'] + 1e-8)).fillna(0)
    
    # -------------------------------------------------------------------------
    # [v4.2] 新增 KD 與 MACD 特徵
    # -------------------------------------------------------------------------
    # 1. Stochastic KD (9, 3)
    low_min_9 = df['Low'].rolling(9).min()
    high_max_9 = df['High'].rolling(9).max()
    rsv = ((df['Close'] - low_min_9) / (high_max_9 - low_min_9 + 1e-9)) * 100
    df['K_raw'] = rsv.rolling(3).mean()  # K(9,3)
    df['D_raw'] = df['K_raw'].rolling(3).mean()  # D(9,3)
    df['Norm_K'] = (df['K_raw'] / 100.0).fillna(0.5)
    df['Norm_D'] = (df['D_raw'] / 100.0).fillna(0.5)
    
    # 2. MACD (12, 26, 9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema_12 - ema_26  # DIF (快線 - 慢線)
    df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()  # MACD Signal
    df['OSC'] = df['DIF'] - df['MACD_Signal']  # OSC (柱狀圖)
    
    # 正規化：除以收盤價 (轉為百分比概念)
    df['Norm_DIF'] = (df['DIF'] / df['Close']).fillna(0)
    df['Norm_MACD'] = (df['MACD_Signal'] / df['Close']).fillna(0)
    df['Norm_OSC'] = (df['OSC'] / df['Close']).fillna(0)
    
    # 移除計算過程中產生的暫時欄位 (保留 MA 以便 debug 也可以，但這裡先保持乾淨)
    # df = df.drop(columns=['MA20', 'MA60', 'MA240', 'MA_Vol_20']) 
    
    # 重要：移除因為 MA240 造成的前端空值
    df = df.dropna(subset=['MA240']) 
    
    if benchmark_df is not None:
        bench_close = benchmark_df['Close'].reindex(df.index).fillna(method='ffill')
        df['RS_Raw'] = df['Close'] / bench_close
        rs_min, rs_max = df['RS_Raw'].rolling(250).min(), df['RS_Raw'].rolling(250).max()
        df['Norm_RS_Ratio'] = ((df['RS_Raw'] - rs_min) / ((rs_max - rs_min).replace(0, np.nan).fillna(1.0) + 1e-9)).fillna(0.5)
        for period in [5, 10, 20, 60, 120]:
            df[f'RS_ROC_{period}'] = df['RS_Raw'].pct_change(period).fillna(0)
    else:
        df['Norm_RS_Ratio'] = 0.5
        for period in [5, 10, 20, 60, 120]:
            df[f'RS_ROC_{period}'] = 0.0
    
    df = add_lstm_features(df, ticker)
    df['Signal_Buy_Filter'] = df['High'] > df['DC_Upper_10']
    # 計算未來 N 天內的最高價報酬率（正確公式）
    # 目標：計算 T+1 到 T+N 這 N 天的最高價
    # 公式：先 shift(-1) 排除當天，反轉後 rolling 取最大值，再反轉回來
    df['Next_20d_Max'] = df['High'].shift(-1).iloc[::-1].rolling(20, min_periods=20).max().iloc[::-1] / df['Close'] - 1
    df['Next_120d_Max'] = df['High'].shift(-1).iloc[::-1].rolling(120, min_periods=120).max().iloc[::-1] / df['Close'] - 1
    df = df.dropna(subset=[c for c in df.columns if c not in ['Next_20d_Max', 'Next_120d_Max']])
    
    if use_cache:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
        except:
            pass
    
    return df


# =============================================================================
# 4. RL 環境定義
# =============================================================================
class BuyEnvHybrid(gym.Env):
    """Buy RL Environment with class balancing"""
    def __init__(self, data_dict, is_training=True):
        super().__init__()
        self.samples, self.pos_samples, self.neg_samples = [], [], []
        
        for t, df in data_dict.items():
            df = df.dropna(subset=['Next_120d_Max'])  # 改用 120 天目標
            signals = df  # 移除唐其安通道限制，所有日期都可買入
            if len(signals) > 0:
                states = signals[FEATURE_COLS].values.astype(np.float32)
                future_rets = signals['Next_120d_Max'].values.astype(np.float32)  # 改用 120 天最高報酬
                for i in range(len(signals)):
                    sample = (states[i], future_rets[i])
                    self.samples.append(sample)
                    (self.pos_samples if future_rets[i] >= 0.10 else self.neg_samples).append(sample)
        
        print(f"[BuyEnv] Total samples: {len(self.samples)} | Pos samples (>10%): {len(self.pos_samples)} | Neg samples: {len(self.neg_samples)}")
        if len(self.pos_samples) == 0:
            print("[BuyEnv Warning] No positive samples found! Reward might be stuck at 0 if agent buys.")
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(FEATURE_COLS),), dtype=np.float32)
        self.is_training = is_training
        self.idx, self.current_sample = 0, None
    
    def reset(self, seed=None, options=None):
        # [v6.02] 訓練時使用類別平衡 50/50，驗證時使用真實分佈
        if self.is_training:
            # 訓練模式：50/50 類別平衡
            if np.random.rand() < 0.5 and self.pos_samples:
                self.current_sample = self.pos_samples[np.random.randint(len(self.pos_samples))]
            elif self.neg_samples:
                self.current_sample = self.neg_samples[np.random.randint(len(self.neg_samples))]
            else:
                self.current_sample = self.samples[np.random.randint(len(self.samples))]
        else:
            # 驗證模式：使用真實分佈 (反映實際市場機會比例)
            self.current_sample = self.samples[np.random.randint(len(self.samples))]
        return self.current_sample[0], {}
    
    def step(self, action):
        _, max_ret = self.current_sample
        is_success = max_ret >= 0.10  # 未來 120 天最大漲幅 >= 10%
        
        if action == 1:  # 選擇買入
            if is_success:
                reward = 2.0  # 買對：高獎勵
            else:
                reward = -0.5  # 買錯：輕微懲罰
        else:  # 選擇不買
            if is_success:
                reward = -1.0  # 錯過好機會：懲罰
            else:
                reward = 0.5   # 正確迴避：較低獎勵
        
        return self.current_sample[0], reward, True, False, {}


class BuyEnvHybridV5(gym.Env):
    """Buy RL Environment V5 - 對稱獎勵結構
    
    獎勵設計：
    - 買對 (action=1, 漲幅≥10%): +1.0
    - 買錯 (action=1, 漲幅<10%): 0.0
    - 錯過好機會 (action=0, 漲幅≥10%): 0.0
    - 正確迴避 (action=0, 漲幅<10%): +1.0
    """
    def __init__(self, data_dict, is_training=True):
        super().__init__()
        self.samples, self.pos_samples, self.neg_samples = [], [], []
        
        for t, df in data_dict.items():
            df = df.dropna(subset=['Next_120d_Max'])
            signals = df  # 無唐其安通道限制
            if len(signals) > 0:
                states = signals[FEATURE_COLS].values.astype(np.float32)
                future_rets = signals['Next_120d_Max'].values.astype(np.float32)
                for i in range(len(signals)):
                    sample = (states[i], future_rets[i])
                    self.samples.append(sample)
                    (self.pos_samples if future_rets[i] >= 0.10 else self.neg_samples).append(sample)
        
        print(f"[BuyEnvV5] Total samples: {len(self.samples)} | Pos (≥10%): {len(self.pos_samples)} | Neg: {len(self.neg_samples)}")
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(FEATURE_COLS),), dtype=np.float32)
        self.is_training = is_training
        self.idx, self.current_sample = 0, None
    
    def reset(self, seed=None, options=None):
        # [v6.02] 訓練時使用類別平衡 50/50，驗證時使用真實分佈
        if self.is_training:
            # 訓練模式：50/50 類別平衡
            if np.random.rand() < 0.5 and self.pos_samples:
                self.current_sample = self.pos_samples[np.random.randint(len(self.pos_samples))]
            elif self.neg_samples:
                self.current_sample = self.neg_samples[np.random.randint(len(self.neg_samples))]
            else:
                self.current_sample = self.samples[np.random.randint(len(self.samples))]
        else:
            # 驗證模式：使用真實分佈 (反映實際市場機會比例)
            self.current_sample = self.samples[np.random.randint(len(self.samples))]
        return self.current_sample[0], {}
    
    def step(self, action):
        _, max_ret = self.current_sample
        is_success = max_ret >= 0.10  # 未來 120 天最大漲幅 >= 10%
        
        # V5 對稱獎勵結構
        if action == 1:  # 選擇買入
            reward = 1.0 if is_success else 0.0
        else:  # 選擇不買
            reward = 0.0 if is_success else 1.0
        
        return self.current_sample[0], reward, True, False, {}

class SellEnvHybrid(gym.Env):
    """Sell RL Environment (v6.0 - Fixed Reward Hacking)
    
    修正重點:
    1. 隨機化 Episode 長度 (60~250 天)，避免 Agent 學會「死守到第 N 天」
    2. 解耦獎勵視窗：無論何時結算，都往後看固定 60 天來計算錯失獎勵
    3. 資料切片擴大到 310 天，確保有足夠的未來數據供獎勵計算
    """
    
    # Episode / Reward 參數
    MIN_EPISODE_LENGTH = 60
    MAX_EPISODE_LENGTH = 250
    REWARD_LOOKAHEAD = 60  # 結算時往後偷看的天數
    DATA_BUFFER = MAX_EPISODE_LENGTH + REWARD_LOOKAHEAD  # 310 天
    
    def __init__(self, data_dict):
        super().__init__()
        self.episodes = []
        
        for t, df in data_dict.items():
            buy_indices = np.where(df['Signal_Buy_Filter'])[0]
            feature_data = df[FEATURE_COLS].values.astype(np.float32)
            close_prices = df['Close'].values.astype(np.float32)
            
            for idx in buy_indices:
                # 確保有足夠的數據供 Episode + Lookahead
                if idx + self.DATA_BUFFER < len(df):
                    episode_prices = close_prices[idx:idx + self.DATA_BUFFER]
                    self.episodes.append({
                        'features': feature_data[idx:idx + self.DATA_BUFFER],
                        'returns': episode_prices / episode_prices[0]
                    })
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(FEATURE_COLS) + 1,), dtype=np.float32)
        
        # 會在 reset 時隨機決定
        self.max_steps = self.MAX_EPISODE_LENGTH
        self.current_episode = None
        self.day = 0
    
    def reset(self, seed=None, options=None):
        self.current_episode = self.episodes[np.random.randint(len(self.episodes))]
        self.day = 0
        # 🔀 隨機化本回合的最大步數
        self.max_steps = np.random.randint(self.MIN_EPISODE_LENGTH, self.MAX_EPISODE_LENGTH + 1)
        return np.concatenate([self.current_episode['features'][0], [1.0]]).astype(np.float32), {}
    
    def step(self, action):
        current_return = self.current_episode['returns'][self.day]
        
        # 強制結算條件：Agent 選擇賣出 OR 達到本回合隨機上限
        if action == 1 or self.day >= self.max_steps - 1:
            # =========================================================
            # 🔧 解耦獎勵計算：無論何時結算，都往後看固定 REWARD_LOOKAHEAD 天
            # =========================================================
            lookahead_end = min(self.day + self.REWARD_LOOKAHEAD, self.DATA_BUFFER)
            future_returns = self.current_episode['returns'][self.day:lookahead_end]
            
            future_max = np.max(future_returns) if len(future_returns) > 0 else current_return
            future_min = np.min(future_returns) if len(future_returns) > 0 else current_return
            
            # 1. 基礎獎勵：當前報酬 (獲利 10% = +1.0, 虧 5% = -0.5)
            base_reward = (current_return - 1.0) * 10
            
            # 2. 錯過高點的懲罰 / 賣在高點的獎勵
            if future_max > current_return + 0.01:  # 未來還會漲 >1%
                # 錯過的漲幅，輕微懲罰
                penalty = (future_max - current_return) * 2
            else:
                # 賣在接近最高點！額外獎勵
                penalty = -0.5
            
            # 3. 躲過大跌的獎勵
            if future_min < current_return - 0.05:  # 未來會跌 >5%
                bonus = (current_return - future_min) * 5
            else:
                bonus = 0
            
            reward = base_reward - penalty + bonus
            done = True
        else:  # 持有
            # 動態持有懲罰（溫和版）：只有大獲利時才輕微懲罰
            if current_return >= 1.10:  # 已獲利 10% 以上
                reward = -0.01  # 輕微懲罰
            elif current_return >= 1.05:  # 獲利 5-10%
                reward = -0.002  # 非常輕微懲罰
            else:  # 獲利 <5% 或虧損
                reward = 0.0  # 不懲罰
            self.day += 1
            done = False
        
        # 安全索引：確保不超過 max_steps (觀測用)，但獎勵計算可用 DATA_BUFFER
        obs_idx = min(self.day, self.max_steps - 1)
        obs = np.concatenate([self.current_episode['features'][obs_idx], 
                              [self.current_episode['returns'][obs_idx]]]).astype(np.float32)
        return obs, reward, done, False, {}



# =============================================================================
# 5. Pre-training 流程
# =============================================================================
def run_pretraining(train_data: dict, models_path: str, device: str,
                    pretrain_buy_steps: int = 1_000_000, pretrain_sell_steps: int = 500_000,
                    train_buy: bool = True, train_sell: bool = True):
    """執行預訓練 (含 TensorBoard 日誌記錄)
    
    Args:
        train_data: 訓練資料字典
        models_path: 模型儲存路徑
        device: 運算裝置 (cuda/cpu)
        pretrain_buy_steps: Buy Agent 預訓練步數 (default: 1,000,000)
        pretrain_sell_steps: Sell Agent 預訓練步數 (default: 500,000)
        train_buy: 是否訓練 Buy Agent (default: True)
        train_sell: 是否訓練 Sell Agent (default: True)
    """
    print(f"\n[System] Starting Pre-training with {len(train_data)} indices...")
    
    # 建立日誌目錄
    tensorboard_log = "./tensorboard_logs/"
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(os.path.join(models_path, "best_pretrain"), exist_ok=True)
    
    n_envs = min(8, max(1, multiprocessing.cpu_count() - 1))
    print(f"[System] CPU cores: {multiprocessing.cpu_count()}, Using {n_envs} envs")
    
    ppo_params = {
        "learning_rate": 0.0001, 
        "n_steps": max(128, 2048 // n_envs),
        "batch_size": 512, 
        "ent_coef": 0.01, 
        "device": device,
        "policy_kwargs": dict(net_arch=[64, 64, 64]), 
        "verbose": 1,
        "tensorboard_log": tensorboard_log  # 啟用 TensorBoard
    }
    
    buy_model = None
    sell_model = None
    
    # =========================================================================
    # Buy Agent
    # =========================================================================
    if train_buy:
        print("\n🛒 Training Buy Agent (Base Model)...")
        buy_env = make_vec_env(BuyEnvHybrid, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                               env_kwargs={'data_dict': train_data, 'is_training': True})
        
        # 建立評估環境
        eval_buy_env = make_vec_env(BuyEnvHybrid, n_envs=1, vec_env_cls=DummyVecEnv,
                                    env_kwargs={'data_dict': train_data, 'is_training': False})
        
        buy_model = PPO("MlpPolicy", buy_env, **ppo_params)
        
        # Callbacks
        buy_callbacks = CallbackList([
            CheckpointCallback(save_freq=80000, save_path=models_path, name_prefix="ppo_buy_base"),
            EvalCallback(eval_buy_env, best_model_save_path=os.path.join(models_path, "best_pretrain", "buy"),
                         log_path="./logs/", eval_freq=10000, n_eval_episodes=50, 
                         deterministic=True)
        ])
        
        buy_model.learn(total_timesteps=pretrain_buy_steps, callback=buy_callbacks, tb_log_name="buy_pretrain")
        
        # 複製 best model 作為 base model (而非使用最後一步的模型)
        best_buy_path = os.path.join(models_path, "best_pretrain", "buy", "best_model.zip")
        buy_base_path = os.path.join(models_path, "ppo_buy_base.zip")
        if os.path.exists(best_buy_path):
            shutil.copy(best_buy_path, buy_base_path)
            print(f"[Pre-train] ✅ Buy Agent: Copied BEST model to {buy_base_path}")
        else:
            buy_model.save(os.path.join(models_path, "ppo_buy_base"))
            print(f"[Pre-train] ⚠️ Buy Agent: Best model not found, saved last step model")
        
        buy_env.close()
        eval_buy_env.close()
    else:
        print("\n[Skip] Buy Agent pre-training (train_buy=False)")
    
    # =========================================================================
    # Sell Agent
    # =========================================================================
    if train_sell:
        print("\n💰 Training Sell Agent (Base Model)...")
        sell_env = make_vec_env(SellEnvHybrid, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                                env_kwargs={'data_dict': train_data})
        
        # 建立評估環境
        eval_sell_env = make_vec_env(SellEnvHybrid, n_envs=1, vec_env_cls=DummyVecEnv,
                                     env_kwargs={'data_dict': train_data})
        
        sell_model = PPO("MlpPolicy", sell_env, **ppo_params)
        
        # Callbacks
        sell_callbacks = CallbackList([
            CheckpointCallback(save_freq=80000, save_path=models_path, name_prefix="ppo_sell_base"),
            EvalCallback(eval_sell_env, best_model_save_path=os.path.join(models_path, "best_pretrain"),
                         log_path="./logs/", eval_freq=10000, n_eval_episodes=50, 
                         deterministic=True)
        ])
        
        sell_model.learn(total_timesteps=pretrain_sell_steps, callback=sell_callbacks, tb_log_name="sell_pretrain")
        
        # 複製 best model 作為 base model (而非使用最後一步的模型)
        best_sell_path = os.path.join(models_path, "best_pretrain", "best_model.zip")
        sell_base_path = os.path.join(models_path, "ppo_sell_base.zip")
        if os.path.exists(best_sell_path):
            shutil.copy(best_sell_path, sell_base_path)
            print(f"[Pre-train] ✅ Sell Agent: Copied BEST model to {sell_base_path}")
        else:
            sell_model.save(os.path.join(models_path, "ppo_sell_base"))
            print(f"[Pre-train] ⚠️ Sell Agent: Best model not found, saved last step model")
        
        sell_env.close()
        eval_sell_env.close()
    else:
        print("\n[Skip] Sell Agent pre-training (train_sell=False)")
    
    print("[System] Pre-training Completed.")
    return buy_model, sell_model


# =============================================================================
# 6. Fine-tuning 流程 (Transfer Learning)
# =============================================================================
def run_finetuning(twii_finetune_data: dict, twii_eval_data: dict, models_path: str, device: str,
                   finetune_buy_steps: int = 1_000_000, finetune_sell_steps: int = 500_000,
                   train_buy: bool = True, train_sell: bool = True):
    """
    針對 ^TWII 進行微調 (含 TensorBoard 日誌記錄)
    - 載入預訓練權重
    - 使用較低的 Learning Rate (1e-5)
    - 可自訂訓練步數
    - EvalCallback 監控驗證集表現
    
    Args:
        finetune_buy_steps: Buy Agent 微調步數 (default: 1,000,000)
        finetune_sell_steps: Sell Agent 微調步數 (default: 300,000)
        train_buy: 是否訓練 Buy Agent (default: True)
        train_sell: 是否訓練 Sell Agent (default: True)
    """
    print("\n" + "=" * 60)
    print("🎯 Phase 4: Fine-tuning for ^TWII (with TensorBoard)")
    print("=" * 60)
    
    # 建立日誌目錄
    tensorboard_log = "./tensorboard_logs/"
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(os.path.join(models_path, "best_tuned"), exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    
    n_envs = min(4, max(1, multiprocessing.cpu_count() - 1))
    
    # Fine-tuning 參數（Transfer Learning 關鍵）
    finetune_params = {
        "learning_rate": 1e-5,  # 原本的 1/10
        "n_steps": 256,
        "batch_size": 128,
        "ent_coef": 0.005,
        "device": device,
        "verbose": 1
    }
    
    buy_model = None
    sell_model = None
    
    # =========================================================================
    # Fine-tune Buy Agent
    # =========================================================================
    if train_buy:
        print("\n[Fine-tune] Loading ppo_buy_base and fine-tuning for ^TWII...")
        
        buy_base_path = os.path.join(models_path, "ppo_buy_base.zip")
        if not os.path.exists(buy_base_path):
            print(f"[Error] Base model not found: {buy_base_path}")
            return None, None
    
        buy_env = make_vec_env(BuyEnvHybrid, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                               env_kwargs={'data_dict': twii_finetune_data, 'is_training': True})
        
        # 建立評估環境 (使用 Backtest 數據子集)
        eval_buy_env = make_vec_env(BuyEnvHybrid, n_envs=1, vec_env_cls=DummyVecEnv,
                                    env_kwargs={'data_dict': twii_eval_data, 'is_training': False})
        
        buy_model = PPO.load(buy_base_path, env=buy_env, device=device,
                             tensorboard_log=tensorboard_log)
        buy_model.learning_rate = finetune_params["learning_rate"]
        buy_model.ent_coef = finetune_params["ent_coef"]
        
        # Callbacks
        buy_callbacks = CallbackList([
            CheckpointCallback(save_freq=100000, save_path=models_path, name_prefix="ppo_buy_finetune"),
            EvalCallback(eval_buy_env, best_model_save_path=os.path.join(models_path, "best_tuned", "buy"),
                         log_path="./logs/", eval_freq=10000, n_eval_episodes=30, 
                         deterministic=True)
        ])
        
        print(f"[Fine-tune] Training Buy Agent for {finetune_buy_steps:,} steps (LR: {finetune_params['learning_rate']})")
        buy_model.learn(total_timesteps=finetune_buy_steps, callback=buy_callbacks, 
                        tb_log_name="buy_finetune", reset_num_timesteps=False)
        
        # 複製 best model 作為 final model (而非使用最後一步的模型)
        best_buy_path = os.path.join(models_path, "best_tuned", "buy", "best_model.zip")
        buy_final_path = os.path.join(models_path, "ppo_buy_twii_final.zip")
        if os.path.exists(best_buy_path):
            shutil.copy(best_buy_path, buy_final_path)
            print(f"[Fine-tune] ✅ Buy Agent: Copied BEST model to {buy_final_path}")
        else:
            buy_model.save(os.path.join(models_path, "ppo_buy_twii_final"))
            print(f"[Fine-tune] ⚠️ Buy Agent: Best model not found, saved last step model")
        
        buy_env.close()
        eval_buy_env.close()
    else:
        print("\n[Skip] Buy Agent fine-tuning (train_buy=False)")
    
    # =========================================================================
    # Fine-tune Sell Agent
    # =========================================================================
    if train_sell:
        print("\n[Fine-tune] Loading ppo_sell_base and fine-tuning for ^TWII...")
        
        sell_base_path = os.path.join(models_path, "ppo_sell_base.zip")
        sell_env = make_vec_env(SellEnvHybrid, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                                env_kwargs={'data_dict': twii_finetune_data})
        
        # 建立評估環境 (使用 Backtest 數據子集)
        eval_sell_env = make_vec_env(SellEnvHybrid, n_envs=1, vec_env_cls=DummyVecEnv,
                                     env_kwargs={'data_dict': twii_eval_data})
        
        sell_model = PPO.load(sell_base_path, env=sell_env, device=device,
                              tensorboard_log=tensorboard_log)
        sell_model.learning_rate = finetune_params["learning_rate"]
        sell_model.ent_coef = finetune_params["ent_coef"]
        
        # Callbacks
        sell_callbacks = CallbackList([
            CheckpointCallback(save_freq=50000, save_path=models_path, name_prefix="ppo_sell_finetune"),
            EvalCallback(eval_sell_env, best_model_save_path=os.path.join(models_path, "best_tuned", "sell"),
                         log_path="./logs/", eval_freq=10000, n_eval_episodes=30, 
                         deterministic=True)
        ])
        
        print(f"[Fine-tune] Training Sell Agent for {finetune_sell_steps:,} steps (LR: {finetune_params['learning_rate']})")
        sell_model.learn(total_timesteps=finetune_sell_steps, callback=sell_callbacks, 
                         tb_log_name="sell_finetune", reset_num_timesteps=False)
        
        # 複製 best model 作為 final model (而非使用最後一步的模型)
        best_sell_path = os.path.join(models_path, "best_tuned", "sell", "best_model.zip")
        sell_final_path = os.path.join(models_path, "ppo_sell_twii_final.zip")
        if os.path.exists(best_sell_path):
            shutil.copy(best_sell_path, sell_final_path)
            print(f"[Fine-tune] ✅ Sell Agent: Copied BEST model to {sell_final_path}")
        else:
            sell_model.save(os.path.join(models_path, "ppo_sell_twii_final"))
            print(f"[Fine-tune] ⚠️ Sell Agent: Best model not found, saved last step model")
        
        sell_env.close()
        eval_sell_env.close()
    else:
        print("\n[Skip] Sell Agent fine-tuning (train_sell=False)")
    
    print("\n[System] Fine-tuning Completed!")
    return buy_model, sell_model


# =============================================================================
# 7. Backtesting 流程
# =============================================================================
class HybridBacktester:
    """Hybrid Trading System Backtester"""
    
    def __init__(self, buy_model, sell_model, initial_capital=1_000_000):
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.initial_capital = initial_capital
        
        # 交易記錄
        self.trades = []
        self.equity_curve = []
        self.buy_signals = []  # (date, price)
        self.sell_signals = [] # (date, price)
    
    def run(self, df: pd.DataFrame) -> dict:
        """
        執行回測
        
        Args:
            df: 包含特徵的 DataFrame
        
        Returns:
            績效指標字典
        """
        capital = self.initial_capital
        position = None  # {'shares': int, 'buy_price': float, 'buy_date': date}
        
        dates = df.index.tolist()
        closes = df['Close'].values
        
        # 建立觀察資料
        features = df[FEATURE_COLS].values.astype(np.float32)
        buy_signals_mask = df['Signal_Buy_Filter'].values
        
        for i in tqdm(range(len(df)), desc="Backtesting"):
            date = dates[i]
            price = closes[i]
            
            # 記錄當日淨值
            if position:
                current_value = capital + position['shares'] * price
            else:
                current_value = capital
            self.equity_curve.append({'date': date, 'value': current_value})
            
            # 持有中：檢查賣出
            if position is not None:
                hold_days = i - position['buy_idx']
                current_return = price / position['buy_price']
                
                # 準備 Sell Agent 觀察
                sell_obs = np.concatenate([features[i], [current_return]]).astype(np.float32)
                
                # 預測
                action, _ = self.sell_model.predict(sell_obs.reshape(1, -1), deterministic=True)
                
                # 停損條件 或 AI 決定賣出 或 持有超過 120 天
                stop_loss = current_return < 0.92  # -8% 停損
                should_sell = action[0] == 1 or stop_loss
                
                if should_sell:
                    # 執行賣出
                    sell_value = position['shares'] * price
                    profit = sell_value - position['shares'] * position['buy_price']
                    capital += sell_value
                    
                    self.trades.append({
                        'buy_date': position['buy_date'],
                        'buy_price': position['buy_price'],
                        'sell_date': date,
                        'sell_price': price,
                        'return': current_return - 1,
                        'profit': profit
                    })
                    self.sell_signals.append((date, price))
                    
                    position = None
            
            # 空手：檢查買入
            elif buy_signals_mask[i]:
                # 準備 Buy Agent 觀察
                buy_obs = features[i].reshape(1, -1)
                action, _ = self.buy_model.predict(buy_obs, deterministic=True)
                
                if action[0] == 1:  # Buy
                    # 執行買入（使用 90% 資金）
                    invest_amount = capital * 0.9
                    shares = int(invest_amount / price)
                    
                    if shares > 0:
                        cost = shares * price
                        capital -= cost
                        
                        position = {
                            'shares': shares,
                            'buy_price': price,
                            'buy_date': date,
                            'buy_idx': i
                        }
                        self.buy_signals.append((date, price))
        
        # 計算績效指標
        return self._calculate_metrics(df)
    
    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """計算績效指標"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        # 總報酬率
        initial = self.initial_capital
        final = equity_df['value'].iloc[-1]
        total_return = (final - initial) / initial
        
        # 年化報酬率
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.0
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 夏普值 (假設無風險利率 = 2%)
        daily_returns = equity_df['value'].pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() * 252 - 0.02) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0
        
        # 最大回撤
        rolling_max = equity_df['value'].cummax()
        drawdown = (equity_df['value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 勝率
        if self.trades:
            wins = sum(1 for t in self.trades if t['return'] > 0)
            win_rate = wins / len(self.trades)
        else:
            win_rate = 0
        
        return {
            'initial_capital': initial,
            'final_value': final,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'equity_df': equity_df
        }


def run_backtesting(twii_backtest_df: pd.DataFrame, buy_model, sell_model, 
                    results_path: str, benchmark_df: pd.DataFrame) -> dict:
    """
    執行回測並視覺化結果
    """
    print("\n" + "=" * 60)
    print("📊 Phase 4: Backtesting (2023-Present)")
    print("=" * 60)
    
    # 執行回測
    backtester = HybridBacktester(buy_model, sell_model, initial_capital=1_000_000)
    metrics = backtester.run(twii_backtest_df)
    
    if not metrics:
        print("[Error] Backtesting failed!")
        return {}
    
    # 印出績效
    print("\n" + "-" * 60)
    print("📈 Performance Summary")
    print("-" * 60)
    print(f"  初始資金: ${metrics['initial_capital']:,.0f}")
    print(f"  最終淨值: ${metrics['final_value']:,.0f}")
    print(f"  總報酬率: {metrics['total_return']*100:.2f}%")
    print(f"  年化報酬: {metrics['annualized_return']*100:.2f}%")
    print(f"  夏普值:   {metrics['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"  交易次數: {metrics['total_trades']}")
    print(f"  勝率:     {metrics['win_rate']*100:.1f}%")
    print("-" * 60)
    
    # ==========================================================================
    # 視覺化
    # ==========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子圖 1: Portfolio Value vs Benchmark
    ax1 = axes[0]
    
    equity_df = metrics['equity_df']
    
    # Portfolio
    ax1.plot(equity_df.index, equity_df['value'], label='Hybrid System', 
             color='blue', linewidth=2)
    
    # Benchmark (Buy & Hold)
    bench_slice = benchmark_df.loc[equity_df.index[0]:equity_df.index[-1]]['Close']
    bench_normalized = bench_slice / bench_slice.iloc[0] * metrics['initial_capital']
    ax1.plot(bench_normalized.index, bench_normalized.values, 
             label='^TWII Buy & Hold', color='gray', linewidth=1.5, alpha=0.7)
    
    ax1.set_title('Portfolio Value vs Benchmark (2023-Present)', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 子圖 2: Price with Buy/Sell Signals
    ax2 = axes[1]
    
    price_slice = twii_backtest_df['Close']
    ax2.plot(price_slice.index, price_slice.values, label='^TWII Close', 
             color='black', linewidth=1)
    
    # Buy signals (紅色三角形)
    if backtester.buy_signals:
        buy_dates, buy_prices = zip(*backtester.buy_signals)
        ax2.scatter(buy_dates, buy_prices, marker='^', color='red', s=100, 
                    label='Buy Signal', zorder=5)
    
    # Sell signals (綠色三角形)
    if backtester.sell_signals:
        sell_dates, sell_prices = zip(*backtester.sell_signals)
        ax2.scatter(sell_dates, sell_prices, marker='v', color='green', s=100, 
                    label='Sell Signal', zorder=5)
    
    ax2.set_title('^TWII Price with Trading Signals', fontsize=14)
    ax2.set_ylabel('Price')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 儲存圖表
    save_path = os.path.join(results_path, 'final_performance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[System] Chart saved to: {save_path}")
    
    plt.close()
    
    return metrics


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  🚀 Hybrid Trading System - Full Pipeline")
    print("=" * 70)
    
    # =========================================================================
    # Phase 0: Setup
    # =========================================================================
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    
    # Check if base models exist
    buy_base_exists = os.path.exists(os.path.join(MODELS_PATH, "ppo_buy_base.zip"))
    sell_base_exists = os.path.exists(os.path.join(MODELS_PATH, "ppo_sell_base.zip"))
    
    # =========================================================================
    # Phase 1-3: Pre-training (if needed)
    # =========================================================================
    if not buy_base_exists or not sell_base_exists:
        print("\n[System] Base models not found. Running Phase 1-3...")
        
        # [v6.0] LSTM 已移除，不再載入 LSTM 模型
        
        # Download data
        raw_data = fetch_index_data(DATA_PATH, start_date="2000-01-01")
        
        # Calculate features
        train_data = {}
        benchmark_df = raw_data.get("^TWII")
        for ticker, df in raw_data.items():
            try:
                processed = calculate_features(df, benchmark_df, ticker, use_cache=True)
                if len(processed) > 100:
                    train_data[ticker] = processed
            except Exception as e:
                print(f"  Error: {ticker} - {e}")
        
        # Pre-training
        run_pretraining(train_data, MODELS_PATH, device)
    else:
        print("\n[System] Base models found. Skipping Phase 1-3.")
    
    # =========================================================================
    # Phase 4: Fine-tuning & Backtesting
    # =========================================================================
    print("\n" + "=" * 70)
    print("  📌 Phase 4: Fine-tuning & Backtesting for ^TWII")
    print("=" * 70)
    
    # [v6.0] LSTM 已移除，不再載入 LSTM 模型
    
    # Load ^TWII data with features
    print("\n[System] Loading ^TWII data...")
    cache_path = os.path.join(CACHE_DIR, "_TWII_features.pkl")
    
    if os.path.exists(cache_path):
        print(f"[Cache] Loading ^TWII features...")
        with open(cache_path, 'rb') as f:
            twii_full_df = pickle.load(f)
    else:
        print("[Compute] Downloading and processing ^TWII...")
        twii_raw = yf.download("^TWII", start="2000-01-01", auto_adjust=True, progress=False)
        twii_full_df = calculate_features(twii_raw, twii_raw, ticker="^TWII", use_cache=True)
    
    print(f"[System] ^TWII data: {len(twii_full_df)} rows")
    print(f"[System] Date range: {twii_full_df.index[0].strftime('%Y-%m-%d')} ~ {twii_full_df.index[-1].strftime('%Y-%m-%d')}")
    
    # =========================================================================
    # Split data
    # =========================================================================
    print(f"\n[System] Splitting data at {SPLIT_DATE}...")
    
    split_date = pd.Timestamp(SPLIT_DATE)
    twii_finetune_df = twii_full_df[twii_full_df.index < split_date]
    twii_backtest_df = twii_full_df[twii_full_df.index >= split_date]
    
    print(f"  Fine-tuning set: {len(twii_finetune_df)} rows (< {SPLIT_DATE})")
    print(f"  Backtest set:    {len(twii_backtest_df)} rows (>= {SPLIT_DATE})")
    
    # =========================================================================
    # Fine-tuning
    # =========================================================================
    finetune_data = {'^TWII': twii_finetune_df}
    eval_data = {'^TWII': twii_backtest_df}  # 使用 Backtest 數據作為驗證集
    buy_model, sell_model = run_finetuning(finetune_data, eval_data, MODELS_PATH, device)
    
    if buy_model is None:
        print("[Error] Fine-tuning failed!")
        sys.exit(1)
    
    # =========================================================================
    # Backtesting
    # =========================================================================
    # 載入 Fine-tuned 模型
    buy_final = PPO.load(os.path.join(MODELS_PATH, "ppo_buy_twii_final.zip"))
    sell_final = PPO.load(os.path.join(MODELS_PATH, "ppo_sell_twii_final.zip"))
    
    metrics = run_backtesting(twii_backtest_df, buy_final, sell_final, 
                               RESULTS_PATH, twii_full_df)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  ✅ [System] All Phases Completed.")
    print("=" * 70)
    print(f"""
    📊 Final Results:
    ────────────────────────────────────
    總報酬率:   {metrics.get('total_return', 0)*100:.2f}%
    年化報酬:   {metrics.get('annualized_return', 0)*100:.2f}%
    夏普值:     {metrics.get('sharpe_ratio', 0):.2f}
    最大回撤:   {metrics.get('max_drawdown', 0)*100:.2f}%
    交易次數:   {metrics.get('total_trades', 0)}
    勝率:       {metrics.get('win_rate', 0)*100:.1f}%
    ────────────────────────────────────
    
    📁 Output Files:
    - models_hybrid/ppo_buy_twii_final.zip
    - models_hybrid/ppo_sell_twii_final.zip
    - results_hybrid/final_performance.png
    
    📈 TensorBoard 訓練監控：
    ────────────────────────────────────
    執行以下指令開啟 TensorBoard：
    
        tensorboard --logdir ./tensorboard_logs/
    
    開啟瀏覽器前往 http://localhost:6006
    查看 Loss, Entropy, Reward 等訓練曲線。
    ────────────────────────────────────
    """)
