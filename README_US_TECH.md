# US Tech Stock Buy Agent - RL 交易訓練系統

使用強化學習 (Reinforcement Learning) 訓練 Buy Agent，辨識 10 隻美股科技股的「起漲點」。

## 目標標的

`NVDA`, `MSFT`, `AAPL`, `AMZN`, `META`, `AVGO`, `GOOGL`, `TSLA`, `NFLX`, `PLTR`

## 核心訓練策略

### 兩階段訓練

| 階段 | 說明 |
|------|------|
| **Phase 1: Pre-training** | 整合 10 隻股票的歷史數據進行大規模預訓練，建立通用科技股動能模型 |
| **Phase 2: Fine-tuning** | 針對每隻股票載入預訓練權重，進行個股微調 |

### 訓練/驗證期間

```
訓練集: 2000-01-01 ~ 2017-10-15 + 2023-10-16 ~ 2025-12-31
驗證集: 2017-10-16 ~ 2023-10-15
```

### 目標標籤

Buy Agent 預測：**未來 20 交易日內，最高價報酬率是否達到 +10% 以上**

### 獎勵機制 (對稱獎勵結構)

| 情境 | 獎勵 |
|------|------|
| 買對 (action=1, 漲幅≥10%) | +1.0 |
| 買錯 (action=1, 漲幅<10%) | 0.0 |
| 錯過 (action=0, 漲幅≥10%) | 0.0 |
| 正確迴避 (action=0, 漲幅<10%) | +1.0 |

---

## 特徵工程

### 基準指數
`^IXIC` (Nasdaq Composite) - 用於計算相對強度 (RS)

### 特徵列表 (32 個)

| 類別 | 特徵 |
|------|------|
| **價格正規化** | Norm_Close, Norm_Open, Norm_High, Norm_Low, Norm_DC_Lower |
| **Heikin Ashi** | Norm_HA_Open, Norm_HA_High, Norm_HA_Low, Norm_HA_Close |
| **SuperTrend** | Norm_SuperTrend_1 (14,2), Norm_SuperTrend_2 (21,1) |
| **動量指標** | Norm_RSI, Norm_K, Norm_D, Norm_DIF, Norm_MACD, Norm_OSC |
| **趨勢特徵** | Feat_MA20_Slope, Feat_Trend_Gap, Feat_Bias_MA20, Feat_Dist_MA60, Feat_Dist_MA240 |
| **波動率代理** | Feat_ATR_Ratio, Feat_HV20, Feat_Price_Pos |
| **相對強度** | Norm_RS_Ratio, RS_ROC_5, RS_ROC_10, RS_ROC_20, RS_ROC_60, RS_ROC_120 |

### 新增波動率指標 (替代 Volume)

| 指標 | 公式 | 用途 |
|------|------|------|
| `Feat_ATR_Ratio` | ATR(5) / ATR(20) | 偵測價格波動擴張 |
| `Feat_HV20` | 20日年化歷史波動率 | 偵測波動擠壓後的釋放 |
| `Feat_Price_Pos` | (Close - Low20) / (High20 - Low20) | 判斷價格相對於近期盤整區的位置 |

---

## 使用方式

### 1. 安裝依賴

```bash
pip install stable-baselines3 gymnasium yfinance ta pandas numpy tqdm
```

### 2. 執行訓練

```bash
python train_us_tech_buy_agent.py
```

訓練流程會自動：
1. 下載/更新股票資料 (CSV 增量更新)
2. 計算特徵並快取
3. 執行 Pre-training (所有股票)
4. 執行 Fine-tuning (各股票獨立)
5. 生成 `model_manifest.json`

### 3. 監控訓練

```bash
tensorboard --logdir ./tensorboard_logs/
```

開啟 http://localhost:6006 查看：
- `buy_pretrain_us_tech` - 預訓練曲線
- `buy_finetune_{TICKER}` - 各股票微調曲線

---

## 輸出結構

```
models_v5/
├── ppo_buy_base_us_tech.zip           # 預訓練基礎模型
├── finetuned/
│   ├── NVDA/
│   │   ├── ppo_buy_NVDA_final.zip     # 微調後模型
│   │   └── best/best_model.zip
│   ├── MSFT/
│   │   └── ...
│   └── PLTR/
│       └── ...
└── model_manifest.json                 # 模型清單
```

### model_manifest.json 範例

```json
{
  "version": "v5_us_tech",
  "base_model": "ppo_buy_base_us_tech.zip",
  "tickers": {
    "NVDA": {
      "model_path": "models_v5/finetuned/NVDA/ppo_buy_NVDA_final.zip",
      "train_end_date": "2025-12-31",
      "val_win_rate": 0.491,
      "actual_training_days": 4520,
      "base_model_version": "ppo_buy_base_us_tech"
    }
  }
}
```

---

## 訓練參數

| 參數 | Pre-training | Fine-tuning |
|------|--------------|-------------|
| 步數 | 1,000,000 | 750,000 |
| Learning Rate | 1e-4 | 1e-5 (1/10) |
| Batch Size | 512 | 128 |
| Entropy Coef | 0.01 | 0.01 |
| Network | MLP [64, 64, 64] | 同左 |
| Device | CPU | CPU |

---

## 評估與分析工具

### 1. 決策表現評估

```bash
python test_buy_agent_performance.py
```

評估 Agent 的 Precision 與 Recall，輸出：
- `test_results/test_results_summary.csv`
- `test_results/test_results_chart.png`

### 2. 信心度分層分析

```bash
python test_confidence_calibration.py
```

分析不同信心度區間 (50-60%, 60-70%, ..., 90-100%) 的決策成功率，輸出：
- `test_results/confidence_calibration_analysis.csv`
- `test_results/confidence_calibration_chart.png`

### 3. 深度回測 (動態停利)

```bash
python backtest_dynamic_trailing.py
```

針對 PLTR, NVDA, TSLA, NFLX 執行回測，包含：
- 信心度門檻: > 90%
- 硬性停損: -8%
- 移動停利啟動: +15%
- 動態回檔停利: 一般區 8% / 高獲利區 11%

輸出：
- `backtest_results/final_backtest_report.csv`
- `backtest_results/equity_curves.png`
- `backtest_results/trade_signals_{TICKER}.png`

### 4. 參數敏感度分析

```bash
python sensitivity_analysis.py
```

網格搜尋 75 組參數組合 (5×5×3)：
- Hard Stop: -3%, -4%, -5%, -6%, -8%
- Callback Base: 3%, 4%, 5%, 6%, 8%
- Callback High: 7%, 9%, 11%

輸出：
- `sensitivity_results/sensitivity_analysis_results.csv` (300 組結果)
- `sensitivity_results/sensitivity_best_params.csv` (最佳參數建議)
- `sensitivity_results/sensitivity_heatmap_{TICKER}.png`

---

## 回測績效參考

**深度回測結果 (2017-10-16 ~ 2023-10-15)：**

| Ticker | 總報酬 | CAGR | Sharpe | MDD | 交易次數 |
|--------|--------|------|--------|-----|----------|
| TSLA | 549.4% | 36.6% | 0.80 | -76.6% | 57 |
| NVDA | 382.7% | 30.0% | 0.76 | -67.7% | 42 |
| ^IXIC B&H | 102.4% | 12.5% | 0.52 | -36.4% | - |

**最佳參數建議 (敏感度分析)：**

| Ticker | Hard Stop | Callback Base | Callback High | Sharpe |
|--------|-----------|---------------|---------------|--------|
| TSLA | -8% | 8% | 9% | 0.87 |
| NVDA | -8% | 8% | 7% | 0.81 |
| PLTR | -6% | 6% | 9% | 0.66 |
| NFLX | -3% | 8% | 9% | 0.46 |

---

## 例外處理

- **股票尚未上市**：自動過濾無效訓練區間，僅使用有效數據
- **暖機期不足**：確保 MA240 等指標計算正確 (前 250 天)
- **NaN 值**：特徵計算後自動移除含 NaN 的資料列

---

## 檔案結構

```
ptrl-v02/
├── train_us_tech_buy_agent.py      # 主訓練腳本
├── test_buy_agent_performance.py   # 決策表現評估
├── test_confidence_calibration.py  # 信心度分層分析
├── backtest_dynamic_trailing.py    # 深度回測
├── sensitivity_analysis.py         # 參數敏感度分析
├── models_v5/                      # 模型儲存
├── test_results/                   # 評估結果
├── backtest_results/               # 回測結果
├── sensitivity_results/            # 敏感度分析結果
└── data/stocks/                    # 股票數據 CSV
```

---

## 參考腳本

本系統基於以下參考腳本開發：
- `reference/ptrl_hybrid_system.py`
- `reference/train_v5_models.py`
