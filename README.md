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

Buy Agent 預設預測：**未來 20 交易日內，最高價報酬率是否達到 +10% 以上**
*(在 sklearn 等輔助腳本中支援參數化，如 120 天 20%)*

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

# 建立虛擬環境 (Windows)
python -m venv .venv
.venv\Scripts\activate

# 建立虛擬環境 (Mac/Linux)
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

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

### 5. 市場濾網回測 (120MA + DC20)

```bash
# 預設期間
python backtest_market_filter.py

# 自訂期間
python backtest_market_filter.py --start 2017-10-16 --end 2025-12-31

# 指定股票
python backtest_market_filter.py --tickers NVDA TSLA
```

市場濾網邏輯：
- **多頭市場**: Nasdaq > 120MA → 准許買入
- **逆勢突破**: Nasdaq ≤ 120MA 且 個股 > DC20_High → 准許買入
- **其餘情況**: 保持空手

輸出目錄依日期範圍命名：`backtest_results_filtered_{START}_{END}/`

---

## Scikit-Learn 輔助分類訓練腳本

除了 PPO 訓練外，本專案提供傳統機器學習演算法的獨立二元分類模型，用於快速驗證特徵與「**未來 20 交易日內是否達到 +10% 報酬**」的關聯性。

### 1. 訓練特徵模型

腳本會自動重用 `train_us_tech_buy_agent.py` 的快取資料與特徵抽取邏輯。支援 RandomForest (`rf`)、AdaBoost (`adaboost`) 與 HistGradientBoosting (`hgb`)。

```bash
# 預設訓練 RF 模型 (針對 NVDA，並處理類別不平衡)
python scripts/train_sklearn_classifier.py --tickers NVDA --model rf --balance-train class_weight_balanced

# 訓練所有 10 檔股票的通用 HGB 模型
python scripts/train_sklearn_classifier.py --model hgb

# 測試資料維度、正類比與切分狀態但不實際訓練
python scripts/train_sklearn_classifier.py --dry-run
```

### 2. 相關參數與驗證

- `--balance-train`: 支援 `none`, `undersample_50_50`, `class_weight_balanced`。
- `--train-ranges`: 支援 Walk-Forward 設定多段訓練區間（如 `2000-01-01:2017-10-15`）。
- `--target-days` 與 `--target-return`: 可自訂預測目標的天數與報酬門檻 (例如：`--target-days 120 --target-return 0.20`)。
- **輸出包含**:
  模型將輸出於 `output_sklearn/run_{model}_{target_days}d_{datetime}/`，涵蓋 Precision/Recall, AUROC, AUPRC, Threshold Sweep 以及 `metrics.json` 中的各特徵重要性 (Feature Importances)。

### 3. Walk-Forward (Rolling) 訓練對抗 Regime Shift

針對容易發生「**特徵意義反轉**（如 2019-2022 年間高分預測反而低勝率）」的問題，專案提供了 `scripts/train_rolling_hgb.py`，支援「每年用過去 N 年的特徵」重新訓練、隔年全量資料驗證的嚴格迴測。

```bash
# 針對 GOOGL 使用前 5 年資料滾動訓練以預測 120天 20%
python scripts/train_rolling_hgb.py --tickers GOOGL --window-years 5 --target-days 120 --target-return 0.20

# 限定驗證範圍並加入防呆檢查 (不執行)
python scripts/train_rolling_hgb.py --tickers GOOGL --val-years 2019 2020 2021 2022 --dry-run
```

腳本會在 `output_rolling_hgb/` 輸出完整的 `rolling_summary.csv`。此總表不僅能追蹤每年的真實可用樣本邊界，更新增了 **Top 5% Hit Rate by Proba** 比較，若發現反向，將自動觸發 `reversal_warning` 以供後續診斷。

為了解決手動尋找最穩定 window year 區間的問題，系統提供 `scripts/run_rolling_grid.py` 包裝器，能一次性自動執行多個年份組合並綜合產出大表：

```bash
# 一次比較 Window_Years 為 3, 5, 7 年的跨年預測穩定性
python scripts/run_rolling_grid.py --tickers GOOGL --window-years-list 3 5 7 --target-days 120 --target-return 0.20
```

#### 進階防禦：自我學習 Regime Feature (推薦)

為了讓模型能主動意識到目前所處的市場狀態（如空頭、高波動等）避免在反轉年失效，執行 Rolling Training 時強烈建議加上 `--use-regime-features` 開關。

這項參數會自動提取專案中的 `BENCHMARK` 大盤指數（如 ^IXIC），計算以下特徵並在 **不修改原生 `FEATURE_COLS`** 的前提下，動態掛載給 HGB 模型：
1. **Trend**: 大盤是否在 200 日均線之上 (`MA200_ABOVE`) 及 均線斜率 (`MA200_SLOPE`)
2. **Volatility**: 20 日年化波動率 (`HV20`) 及其在過去 3 年的百分位數階層 (`HV20_PCTL`)
3. **Momentum**: 中長期 60日 / 120日 的絕對報酬率

**執行範例 (比較對照)：**
```bash
# 1. 計算 Baseline (純個股特徵)
python scripts/train_rolling_hgb.py --tickers GOOGL --window-years 3 --target-days 120 --target-return 0.20 --output-dir output_rolling_baseline --seed 42

# 2. 加入 Regime Features (自學大盤環境)
python scripts/train_rolling_hgb.py --tickers GOOGL --window-years 3 --target-days 120 --target-return 0.20 --use-regime-features --output-dir output_rolling_w_feat --seed 42
```

**驗收與觀察點：**
打開 `output_rolling_w_feat/.../rolling_summary.csv` 檢查成果：
- 檢查 `reversal_warning` 是否在歷史上的熊市 (如 2019/2022) 成功從 `True` 轉為 `False`。
- 觀察 `rolling_summary.csv` 中新增的統計欄位（如該年的 `regime_above_ma200_rate` 平均低於 0.5 時，模型的命中率有無穩住），藉此驗收模型是否成功靠大盤特徵避開了不佳市況。

### 4. 自動化網格搜尋 (Window Years Grid)

```bash
# 一次比較 Window_Years 為 3, 5, 7 年的跨年預測穩定性
python scripts/run_rolling_grid.py --tickers GOOGL --window-years-list 3 5 7 --target-days 120 --target-return 0.20
```

腳本會在 `output_rolling_grid/{TICKER}_.../` 目錄內：
1. 為每一組 `window_years` 保留獨立的年份預測輸出至 `windows/wX/`
2. 自動產出 `grid_summary.csv`，列舉各個 `window_years` 的 `mean_roc_auc`、反向發生次數 `reversal_year_count` 以及最糟表現年度，方便一眼選出最抗跌的滑動區間。

### 5. Regime Gate 離線防禦評估

在 Rolling 網格搜尋完成後，我們可以使用 `scripts/eval_regime_gate_flip.py` 來進行離線 Regime Gate 驗證。透過大盤 (Benchmark) 特徵判斷市況，將測出為 "Reversal Regime" 時期的預測分數反轉 (`1 - y_proba`)，以此拯救模型在極端反向年（如 2019 或 2022）的預測失靈。

```bash
# 對已經跑好的 GOOGL w5 rolling 預測結果進行 Regime Gate 評估 (評估 Top 5% 命中率變化)
python scripts/eval_regime_gate_flip.py --ticker GOOGL --pred-dir output_rolling_grid/GOOGL_120d20pct_.../windows/w5 --topk-pct 5
```

會輸出 `output_gate_eval/gate_eval_summary_{TICKER}.csv` 總表，包含 4 種 Gate 邏輯（Trend, Volatility, Momentum, Combo）相較於原始預測的命中率 (Precision@k) 提升幅度與發動反轉的比例，方便您判斷哪種市況濾網最適合目前的目標策略。

---

## PPO 離線單步推論評估

為了能夠在相同的 Validation 集上公平地與 Sklearn 模型（或其他基準模型）比較，專案提供針對已訓練好 PPO (`best_model.zip` 或 `final_model.zip`) 的獨立離線評估腳本。該腳本**不觸發重新訓練 (learn)、不呼樣 (Resample)，只進行純驗證集的 Metrics 生成**。

### 1. 執行推論與評估

此腳本會調用所選 PPO 模型的 policy network，透過 `model.policy.get_distribution()` 在 `no_grad` 模式下提取 $P(action_{buy}|x)$ 機率，對齊 Sklearn 工具相同格式的指標陣列。

```bash
# 針對特定股票群使用單一部署模型評估 (預設 Threshold = 0.5)
python scripts/eval_ppo_classifier.py --model-path models_v5/ppo_buy_base_us_tech.zip --tickers NVDA MSFT TSLA --threshold 0.5

# 針對各 ticker 獨立載入其對應微調後的 best_model.zip 進行評估
# (使用 {ticker} 變數，腳本將自動幫每個 ticker 尋找並載入該專屬模型)
python scripts/eval_ppo_classifier.py --model-path "models_v5/finetuned/{ticker}/best/best_model.zip" --tickers NVDA MSFT TSLA

# 查看推論資料列與狀態分布 (Dry Run 不做實際推論)
python scripts/eval_ppo_classifier.py --model-path models_v5/ppo_buy_base_us_tech.zip --tickers NVDA --dry-run
```

### 2. 輸出與指標

與 `train_sklearn_classifier.py` 100% 對齊：
- **輸出路徑**: 預設於 `output_eval_ppo/eval_ppo_{model_name}_{datetime}/`
- 提供完整的 `metrics.json` (包含 P@k 與 Threshold sweep)
- 輸出具時間與股票代碼標記的 CSV `val_predictions.csv`，便於視覺化或自訂評估策略。

---

## 日常買點輔助推論小工具

如果您已經有訓練好的 `.joblib` (Sklearn) 或是 `.zip` (PPO) 檔案，可以直接透過 `scripts/predict_today.py` 每日執行，腳本會自動抓取最新的 YF 收盤數據、轉換特徵，並產出一份當日買點判定清單。

```bash
# 基本推論：使用已經訓練好的 20 天 10% HGB 模型判斷預設 10 檔股票的最新買點
python scripts/predict_today.py --model-path output_sklearn/run_hgb_123/model.joblib

# 自訂目標推論：若模型是特別訓練 120 天 20% 的目標，需搭配相同設定讓列印資訊正確
python scripts/predict_today.py --model-path output_sklearn/run_hgb_120d_123/model.joblib --target-days 120 --target-return 0.20

# 載入各自的 PPO 微調模型，預測指定 3 檔最新買點
python scripts/predict_today.py --model-path "models_v5/finetuned/{ticker}/best/best_model.zip" --tickers NVDA MSFT TSLA
```

**輸出範例**:
```
📊 今日推論結果 (Prediction for Latest Close)
-----------------------------------------------------------------
Ticker   | Latest Date  | P(Buy) %     | Decision
-----------------------------------------------------------------
NVDA     | 2026-02-20   |  20.72%      | WAIT ⚪
MSFT     | 2026-02-20   |  11.31%      | WAIT ⚪
TSLA     | 2026-02-20   |  58.91%      | BUY 🟢
-----------------------------------------------------------------
🎯 總計符合買進門檻 (0.5): 1 檔
```

---

## 特徵飄移診斷 (Regime Shift Analytics)

針對長天期預測（例如 120天）可能發生的模型失效（如 Validation ROC-AUC < 0.5），專案提供 `analyze_topk_feature_shifts.py` 自動分析各年份的極端分數群體，以此釐清是哪些特徵不再適用於近年的市場（發生了 Regime Shift）。

### 使用方式

只需要傳入預測完成產出的 `val_predictions.csv`：

```bash
# 對特定型號與標的，取預測分數最極端的 Top 5% 來比對差異
python scripts/analyze_topk_feature_shifts.py --val-predictions output_sklearn/run_hgb_120d_123/val_predictions.csv --ticker GOOGL --topk-pct 5 --output-dir output_analysis
```

### 診斷輸出

輸出目錄下會依照各年份產生統計對比表，例如：
- `YYYY_feature_diff_A_vs_B.csv`：排列出高分群(A)與低分群(B)之間，**標準化差異 (Standardized Diff) 最大**的反轉特徵。
- `summary.json`：總覽各年度的 Precision@k 表現，若低分群的真實勝率大於高分群，會留下警告標記與特徵翻轉排名。

---

## 回測績效參考

### 無濾網版本 (2017-10-16 ~ 2023-10-15)

| Ticker | 總報酬 | CAGR | Sharpe | MDD |
|--------|--------|------|--------|-----|
| TSLA | 725.2% | 42.2% | 0.87 | -74.9% |
| NVDA | 421.1% | 31.7% | 0.79 | -66.7% |
| ^IXIC B&H | 102.4% | 12.5% | 0.52 | -36.4% |

### 市場濾網版本 (120MA + DC20)

| Ticker | 總報酬 | Sharpe | MDD | MDD 改善 |
|--------|--------|--------|-----|---------|
| **TSLA** | **1242.5%** | **1.12** | **-40.2%** | **+34.7%** |
| **NVDA** | **568.5%** | **0.99** | **-40.8%** | **+25.9%** |
| NFLX | 49.9% | 0.31 | **-37.5%** | **+34.2%** |
| PLTR | -11.4% | 0.09 | -69.1% | -1.0% |

> ✅ **關鍵發現**: 市場濾網成功將 TSLA/NVDA/NFLX 的 MDD 從 -65%~-75% 降至 -40% 以下，同時提升報酬率與 Sharpe

---

## NVDA 專屬跟單系統

### 1. NVDA 跟單回測腳本

```bash
# 執行回測
python backtest_nvda_follow.py --start 2020-01-01 --end 2023-12-31

# 只指定起始日（結束日自動設為今天）
python backtest_nvda_follow.py --start 2025-12-09
```

#### 核心特色

| 功能 | 說明 |
|------|------|
| **年度資金注入** | 起始日注入 $2,400，每年第一個交易日再注入 $2,400 |
| **信心度分級買入** | >95%: 25%, 90-95%: 15%, <90%: 不買 |
| **市場濾網** | Nasdaq > 120MA 或 個股 > DC20 突破 |
| **資金回流** | 賣出後資金回到資金池 |
| **Nasdaq B&H 比較** | 同等資金注入的基準對比 |

#### 輸出檔案

```
backtest_results_nvda/{start}_{end}/
├── end_date_summary_NVDA_{start}_{end}.txt  # 跟單總結（含明日操作建議）
├── equity_curve_nvda_follow.png             # 淨值曲線圖
└── trade_log_NVDA_{start}_{end}.csv         # 交易紀錄
```

### 2. 風險管理參數網格搜尋

```bash
python grid_search_nvda_params.py
```

#### 最終優化參數（經 4 輪測試）

```python
HARD_STOP_PCT = -0.08          # 硬性停損 -8%
TRAILING_ACTIVATION = 0.20     # 移動停利啟動 +20%
HIGH_PROFIT_THR = 0.25         # 高利潤門檻 25%
CALLBACK_BASE = 0.08           # 基礎回檔停利 8%
CALLBACK_HIGH = 0.17           # 高利潤回檔停利 17% ⭐
```

#### 優化績效（2017-10-16 ~ 2023-10-15）

| 策略 | Return | Sharpe | MDD | 特點 |
|------|--------|--------|-----|------|
| **最優 (CB=17%)** | **+544.9%** | **1.27** | **-31.6%** | 最佳平衡 ⭐ |
| 保守 (CB=7%) | +404% | 1.22 | -32.1% | 穩健，高勝率 65% |
| 激進 (CB=75%) | +649% | 1.25 | -63.6% | 高報酬高風險 ⚠️ |

#### 網格搜尋輸出

```
grid_search_results_nvda/
├── grid_search_results.csv        # 完整結果表
├── parameter_heatmaps.png         # 參數熱力圖
├── parameter_impact.png           # 單參數影響分析
└── performance_scatter.png        # 績效散點圖
```

#### 關鍵發現

1. **CALLBACK_HIGH = 17%** 是報酬與風險的完美平衡點
2. **12-15%** 是效率谷底，應避開
3. **HARD_STOP = -8% 到 -10%** 較寬的停損讓 NVDA 有更多波動空間
4. **TRAILING_ACTIVATION = 20%** 讓利潤充分發展再啟動保護

---

## 例外處理

- **股票尚未上市**：自動過濾無效訓練區間，僅使用有效數據
- **暖機期不足**：確保 MA240 等指標計算正確 (前 250 天)
- **NaN 值**：特徵計算後自動移除含 NaN 的資料列

---

## 快取機制

### 特徵快取自動失效 (2026-01-22)

`calculate_features` 函數會將計算好的特徵資料快取至 `data/processed/{TICKER}_features_ustech.pkl`。

**自動失效邏輯**：
- 載入快取時，比較**快取資料的最後日期**與**輸入資料的最後日期**
- 如果快取資料較舊，自動失效並重新計算特徵

```
[Cache] Loading features for NVDA (up to 2026-01-21)...     # 使用有效快取
[Cache] Invalidating stale cache for NVDA: 2026-01-16 < 2026-01-21  # 快取過期
[Compute] Generating features for NVDA...                    # 重新計算
```

**受益腳本**（所有導入 `calculate_features` 的腳本）：
- `backtest_nvda_follow.py`
- `backtest_market_filter.py`
- `backtest_dynamic_trailing.py`
- `sensitivity_analysis.py`
- `test_buy_agent_performance.py`
- `test_confidence_calibration.py`

## Pandas 兼容性修復 (2026-01-30)

解決在 Pandas 2.0+ 版本中出現的 `TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'` 錯誤。

**修復內容**：
- 將 `fillna(method='ffill')` 替換為 `ffill()`
- 將 `fillna(method='bfill')` 替換為 `bfill()`

**受影響並已修復的腳本**：
- `backtest_nvda_follow.py`
- `backtest_market_filter.py`
- `backtest_dynamic_trailing.py`
- `train_us_tech_buy_agent.py`

## 檔案結構

```
ptrl-v02/
├── train_us_tech_buy_agent.py      # 主訓練腳本
├── test_buy_agent_performance.py   # 決策表現評估
├── test_confidence_calibration.py  # 信心度分層分析
├── test_us_tech_quick.py           # 快速測試腳本
├── backtest_dynamic_trailing.py    # 深度回測 (無濾網)
├── backtest_market_filter.py       # 市場濾網回測
├── backtest_nvda_follow.py         # NVDA 專屬跟單回測 ⭐
├── grid_search_nvda_params.py      # NVDA 參數網格搜索 ⭐
├── regenerate_best_params.py       # 參數重新生成輔助腳本
├── sensitivity_analysis.py         # 參數敏感度分析
├── scripts/                        # 獨立分析與訓練工具
│   ├── train_sklearn_classifier.py # sklearn 二元分類訓練腳本
│   ├── train_rolling_hgb.py        # Walk-Forward 滾動時間窗訓練
│   ├── run_rolling_grid.py         # Window Years 自動網格搜尋與統整
│   ├── eval_regime_gate_flip.py    # 離線 Regime Gate 預測翻轉評估
│   ├── eval_ppo_classifier.py      # PPO 離線推論單步評估腳本
│   ├── predict_today.py            # 日常買點預測推論工具
│   └── analyze_topk_feature_shifts.py # 特徵翻轉與 Regime Shift 診斷
├── src/
│   ├── features/
│   │   └── regime_features.py      # 萃取大盤狀態 (MA200, HV20) 給模型防禦的函數
│   └── train/
│       └── sklearn_utils.py        # 共用的 sklearn 指標與類別轉換工具
├── models_v5/                      # 模型儲存
├── output_sklearn/                 # sklearn 訓練結果輸出
├── output_eval_ppo/                # PPO 離線推論評估輸出
├── data/stocks/                    # 股票數據 CSV
├── logs/                           # 系統日誌
├── tensorboard_logs/               # TensorBoard 監控日誌
├── test_results/                   # 評估結果
├── backtest_results/               # 回測結果
├── backtest_results_filtered_*/    # 濾網回測結果 (依日期)
├── backtest_results_nvda/          # NVDA 跟單回測結果 ⭐
├── grid_search_results_nvda/       # 網格搜索結果 ⭐
└── sensitivity_results/            # 敏感度分析結果
```

---

## 參考腳本

本系統基於以下參考腳本開發：
- `reference/ptrl_hybrid_system.py`
- `reference/train_v5_models.py`
