# US Tech Stock Buy Agent - RL 交易訓練系統

使用強化學習 (Reinforcement Learning) 訓練 Buy Agent，辨識 10 隻美股科技股的「起漲點」。

## 目標標的

`NVDA`, `MSFT`, `AAPL`, `AMZN`, `META`, `AVGO`, `GOOGL`, `TSLA`, `NFLX`, `PLTR`, `TSM`

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

#### 實例 1：基本使用

若想針對單一標的（例如 GOOGL）跑 5 年窗口、目標 120 天漲幅 20%：

```bash
python scripts/train_rolling_hgb.py --tickers GOOGL --window-years 5 --target-days 120 --target-return 0.20

# 限定驗證範圍並加入防呆檢查 (不執行)
python scripts/train_rolling_hgb.py --tickers GOOGL --val-years 2019 2020 2021 2022 --dry-run
```

腳本會在 `output_rolling_hgb/` 輸出完整的 `rolling_summary.csv`。此總表不僅能追蹤每年的真實可用樣本邊界，更支援最新的 **V2 Reversal 雙保險診斷** (搭配 Top5 與 Top10 Gap 監控)。當發現反轉 (Gap <= `-0.10`)，將自動觸發 `reversal_warning`。

#### Baseline vs Regime 差分對照
新增了 `compare_rolling_summaries.py` 自動比對 Baseline (未加特徵) 與 Regime (加特徵) 的逐年 Rolling 成效，以量化驗證防禦機制的成功率：

```bash
# 輸入兩份 CSV 進行 Inner Join 比較
python scripts/compare_rolling_summaries.py --baseline output_rolling_baseline/run.../rolling_summary.csv --regime output_rolling_w_feat/run.../rolling_summary.csv --output-dir output_compare
```
將會自動輸出 `yearly_diff.csv` (觀察逐年 delta 差距) 以及 `aggregate_compare.json` (統整自 2017 起的 Worst Gap 與 Reversal 發生總次數改善情況)。

為了解決手動尋找最穩定 window year 區間的問題，系統提供 `scripts/run_rolling_grid.py` 包裝器，能一次性自動執行多個年份組合並綜合產出大表：

```bash
# 一次比較 Window_Years 為 3, 5, 7 年的跨年預測穩定性
python scripts/run_rolling_grid.py --tickers GOOGL --window-years-list 3 5 7 --target-days 120 --target-return 0.20
```

#### 進階防禦：大盤與個股 Regime Feature 雙重掛載與 HGB 正規化 (推薦)

為了讓模型能主動意識到目前所處的市場狀態（如空頭、高波動等）避免在反轉年失效，執行 Rolling Training 時強烈建議加上 `--use-regime-features` 開關。同時，為了壓制空頭強勢段（如 2022 年）所產生的模型預測分數過度自信飽和情況，建議掛載 `--hgb-reg-preset regularized` 開關（自動啟用 `min_samples_leaf=50, max_depth=3, l2=0.1`）降低決策樹敏感度。

`--use-regime-features` 參數會自動提取專案中的 `BENCHMARK` 大盤指數（如 ^IXIC）以及 **個股歷史股價**，計算以下特徵動態掛載給 HGB 模型：
1. **大盤趨勢 (Trend)**: 大盤是否在 200 日均線之上 (`MA200_ABOVE`) 及 均線斜率 (`MA200_SLOPE`)
2. **大盤波動率 (Benchmark Volatility)**: 大盤 20 日年化波動率 (`HV20`) 及其在過去 3 年的歷史百分位數 (`HV20_PCTL`)
3. **大盤動能 (Benchmark Momentum)**: 大盤中長期 60日 / 120日 的絕對報酬率
4. **個股獨立防禦 (Stock-Specific Regime)**:
   - 個股獨立 20 日波動率 (`HV20`) 與其過去三年歷史百分位。
   - 個股相對於大盤基準的 120 日動能強弱差 (`RS120`)。
   - 個股嚴重乖離/極端過熱標記 (`EXTREME_DIST_MA240_FLAG`)。

**執行範例 (針對 TSM 佈署完整抗跌配備)：**
```powershell
python scripts/train_rolling_hgb.py --tickers TSM `
  --window-years 3 --target-days 120 --target-return 0.20 `
  --use-regime-features --reversal-gap-margin 0.10 `
  --hgb-reg-preset regularized --output-dir output_rolling_tsm_v3 `
  --seed 42
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

### 5. 全市場批次驗證與總表 (Batch Rolling & Summary)

在跑完基礎的 Rolling 驗證後，我們可以使用 `scripts/run_rolling_all_tickers.py` 批次對所有 10 檔目標科技股啟動滾動測試，並透過 `scripts/summarize_all_tickers.py` 一鍵產出跨股票橫向比較的「最佳實務總表」。這能讓您立刻看出哪些股票在哪個年份最具反轉抵抗力！

#### (1) 執行批次 Rolling 測試
```powershell
# 對全部的 tickers 以 3 年窗格、120天 20% 目標，開啟大盤防禦進行批次驗證 (支援並行加速)
python scripts/run_rolling_all_tickers.py `
  --tickers GOOGL NVDA MSFT AMZN META AVGO NFLX AAPL TSLA PLTR TSM `
  --output-dir output_rolling_all `
  --window-years 3 --target-days 120 --target-return 0.20 `
  --use-regime-features --reversal-gap-margin 0.10 `
  --hgb-reg-preset regularized `
  --val-years 2018 2019 2020 2021 2022 2023 2024 2025 `
  --max-workers 2 `
  --seed 42
```
- `--use-regime-features`：掛載大盤（SPY/QQQ）總體經濟特徵，以及針對個別股票（如 TSM）的自身波動率與相對強度（RS120）特徵，幫助模型迴避市場崩盤與極端乖離段。
- `--hgb-reg-preset`：(新增) 提供 HGB 決策樹的正規化微調，可選 `default` 或 `regularized` (強啟動 min_samples_leaf=50, max_depth=3, l2=0.1)，幫助壓制如 2022 年空頭年的分數過度飽和現象。
- `--reversal-gap-margin`：容忍的 Hit-Gap 誤差值 (預設 0.10)。
這會在 `output_rolling_all/` 底下自動建立各個 Ticker 的專屬資料夾，並寫入該股各自的 `rolling_summary.csv` 與每個年份的 Metrics。

#### (2) 彙整全股票超級總表
```powershell
# 掃描 output-dir 底下的各股報表，並彙整出指定起訖年份的單張 CSV
python scripts/summarize_all_tickers.py `
  --input-dir output_rolling_all `
  --output-dir output_rolling_all `
  --years-from 2018 --years-to 2025 `
  --topk 10 `
  --sort-by mean_top10_hit_proba
```
這會產出終極的 `all_tickers_summary.csv` 大表與 Json，提供包含：`mean_roc_auc`、各股最差年度的 Gap、Top 10 的平均命中率、甚至列舉該股票是否發生 `reversal_year_count_v2`（雙保險反轉警報）等一覽無遺的評比！

### 6. Regime Gate 離線防禦評估

在 Rolling 完成部分實驗後，我們可以使用 `scripts/eval_regime_gate_flip.py` 來進行離線 Regime Gate 驗證。透過大盤 (Benchmark) 特徵判斷市況，將測出為 "Reversal Regime" 時期的預測分數反轉 (`1 - y_proba`)，以此拯救模型在極端反向年（如 2019 或 2022）的預測失靈。

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

## 日常實盤推論系統 (Daily Train & Predict)

為了落實真正即時的交易，專案提供高度進化的 `scripts/predict_today.py`。它捨棄了依賴過期靜態模型的舊思維，轉為**「每天拉取最新資料、單檔獨立滾動建模」**的設計（情境 A）。同時內建了以大盤指標作防禦的 Proxy 風控降槓桿機制。

### 1. 單檔每日智能訓練 (Single-Ticker Daily Train)

這是未來上線每天 Cronjob 預設的操作方式，不需要給定模型路徑，腳本會依照下列邏輯全自動運行：
1. 自動抓取過去 8 年股市/大盤資料以便確保特徵暖機無虞。
2. 切取這 10 檔目標股票各自**最近 3 年**的有效特徵資料 (`[Today - 3y, Today]`) 作為該檔專屬的訓練集。
3. 把 Regime 防禦特徵掛載上去，自動跑出 10 顆 `HistGradientBoostingClassifier` 並快取至 `output_daily/YYYYMMDD/{ticker}/model.joblib`，加速今日內反覆推論的效率。

```bash
# 全自動每日訓練及推論，並輸出至 output_daily/ 當天目錄
python scripts/predict_today.py

# 強迫重新訓練模型 (不讀取當日快取)
python scripts/predict_today.py --force-retrain
```

### 2. 動態排名與風控決策 

不再使用死板的固定 `Threshold 0.5` 機制！新版加入了單檔自身歷史分位數判定：
- **百分位評等 (Percentile Rank)**：這顆剛出爐的新模型，會先回推計算自己這檔股票過去 252 交易日出現過的分數 (P)，並計算「**今天這筆推論分數在歷史上的相對位階** (`pct_rank_today`)」。
- **High Risk 風控降頻**:
  - `Normal (正常)`: 當日 Regime 正常。只要今日分數落於自身歷史的 Top 10% 內 (`--topk-threshold-pct 0.90`)，就會標記 **`BUY`** (資金池配給 Position 1.0 倍)。
  - `High Risk (高危)`: 例如「大盤跌破 MA 200 且波動率百分位 > 80%」時。出手機制瞬間嚴格化至 Top 5% (`--risk-threshold-pct 0.95`)。就算有股票成功入圍，也會被標註為 **`BUY_REDUCED`** (資金池配給強制縮為 0.5 倍)；不達標的更會亮起紅燈 **`SKIP_RISK`**。

#### 輸出範例

除了在 Console 印出美觀的報告，腳本更會自動匯出 `predictions.csv` 與 `run_summary.json` 便於後續串接自動下單機。

```
📊 今日推論結果 (Single Ticker Approach)
----------------------------------------------------------------------------------------
Ticker   | Latest Date  | Score(p)   | PctRank  | Act Thresh | Action          | Pos Scale
----------------------------------------------------------------------------------------
NVDA     | 2026-02-20   |  99.83%    | 50.4%    | >=0.9      | WATCH           | x0.0
AAPL     | 2026-02-20   |   0.06%    | 36.1%    | >=0.9      | WATCH           | x0.0
AMZN     | 2026-02-20   |  99.99%    | 97.6%    | >=0.9      | BUY             | x1.0
NFLX     | 2026-02-20   |  99.99%    | 98.4%    | >=0.9      | BUY             | x1.0
----------------------------------------------------------------------------------------
📝 報告輸出完成於: output_daily/20260221
✅ predictions.csv 與 run_summary.json 已更新檔案
```

### 3. 靜態單步推論 (Legacy Mode)

當然，如果您想保留以前的手感，直接拿特定寫死好的 sklearn / PPO 模型來測今天的漲幅預測也是 100% 相容的（不經過分位數計算）：

```bash
# 載入 sklearn 模型進行今日推論（門檻大於 0.5 才買）
python scripts/predict_today.py --model-path output_sklearn/run_hgb_123/model.joblib --threshold 0.5

# 載入 10 檔不同路徑下的 PPO best_model.zip 
python scripts/predict_today.py --model-path "models_v5/finetuned/{ticker}/best/best_model.zip" --tickers NVDA MSFT TSLA
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
│   ├── run_rolling_all_tickers.py  # (新增) 全股票批次 Rolling 並行啟動器
│   ├── summarize_all_tickers.py    # (新增) 掃描收集批次 Rolling 之橫向比較大表
│   ├── eval_regime_gate_flip.py    # 離線 Regime Gate 預測翻轉評估
│   ├── eval_ppo_classifier.py      # PPO 離線推論單步評估腳本
│   ├── predict_today.py            # 單檔自動每日即時訓練與風控推論實盤系統
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
