@echo off
chcp 65001 >nul
echo =======================================================
echo US Tech Stock - Daily Predictor
echo =======================================================
cd /d "%~dp0"

echo 正在啟動虛擬環境並執行預測腳本...
.\.venv\Scripts\python.exe scripts\predict_today.py --use-regime-features true --profiles-path configs\rolling_profiles.json

echo.
echo =======================================================
echo 執行完畢，請確認 output_daily 內的結果。
pause
