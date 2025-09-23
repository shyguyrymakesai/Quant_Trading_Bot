@echo off
echo 🚀 Activating Crypto Momentum Bot Environment...
call "%~dp0venv\Scripts\activate.bat"
echo ✅ Virtual environment activated!
echo.
echo Available commands:
echo   python scripts/run_bot.py          - Run the trading bot
echo   python research/backtest_macd_adx_talib.py - Run backtest
echo   uvicorn src.app:app --reload --port 8000   - Start admin API
echo.
echo 📝 Don't forget to update your API keys in .env file!
cmd /k