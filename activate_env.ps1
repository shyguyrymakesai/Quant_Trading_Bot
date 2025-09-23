# 🚀 Activate Crypto Momentum Bot Environment (PowerShell)
Write-Host "🚀 Activating Crypto Momentum Bot Environment..." -ForegroundColor Green

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

Write-Host "✅ Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host "  python scripts/run_bot.py                        - Run the trading bot" -ForegroundColor Cyan
Write-Host "  python research/backtest_macd_adx_talib.py        - Run backtest" -ForegroundColor Cyan
Write-Host "  uvicorn src.app:app --reload --port 8000          - Start admin API" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Don't forget to update your API keys in .env file!" -ForegroundColor Yellow