# Setup script for Coinbase API credentials in AWS SSM Parameter Store
# Run this ONCE after creating your Coinbase Advanced Trade API keys

param(
    [Parameter(Mandatory=$true)]
    [string]$ApiKey,
    
    [Parameter(Mandatory=$true)]
    [string]$ApiSecret,
    
    [string]$Region = "us-east-1"
)

$PFX = "/quant-bot/coinbase"

Write-Host "Setting up Coinbase Advanced Trade API credentials in SSM Parameter Store..." -ForegroundColor Green

try {
    # Store API Key
    Write-Host "Storing API Key..." -ForegroundColor Yellow
    aws ssm put-parameter --name "$PFX/API_KEY" --type SecureString --value $ApiKey --overwrite --region $Region
    
    # Store API Secret  
    Write-Host "Storing API Secret..." -ForegroundColor Yellow
    aws ssm put-parameter --name "$PFX/API_SECRET" --type SecureString --value $ApiSecret --overwrite --region $Region
    
    Write-Host "Coinbase credentials stored successfully!" -ForegroundColor Green
    Write-Host "   Parameters created:" -ForegroundColor Cyan
    Write-Host "   - $PFX/API_KEY" -ForegroundColor Cyan  
    Write-Host "   - $PFX/API_SECRET" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "   1. Apply Terraform to give ECS permission to read these params"
    Write-Host "   2. Deploy live trading bot"
    Write-Host "   3. Monitor first live trades carefully"
    
} catch {
    Write-Host "Error setting up SSM parameters: $_" -ForegroundColor Red
    exit 1
}