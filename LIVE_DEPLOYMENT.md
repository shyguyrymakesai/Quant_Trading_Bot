# 🚀 Live Trading Deployment Guide

This guide walks through deploying your bot for live trading with secure Coinbase API integration.

## Prerequisites

- [ ] Coinbase account with trading enabled
- [ ] AWS CLI configured with appropriate permissions
- [ ] Terraform installed

## Step 1: Create Coinbase Advanced Trade API Keys

1. Go to [Coinbase Developer Portal](https://www.coinbase.com/cloud)
2. Create a new API key with these permissions:
   - **View**: ✅ (required for balance checks)
   - **Trade**: ✅ (required for order placement) 
   - **Transfer**: ❌ (keep OFF for security)
3. **Important**: Select "Advanced Trade" not "Coinbase Exchange"
4. Note down your:
   - API Key (starts with `organizations/...`)
   - API Secret (long base64 string)

## Step 2: Store Credentials in AWS SSM (Secure)

Run the PowerShell script with your actual credentials:

```powershell
.\setup-coinbase-ssm.ps1 -ApiKey "your_api_key" -ApiSecret "your_api_secret"
```

This stores them encrypted in AWS SSM Parameter Store at:
- `/quant-bot/coinbase/API_KEY`
- `/quant-bot/coinbase/API_SECRET`

## Step 3: Deploy Infrastructure Changes

Apply Terraform to give ECS permission to read SSM parameters:

```powershell
cd infra
terraform plan  # Review changes
terraform apply -auto-approve
cd ..
```

This adds:
- SSM read permissions for `/quant-bot/coinbase/*` parameters
- KMS decrypt permissions for SSM
- Updates container environment to live mode

## Step 4: Install boto3 and Test Locally

```powershell
pip install boto3
python smoke_test_live.py
```

This validates:
- ✅ SSM credentials loading
- ✅ Coinbase API connection  
- ✅ Smart sizing logic
- ✅ Account balance check

## Step 5: Build and Deploy Live Image

```powershell
# Build new image with boto3
docker build -t quant-bot-live .

# Tag and push to ECR
$ECR_URI = "969932165253.dkr.ecr.us-east-1.amazonaws.com/quant-bot-app"
docker tag quant-bot-live:latest $ECR_URI:latest

# Login to ECR and push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
docker push $ECR_URI:latest
```

## Step 6: Live Smoke Test (One Manual Run)

Run one ECS task manually to test before enabling schedule:

```powershell
# Update task override to live mode (already done)
aws ecs run-task `
  --task-definition quant-bot-task:2 `
  --cluster quant-bot-cluster `
  --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={subnets=[subnet-0bf6067c27c20cafb,subnet-07ec095ea8d54e24d],securityGroups=[sg-0cfe1946724ffa248],assignPublicIp=ENABLED}" `
  --overrides file://task-override.json `
  --region us-east-1
```

## Step 7: Monitor First Live Run

1. **CloudWatch Logs**: Check `/quant-bot/app` for errors
2. **S3 Artifacts**: Look for `order_plan` in run artifacts  
3. **Coinbase Account**: Verify any orders placed correctly

## Step 8: Enable Full Schedule (If Smoke Test Passed)

The bot is now configured for live trading with:

- 🔐 **Secure**: API keys stored in encrypted SSM, not plaintext
- 💡 **Smart Sizing**: $1 minimum respects exchange limits  
- 🎯 **Maker-Only**: 1.5bps offset reduces fees
- ⏰ **Scheduled**: Runs every 30 minutes automatically
- 📊 **Multi-Symbol**: BTC-USD and ETH-USD

## Safety Features Active

- ✅ Maker-only orders (post_only flag)
- ✅ Dynamic exchange minimums ($1 for BTC/ETH)
- ✅ 4-bar cooldown after exits
- ✅ ADX threshold filtering (>20)
- ✅ Position sizing limits
- ✅ S3 trade logging for audit

## Emergency Stop

To stop live trading immediately:

```powershell
# Disable the schedule
aws events disable-rule --name quant-bot-every-30 --region us-east-1

# Or change back to paper mode
# Update infra/variables.tf: MODE = "paper"
# terraform apply -auto-approve
```

## 📈 Monitoring Dashboard

Key metrics to watch:
- **CloudWatch**: `/quant-bot/app` logs for errors
- **S3 Bucket**: `quant-bot-trades-*` for artifacts  
- **Coinbase**: Account balance and order history
- **Signals**: Look for `order_plan` in artifacts when BUY/SELL signals trigger

---

🎯 **Your bot is now ready for live trading with institutional-grade security and risk management!**