variable "project" { default = "quant-bot" }
variable "region"  { default = "us-east-1" }
variable "aws_profile" { default = "quant-bot" }

# schedule: every 30 mins
variable "schedule_expression" {
  default = "cron(0,30 * * * ? *)"
}

variable "cpu"    { default = 256 }   # 0.25 vCPU
variable "memory" { default = 512 }   # 0.5 GB

variable "container_env" {
  type = map(string)
  default = {
    SYMBOLS                = "[\"BTC-USD\",\"ETH-USD\"]"
    TF                     = "1h"
    MODE                   = "live"
    PYTHONPATH             = "/app/src"
    COINBASE_PARAM_PREFIX  = "/quant-bot/coinbase"
    COINBASE_API_MODE      = "advanced_trade"
    
    # Smart sizing with $1 floor (respects exchange minimums)
    MIN_NOTIONAL_FLOOR     = "1"
    FLAT_ORDER_NOTIONAL    = "1"
    MAKER_ONLY             = "true"
    MAKER_OFFSET           = "0.00015"
    
    # Strategy parameters
    ADX_THRESHOLD          = "20"
    COOLDOWN_BARS          = "4"
    TARGET_VOL             = "0.02"
  }
}
