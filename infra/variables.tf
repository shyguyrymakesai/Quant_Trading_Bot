variable "project" { default = "quant-bot" }
variable "region"  { default = "us-east-1" }

# schedule: every 30 mins
variable "schedule_expression" {
  default = "cron(0,30 * * * ? *)"
}

variable "cpu"    { default = 256 }   # 0.25 vCPU
variable "memory" { default = 512 }   # 0.5 GB

variable "container_env" {
  type = map(string)
  default = {
    AWS_REGION       = "us-east-1"
    S3_BUCKET_TRADES = ""            # filled after apply via outputs (you can set later with tfvars)
    SYMBOLS          = "BTC-USD,ETH-USD"
    TF               = "1h"
  }
}
