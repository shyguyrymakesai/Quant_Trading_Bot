output "ecr_repo_url"     { value = aws_ecr_repository.repo.repository_url }
output "s3_bucket_trades" { value = aws_s3_bucket.trades.bucket }
output "ecs_cluster"      { value = aws_ecs_cluster.this.name }
output "log_group"        { value = aws_cloudwatch_log_group.app.name }
