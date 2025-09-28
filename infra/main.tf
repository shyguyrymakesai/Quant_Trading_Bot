data "aws_caller_identity" "me" {}
data "aws_region" "current" {}

locals {
  merged_env = merge(
    { AWS_REGION = var.region },
    var.container_env,
    { S3_BUCKET_TRADES = aws_s3_bucket.trades.bucket }
  )
}

# Use default VPC + its default subnets (keeps cost low, no NAT)
data "aws_vpc" "default" { default = true }
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ECR repo for your image
resource "aws_ecr_repository" "repo" {
  name                 = "${var.project}-app"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# CloudWatch Logs group
resource "aws_cloudwatch_log_group" "app" {
  name              = "/${var.project}/app"
  retention_in_days = 14
}

# S3 bucket to store artifacts/trades (easy + cheap)
resource "aws_s3_bucket" "trades" {
  bucket        = "${var.project}-trades-${data.aws_caller_identity.me.account_id}"
  force_destroy = true
}

# IAM roles
data "aws_iam_policy_document" "assume_ecs" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task" {
  name               = "${var.project}-task-role"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs.json
}

resource "aws_iam_role" "exec" {
  name               = "${var.project}-exec-role"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs.json
}

# task permissions: write to S3 & Logs
data "aws_iam_policy_document" "task_policy" {
  statement {
    actions   = ["s3:PutObject","s3:PutObjectAcl","s3:AbortMultipartUpload","s3:ListBucket","s3:GetObject"]
    resources = [aws_s3_bucket.trades.arn, "${aws_s3_bucket.trades.arn}/*"]
  }
  statement {
    actions   = ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"]
    resources = ["*"]
  }
}
resource "aws_iam_policy" "task" {
  name   = "${var.project}-task-policy"
  policy = data.aws_iam_policy_document.task_policy.json
}
resource "aws_iam_role_policy_attachment" "task_attach" {
  role       = aws_iam_role.task.name
  policy_arn = aws_iam_policy.task.arn
}

# execution role needs ECR + logs
resource "aws_iam_role_policy_attachment" "exec_attach" {
  role       = aws_iam_role.exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Cluster
resource "aws_ecs_cluster" "this" { name = "${var.project}-cluster" }

# Task Definition (single container)
resource "aws_ecs_task_definition" "app" {
  family                   = "${var.project}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.exec.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = "app"
    image     = "${aws_ecr_repository.repo.repository_url}:latest"
    essential = true
    environment = [for k, v in local.merged_env : { name = k, value = v }]
    logConfiguration = {
      logDriver = "awslogs",
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name,
        awslogs-region        = var.region,
        awslogs-stream-prefix = "ecs"
      }
    }
  }])
}

# Allow reading Coinbase SSM parameters
data "aws_iam_policy_document" "ssm_read" {
  statement {
    actions   = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"]
    resources = [
      "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.me.account_id}:parameter/quant-bot/coinbase/*"
    ]
  }
  statement {
    actions   = ["kms:Decrypt"]
    resources = ["*"] # default AWS managed key for SSM
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${data.aws_region.current.name}.amazonaws.com"]
    }
  }
}

resource "aws_iam_policy" "ssm_read" {
  name   = "${var.project}-ssm-read"
  policy = data.aws_iam_policy_document.ssm_read.json
}

resource "aws_iam_role_policy_attachment" "task_ssm_attach" {
  role       = aws_iam_role.task.name
  policy_arn = aws_iam_policy.ssm_read.arn
}

# EventBridge schedule → run Fargate task
resource "aws_iam_role" "events_invoke" {
  name               = "${var.project}-events-invoke"
  assume_role_policy = jsonencode({
    Version="2012-10-17",
    Statement=[{ Action="sts:AssumeRole", Effect="Allow", Principal={ Service="events.amazonaws.com" }}]
  })
}

resource "aws_iam_role_policy" "events_invoke" {
  name = "${var.project}-events-invoke-policy"
  role = aws_iam_role.events_invoke.id
  policy = jsonencode({
    Version="2012-10-17",
    Statement=[
      { Effect="Allow", Action=["ecs:RunTask"], Resource=[aws_ecs_task_definition.app.arn] },
      { Effect="Allow", Action=["iam:PassRole"], Resource=[aws_iam_role.task.arn, aws_iam_role.exec.arn] }
    ]
  })
}

resource "aws_cloudwatch_event_rule" "every_30" {
  name                = "${var.project}-every-30"
  schedule_expression = var.schedule_expression
}

resource "aws_cloudwatch_event_target" "run_task" {
  rule      = aws_cloudwatch_event_rule.every_30.name
  target_id = "run-task"
  arn       = aws_ecs_cluster.this.arn
  role_arn  = aws_iam_role.events_invoke.arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.app.arn
    launch_type         = "FARGATE"
    platform_version    = "LATEST"
    task_count          = 1

    network_configuration {
      subnets          = data.aws_subnets.default.ids
      assign_public_ip = true
      # security_groups = []  # optional
    }
  }
}
