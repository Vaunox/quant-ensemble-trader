#!/bin/bash
# setup_cloudwatch.sh
# Verifies this EC2 instance can send metrics to CloudWatch.
# Run once after attaching the IAM role (see PREREQUISITE below).
#
# PREREQUISITE — do this once in the AWS console:
#   1. IAM → Roles → Create Role → AWS Service → EC2
#      → Permissions: attach "CloudWatchFullAccess" (or custom with cloudwatch:PutMetricData)
#      → Name it: "algo_v2_cloudwatch_role"
#   2. EC2 console → select the c6a instance
#      → Actions → Security → Modify IAM Role → select "algo_v2_cloudwatch_role"
#
# After that, run: bash setup_cloudwatch.sh
#
# Metrics appear in: AWS Console → CloudWatch → Metrics → Custom Namespaces → algo_v2/training
# Dashboard:         CloudWatch → Dashboards → Create dashboard → add algo_v2/training metrics

set -e
cd "$(dirname "$0")/.."
source "${FINRL_ENV:-/home/ubuntu/finrl_env}/bin/activate"

echo "Testing CloudWatch access..."
python3 - << 'EOF'
import os
import boto3

cw = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
cw.put_metric_data(
    Namespace="algo_v2/test",
    MetricData=[{
        "MetricName": "SetupTest",
        "Value": 1.0,
        "Unit": "None",
        "Dimensions": [{"Name": "Status", "Value": "ok"}]
    }]
)
print("[OK] CloudWatch access verified.")
print("")
print("During training, metrics appear at:")
print("  AWS Console → CloudWatch → Metrics → Custom Namespaces → algo_v2/training")
print("")
print("Useful filters in CloudWatch dashboard:")
print("  Metric: Sharpe     | Dimension: Algorithm=ppo   (see all PPO bots)")
print("  Metric: Sharpe     | Dimension: Bot=bot_01_ppo_pure_mlp  (single bot)")
print("  Metric: RewardBest | Dimension: Algorithm=sac   (see SAC progress)")
EOF
