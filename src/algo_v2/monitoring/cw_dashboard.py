"""
setup_cloudwatch_dashboard.py
Run once to create the CloudWatch dashboard + Logs Insights saved queries.

Usage:
  python3 setup_cloudwatch_dashboard.py

Dashboard: AWS Console → CloudWatch → Dashboards → algo_v2-training
Queries:   AWS Console → CloudWatch → Logs Insights → Saved queries
"""

import json
import boto3

from algo_v2.config import AWS_REGION

REGION = AWS_REGION
NS = "algo_v2/training"
LOG_GROUP = "algo_v2/training-logs"
DASH_NAME = "algo_v2-training"

ALGOS = ["ppo", "appo", "sac", "impala", "tqc", "marwil"]

# ── Dashboard widgets ──────────────────────────────────────────────────────────


def _metric(name, dim_name, dim_val, label=None):
    m = [NS, name, dim_name, dim_val]
    if label:
        return {
            "expression": None,
            "id": None,
            "label": label,
            "metricStat": {
                "metric": {"namespace": NS, "metricName": name, "dimensions": [{"name": dim_name, "value": dim_val}]},
                "period": 60,
                "stat": "Average",
            },
        }
    return [NS, name, dim_name, dim_val, {"label": dim_val, "period": 60, "stat": "Average"}]


def algo_sharpe_widget(algo, x, y):
    return {
        "type": "metric",
        "x": x,
        "y": y,
        "width": 8,
        "height": 6,
        "properties": {
            "title": f"Sharpe — {algo.upper()}",
            "view": "timeSeries",
            "stacked": False,
            "region": REGION,
            "period": 60,
            "stat": "Average",
            "metrics": [[NS, "Sharpe", "Algorithm", algo]],
            "yAxis": {"left": {"min": -1}},
        },
    }


def reward_widget(algo, x, y):
    return {
        "type": "metric",
        "x": x,
        "y": y,
        "width": 8,
        "height": 6,
        "properties": {
            "title": f"RewardBest — {algo.upper()}",
            "view": "timeSeries",
            "stacked": False,
            "region": REGION,
            "period": 60,
            "stat": "Maximum",
            "metrics": [[NS, "RewardBest", "Algorithm", algo]],
        },
    }


def gan_widget(x, y):
    return {
        "type": "metric",
        "x": x,
        "y": y,
        "width": 12,
        "height": 6,
        "properties": {
            "title": "GAN Training Losses (Phase 1)",
            "view": "timeSeries",
            "stacked": False,
            "region": REGION,
            "period": 60,
            "stat": "Average",
            "metrics": [
                [NS, "GAN_DiscriminatorLoss", "Phase", "GAN", {"label": "Discriminator"}],
                [NS, "GAN_GeneratorLoss", "Phase", "GAN", {"label": "Generator"}],
            ],
        },
    }


def log_widget(x, y):
    return {
        "type": "log",
        "x": x,
        "y": y,
        "width": 24,
        "height": 6,
        "properties": {
            "title": "Live Training Log (last 200 lines)",
            "region": REGION,
            "logGroupNames": [LOG_GROUP],
            "query": "fields @timestamp, @message | sort @timestamp desc | limit 200",
            "view": "table",
        },
    }


def progress_widget(x, y):
    return {
        "type": "log",
        "x": x,
        "y": y,
        "width": 12,
        "height": 6,
        "properties": {
            "title": "Bot Completions",
            "region": REGION,
            "logGroupNames": [LOG_GROUP],
            "query": "fields @timestamp, @message | filter @message like /SUCCESS|FATAL|SKIP/ | sort @timestamp desc | limit 100",
            "view": "table",
        },
    }


def iter_widget(x, y):
    return {
        "type": "log",
        "x": x,
        "y": y,
        "width": 12,
        "height": 6,
        "properties": {
            "title": "Iteration Progress",
            "region": REGION,
            "logGroupNames": [LOG_GROUP],
            "query": "fields @timestamp, @message | filter @message like /iter.*elapsed/ | sort @timestamp desc | limit 100",
            "view": "table",
        },
    }


def failures_widget(x, y):
    return {
        "type": "log",
        "x": x,
        "y": y,
        "width": 12,
        "height": 6,
        "properties": {
            "title": "Failures & Retries",
            "region": REGION,
            "logGroupNames": [LOG_GROUP],
            "query": "fields @timestamp, @message | filter @message like /FATAL|FAILED|OOM|killed/ | sort @timestamp desc | limit 50",
            "view": "table",
        },
    }


def build_widgets():
    # Layout: row 0 — Sharpe per algo (3 across, 2 rows = 6 algos)
    #         row 2 — RewardBest per algo
    #         row 4 — GAN losses + live log
    #         row 5 — completions + failures
    widgets = []
    for i, algo in enumerate(ALGOS):
        col = (i % 3) * 8
        row = (i // 3) * 6
        widgets.append(algo_sharpe_widget(algo, col, row))

    for i, algo in enumerate(ALGOS):
        col = (i % 3) * 8
        row = 12 + (i // 3) * 6
        widgets.append(reward_widget(algo, col, row))

    widgets.append(gan_widget(0, 24))
    widgets.append(log_widget(0, 30))
    widgets.append(progress_widget(0, 36))
    widgets.append(iter_widget(12, 36))
    widgets.append(failures_widget(0, 42))
    return widgets


SAVED_QUERIES = [
    {
        "name": "algo_v2 — Bot Progress",
        "query": (
            "fields @timestamp, @message\n"
            "| filter @message like /Assembling|SUCCESS|FATAL|SKIP|iter/\n"
            "| sort @timestamp asc\n"
            "| limit 500"
        ),
    },
    {
        "name": "algo_v2 — Sharpe Summary",
        "query": (
            "fields @timestamp, @message\n"
            "| filter @message like /Sharpe:/\n"
            '| parse @message "Sharpe: *" as sharpe\n'
            "| sort @timestamp desc\n"
            "| limit 200"
        ),
    },
    {
        "name": "algo_v2 — Failures & OOM",
        "query": (
            "fields @timestamp, @message\n"
            "| filter @message like /FATAL|FAILED|OOM|killed|out of memory/\n"
            "| sort @timestamp desc\n"
            "| limit 100"
        ),
    },
    {
        "name": "algo_v2 — Training Speed (iter times)",
        "query": (
            "fields @timestamp, @message\n| filter @message like /iter.*elapsed/\n| sort @timestamp desc\n| limit 200"
        ),
    },
]


def main():
    cw = boto3.client("cloudwatch", region_name=REGION)
    logs = boto3.client("logs", region_name=REGION)

    # Default time window: last 12 hours — always shows current run without manual adjustment
    dashboard_body = json.dumps(
        {
            "start": "-PT12H",
            "end": "PT0H",
            "widgets": build_widgets(),
        }
    )

    # ── Create / update dashboard ──────────────────────────────────────────────
    cw.put_dashboard(DashboardName=DASH_NAME, DashboardBody=dashboard_body)
    print(f"[OK] Dashboard created: {DASH_NAME}")
    print(f"     https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#dashboards:name={DASH_NAME}")

    # ── Logs Insights saved queries (Part C) ───────────────────────────────────
    for q in SAVED_QUERIES:
        try:
            logs.put_query_definition(
                name=q["name"],
                logGroupNames=[LOG_GROUP],
                queryString=q["query"],
            )
            print(f"[OK] Saved query: {q['name']}")
        except Exception as e:
            print(f"[WARN] Could not save query '{q['name']}': {e}")

    print("\nAll done. Open the dashboard at the URL above.")


if __name__ == "__main__":
    main()
