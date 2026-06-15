"""
cw_sidecar.py — CloudWatch metrics + logs sidecar.

Streams ALL log files to CloudWatch Logs + pushes training metrics:
  • phase6.log            → stream: phase6-master-{ts}
  • parallel_logs/group*.log → stream: phase6-group0-{ts}, phase6-group1-{ts}, ...

Each run creates fresh streams (timestamp suffix). Only NEW lines are shipped —
no replay of logs from previous runs.

Metrics:   AWS Console → CloudWatch → Metrics → algo_v2/training
Logs:      AWS Console → CloudWatch → Logs Insights → algo_v2/training-logs
           Dashboard:   CloudWatch → Dashboards → algo_v2-training
"""

import glob
import os
import re
import threading
import time
from datetime import datetime, timezone

import boto3

from algo_v2.config import AWS_REGION, PROJECT_ROOT

LOG_PATH = os.path.join(PROJECT_ROOT, "phase6.log")
PARALLEL_DIR = os.path.join(PROJECT_ROOT, "parallel_logs")
REGION = AWS_REGION
NS = "algo_v2/training"
LOG_GROUP = "algo_v2/training-logs"
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

cw = boto3.client("cloudwatch", region_name=REGION)
logs = boto3.client("logs", region_name=REGION)

try:
    logs.create_log_group(logGroupName=LOG_GROUP)
except Exception:
    pass

bot_stats = {}
_metrics_lock = threading.Lock()


# ── CloudWatch Logs helpers ────────────────────────────────────────────────────


def _ensure_stream(stream_name):
    try:
        logs.create_log_stream(logGroupName=LOG_GROUP, logStreamName=stream_name)
    except Exception:
        pass


def _flush(stream_name, buf):
    if not buf:
        return []
    batch, rest = buf[:50], buf[50:]
    try:
        logs.put_log_events(logGroupName=LOG_GROUP, logStreamName=stream_name, logEvents=batch)
    except Exception:
        pass
    return rest


# ── Metrics parsing ────────────────────────────────────────────────────────────


def _parse_metrics(line, current_bot):
    m = re.search(r"---> Assembling (bot_\S+?) \.\.\.", line)
    if m:
        current_bot = m.group(1)

    if not current_bot:
        return current_bot

    with _metrics_lock:
        if current_bot not in bot_stats:
            bot_stats[current_bot] = {}

    algo = current_bot.split("_")[2] if len(current_bot.split("_")) > 2 else "unknown"
    dims_bot = [{"Name": "Bot", "Value": current_bot}, {"Name": "Algorithm", "Value": algo}]
    dims_algo = [{"Name": "Algorithm", "Value": algo}]
    dims_gan = [{"Name": "Phase", "Value": "GAN"}]

    m = re.search(r"\[(\d+)/2000\] Loss_D: ([\d.\-]+) Loss_G: ([\d.\-]+)", line)
    if m:
        try:
            cw.put_metric_data(
                Namespace=NS,
                MetricData=[
                    {
                        "MetricName": "GAN_DiscriminatorLoss",
                        "Value": float(m.group(2)),
                        "Unit": "None",
                        "Dimensions": dims_gan,
                    },
                    {
                        "MetricName": "GAN_GeneratorLoss",
                        "Value": float(m.group(3)),
                        "Unit": "None",
                        "Dimensions": dims_gan,
                    },
                ],
            )
        except Exception:
            pass

    if "total_reward:" in line:
        try:
            val = float(line.split(":")[-1].strip())
            cw.put_metric_data(
                Namespace=NS,
                MetricData=[{"MetricName": "RewardBest", "Value": val, "Unit": "None", "Dimensions": dims_bot}],
            )
        except Exception:
            pass

    elif "Sharpe:" in line:
        try:
            val = float(line.split(":")[-1].strip())
            cw.put_metric_data(
                Namespace=NS,
                MetricData=[
                    {"MetricName": "Sharpe", "Value": val, "Unit": "None", "Dimensions": dims_bot},
                    {"MetricName": "Sharpe", "Value": val, "Unit": "None", "Dimensions": dims_algo},
                ],
            )
        except Exception:
            pass

    return current_bot


# ── Log file tailer ────────────────────────────────────────────────────────────


def tail_log(log_path, stream_name):
    """Tail log file from current end. Only new lines are shipped — no old-log replay."""
    _ensure_stream(stream_name)
    current_bot = None
    buf = []
    last_flush = time.time()

    # Wait for file to exist (phase6.log may not exist yet at startup)
    while not os.path.exists(log_path):
        time.sleep(3)

    with open(log_path, "r", errors="replace") as f:
        f.seek(0, 2)  # start at end — ignore anything written before sidecar started
        while True:
            line = f.readline()
            if not line:
                if (time.time() - last_flush) >= 5:
                    buf = _flush(stream_name, buf)
                    last_flush = time.time()
                time.sleep(2)
                continue
            buf.append({"timestamp": int(time.time() * 1000), "message": line.rstrip()})
            current_bot = _parse_metrics(line, current_bot)
            if len(buf) >= 50 or (time.time() - last_flush) >= 5:
                buf = _flush(stream_name, buf)
                last_flush = time.time()


# ── Group log watcher ──────────────────────────────────────────────────────────


def watch_group_logs():
    """Polls parallel_logs/ every 10s for new group*.log files and spawns a tailer per file."""
    known = set()
    while True:
        try:
            files = glob.glob(os.path.join(PARALLEL_DIR, "group*.log"))
        except Exception:
            files = []

        for fpath in sorted(files):
            if fpath in known:
                continue
            known.add(fpath)
            basename = os.path.basename(fpath)  # group0_bots1-2.log
            m_grp = re.match(r"group(\d+)", basename)
            group_num = int(m_grp.group(1)) if m_grp else 0
            stream = f"phase6-group{group_num:02d}-{RUN_TS}"  # group00, group01, ..., group20
            t = threading.Thread(
                target=tail_log,
                args=(fpath, stream),
                daemon=True,
                name=f"tailer-{group_num}",
            )
            t.start()
            print(f"[watcher] {basename} → CloudWatch stream: {stream}")

        time.sleep(10)


# ── Entry point ────────────────────────────────────────────────────────────────


def main():
    print(f"Run ID: {RUN_TS}")
    print(f"CloudWatch Logs group: {LOG_GROUP}")

    # Master log tailer
    threading.Thread(
        target=tail_log, args=(LOG_PATH, f"phase6-master-{RUN_TS}"), daemon=True, name="tailer-master"
    ).start()
    print(f"Tailing phase6.log → phase6-master-{RUN_TS}")

    # Group log watcher
    threading.Thread(target=watch_group_logs, daemon=True, name="group-watcher").start()
    print(f"Watching {PARALLEL_DIR}/group*.log ...")

    # Keep main thread alive
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
