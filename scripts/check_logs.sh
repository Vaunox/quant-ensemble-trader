#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/phase6.log"
LAST="$ROOT/.last_check"
NEW_LINES=$(wc -l < "$LOG")
if [ -f "$LAST" ]; then
    OLD_LINES=$(cat "$LAST")
    START=$((OLD_LINES - 2000))
    if [ $START -lt 1 ]; then START=1; fi
else
    START=1
fi
echo "$NEW_LINES" > "$LAST"

# Delta scan: ignore "raise an error in the future" and "DeprecationWarning"
tail -n +"$START" "$LOG" | grep -a -i -A 5 -B 2 -e "error" -e "exception" -e "traceback" -e "fatal" | grep -v -i "deprecationwarning" | grep -v -i "future!" > /tmp/scan_errors.log

BOT_NAME=$(grep -a "Commencing Live Deep Training" "$LOG" | tail -n 1 | awk -F'[][]' '{print $2}')
ITER=$(grep -a -e "-> Iteration" "$LOG" | tail -n 1 | awk '{print $3}' | tr -d ':')

if [ ! -s /tmp/scan_errors.log ]; then
    echo "STATUS_OK: Bot: $BOT_NAME, Progress: $ITER%"
    echo "--- LAST 5 LOG LINES ---"
    tail -n 5 "$LOG"
else
    cat /tmp/scan_errors.log
fi