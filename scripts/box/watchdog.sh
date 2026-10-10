#!/bin/bash
# Independent backstop: self-stop after 30 min with no python/curl worker running.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | sed 's/^/export /')"
idle=0
while true; do
  if ps -eo args | grep -E "[p]ython scripts/12_|[c]url -sL" > /dev/null; then idle=0; else idle=$((idle+1)); fi
  if [ "$idle" -ge 30 ]; then
    curl -s -X PUT -H "Authorization: Bearer $CONTAINER_API_KEY" -d '{"state":"stopped"}' \
      "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" > /root/watchdog_stop.log 2>&1
    exit 0
  fi
  sleep 60
done
