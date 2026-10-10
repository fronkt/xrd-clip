#!/bin/bash
# Phase 0 box queue: strong-baseline sweep (4 parallel workers, VC on GPU) + opXRD download.
# Ends with a self-stop after a grace period for the pull; watchdog.sh is the independent backstop.
cd /root/xrd-clip
export PYTHONPATH=src OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 NTHREADS=6
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | sed 's/^/export /')"
selfstop() { curl -s -X PUT -H "Authorization: Bearer $CONTAINER_API_KEY" -d '{"state":"stopped"}' \
  "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" > /root/selfstop.log 2>&1; }
mkdir -p results logs
run() { python scripts/12_strong_baselines.py --only "$1" --out results/sb_$2.json \
          --ranks_out results/sb_$2_ranks.npz > logs/sb_$2.log 2>&1; echo "EXIT $?" >> logs/sb_$2.log; }
run clean,zero_offset w1 &
run strain,domain_nm w2 &
run march,counts,thermal w3 &
run field,sign w4 &
( mkdir -p /root/opxrd && cd /root/opxrd && \
  curl -sL -o opxrd.zip https://zenodo.org/api/records/15298026/files/opxrd.zip/content && \
  unzip -q opxrd.zip && echo OPXRD_OK > /root/opxrd/STATUS || echo OPXRD_FAIL > /root/opxrd/STATUS ) &
wait
touch /root/DONE
sleep 2700          # 45 min for the pull, then stop (not destroy: disk survives)
selfstop
