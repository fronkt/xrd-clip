#!/usr/bin/env bash
# R5 (open stand-in): COD CIFs by rsync, then stick simulation with the unchanged stage-2 simulator. Open data only
# (COD, CC0). Needs cod/library_meta.json + cod/rsync_paths.txt from scripts/31 (paths) next to the repo code.
#   setsid nohup bash scripts/box/cod_box.sh > /workspace/cod.log 2>&1 &
# Self-stop 2 h after COD_DONE, or after 30 min with no rsync/python worker.
set -u
cd /workspace/xrd || exit 1
selfstop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "[cod] SELF-STOP $(date -u +%FT%TZ) ($1)"
  curl -s -X PUT -H "Authorization: Bearer $key" -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/$id/"
}
(
  idle=0
  while true; do
    sleep 60
    if ps -eo args | grep -qE "[r]sync|[x]rdclip.data.simulate|[3]1_cod_library|[c]ritic2"; then idle=0; else idle=$((idle + 1)); fi
    [ -f /workspace/COD_DONE ] && exit 0
    if [ $idle -ge 30 ]; then selfstop "watchdog: no worker for 30 min"; exit 0; fi
  done
) &
echo "[cod] start $(date -u +%FT%TZ)"
command -v rsync > /dev/null || (apt-get update -qq && apt-get install -y -qq rsync > /dev/null)
pip install -q pymatgen gemmi > /dev/null 2>&1
quota=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${quota:-0}" -gt 0 ] || quota=$(nproc)
W=$(( quota - 4 )); [ $W -gt 64 ] && W=64; [ $W -lt 2 ] && W=2
mkdir -p cod/cif logs
for try in 1 2 3 4; do
  rsync -a --files-from=cod/rsync_paths.txt rsync://www.crystallography.net/cif/ cod/cif/ >> logs/cod_rsync.log 2>&1 && break
  echo "[cod] rsync attempt $try failed, retrying" ; sleep 60
done
echo "[cod] CIFs: $(find cod/cif -name '*.cif' | wc -l) of $(wc -l < cod/rsync_paths.txt)"
export PYTHONPATH=src OMP_NUM_THREADS=1
python scripts/31_cod_library.py shards --meta cod/library_meta.json --cif_root cod/cif --out cod/shards --n_shards 6
python -m xrdclip.data.simulate --in cod/shards --out cod/sticks --workers "$W" > logs/cod_sim.log 2>&1
echo "EXIT $?" >> logs/cod_sim.log
tar -cf /workspace/cod_out.tar cod/sticks logs/cod_sim.log logs/cod_rsync.log
md5sum /workspace/cod_out.tar > /workspace/cod_out.tar.md5
touch /workspace/COD_DONE
echo "[cod] COD_DONE $(du -h /workspace/cod_out.tar | cut -f1) $(date -u +%FT%TZ)"
for i in $(seq 1 120); do
  [ -f /workspace/COD_PULLED ] && { echo "[cod] pulled"; exit 0; }
  sleep 60
done
selfstop "2 h after COD_DONE"
