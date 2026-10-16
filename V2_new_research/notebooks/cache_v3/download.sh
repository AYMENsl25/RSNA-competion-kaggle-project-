#!/bin/bash
# Wait for each v3 cache shard kernel, download its output, verify sha256. Log: outputs/cache_v3/download.log
OUT="/c/Users/msi/Downloads/kaggle project RSNA/kaggle project RSNA/V2_new_research/outputs/cache_v3"
mkdir -p "$OUT"; LOG="$OUT/download.log"
for k in 0 1 2 3 4; do
  K=aymenslimani/rsna-knee-v3-cache-s$k
  until st=$(kaggle kernels status $K 2>&1); echo "$st" | grep -qiE "complete|error|cancel"; do sleep 60; done
  echo "$(date +%T) s$k status: $st" >> "$LOG"
  echo "$st" | grep -qi complete || continue
  D="$OUT/s$k"; mkdir -p "$D"
  [ -f "$D/OK" ] && continue
  echo "$(date +%T) s$k download start" >> "$LOG"
  kaggle kernels output $K -p "$D" >> "$LOG" 2>&1
  if (cd "$D" && sha256sum -c checksums.sha256 >> "$LOG" 2>&1); then touch "$D/OK"; echo "$(date +%T) s$k VERIFIED $(du -sh "$D"|cut -f1)" >> "$LOG"; else echo "$(date +%T) s$k CHECKSUM FAIL" >> "$LOG"; fi
done
echo "$(date +%T) ALL DONE" >> "$LOG"
