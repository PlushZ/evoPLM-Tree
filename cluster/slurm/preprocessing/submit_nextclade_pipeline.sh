#!/usr/bin/env bash
set -euo pipefail

# ---- config (simple & safe) ----
SHARD_LIST="data/raw/ncbi_virus/shards/shard_list.txt"
ARRAY_JOB="cluster/slurm/preprocessing/03_nextclade_array.sh"
MERGE_JOB="cluster/slurm/preprocessing/04_merge_nextclade.sh"

BATCH_SIZE=${BATCH_SIZE:-50}   # submit at most this many tasks at once (fits 50 limit)
POLL_SECS=${POLL_SECS:-120}     # how often to check if the wave finished
THROTTLE=${THROTTLE:-$BATCH_SIZE}  # cap concurrency inside each wave (can leave as default)

mkdir -p logs


N=$(wc -l < "$SHARD_LIST")
echo "Found $N shards."

start=1
while [ $start -le $N ]; do
  end=$(( start + BATCH_SIZE - 1 ))
  if [ $end -gt $N ]; then end=$N; fi

  count=$(( end - start + 1 ))
  array_spec="1-${count}%${THROTTLE}"
  offset=$(( start - 1 ))   # maps local 1..count -> global shard lines start..end

  echo "Submitting wave for shards ${start}-${end} (count=${count})"
  cmd=( sbatch --parsable
        --array="$array_spec"
        --export=ALL,OFFSET="$offset",SHARD_LIST="$SHARD_LIST"
        "$ARRAY_JOB" )
  echo "Submitting: ${cmd[*]}"
  jid=$("${cmd[@]}")
  echo "Wave job id: $jid"

  # ---- wait for this wave to finish ----
  echo "Waiting for wave to complete (polling every ${POLL_SECS}s)..."
  while : ; do
    # 'squeue -h -j $jid' prints nothing when the array + its tasks are gone
    if [ -z "$(squeue -h -j "$jid" -o %i 2>/dev/null)" ]; then
      echo "Wave ${start}-${end} finished."
      break
    fi
    running=$(squeue -h -j "$jid" | wc -l | tr -d ' ')
    echo "  $(date '+%H:%M:%S') still active tasks: ${running}"
    sleep "$POLL_SECS"
  done

  start=$(( end + 1 ))
done

# ----  merge step after all waves ----
echo "Submitting merge job..."
jid_merge=$(sbatch --parsable "$MERGE_JOB")
echo "Submitted merge -> $jid_merge"
