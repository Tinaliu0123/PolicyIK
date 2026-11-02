#!/usr/bin/env bash
# Record the scripted demonstrations for the tasks in data/demo_tasks.json on the real arm (into data/demos.npz).
# An existing output is appended to, so an interrupted collection continues where it stopped.
#
#   bash scripts/collect_demos.sh --dry-run          # print the plan, move nothing
#   bash scripts/collect_demos.sh                    # all remaining tasks
#   bash scripts/collect_demos.sh --start 40 --n 10  # a slice
set -eu
cd "$(dirname "$0")/.."
exec python3 -m policyik_hw.collect_demos --config config.yaml --task-file data/demo_tasks.json "$@"
