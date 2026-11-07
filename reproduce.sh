#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA="${DATA:-data}"
RUNS="${RUNS:-runs}"
SCORES="${SCORES:-results/scores}"
for config in $(cd configs && find . -name '*.yaml' | sed 's|^\./||; s|\.yaml$||' | sort); do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        python -m ccw_net.run --config "$config" --seed "$seed" --data-root "$DATA" --runs "$RUNS"
    done
done
python -m ccw_net.scoring --runs "$RUNS" --scores "$SCORES" --data-root "$DATA"
python -m ccw_net.tables --scores "$SCORES"
