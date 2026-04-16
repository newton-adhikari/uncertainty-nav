#!/usr/bin/env bash
# Train all methods in parallel with terminal output.
# Wave 1: 5 ensemble members | Wave 2: 4 baselines | Wave 3: 5 extra members
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs