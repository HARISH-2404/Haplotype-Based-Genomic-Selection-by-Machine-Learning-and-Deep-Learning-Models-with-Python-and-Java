#!/usr/bin/env bash
# usage: ./run.sh [--schemes random_rep1,family_out] [--no-mlp]
cd "$(dirname "$0")"
./build.sh && java -Xmx2g -cp out org.hgs.Main --root .. "$@"
