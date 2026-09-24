#!/usr/bin/env bash
# feat-186 (results/frontier_levels_note.md): three judge-only calls, one per idle card.
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
go() { local gpu=$1 tag=$2 arms=$3
  ( CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python analysis/frontier_levels.py --arms "$arms" --tag "$tag" \
      --out results > output/logs/feat186_$tag.log 2>&1 \
    && touch output/logs/feat186_$tag.done || touch output/logs/feat186_$tag.fail ) & }
go 1 a sel_n1,sel_n2,sel_n4,sel_n8,sel_n16
go 2 b sel_n32,sel_n64,anchor_k0,met_k0.5,met_k1
go 4 c met_k3,met_k5,met_k10,met_k20
wait
