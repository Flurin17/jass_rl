# Training and qualification on an M3 Pro

The supported local target is an Apple M3 Pro with 36 GB unified memory.  The
training path is MaskablePPO with centralized training/decentralized execution:
the critic may use the remaining hands during training, while the actor and all
evaluation policies receive only the 1,603-value canonical public observation.

## Install and verify

```bash
uv sync --extra rl --extra dev
MPLCONFIGDIR=./.cache/matplotlib XDG_CACHE_HOME=./.cache \
  .venv/bin/pytest -q
```

The measured local microbenchmark is saved in
`models/experiments/m3_training_resource_benchmark.json`.  With two environments,
a 128x128 network, and 512 requested updates it measured:

| Device | Steps/s | Peak RSS | Peak accelerator allocation |
| --- | ---: | ---: | ---: |
| CPU (4 threads) | 2,238 | 293 MiB | n/a |
| MPS | 217 | 415 MiB | 11 MiB |

For this small masked policy workload CPU is materially faster, so the audited
runs use `--device cpu`.  MPS remains available, but should be selected only
after rerunning `python -m rl.benchmark_training` with the intended network and
rollout sizes.  Both measurements are far below the 28 GB project ceiling.

## Reproduce the full-standard continuation

The strongest full Schieber checkpoint was continued from the strict trump-only
checkpoint with the complete standard profile and a deterministic 80/20
strategic/random opponent curriculum:

```bash
MPLCONFIGDIR=./.cache/matplotlib XDG_CACHE_HOME=./.cache \
  .venv/bin/python -m rl.train_selfplay \
  --fork-from models/experiments/ppo_verardo_full_v1/20260715_093548/model_final.zip \
  --allow-environment-fork \
  --profile standard \
  --opponent-mixture strategic=0.8,random=0.2 \
  --total-steps 3000000 --iterations 24 \
  --n-envs 8 --vec-env dummy --n-steps 1024 --batch-size 512 --n-epochs 4 \
  --net-arch 512,256 --learning-rate 0.00005 --ent-coef 0.002 \
  --gamma 1.0 --gae-lambda 0.98 \
  --reward-scale 0.004 --terminal-win-bonus 0.5 \
  --normalize-contract-reward --device cpu --seed 20260720 \
  --save-dir models/experiments/ppo_full_consolidation_v1
```

Every run writes an atomic `run_manifest.json`, hashed periodic checkpoints, and
a hashed final archive.  Resume with the same environment and optimization
arguments plus `--resume <checkpoint.zip>` and a larger `--total-steps`; the
loader rejects incompatible observations, rules, or opponent curricula unless
an explicit, recorded fork is requested.

## Qualification

The neural actor is combined with bounded public-information Monte Carlo search.
The actor may override search only for a high-confidence action inside the
configured raw-point gap.  The formal CLI validates the run manifest, model
hash, actor privacy boundary, rules profile, paired schedule, and every gate:

```bash
# Full Schieber, run once for each opponent (4,000 games / 2,000 pairs each).
.venv/bin/python -m rl.qualify_hybrid \
  models/experiments/ppo_full_consolidation_v1/20260715_095439/model_final.zip \
  --benchmark full --opponents strategic --episodes 4000 --seed 49001 \
  --determinizations 24 --max-rollouts 216 --common-random-numbers \
  --opponent-rollout-policy generic --max-search-gap 3 \
  --min-neural-probability 0.5 --device cpu \
  --output models/experiments/formal_full_strategic_4000_seed49001.json

# Pinned clean-room reference. Bid inference is public but reference-specific,
# so it may not be mislabeled or reused for the random-opponent report.
.venv/bin/python -m rl.qualify_hybrid \
  models/experiments/ppo_verardo_full_v1/20260715_093548/model_final.zip \
  --benchmark external --opponents verardo --episodes 4000 --seed 59001 \
  --determinizations 24 --max-rollouts 216 --common-random-numbers \
  --opponent-rollout-policy verardo --infer-opponent-bids \
  --max-search-gap 3 --min-neural-probability 0.5 --device cpu \
  --output models/experiments/formal_external_reference_4000_seed59001.json
```

Development screens are not qualification.  The exact gates, tie handling, and
pinned external commit are defined in [`benchmark_goal.md`](benchmark_goal.md).
Visualization work is deliberately blocked until all full and external reports
qualify.
