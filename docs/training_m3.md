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
a 128x128 network, and 512 requested timesteps it measured:

| Device | Steps/s | Peak RSS | Peak accelerator allocation |
| --- | ---: | ---: | ---: |
| CPU (4 threads) | 2,238 | 293 MiB | n/a |
| MPS | 217 | 415 MiB | 11 MiB |

For this small masked policy workload CPU is materially faster, so the audited
runs use `--device cpu`.  MPS remains available, but should be selected only
after rerunning `python -m rl.benchmark_training` with the intended network and
rollout sizes.  Both measurements are far below the 28 GB project ceiling.

Training now sets PyTorch's intra-op pool explicitly instead of accepting its
host-dependent default.  On this 12-core M3 Pro, three runs of the actual
512x256 CTDE network (`n_steps=1024`, `n_envs=8`, four epochs) averaged 2,708
steps/s with eight threads, compared with 2,638 steps/s with PyTorch's former
six-thread setting.  Twelve threads averaged 2,589 steps/s: the extra
coordination cost outweighed the added cores for this network.  The portable
default is therefore `--cpu-threads 8` (or every available CPU on smaller
machines).  Use `--cpu-threads 12` when experimenting with larger networks, or
reduce it to keep more desktop headroom; every value is recorded in the run
manifest.

The thread sweep used seeds `20260717`–`20260719`, one 1,024-step rollout per
environment (8,192 transitions per run), eight environments, batch size 512,
four epochs, and the 512/256 network:

| PyTorch threads | Seed results (steps/s) | Mean steps/s |
| ---: | --- | ---: |
| 4 | 2,515.4 · 2,747.3 · 2,712.5 | 2,658.4 |
| 6 | 2,769.9 · 2,518.8 · 2,625.3 | 2,638.0 |
| 8 | 2,771.1 · 2,698.6 · 2,653.2 | **2,707.6** |
| 12 | 2,572.3 · 2,597.3 · 2,596.4 | 2,588.7 |

Reproduce one cell with:

```bash
.venv/bin/python -m rl.benchmark_training --devices cpu \
  --n-steps 1024 --n-envs 8 --batch-size 512 --n-epochs 4 --rollouts 1 \
  --net-arch 512,256 --cpu-threads 8 --seed 20260717
```

## Production resource gate

The canonical resource evidence is the clean manifested run at
`models/experiments/m3_resource_gate_stock_v2/20260715_155149/run_manifest.json`.
Unlike the small device comparison above, it exercises the production 512/256
CTDE network, full standard Schieber environment, eight environments, and the
80/20 strategic/random curriculum.  It was run on CPU from clean commit
`dd587ab4a23042cbf5f7b76ef7a48af1c83625aa`:

| CPU threads | Transitions | Learning time | Learning throughput | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 8,192 | 2.6197 s | 3,127.116 steps/s | 471,007,232 bytes (449.1875 MiB) |

The manifest also records seed `20260722`, observation schema 2, the full rules
and environment payload, runtime versions, and SHA-256 values for its checkpoint
and final archive.  This one-rollout run validates the production architecture's
resource envelope and resumable artifact path.  It does not validate playing
strength and was not the invocation that produced the historical qualified
model.

Reproduce its configuration with:

```bash
MPLCONFIGDIR=./.cache/matplotlib XDG_CACHE_HOME=./.cache \
  .venv/bin/python -m rl.train_selfplay \
  --profile standard --total-steps 8192 --iterations 1 \
  --n-envs 8 --vec-env dummy --n-steps 1024 --batch-size 512 --n-epochs 4 \
  --cpu-threads 8 --net-arch 512,256 \
  --learning-rate 0.00005 --ent-coef 0.002 \
  --gamma 1.0 --gae-lambda 0.98 \
  --reward-scale 0.004 --terminal-win-bonus 0.5 \
  --opponent-mixture strategic=0.8,random=0.2 \
  --normalize-contract-reward --device cpu --seed 20260722 \
  --save-dir models/experiments/m3_resource_gate_stock_v2
```

## Full-standard continuation

The strongest full Schieber checkpoint was continued from the strict trump-only
checkpoint with the complete standard profile and a deterministic 80/20
strategic/random opponent curriculum:

The qualified historical manifest does not record its thread count.  It likely
used PyTorch's then-observed six-thread host default.  To reproduce that likely
resource setting, use `--cpu-threads 6`; exact weights still require the original
manifested lineage and deterministic inputs.  The command below uses the newly
measured eight-thread recommendation.  It reproduces the training design, not
the historical invocation, and intentionally produces a new model hash and
trajectory.

```bash
MPLCONFIGDIR=./.cache/matplotlib XDG_CACHE_HOME=./.cache \
  .venv/bin/python -m rl.train_selfplay \
  --fork-from models/experiments/ppo_verardo_full_v1/20260715_093548/model_final.zip \
  --allow-environment-fork \
  --profile standard \
  --opponent-mixture strategic=0.8,random=0.2 \
  --total-steps 3000000 --iterations 24 \
  --n-envs 8 --vec-env dummy --n-steps 1024 --batch-size 512 --n-epochs 4 \
  --cpu-threads 8 \
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
# Full Schieber Stock-aware v2 shard. Repeat for `random` and `strategic`, with
# seeds 49001, 49501, 50001, and 50501 (1,000 games / 500 pairs per shard).
.venv/bin/python -m rl.qualify_hybrid \
  models/experiments/ppo_full_consolidation_v1/20260715_095439/model_final.zip \
  --benchmark full --opponents strategic --episodes 1000 --seed 49001 \
  --determinizations 24 --max-rollouts 216 --common-random-numbers \
  --opponent-rollout-policy generic --max-search-gap 3 \
  --min-neural-probability 0.5 --device cpu \
  --output models/experiments/formal_full_strategic_stock_v2_shard_49001_1000.json

# Substitute `--opponents random` and this output pattern for the other track:
# models/experiments/formal_full_random_stock_v2_shard_49001_1000.json

# External reference shard. Repeat with seeds 59001, 59501, 60001, and 60501.
.venv/bin/python -m rl.qualify_hybrid \
  models/experiments/ppo_verardo_full_v1/20260715_093548/model_final.zip \
  --benchmark external --opponents verardo --episodes 1000 --seed 59001 \
  --determinizations 24 --max-rollouts 216 --common-random-numbers \
  --opponent-rollout-policy verardo --infer-opponent-bids \
  --terminal-win-bonus 30 --max-search-gap 3 \
  --min-neural-probability 0.5 --device cpu \
  --output models/experiments/formal_external_reference_shard_59001_1000.json

# External-random shard. Repeat with seeds 61001, 61501, 62001, and 62501.
.venv/bin/python -m rl.qualify_hybrid \
  models/experiments/ppo_verardo_full_v1/20260715_093548/model_final.zip \
  --benchmark external --opponents verardo-random --episodes 1000 --seed 61001 \
  --determinizations 24 --max-rollouts 216 --common-random-numbers \
  --opponent-rollout-policy random --terminal-win-bonus 60 \
  --max-search-gap 3 --min-neural-probability 0.5 --device cpu \
  --output models/experiments/formal_external_random_shard_61001_1000.json

# Merge each four-shard family; the merger rejects overlapping deal seeds,
# policy-implementation/configuration drift, dirty provenance, and totals other
# than 4,000 games. The canonical full policy implementation SHA-256 is
# d99631fb1b71693cd1dea729ae7ba413ba09865bd286e3017d260ce4183cb160.
.venv/bin/python -m rl.merge_evaluations \
  models/experiments/formal_full_random_stock_v2_shard_*_1000.json \
  --required-games 4000 \
  --output models/experiments/formal_final_full_random_stock_v2_4000_merged.json
.venv/bin/python -m rl.merge_evaluations \
  models/experiments/formal_full_strategic_stock_v2_shard_*_1000.json \
  --required-games 4000 \
  --output models/experiments/formal_final_full_strategic_stock_v2_4000_merged.json
.venv/bin/python -m rl.merge_evaluations \
  models/experiments/formal_external_reference_shard_*_1000.json \
  --required-games 4000 \
  --output models/experiments/formal_final_external_reference_4000_merged.json
.venv/bin/python -m rl.merge_evaluations \
  models/experiments/formal_external_random_shard_*_1000.json \
  --required-games 4000 \
  --output models/experiments/formal_final_external_random_4000_merged.json
```

Development screens are not qualification.  The exact gates, tie handling, and
pinned external commit are defined in [`benchmark_goal.md`](benchmark_goal.md).
The visualization and publication assets were generated only after all full
and external reports qualified; their reproducible command is in
[`assets/linkedin/README.md`](assets/linkedin/README.md).
