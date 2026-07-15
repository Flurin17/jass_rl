# Jass RL

Rule-correct Swiss Schieber simulation, local MaskablePPO training, imperfect-
information search, paired evaluation, and a real-game advisor.  The
project is designed for an Apple M3 Pro with 36 GB unified memory and does not
require CUDA or a hosted service.

## Current status

- One deterministic transition engine is shared by the core game, CLI, replay,
  and PettingZoo environment.
- The standard profile supports trump, Obeabe, Uneufe, Schieben, Puur and
  undertrump rules, Match, configurable factors, Weis, and Stöck.
- The canonical actor observation is acting-seat-relative and public-only.  A
  CTDE critic may see remaining hands during training, but that private suffix
  is structurally excluded from the actor.
- Training is resumable and writes environment manifests plus hashes for every
  checkpoint and final archive.
- Evaluation uses independently shuffled duplicate deals, team swaps, balanced
  starters, strict ties, Wilson intervals, per-contract metrics, and measured
  inference time.
- All `247` tests and Ruff currently pass.  The PettingZoo API contract passes.

The latest qualification evidence for the neural-guided PIMC candidate is:

| Track | Opponent | Games | Raw wins | Paired wins | Mean difference | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Full Schieber, Stock-aware v2 | random | 4,000 | 85.80% | 96.30% | +209.89625 | qualified |
| Full Schieber, Stock-aware v2 | strategic | 4,000 | 62.95% | 81.10% | +71.444 | qualified |
| `verardo-v1` | clean-room reference | 4,000 | 70.05% | 89.10% | +39.19 | qualified |
| `verardo-v1` | random | 4,000 | 87.88% | 97.75% | +77.81 | qualified |

Every formal gate uses 4,000 games and 2,000 independent pairs.  All four gates
pass.  Each canonical full-profile result is a strict merge of four disjoint
1,000-game shards and is bound to Stock-aware policy implementation
`d99631fb1b71693cd1dea729ae7ba413ba09865bd286e3017d260ce4183cb160`.
The full and `verardo-v1` rows use different manifest-backed models and policy
settings and are never presented as evidence for one another.  See
[`docs/benchmark_goal.md`](docs/benchmark_goal.md) for the gate definitions and
the exact comparison caveat.

## Install and test

Python 3.11–3.13 is supported.  The lock file is the reproducible path:

```bash
uv sync --extra rl --extra dev
MPLCONFIGDIR=./.cache/matplotlib XDG_CACHE_HOME=./.cache \
  .venv/bin/pytest -q
.venv/bin/ruff check .
```

## Play and replay

```bash
.venv/bin/python -m cli.play --mode trump --trump-suit schilten
.venv/bin/python -m cli.play --mode obeabe --replay-out /tmp/jass.json
.venv/bin/python -m cli.replay /tmp/jass.json
```

Swiss suit names are used throughout: `schellen`, `rosen`, `schilten`, and
`eicheln`.

## Train locally

The measured M3 benchmarks favor CPU for this masked network workload.  A clean
production-architecture resource validation processed 8,192 transitions at
2,750.622 learning steps/s using eight CPU threads, with 471,269,376 bytes peak
RSS.  Its clean manifest records the chip and 36 GB physical memory directly.
This validates the current resource envelope; the historical model manifest did
not record thread count and likely used PyTorch's then-observed six-thread host
default.  A basic new
full-profile run is:

```bash
MPLCONFIGDIR=./.cache/matplotlib XDG_CACHE_HOME=./.cache \
  .venv/bin/python -m rl.train_selfplay \
  --profile standard --total-steps 1000000 --iterations 10 \
  --n-envs 8 --n-steps 1024 --batch-size 512 --n-epochs 4 \
  --cpu-threads 8 \
  --opponent-mixture strategic=0.8,random=0.2 \
  --normalize-contract-reward --device cpu --seed 0
```

The default output is `models/<timestamp>/`; a named `--save-dir` can add a run
directory.  Use `--resume` for an interrupted compatible run.  Use `--fork-from`
together with `--allow-environment-fork` for an explicit curriculum transition.
Exact M3 settings, measured memory, the audited continuation command, and
qualification commands are in
[`docs/training_m3.md`](docs/training_m3.md).

## Evaluate

Evaluate a standalone manifest-backed model against all baseline protocols:

```bash
.venv/bin/python -m rl.eval models/<run>/model_final.zip \
  --episodes 4000 --opponents random strategic verardo-random verardo
```

Evaluate the production neural-guided search policy with provenance and every
gate in the output JSON:

```bash
.venv/bin/python -m rl.qualify_hybrid models/<run>/model_final.zip \
  --benchmark full --opponents strategic --episodes 4000 \
  --determinizations 24 --max-rollouts 216 --common-random-numbers \
  --opponent-rollout-policy generic --max-search-gap 3 \
  --min-neural-probability 0.5 --device cpu
```

The search boundary accepts only the canonical observation and legal-action
mask.  It never receives `GameState`, the environment object, or another
player's hand.  CTDE model inference pads the private suffix with zeros.

## Use the advisor

Rank the legal cards in a public game state from the terminal:

```bash
.venv/bin/python -m rl.advisor \
  models/experiments/ppo_full_consolidation_v1/20260715_095439/model_final.zip \
  examples/advisor_state.json
```

Start the local visual advisor with qualification evidence that matches that
exact model and policy configuration:

```bash
.venv/bin/python -m rl.advisor_web \
  models/experiments/ppo_full_consolidation_v1/20260715_095439/model_final.zip \
  --report models/experiments/formal_final_full_random_stock_v2_4000_merged.json \
  --report models/experiments/formal_final_full_strategic_stock_v2_4000_merged.json
```

Open `http://127.0.0.1:8765/`.  Reports with a different checkpoint, rules
profile, search configuration, or neural-guidance configuration are rejected at
startup.  State format, relative-seat conventions, and real-table usage are in
[`docs/advisor.md`](docs/advisor.md).

Qualification JSON and run manifests are tracked.  The 24 MB qualified model
archive remains a separate local artifact; see [`models/README.md`](models/README.md)
before expecting this exact command to work in a fresh clone.

## External comparison

The matched external protocol pins
[`AlessioVerardo/jass-reinforcement-learning`](https://github.com/AlessioVerardo/jass-reinforcement-learning)
at commit `4ec7c665ce740fa1d792f653875a63f755551e7c`.  That repository declares MIT
in `pyproject.toml` but has no standalone license file, and it does not ship the
reported checkpoint.  This project uses an independently implemented,
documented proxy for the published rule policy and a stricter paired protocol;
it does not claim a direct checkpoint victory.  A direct head-to-head gate must
be added if the original checkpoint becomes available.

## Repository layout

- `core/`: cards, bidding, legal play, rankings, scoring, announcements, state,
  transitions, and game loop.
- `env/`: canonical PettingZoo AEC environment and action masks.
- `rl/`: baselines, CTDE policy, training, PIMC/hybrid policy, tournaments,
  manifest-aware evaluation, and qualification CLI.
- `cli/`: interactive play and deterministic replay.
- `tests/`: deterministic unit, property, privacy, protocol, and API tests.
- `docs/`: rules, benchmark contract, and M3 workflow.
- `docs/assets/linkedin/`: reproducible social-asset command and an honest
  publication caption.
