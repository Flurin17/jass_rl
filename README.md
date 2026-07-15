# Jass RL

Rule-correct Swiss Schieber simulation, local MaskablePPO training, imperfect-
information search, paired evaluation, and an eventual real-game advisor.  The
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
- `220` tests and Ruff currently pass.  The PettingZoo API contract passes.

The latest qualification evidence for the neural-guided PIMC candidate is:

| Track | Opponent | Games | Raw wins | Paired wins | Mean difference | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Full Schieber | random | 4,000 | 85.425% | 95.55% | +208.38 | qualified |
| Full Schieber | strategic | 4,000 | 62.60% | 78.60% | +67.14 | qualified |
| `verardo-v1` | clean-room reference | 400 | 71.75% | 88.00% | +38.09 | screen pass |
| `verardo-v1` | random | 1,200 pooled | 89.00% | 97.33% | +80.47 | screen pass |

Formal gates require 4,000 games and 2,000 independent pairs per opponent.  The
full-project gates are complete, but the external gates are not.  The
visualization/advisor is deliberately not built until every gate passes; see
[`docs/benchmark_goal.md`](docs/benchmark_goal.md).

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

The measured M3 benchmark favors CPU for this masked network workload.  A basic
full-profile run is:

```bash
MPLCONFIGDIR=./.cache/matplotlib XDG_CACHE_HOME=./.cache \
  .venv/bin/python -m rl.train_selfplay \
  --profile standard --total-steps 1000000 --iterations 10 \
  --n-envs 8 --n-steps 1024 --batch-size 512 --n-epochs 4 \
  --opponent-mixture strategic=0.8,random=0.2 \
  --normalize-contract-reward --device cpu --seed 0
```

Models are stored below `models/<run>/<timestamp>/`.  Use `--resume` for an
interrupted compatible run and `--fork-from --allow-environment-fork` for an
explicit curriculum transition.  Exact M3 settings, measured memory, the
audited continuation command, and qualification commands are in
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

## External comparison

The matched external protocol pins
[`AlessioVerardo/jass-reinforcement-learning`](https://github.com/AlessioVerardo/jass-reinforcement-learning)
at commit `4ec7c665ce740fa1d792f653875a63f755551e7c`.  That repository does not ship
the reported checkpoint and has no declared license, so this project neither
copies its source nor claims a direct checkpoint victory.  It uses an
independently implemented, documented proxy for the published rule policy and
a stricter paired protocol.  A direct head-to-head gate must be added if the
original checkpoint becomes available.

## Repository layout

- `core/`: cards, bidding, legal play, rankings, scoring, announcements, state,
  transitions, and game loop.
- `env/`: canonical PettingZoo AEC environment and action masks.
- `rl/`: baselines, CTDE policy, training, PIMC/hybrid policy, tournaments,
  manifest-aware evaluation, and qualification CLI.
- `cli/`: interactive play and deterministic replay.
- `tests/`: deterministic unit, property, privacy, protocol, and API tests.
- `docs/`: rules, benchmark contract, and M3 workflow.
