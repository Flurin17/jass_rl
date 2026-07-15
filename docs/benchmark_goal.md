# Model qualification and external benchmark goal

This document is a completion gate, not a roadmap suggestion.  A run is not
"promising" until it satisfies every applicable gate below with a saved run
manifest and reproducible evaluation output.

## External reference

Reference repository:
[`AlessioVerardo/jass-reinforcement-learning`](https://github.com/AlessioVerardo/jass-reinforcement-learning/tree/main)
at commit `4ec7c665ce740fa1d792f653875a63f755551e7c` (inspected 2026-07-15).

Its published PPO uses:

- a 1,681-dimensional actor observation with acting-seat-relative ordered trick
  history and inferred void suits;
- centralized-training/decentralized-execution (CTDE), where the critic can see
  all hands but the actor cannot;
- PPO with frozen historical opponents, 8 environments, a 512/256 card actor,
  and separate trump/card heads;
- a trump-only Schieber subset (four suits plus Schieben), Match, but no
  Obeabe/Undenufe, Weis, Stoeck, or configurable contract multipliers;
- a reported checkpoint at roughly 25 million episodes.

Published results are 84.1% versus random with +67.1 average raw point
difference and 63.4% versus its deterministic rule-based agent with +29.1 over
1,000 games.  The published tournament fixes the evaluated policy to Team 0,
does not use paired duplicate deals or confidence intervals, does not ship the
referenced checkpoint. It declares MIT in `pyproject.toml` but has no standalone
license file. Therefore a direct checkpoint head-to-head is not currently
reproducible. The reproducible comparison available today is a
clean-room proxy: an independently implemented, behavior-compatible baseline
derived from the published strategy description and run on a matched rules
profile. Clearing that proxy means exceeding the published headline metrics
under the stricter protocol below; it does not mean defeating the unavailable
original checkpoint directly.

## Matched external protocol

The `verardo-v1` benchmark profile must use four trump contracts plus Schieben,
raw 157-point scoring plus Match, no announcements, and a pinned clean-room
implementation of the documented reference rule-based behavior. Its legal-move
profile matches the pinned engine, including the edge case where an all-trump
hand must overtrump when able and may undertrump only when no overtrump exists.
Each deal is evaluated with team/seat rotation so both policies receive the same
cards and bidding positions. Every
duplicate pair uses its own independently shuffled deal; starter positions are
balanced across pairs rather than reusing one shuffle four times. Report win
rate, game-level and pair-level Wilson 95% intervals, raw score difference,
Match rate, contract, and wall-clock inference cost for at least 2,000 complete
pairs (4,000 games) per opponent.

Strict wins are divided by all games at game level. At pair level, a win requires
a positive summed candidate score difference across the two team-swapped games.
Ties are failures rather than half-wins in both rates and Wilson intervals. To
provisionally exceed the published headline benchmark through the clean-room
proxy, this repository must achieve both:

1. Versus random, both the game-level and pair-level win rates must be at least
   87%, and both Wilson 95% lower bounds must be above the published 84.1%. The
   average per-game raw score difference must be at least +67.1.
2. Versus the pinned reference heuristic, both the game-level and pair-level win
   rates must be at least 67%, and both Wilson 95% lower bounds must be above the
   published 63.4%. The average per-game raw score difference must be at least
   +29.1.

If the reference checkpoint becomes publicly available, add a paired direct
head-to-head gate. Only that direct gate can support a claim about beating the
original model; do not substitute an unverified reimplementation and call it the
original.

## Full-project protocol

The full rule-correct agent must additionally support the configured standard
Schieber profile, including Obeabe, Undenufe, Weis, Stoeck, Match, and contract
factors. For each opponent it uses at least 2,000 independently shuffled,
team-swapped pairs (4,000 games per opponent), with starter positions balanced
across pairs. Against random, both game-level and pair-level win rates must be at
least 60%; against this repository's stronger deterministic heuristic, both must
be at least 55%. In each case both Wilson 95% lower bounds must be above 50%, and
ties count as failures as defined above.

## Local resource gate

Training and evaluation must run on the target Apple M3 Pro with 36 GB RAM.
Resource evidence must exercise the production network architecture and full
standard environment while recording throughput, wall-clock time, peak memory,
device, thread count, seed, code/rules/observation versions, and checkpoints.
The training workflow must be resumable after interruption.  Peak resident
memory should remain below 28 GB so the machine stays usable; no result that
depends on unavailable CUDA hardware counts.  A short resource-validation run
can establish the machine envelope, but it is not evidence of model quality and
cannot replace the formal game gates.

## No-shortcut rule

Passing the existing tests, beating the lowest-action policy, using fixed seats,
reporting an unpaired point estimate, running a toy training job, or rendering a
mock dashboard does not satisfy this goal.  Visualization work starts only after
a saved model clears both the full-project and matched external gates.

## Achieved qualification

The canonical full-profile reports are the Stock-aware v2 reports.  They were
evaluated from a clean tree at commit `ec2b11e` and bind the search/rules code to
policy implementation SHA-256
`d99631fb1b71693cd1dea729ae7ba413ba09865bd286e3017d260ce4183cb160`.
Each is a strict merge of four disjoint 1,000-game shards, with 500 independently
shuffled, team-swapped pairs per shard.  The merger verified identical model,
policy implementation, configuration, environment, and clean provenance, and
rejected overlapping deal-seed ranges.

The external matched-profile reports remain the clean four-shard reports from
commit `00bab39`; Stock is disabled in `verardo-v1`.  Every row below contains
4,000 games and 2,000 pairs:

| Profile/model | Opponent | Game wins | Game Wilson low | Pair wins | Pair Wilson low | Mean difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full standard, Stock-aware v2 (`513e96d90014…`) | random | 85.80% | 84.68% | 96.30% | 95.38% | +209.89625 |
| Full standard, Stock-aware v2 (`513e96d90014…`) | strategic | 62.95% | 61.44% | 81.10% | 79.33% | +71.444 |
| `verardo-v1` (`7e1334b6d922…`) | random | 87.88% | 86.83% | 97.75% | 97.00% | +77.81 |
| `verardo-v1` (`7e1334b6d922…`) | clean-room reference | 70.05% | 68.61% | 89.10% | 87.66% | +39.19 |

All raw, paired, confidence-bound, point-difference, sample-count, privacy,
manifest, hash, and clean-provenance checks pass.  The two profiles use
different checkpoints and policy settings; advisor evidence is therefore bound
to an exact policy identity rather than combining the four rows under one live
model.

The canonical clean production-architecture resource validation is recorded at
`models/experiments/m3_resource_gate_final/20260715_170541/run_manifest.json`.
At commit `b99d7bc`, its 512/256 CTDE policy and full standard environment used
eight CPU threads to process 8,192 transitions in 2.9782 seconds of learning:
2,750.622 learning steps/s with 471,269,376 bytes (449.4375 MiB) peak RSS.  The
manifest directly records Apple M3 Pro, model identifier `Mac15,7`, 38,654,705,664
bytes of physical memory, a clean Git tree, the complete environment, seed,
runtime, observation schema, and hashes for the checkpoint and final archive.
This validates the production architecture's resource envelope, not the exact
historical qualified-model invocation: that model was trained before explicit
thread recording and likely used the then-observed six-thread PyTorch host
default.  Its manifest cannot prove the exact thread count retrospectively.

The external result exceeds the published headline thresholds under the
stricter clean-room proxy protocol, but it remains intentionally described as a
proxy result—not a direct victory over the unavailable upstream checkpoint.
