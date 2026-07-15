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
referenced checkpoint, and declares no software license. Therefore a direct
checkpoint head-to-head is not currently reproducible and its source must not be
copied into this project. The reproducible comparison available today is a
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

Training and evaluation must run on the target Apple M3 Pro with 36 GB RAM.  A
qualification run must record throughput, wall-clock time, peak memory, device,
seed, code/rules/observation versions, and checkpoints.  It must be resumable
after interruption.  Peak resident memory should remain below 28 GB so the
machine stays usable; no result that depends on unavailable CUDA hardware counts.

## No-shortcut rule

Passing the existing tests, beating the lowest-action policy, using fixed seats,
reporting an unpaired point estimate, running a toy training job, or rendering a
mock dashboard does not satisfy this goal.  Visualization work starts only after
a saved model clears both the full-project and matched external gates.
