# Real-game Jass advisor

The advisor ranks every legal card using the trained public actor plus bounded
perfect-information Monte Carlo (PIMC) search.  It accepts only information
visible to all players and the acting player's own hand.  Opponent hands are
sampled possibilities consistent with the public history; they are never input
to the actor.

## Start the visual advisor

Use the full-standard model for real Schieber games:

```bash
.venv/bin/python -m rl.advisor_web \
  models/experiments/ppo_full_consolidation_v1/20260715_095439/model_final.zip \
  --determinizations 24 --max-rollouts 216 \
  --report models/experiments/formal_final_full_random_4000_seed49001.json \
  --report models/experiments/formal_final_full_strategic_4000_seed49001.json
```

Then open `http://127.0.0.1:8765/`.  The server listens on loopback by default,
runs inference locally, and serializes concurrent requests around the shared
model/search state.  Use fewer determinizations only as an explicitly
unqualified latency tradeoff and omit the formal reports: a report is displayed
only when its model hash, rules, search, guidance, artifact, and privacy checks
match the live advisor exactly.

## Enter a table state

Seats are always relative to the person asking for advice:

- `0`: self, the next player to act
- `1`: player to the left
- `2`: partner
- `3`: player to the right

Re-express seats from the new acting player's perspective before every request.
`completed_tricks` is chronological; each trick contains four
`{"player": 1, "card": "rosen:A"}` records in play order.  `current_trick`
contains zero to three records, and its next player must be seat `0`.  Swiss
card codes use `schellen`, `rosen`, `schilten`, or `eicheln` plus
`6,7,8,9,10,J,Q,K,A`.

The remaining fields are:

- `hand`: the acting player's remaining cards.
- `mode` and `trump_suit`: `trump` with a suit, or `obeabe`/`uneufe` with a
  null suit.
- `leader`: relative seat that led or will lead the current trick.
- `team_points`: `[our_team, opponents]`, including publicly awarded Weis and
  Stöck points.
- `bid_starter`, `bid_chooser`, and `bid_pushed`: the public Schieben result.
  Without a push the chooser is the starter; after a push it is the starter's
  partner.
- `announcement_status`: four relative-seat values, each `pass` or `announce`.
  All decisions must be resolved before card advice is requested.

[`examples/advisor_state.json`](../examples/advisor_state.json) is a complete
first-trick example.  The CLI accepts the same JSON and prints machine-readable
rankings:

```bash
.venv/bin/python -m rl.advisor MODEL.zip examples/advisor_state.json \
  --output /tmp/advice.json
```

## Read the result

Every row is legal under the manifested rules profile.  `expected_margin` is
the mean final raw-score margin across successfully sampled worlds;
`standard_error` measures Monte Carlo uncertainty across those worlds;
`neural_probability` is the masked actor prior.  These are decision aids, not
guarantees about the opponents' actual hidden cards.  A `null` margin means the
bounded assignment search could not construct a consistent world and the
deterministic strategic fallback was used.

The server validates the model against its adjacent run manifest before use:
schema and action counts, checkpoint membership and SHA-256, model spaces, and
the CTDE guarantee that private critic inputs cannot change actor logits.
