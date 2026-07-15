# LinkedIn result assets

These images tell a simple story: teaching an AI to play Swiss Jass, testing it,
and showing one card recommendation. Their numbers come directly from the same
verified reports used by the advisor. Generate the example decision first, then
render the two PNGs and animated GIF:

```bash
.venv/bin/python -m rl.advisor \
  models/experiments/ppo_full_consolidation_v1/20260715_095439/model_final.zip \
  examples/advisor_state.json \
  --seed 0 --determinizations 24 --max-rollouts 216 \
  --output /tmp/jass-rl-linkedin-advice.json

uv run --extra media python scripts/render_linkedin_assets.py \
  --full-random models/experiments/formal_final_full_random_stock_v2_4000_merged.json \
  --full-strategic models/experiments/formal_final_full_strategic_stock_v2_4000_merged.json \
  --model-manifest models/experiments/ppo_full_consolidation_v1/20260715_095439/run_manifest.json \
  --advice /tmp/jass-rl-linkedin-advice.json \
  --output-dir docs/assets/linkedin
```

The command produces:

- `jass-rl-linkedin-hero.png`
- `jass-rl-linkedin-decision.png`
- `jass-rl-training-timelapse.gif`

The renderer refuses reports from a dirty checkout, failed qualification, a different
model, or a different advisor implementation. The GIF is kept below LinkedIn's
animated-pixel limit and this project's compact 5 MiB target.

## Suggested caption

I taught an AI to play Swiss Jass.

It practised millions of decisions, learned the rules and teamwork, and then
played 8,000 test games. It won 86% against random play and 63% against a
rule-based strategy.

My favourite part: it does not just name a card. It checks several ways the
hidden cards could be distributed and explains which option came out best. In
one example, its first instinct was the Jack of Schilten—but after thinking it
through, it chose the 9.

There is still plenty to improve, but it can already help me review real Jass
decisions. Full testing details and code are in the repository.
