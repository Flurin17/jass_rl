# Jass RL (Schieber)

## Status (AP0-AP9)

- Core rules (Schieber base) implemented with tests.
- Stoeck + Weis modules implemented (optional).
- CLI play + replay implemented.
- PettingZoo AEC env with action masking implemented.
- RL training/eval scaffolding implemented.
- Baseline-beating performance is NOT validated yet (see TODO).

## Quick start (tests)

```bash
pytest -q
```

## Play (CLI)

```bash
python -m cli.play --mode trump --trump-suit schilten --players low,low,low,low
python -m cli.play --mode obeabe --players human,low,low,low --replay-out /tmp/jass.json
python -m cli.replay /tmp/jass.json
```

## RL setup (requirements)

RL components require these packages (not included in this repo):

- pettingzoo
- gymnasium
- numpy
- sb3-contrib (MaskablePPO)
- stable-baselines3 (dependency of sb3-contrib)

If they are missing, RL tests are skipped.

## Train (MaskablePPO)

```bash
# Agent chooses mode (Schieber bidding enabled by default). Use --no-bidding to disable.
python -m rl.train_selfplay --total-steps 20000
python -m rl.train_selfplay --selfplay --iterations 5 --total-steps 50000
python -m rl.train_selfplay --n-envs 4 --vec-env dummy --device mps --n-steps 512 --batch-size 256
```

Models are saved under `models/<timestamp>/` for each run (e.g., `models/20251229_094512/model_final.zip`).

## Interpreting training logs

- `ep_len_mean` should be 9 when bidding/weis are disabled, 10 when bidding+announce are enabled.
- `ep_rew_mean` is **team A points** accumulated during the episode (includes Weis if enabled).
- Use `rl.eval` to judge quality (win-rate vs baseline), not just training loss.

## Full training run (example)

```bash
MPLCONFIGDIR=./.cache/matplotlib XDG_CACHE_HOME=./.cache \
  .venv/bin/python -m rl.train_selfplay \
  --total-steps 500000 \
  --iterations 10 \
  --n-envs 4 \
  --vec-env dummy \
  --n-steps 512 \
  --batch-size 256 \
  --device mps \
  --save-dir models \
  --seed 0
```

If you want the **environment** to choose the mode instead, pass `--no-bidding`.

## Eval

```bash
python -m rl.eval models/model_final.zip --episodes 200
```

## Debugging tips

- Illegal moves: check `core/legal_moves.py` and `core/rankings.py`.
- Scoring anomalies: check `core/scoring.py` (Obeabe/Uneufe tables).
- Replay mismatch: ensure same `seed`, `mode`, and `leader` are used.
- RL masking: use `env/jass_aec_env.py` and `rl/single_agent_env.py` to inspect action masks.

## TODO (AP9 remaining goal)

- Validate that the trained agent reliably beats baselines (e.g. >60% win rate over 1,000 rounds with fixed seeds).
- Add training benchmark configs / pinned dependency versions for reproducibility.
