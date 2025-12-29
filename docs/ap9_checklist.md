# AP9 Acceptance Checklist

## Game-fertig

- [x] Rules in scope implemented (Schieber + optional Stoeck/Weis)
- [x] Tests for rules and scoring are green
- [x] CLI play works (human/AI mix)
- [x] Replay save/load works

## RL-fertig

- [x] Train command available (`python -m rl.train_selfplay`)
- [x] Eval command available (`python -m rl.eval`)
- [x] Play command available (`python -m cli.play`)
- [ ] Agent beats baselines consistently (>60% win rate over 1,000 rounds, fixed seeds)
- [x] Documentation for training and debugging (see README.md)

## Notes

The baseline-beating criterion is not validated yet because it depends on training
runs and pinned dependency versions. When you run a benchmark, fill in:

- Model checkpoint:
- Seeds used:
- Win rate:
- Average points:
- Dependency versions:
