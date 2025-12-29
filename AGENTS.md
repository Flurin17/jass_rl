# Repository Guidelines

## Project Structure & Module Organization

- `core/`: game domain logic (cards, scoring, rankings, legal moves, bidding, ruleset, state, game loop).
- `core/announcements/`: Stöck and Weis modules.
- `env/`: PettingZoo AEC environment for RL.
- `rl/`: single-agent wrapper, training, and evaluation scripts.
- `cli/`: playable CLI and replay tools.
- `tests/`: pytest test suite.
- `docs/`: rules scope and acceptance notes.

## Build, Test, and Development Commands

- `pytest -q`: run the full test suite.
- `python -m cli.play --mode trump --trump-suit schilten`: play a round (CLI).
- `python -m cli.replay /path/to/replay.json`: replay a saved game.
- `python -m rl.train_selfplay --total-steps 20000`: train with MaskablePPO.
- `python -m rl.eval models/<timestamp>/model_final.zip --episodes 200`: evaluate a model.

If using the uv venv, prefer `.venv/bin/python -m ...`.

## Coding Style & Naming Conventions

- Python code uses 4-space indentation.
- Filenames are snake_case (e.g., `legal_moves.py`).
- Public APIs are exported in `core/__init__.py` and `env/__init__.py`.
- Keep data structures simple and deterministic; avoid hidden randomness outside seeded RNGs.

## Testing Guidelines

- Framework: `pytest`.
- Tests live in `tests/` and are named `test_*.py`.
- Keep tests deterministic (use fixed seeds).
- RL tests skip if `pettingzoo`, `gymnasium`, or `numpy` are missing.

## Commit & Pull Request Guidelines

- No explicit commit convention is defined in this repo; use concise, imperative messages (e.g., "Add AEC env masking").
- PRs should include a short summary, test results, and any new CLI/usage commands.
- If changing rules or scoring, include references and update relevant tests and docs.

## Configuration Notes

- Swiss suits are used throughout: `schellen`, `rosen`, `schilten`, `eicheln`.
- RL training saves models under `models/<timestamp>/`.
- Use `MPLCONFIGDIR=./.cache/matplotlib` and `XDG_CACHE_HOME=./.cache` if matplotlib cache warnings appear.
