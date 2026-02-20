# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-first Python project for RL training and backtesting of US tech equities.
- Core training: `train_us_tech_buy_agent.py`
- Evaluation/backtests: `test_*.py`, `backtest_*.py`, `sensitivity_analysis.py`, `grid_search_nvda_params.py`
- Model artifacts: `models_v5/` (tracked manifest in `models_v5/model_manifest.json`)
- Reference implementations: `reference/`
- Generated outputs (not always tracked): `test_results/`, `backtest_results*/`, `grid_search_results_nvda/`, `sensitivity_results/`

Keep new analysis tools as top-level scripts unless a reusable module is clearly needed.

## Build, Test, and Development Commands
- `python -m venv .venv` then `.venv\\Scripts\\activate` (Windows): create/activate local environment.
- `pip install -r requirements.txt`: install dependencies.
- `python train_us_tech_buy_agent.py`: run end-to-end model training.
- `python test_buy_agent_performance.py`: generate precision/recall style evaluation outputs.
- `python test_confidence_calibration.py`: run confidence-bin calibration analysis.
- `python backtest_market_filter.py --start 2017-10-16 --end 2025-12-31`: run market-filtered backtest.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables/files; constants in `UPPER_SNAKE_CASE`.
- Keep scripts deterministic and parameterized via CLI flags (e.g., `--start`, `--end`, `--tickers`).
- Prefer clear, small helper functions over long inline logic blocks.

## Testing Guidelines
- Test scripts use `test_*.py` naming and are executed directly with Python.
- Add tests next to existing top-level test scripts using the same naming pattern.
- For strategy changes, include at least one reproducible backtest command and output path in your PR notes.

## Commit & Pull Request Guidelines
Recent history favors Conventional Commit style (`feat:`, `docs:`, `chore:`), sometimes with scoped descriptions.
- Commit format: `type: short imperative summary`.
- PRs should include: purpose, key parameter/settings changes, run commands used, and before/after metrics.
- Link related issues and attach charts/tables when behavior or performance changes.

## Security & Configuration Tips
- Do not commit API keys, raw brokerage credentials, or large generated result artifacts.
- Keep local data files under ignored paths and verify `.gitignore` before committing.
