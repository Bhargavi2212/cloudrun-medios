# Contributing Guidelines

Thank you for helping build Medi OS Kiroween Edition v2.0. This project follows `.cursorrules`; review that file before contributing.

## Local Setup
1. Install Python 3.11 and Node.js 18 (see `.tool-versions` or `.nvmrc`).
2. Install Poetry: `pipx install poetry` or follow https://python-poetry.org/docs/#installation.
3. Run `poetry install` to bootstrap the shared tooling environment.
4. Install frontend dependencies once the `apps/frontend` project is scaffolded (`npm install` or `pnpm install`).

## Development Workflow
- Activate the Poetry shell (`poetry shell`) or use `poetry run` for commands.
- Run `pre-commit install` to enable linting hooks.
- Format and lint before committing:
  - `poetry run ruff check .`
  - `poetry run ruff format .`
  - `poetry run black .`
  - `poetry run mypy .`
  - `poetry run pytest --collect-only`
- Add or update tests alongside code changes. No functionality should rely on fake outputs or skipped tests.

## Pull Request Checklist
- Follow commit and branch conventions in `.cursorrules`.
- Ensure new modules include type hints, Google-style docstrings, and emoji-enhanced logging where applicable.
- Update relevant README files and `FEDERATED_BUILD_GUIDE.md` when behavior changes.
- Provide deployment notes if infrastructure or environment variables change.

## Code of Conduct
By participating you agree to uphold the standards set out in `CODE_OF_CONDUCT.md`. Report issues privately to the maintainers at `engineering@medios.health`.

