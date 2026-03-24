# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python 3.11 workspace for multiple AI Devs tasks. Shared code lives in `devs_utilities/` (`env.py`, `http.py`, `logging.py`), while each task keeps its own solver and notes in folders such as `categorize/`, `railway/`, `sensors/`, `firmware/`, and `proxy/`. Tests are colocated with the code as `test_*.py` files, for example `railway/test_route_agent.py` and `devs_utilities/test_http.py`. The interactive launcher is `task_menu.py`. The `proxy/` app is a separate package with source under `proxy/src/proxy_api_server/`.

## Build, Test, and Development Commands
Create and activate the local environment before working:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Key commands:

- `python task_menu.py` runs the interactive task launcher.
- `python sensors\solve_sensors.py --verify` runs a task solver with remote verification.
- `python railway\route_agent.py "Sprawdz status trasy b-7"` runs the Railway agent directly.
- `docker compose --env-file ..\.env up --build` starts the proxy service from `proxy/`.
- `.venv\Scripts\python.exe -m unittest discover -q` runs the current test suite.

## Coding Style & Naming Conventions
Match the existing Python style: 4-space indentation, module docstrings, and `from __future__ import annotations` in new modules. Prefer explicit type hints, `Path` over raw string path logic, and small focused helpers. Use `snake_case` for files, functions, and variables; `PascalCase` for classes; and keep task entrypoints named `solve_<task>.py` or similarly descriptive (`route_agent.py`). No repo-wide formatter is configured, so keep imports tidy and changes consistent with nearby code.

## Testing Guidelines
Tests use `unittest` naming and are discoverable by `unittest` or `pytest` if installed. Name new files `test_<module>.py` and keep them beside the module they cover. Favor deterministic unit tests with `unittest.mock.patch` for HTTP and model calls. Run `.venv\Scripts\python.exe -m unittest discover -q` before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Add interactive task menu...` and `Refactor HTTP utilities...`. Follow that pattern: start with a verb, keep the first line specific, and scope the change to one task or utility when possible. PRs should explain which task folder changed, list the command used for verification, mention `.env` or API-key impacts, and include screenshots only for UI or proxy log view changes.

## Security & Configuration Tips
Use the repository-root `.env` (`C:\Users\fipci\PycharmProjects\Ai-devs-4-solutions\.env`) for secrets and runtime-tunable settings. Keep shared OpenRouter settings in the `OPENROUTER_*` section instead of duplicating per-task model or header variables. Stable AG3NTS hub endpoints live in shared helpers under `devs_utilities/ag3nts.py`; do not duplicate them in task-local env variables. New scripts should load configurable values through `repo_env.py` or `devs_utilities.env`. Never commit secrets. Generated artifacts such as `sendit/analysis/`, `sensors/dataset/`, `failure/failure.log`, and `*.json` are ignored and should stay out of review unless a change explicitly requires them.
