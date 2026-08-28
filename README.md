# RL

Reinforcement Learning projects — experiments in RLVR (RL from Verifiable Rewards), reward shaping, and LLM-based agents.

## Projects

| Project | What it does | Status |
|---------|--------------|--------|
| [**information_extraction**](./information_extraction/) | Extracts structured data from invoice emails and PDF receipts with Claude tool-use, then scores it with composable weighted verifiers to produce a Harbor-style RLVR reward | Evaluation + verifier framework complete · RL training not yet started |

For that project: [current state](./information_extraction/README.md#current-project-state) ·
[not yet implemented](./information_extraction/README.md#not-yet-implemented) ·
[next milestone](./information_extraction/README.md#next-technical-milestone) ·
[technical map](./information_extraction/README.md#technical-map)

## Getting Started

Each sub-project is self-contained with its own README, dependencies, and tests:

```bash
cd information_extraction
uv sync --extra dev
uv run python -m pytest -q
```

## Conventions

- `main` is the only long-lived branch and is kept green; work lands through short-lived
  `feat/` · `fix/` · `docs/` · `chore/` branches and a pull request.
- Each project pins its own dependencies with [uv](https://docs.astral.sh/uv/) and ships tests
  that run without network or API access.
- Personal source documents stay out of the repository; every committed fixture is synthetic.
