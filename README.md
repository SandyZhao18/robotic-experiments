# robotic-experiments
Experiments with robotic arms at Viam

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
cp .env.example .env   # then fill in the API key, key ID, and machine address
```

Credentials come from the machine's **CONNECT** tab in the Viam app. `.env` is gitignored — never commit keys.

## Scripts

- `uv run inspect-machine` — connects and runs a read-only check on every known resource (nothing actuates).
