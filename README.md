# Solo-Coder

A local agentic coding bot powered by [Ollama](https://ollama.com) (`qwen2.5-coder:3b`).

## How it works

Solo-Coder runs a **Think → Act → Observe** loop:

| Phase | What happens |
|-------|-------------|
| **Think** | The LLM plans the next step and emits a JSON action |
| **Act** | The agent executes the action (search, scrape, write file, run tests) |
| **Observe** | The result is fed back to the LLM as context |
| **Refine** | If tests fail the LLM analyses the error and tries again |

Once all tests pass the agent commits the generated code and pushes it to GitHub.

## Prerequisites

| Tool | Purpose |
|------|---------|
| Docker + Docker Compose | Sandbox isolation and agent container |
| [Ollama](https://ollama.com) running on the host | LLM inference |
| `qwen2.5-coder:3b` pulled in Ollama | The model used |

Pull the model once:
```bash
ollama pull qwen2.5-coder:3b
```

## Quick start

```bash
# 1. Configure
cp .env.example .env
$EDITOR .env          # set TASK, GITHUB_PAT, GITHUB_REPO, etc.

# 2. Build images
docker compose build

# 3. Run
docker compose --env-file .env up
```

Logs stream to stdout. The agent exits when the task is done or after 6 iterations.

## Running the agent directly (no Docker)

Useful for development — requires Python 3.10+ and a running sandbox container.

```bash
pip install -r requirements.txt

export TASK="Write a Python function that reverses a string and test it."
export OLLAMA_HOST="http://localhost:11434"
export WORKSPACE="./workspace"
export GITHUB_PAT=""   # leave blank to skip push

python3 agent.py
```

## Project layout

```
solo-coder/
├── agent.py              # Main agent loop
├── docker-compose.yml    # Agent + sandbox services
├── Dockerfile.agent      # Agent image
├── sandbox/
│   └── Dockerfile.sandbox  # Isolated execution environment
├── requirements.txt
├── .env.example
└── README.md
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TASK` | Fibonacci example | Natural-language task description |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `qwen2.5-coder:3b` | Model tag |
| `WORKSPACE` | `/workspace` | Shared directory for generated files |
| `GITHUB_PAT` | _(empty)_ | GitHub personal access token |
| `GITHUB_REPO` | _(empty)_ | `owner/repo` to push to |
| `GITHUB_USER` | `solo-coder-bot` | Git committer name |
| `GITHUB_EMAIL` | `bot@solo-coder.local` | Git committer email |
