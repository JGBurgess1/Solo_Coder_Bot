"""
Solo-Coder agent — Think-Act-Observe loop
==========================================
Environment variables (all required unless noted):
  TASK          – natural-language coding task description
  OLLAMA_HOST   – Ollama base URL  (default: http://localhost:11434)
  OLLAMA_MODEL  – model tag        (default: qwen2.5-coder:3b)
  WORKSPACE     – directory shared with the sandbox container (default: /workspace)
  GITHUB_PAT    – personal access token for push (optional; skips push if absent)
  GITHUB_REPO   – "owner/repo" to push to          (required when PAT is set)
  GITHUB_USER   – git committer name  (default: solo-coder-bot)
  GITHUB_EMAIL  – git committer email (default: bot@solo-coder.local)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import requests
import trafilatura
from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
WORKSPACE    = Path(os.getenv("WORKSPACE", "/workspace"))
TASK         = os.getenv("TASK", "Write a Python function that returns the nth Fibonacci number and add a pytest test for it.")

GITHUB_PAT   = os.getenv("GITHUB_PAT",   "")
GITHUB_REPO  = os.getenv("GITHUB_REPO",  "")
GITHUB_USER  = os.getenv("GITHUB_USER",  "solo-coder-bot")
GITHUB_EMAIL = os.getenv("GITHUB_EMAIL", "bot@solo-coder.local")

SANDBOX_CONTAINER = "solo-coder-sandbox"
MAX_ITERATIONS    = 6   # max Think-Act-Observe cycles before giving up

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("solo-coder")

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_chat(messages: list[dict], temperature: float = 0.2) -> str:
    """Send a chat request to Ollama and return the assistant reply text."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    data = resp.json()
    return data["message"]["content"].strip()


def build_system_prompt() -> str:
    return textwrap.dedent("""\
        You are Solo-Coder, an autonomous coding agent.
        You operate in a Think-Act-Observe loop.

        On each turn you MUST reply with a single JSON object — no prose, no markdown fences.
        The object must have exactly one of these shapes:

        1. Search the web for documentation or examples:
           {"action": "search", "query": "<search terms>"}

        2. Scrape a URL and read its content:
           {"action": "scrape", "url": "<full URL>"}

        3. Write a file to the workspace:
           {"action": "write_file", "path": "<relative path>", "content": "<file content>"}

        4. Run the test suite inside the sandbox:
           {"action": "run_tests", "command": "<shell command>"}

        5. Declare success (only after tests pass):
           {"action": "done", "summary": "<one-sentence summary of what was built>"}

        Rules:
        - Always write code before running tests.
        - Use pytest for Python tests; name test files test_*.py.
        - Keep file paths relative (e.g. "solution.py", "test_solution.py").
        - If tests fail, analyse the error in your next Think step and fix the code.
        - Never emit anything outside the JSON object.
    """)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo web search — returns a formatted list of results."""
    log.info("SEARCH: %s", query)
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"- [{r['title']}]({r['href']})\n  {r['body']}")
    if not results:
        return "No results found."
    return "\n".join(results)


def tool_scrape(url: str) -> str:
    """Fetch a URL and extract readable text via trafilatura."""
    log.info("SCRAPE: %s", url)
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Could not fetch {url}"
        text = trafilatura.extract(downloaded, include_links=False, include_images=False)
        if not text:
            return f"No readable content extracted from {url}"
        # Truncate to avoid overwhelming the context window
        return text[:8000]
    except Exception as exc:  # noqa: BLE001
        return f"Scrape error: {exc}"


def tool_write_file(rel_path: str, content: str) -> str:
    """Write content to a file inside the shared workspace."""
    target = WORKSPACE / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    log.info("WRITE: %s (%d bytes)", target, len(content))
    return f"Written {rel_path} ({len(content)} bytes)"


def tool_run_tests(command: str) -> str:
    """Execute a shell command inside the sandbox container via docker exec."""
    log.info("RUN: %s", command)
    docker_cmd = [
        "docker", "exec",
        "--workdir", "/workspace",
        SANDBOX_CONTAINER,
        "sh", "-c", command,
    ]
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        status = "PASSED" if result.returncode == 0 else "FAILED"
        log.info("TEST %s (exit %d)", status, result.returncode)
        return f"Exit code: {result.returncode}\n{output.strip()}"
    except subprocess.TimeoutExpired:
        return "Exit code: 1\nTest run timed out after 60 seconds."
    except FileNotFoundError:
        # docker not available (e.g. running agent outside compose)
        log.warning("docker not found — running command locally instead")
        result = subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(WORKSPACE),
        )
        return f"Exit code: {result.returncode}\n{(result.stdout + result.stderr).strip()}"


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )


def push_to_github(workspace: Path, repo: str, pat: str, user: str, email: str) -> str:
    """Initialise a git repo in workspace, commit everything, and push."""
    if not pat or not repo:
        return "Skipped: GITHUB_PAT or GITHUB_REPO not set."

    remote_url = f"https://{pat}@github.com/{repo}.git"

    # Init if needed
    if not (workspace / ".git").exists():
        _git(["init"], workspace)

    _git(["config", "user.name",  user],  workspace)
    _git(["config", "user.email", email], workspace)

    # Set / update remote
    remotes = _git(["remote"], workspace).stdout.strip().splitlines()
    if "origin" in remotes:
        _git(["remote", "set-url", "origin", remote_url], workspace)
    else:
        _git(["remote", "add", "origin", remote_url], workspace)

    _git(["add", "-A"], workspace)
    commit = _git(
        ["commit", "-m", "feat: Solo-Coder generated solution\n\nCo-authored-by: Ona <no-reply@ona.com>"],
        workspace,
    )
    if commit.returncode != 0 and "nothing to commit" in commit.stdout + commit.stderr:
        return "Nothing new to commit."

    push = _git(["push", "-u", "origin", "HEAD:main", "--force"], workspace)
    if push.returncode != 0:
        return f"Push failed:\n{push.stderr}"

    return f"Pushed to https://github.com/{repo}"


# ---------------------------------------------------------------------------
# Think-Act-Observe loop
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> dict[str, Any]:
    """Extract the JSON action object from the model reply."""
    # Strip markdown code fences if the model ignores instructions
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from model reply:\n{raw}")


def run_agent(task: str) -> None:
    log.info("Task: %s", task)
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    system = build_system_prompt()
    messages: list[dict] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Complete the following coding task:\n\n{task}\n\n"
                "Start by thinking about what you need to do, then emit your first action."
            ),
        },
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        log.info("--- Iteration %d/%d ---", iteration, MAX_ITERATIONS)

        # THINK: ask the model what to do next
        raw_reply = ollama_chat(messages)
        log.debug("Model reply: %s", raw_reply)
        messages.append({"role": "assistant", "content": raw_reply})

        # Parse action
        try:
            action = parse_action(raw_reply)
        except (ValueError, json.JSONDecodeError) as exc:
            observation = f"ERROR: Could not parse your reply as JSON. {exc}\nPlease reply with a valid JSON action object."
            log.warning(observation)
            messages.append({"role": "user", "content": observation})
            continue

        action_type = action.get("action", "")
        log.info("Action: %s", action_type)

        # ACT + OBSERVE
        if action_type == "search":
            observation = tool_search(action.get("query", ""))

        elif action_type == "scrape":
            observation = tool_scrape(action.get("url", ""))

        elif action_type == "write_file":
            observation = tool_write_file(
                action.get("path", "output.py"),
                action.get("content", ""),
            )

        elif action_type == "run_tests":
            observation = tool_run_tests(action.get("command", "pytest"))

        elif action_type == "done":
            summary = action.get("summary", "Task complete.")
            log.info("DONE: %s", summary)
            # Push to GitHub
            push_result = push_to_github(WORKSPACE, GITHUB_REPO, GITHUB_PAT, GITHUB_USER, GITHUB_EMAIL)
            log.info("GitHub: %s", push_result)
            print(f"\n✅ Solo-Coder finished: {summary}\nGitHub: {push_result}")
            return

        else:
            observation = f"Unknown action '{action_type}'. Use one of: search, scrape, write_file, run_tests, done."

        log.info("Observation: %s", observation[:200])
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    log.error("Reached max iterations (%d) without completing the task.", MAX_ITERATIONS)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_agent(TASK)
