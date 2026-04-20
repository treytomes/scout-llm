# MCP Memory Service

Semantic memory backend for Claude Code sessions on this machine.
Uses SQLite-Vec with ONNX sentence-transformers for local, privacy-first storage.

---

## How it runs

The service is managed by a **systemd user service** that starts automatically
at login and restarts on failure. You never need to start it manually.

```bash
# Check status
systemctl --user status mcp-memory

# Restart if something goes wrong
systemctl --user restart mcp-memory

# View logs
journalctl --user -u mcp-memory -n 50 --no-pager

# Stop / disable
systemctl --user stop mcp-memory
systemctl --user disable mcp-memory
```

Service file: `~/.config/systemd/user/mcp-memory.service`

---

## Web Dashboard

A browser UI for browsing, searching, and inspecting memories:

**http://127.0.0.1:8766/**

Also available:
- **http://127.0.0.1:8766/docs** — REST API (Swagger UI)

This runs as a separate always-on service (`mcp-memory-dashboard`) on port 8766,
using the same database as the MCP server. Changes made through the dashboard are
immediately visible to Claude Code.

```bash
systemctl --user status mcp-memory-dashboard
systemctl --user restart mcp-memory-dashboard
```

---

## Transport

The service runs in **SSE mode** (HTTP-based, no stdin required):

| Property | Value |
|---|---|
| Protocol | Server-Sent Events (SSE) |
| Host | `127.0.0.1` (localhost only) |
| Port | `8765` |
| Endpoint | `http://127.0.0.1:8765/sse` |

To verify it is responding:
```bash
curl -s --max-time 3 http://127.0.0.1:8765/sse | head -2
# Should print: event: endpoint
```

---

## Claude Code integration

Registered in `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "memory": {
      "type": "sse",
      "url": "http://127.0.0.1:8765/sse"
    }
  }
}
```

Claude Code connects automatically when it starts. The server exposes these
MCP tools to Claude:

| Tool | Purpose |
|---|---|
| `store_memory` | Save a new memory with content and tags |
| `retrieve_memory` | Semantic search over stored memories |
| `search_by_tag` | Filter memories by tag |
| `delete_memory` | Remove a memory by ID |
| `list_memories` | Browse recent memories |
| `create_entities` | Store structured entities |
| `search_nodes` | Graph-style entity lookup |

---

## Storage

Data lives at: `~/.local/share/mcp-memory/`

| File | Contents |
|---|---|
| `sqlite_vec.db` | Main semantic memory store (SQLite + vector index) |
| `onnx_models/` | Cached embedding model (downloaded once on first run) |
| `backups/` | Automatic backups |

The ONNX model download (~90MB) happens on the first request after install.
Subsequent starts are fast.

---

## Memory scope

There is no automatic scoping by project — memories are global to this user
account. When storing memories about Scout, tag them appropriately
(e.g., `scout`, `training`, `architecture`) so they can be retrieved by tag
later.

If you want project-isolated memory in the future, the service supports
`X-Agent-ID` headers to scope memories per agent. This would require
configuring the service with an agent ID for each project.

---

## Upgrading

The service runs from the scout-llm virtualenv:
`/home/trey/projects/scout-llm/.venv/bin/memory`

To upgrade:
```bash
cd /home/trey/projects/scout-llm
source .venv/bin/activate
pip install --upgrade mcp-memory-service
systemctl --user restart mcp-memory
```

---

## Toward a full local web server

The memory service is one piece of an eventual always-on local infrastructure.
The scout-llm FastAPI server currently runs on demand via `./start.sh`. To move
toward an Apache-style always-on setup, the same systemd user service pattern
applies to the FastAPI server:

```ini
# ~/.config/systemd/user/scout-llm.service  (not yet created)
[Unit]
Description=Scout LLM Web Server

[Service]
Type=simple
WorkingDirectory=/home/trey/projects/scout-llm
ExecStart=/home/trey/projects/scout-llm/.venv/bin/uvicorn app:app \
    --app-dir ./src/server \
    --host 127.0.0.1 \
    --port 8000
Restart=on-failure

[Install]
WantedBy=default.target
```

A reverse proxy (nginx or Caddy — lighter than Apache for this workload) would
then sit in front of both services, handling TLS and routing:

- `localhost/` → Scout LLM FastAPI (port 8000)
- `localhost/memory/` → MCP memory dashboard (port 8765, HTTP mode)

This is not yet configured — it is the next step in the local infrastructure
buildout.