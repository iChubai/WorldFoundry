# `worldfoundry.mcp`

[Model Context Protocol](https://modelcontextprotocol.io) server that lets an LLM agent
discover models and benchmarks, launch and inspect evaluation runs, and submit Studio
inference jobs. It reuses the same catalog and runtime contracts as the CLI.

## Usage

```bash
python -m pip install -e ".[mcp]"
wf mcp                          # stdio (default)
wf mcp --transport sse           # SSE transport
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `WORLDFOUNDRY_STUDIO_WORKSPACE_URL` | Workspace used by Studio tools | `http://127.0.0.1:7870` |
| `WORLDFOUNDRY_MCP_RUN_ROOT` | Where MCP-launched runs are written | `runs/mcp` |

Python clients can use `worldfoundry.mcp.client.MCPClient`; `convert_result_to_openai_format()`
adapts tool results to OpenAI Chat content blocks.

## Layout

| Path | Purpose |
| --- | --- |
| `__init__.py` | `main()` console entry point and transport selection |
| `server.py` | `create_mcp_server()` — builds the FastMCP server |
| `client.py` | In-process client and result-format conversion |
| `tools/registration.py` | `register_tools()` — the single place tools are registered |
| `tools/discovery.py` | `list_models`, `get_model_info`, `list_benchmarks`, `list_tasks`, … |
| `tools/metrics.py` | `list_metrics`, `show_metric` |
| `tools/readiness.py` | `check_benchmark_datasets` and readiness queries |
| `tools/runs.py` | `preview_run`, `evaluate`, `list_runs`, `get_run_status`, `get_run_result`, `cancel_run`, … |
| `tools/studio.py` | `submit_studio_inference`, `list_studio_jobs`, `wait_for_studio_job`, `get_studio_job_logs`, … |
| `tools/server_info.py`, `context.py`, `query.py`, `responses.py` | Server metadata, shared context, query parsing, response shaping |

New tools go in a module under `tools/` and are added in `tools/registration.py`.
