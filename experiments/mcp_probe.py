"""Generic one-shot MCP stdio client probe (runs in WSL).

Usage:
  PY=/root/.local/share/uv/tools/paper-search-mcp/bin/python
  $PY mcp_probe.py <server-command> '<json-tool-and-args>'

Examples:
  $PY mcp_probe.py /root/.local/bin/paper-search-mcp \
      '{"tool": "search_arxiv", "args": {"query": "...", "max_results": 3}}'
  $PY mcp_probe.py 'docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN \
      ghcr.io/github/github-mcp-server stdio --read-only' \
      '{"tool": "search_code", "args": {"q": "..."}}'
"""
import asyncio
import json
import os
import shlex
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_cmd = sys.argv[1]
    spec = json.loads(sys.argv[2])
    argv = shlex.split(server_cmd)
    env = dict(os.environ)
    if "GITHUB" in server_cmd or "github" in server_cmd:
        with open("/root/.github_mcp_token") as f:
            env["GITHUB_PERSONAL_ACCESS_TOKEN"] = f.read().strip()
    params = StdioServerParameters(command=argv[0], args=argv[1:], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            if spec.get("tool") == "__list__":
                tools = await s.list_tools()
                print([t.name for t in tools.tools])
                return
            res = await s.call_tool(spec["tool"], spec.get("args", {}))
            print(res.content[0].text[:1500])

asyncio.run(main())
