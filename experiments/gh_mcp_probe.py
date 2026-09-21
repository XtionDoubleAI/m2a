"""One-shot MCP client probe for the GitHub official server (read-only).
Runs inside WSL where docker lives. Usage (from WSL):
  /root/.local/share/uv/tools/arxiv-mcp-server/bin/python gh_mcp_probe.py <tool> <json-args>
"""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    tool = sys.argv[1]
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    env = dict(os.environ)
    with open("/root/.github_mcp_token") as f:
        env["GITHUB_PERSONAL_ACCESS_TOKEN"] = f.read().strip()
    params = StdioServerParameters(
        command="docker",
        args=["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
              "ghcr.io/github/github-mcp-server", "stdio", "--read-only"],
        env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool(tool, args)
            payload = json.loads(res.content[0].text)
            items = payload.get("items", payload.get("repositories", []))
            if isinstance(items, list):
                for it in items[:5]:
                    if isinstance(it, dict):
                        print(f"{it.get('full_name', '?')} | stars "
                              f"{it.get('stargazers_count', '?')} | "
                              f"{str(it.get('description'))[:80]}")
            else:
                print(json.dumps(payload, default=str)[:600])

asyncio.run(main())
