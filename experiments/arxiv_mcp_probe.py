"""One-shot MCP client probe for the arxiv-mcp-server (storage: refs/papers).
Usage (from WSL):
  /root/.local/share/uv/tools/arxiv-mcp-server/bin/python arxiv_mcp_probe.py search "<query>"
  /root/.local/share/uv/tools/arxiv-mcp-server/bin/python arxiv_mcp_probe.py download <paper_id>
"""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

STORAGE = "/mnt/e/hx/_DoctorXtion/intern_shxt/Pdev/refs/papers"


async def main():
    action, arg = sys.argv[1], sys.argv[2]
    params = StdioServerParameters(
        command="/root/.local/bin/arxiv-mcp-server",
        args=["--storage-path", STORAGE])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            if action == "search":
                res = await s.call_tool("search_papers",
                                        {"query": arg, "max_results": 5})
                data = json.loads(res.content[0].text)
                papers = data if isinstance(data, list) else data.get("papers", data)
                for p in (papers if isinstance(papers, list) else [])[:5]:
                    print(json.dumps({k: p.get(k) for k in
                                      ("id", "title", "published", "authors")},
                                     ensure_ascii=False)[:220])
            elif action == "download":
                res = await s.call_tool("download_paper", {"paper_id": arg})
                print(res.content[0].text[:300])

asyncio.run(main())
