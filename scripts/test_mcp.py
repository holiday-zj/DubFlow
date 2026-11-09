"""End-to-end MCP client test: spawn mcp_server.py over stdio, call every tool."""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server"],
        cwd="/home/ec2-user/dubflow")
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])

            res = await s.call_tool("dubflow_capabilities", {})
            print("CAPS:", json.loads(res.content[0].text)["languages"]["en"], "…")

            res = await s.call_tool("dubflow_translate", {
                "texts": ["这款产品可以帮你把视频翻译成多种语言"],
                "src_lang": "zh", "tgt_lang": "en", "engine": "marian"})
            print("TRANSLATE:", res.content[0].text[:200])

            res = await s.call_tool("dubflow_submit", {
                "video_path": "/home/ec2-user/dubflow/data/test/sample.mp4",
                "tgt_lang": "en", "whisper_model": "tiny"})
            job = json.loads(res.content[0].text)
            print("SUBMIT:", job)

            res = await s.call_tool("dubflow_wait", {
                "job_id": job["job_id"], "timeout_s": 600})
            final = json.loads(res.content[0].text)
            print("FINAL STATUS:", final["status"], final.get("error") or "")
            if final["status"] == "done":
                print("OUTPUT FILE:", final["output_file"])
                import os
                print("SIZE:", os.path.getsize(final["output_file"]))

asyncio.run(main())
