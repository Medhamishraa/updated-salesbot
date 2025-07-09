# client.py

from agent import match_agent  # Reuse the agent you defined with the prompt/tool
from pydantic_ai.mcp import MCPServerHTTP
from models import SellerInput
import asyncio

# Connect to local MCP SSE server (assumed to be running separately)
server = MCPServerHTTP(url='http://localhost:3001/sse')  
match_agent.mcp_servers = [server]

async def main():
    async with match_agent.run_mcp_servers():
        result = await match_agent.run("Find companies for health monitoring product")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
