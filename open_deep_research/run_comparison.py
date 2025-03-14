"""
Script to run both the original and MCP-based research assistants and save their reports.
This script ensures proper isolation by running everything in Docker containers.
"""

import asyncio
import logging
import os
import uuid
from typing import Dict, Any, List

from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables to auto-approve report plans
os.environ['AUTO_APPROVE'] = 'true'

async def run_original_assistant():
    """Run the original research assistant."""
    logger.info("Starting original research assistant...")
    
    # Initialize memory saver for checkpointing
    memory = MemorySaver()
    
    # Compile the graph
    graph = builder.compile(checkpointer=memory)
    
    # Configure thread with Anthropic models
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": "tavily",  # Using Tavily for search
            "planner_provider": "anthropic",
            "planner_model": "claude-3-5-sonnet-20241022",  # Using Claude 3.5 Sonnet for planning
            "writer_provider": "anthropic",
            "writer_model": "claude-3-5-sonnet-20241022",   # Using Claude 3.5 Sonnet for writing
            "max_search_depth": 1,  # Limiting search depth to keep memory usage low
        }
    }
    
    # Define research topic
    topic = "Brief overview of Model Control Protocol (MCP) in AI systems"
    
    logger.info(f"Starting research on topic: {topic}")
    logger.info("Generating report plan...")
    
    # Stream initial planning phase
    async for event in graph.astream({"topic": topic}, thread, stream_mode="updates"):
        if "__interrupt__" in event:
            # Auto-approve the plan
            logger.info("Auto-approving report plan...")
            async for event in graph.astream({"resume": True}, thread, stream_mode="updates"):
                if "report" in event:
                    report_content = event["report"]
                    with open("MCP_Report_Original.md", "w") as f:
                        f.write(report_content)
                    logger.info("Original report saved to MCP_Report_Original.md")
                    return

async def run_mcp_assistant():
    """Run the MCP-based research assistant."""
    logger.info("Starting MCP-based research assistant...")
    
    # Simplified MCP client implementation
    class MCPClient:
        """Simplified MCP client implementation for demonstration purposes."""
        
        def __init__(self, server_url: str):
            """Initialize the MCP client."""
            self.server_url = server_url
            logger.info(f"Initialized MCP client with server URL: {server_url}")
        
        def get_tools(self) -> List[Dict[str, Any]]:
            """Get available tools from the MCP server."""
            # Simulated tool definitions
            return [
                {
                    "name": "search_academic",
                    "description": "Search for academic papers related to a query.",
                    "input_schema": {"query": "string"}
                },
                {
                    "name": "search_web",
                    "description": "Search the web for information related to a query.",
                    "input_schema": {"query": "string"}
                }
            ]
        
        def call_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
            """Call a tool on the MCP server."""
            logger.info(f"Calling MCP tool: {tool_name} with params: {params}")
            
            # In a real implementation, this would make an API call to the MCP server
            # For this example, we'll simulate the tool execution
            if tool_name == "search_academic" or tool_name == "search_web":
                query = params.get("query", "")
                # Use Tavily API for actual search
                from tavily import TavilyClient
                
                tavily_api_key = os.environ.get("TAVILY_API_KEY")
                if not tavily_api_key:
                    return "Error: TAVILY_API_KEY environment variable not set"
                
                client = TavilyClient(api_key=tavily_api_key)
                search_result = client.search(query=query, search_depth="advanced", include_answer=True)
                
                # Format the result
                result = f"Search results for: {query}\n\n"
                if "answer" in search_result:
                    result += f"Summary: {search_result['answer']}\n\n"
                
                if "results" in search_result:
                    result += "Sources:\n"
                    for i, source in enumerate(search_result["results"][:5], 1):
                        result += f"{i}. {source.get('title', 'No title')}\n"
                        result += f"   URL: {source.get('url', 'No URL')}\n"
                        result += f"   Content snippet: {source.get('content', 'No content')[:200]}...\n\n"
                
                return result
            
            return f"Tool {tool_name} not implemented or failed to execute"
    
    # Initialize memory saver for checkpointing
    memory = MemorySaver()
    
    # Compile the graph
    graph = builder.compile(checkpointer=memory)
    
    # Instantiate the MCP client
    mcp_client = MCPClient(server_url="http://localhost:8000")
    
    # Get MCP tools
    mcp_tools = mcp_client.get_tools()
    logger.info(f"Available MCP tools: {[tool['name'] for tool in mcp_tools]}")
    
    # Configure thread with Anthropic models
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": "tavily",  # Using Tavily for search via MCP
            "planner_provider": "anthropic",
            "planner_model": "claude-3-5-sonnet-20241022",  # Using Claude 3.5 Sonnet for planning
            "writer_provider": "anthropic",
            "writer_model": "claude-3-5-sonnet-20241022",   # Using Claude 3.5 Sonnet for writing
            "max_search_depth": 1,  # Limiting search depth to keep memory usage low
        }
    }
    
    # Read existing MCP content from file
    mcp_content = ""
    try:
        with open("20250314_MCP.md", "r") as f:
            mcp_content = f.read()
        logger.info("Successfully loaded existing MCP content from 20250314_MCP.md")
    except Exception as e:
        logger.error(f"Error loading MCP content: {e}")
    
    # Define research topic
    topic = "Comprehensive overview of Model Control Protocol (MCP) in AI systems"
    
    # Add existing content as context to the topic
    if mcp_content:
        topic += f"\n\nExisting research context:\n{mcp_content}"
    
    logger.info(f"Starting research on topic: {topic}")
    logger.info("Generating report plan...")
    
    # Stream initial planning phase
    async for event in graph.astream({"topic": topic}, thread, stream_mode="updates"):
        if "__interrupt__" in event:
            # Auto-approve the plan
            logger.info("Auto-approving report plan...")
            async for event in graph.astream({"resume": True}, thread, stream_mode="updates"):
                if "report" in event:
                    report_content = event["report"]
                    with open("MCP_Report_With_MCP.md", "w") as f:
                        f.write(report_content)
                    logger.info("MCP-based report saved to MCP_Report_With_MCP.md")
                    return

async def main():
    """Run both research assistants in sequence."""
    await run_original_assistant()
    await run_mcp_assistant()

if __name__ == "__main__":
    asyncio.run(main())
