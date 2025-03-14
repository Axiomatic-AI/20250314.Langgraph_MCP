"""Example script demonstrating the Open Deep Research assistant using Anthropic's Claude.

This script showcases how to use the research assistant to generate a report on Model Control Protocol (MCP)
while running entirely within a Docker container for isolation and security.
It leverages existing content from 20250314_MCP.md as a starting point.
"""

import asyncio
import logging
import os
import uuid

from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run the research assistant to generate a report on Model Control Protocol.
    
    Uses Anthropic's Claude for LLM capabilities and Tavily for search, all running
    within a Docker container for proper isolation and security.
    """
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
    
    # Read existing MCP content from file
    mcp_content = ""
    try:
        with open("20250314_MCP.md", "r") as f:
            mcp_content = f.read()
        logger.info("Successfully loaded existing MCP content from 20250314_MCP.md")
    except Exception as e:
        logger.error(f"Error loading MCP content: {e}")
    
    # Define research topic with reference to existing content
    topic = "Comprehensive overview of Model Control Protocol (MCP) in AI systems, building upon existing research"
    
    # Add existing content as context to the topic
    if mcp_content:
        topic += f"\n\nExisting research context:\n{mcp_content}"
    
    logger.info(f"Starting research on topic: {topic}")
    logger.info("Generating report plan...")
    
    # Stream initial planning phase
    async for event in graph.astream({"topic": topic}, thread, stream_mode="updates"):
        logger.info(event)
    
    logger.info("Accepting plan and generating report...")
    
    # Accept plan and generate report
    async for event in graph.astream({"resume": True}, thread, stream_mode="updates"):
        logger.info(event)
        
        # Save the final report to a file when complete
        if "report" in event:
            report_content = event["report"]
            with open("MCP_Report.md", "w") as f:
                f.write(report_content)
            logger.info("Report saved to MCP_Report.md")

if __name__ == "__main__":
    asyncio.run(main())
