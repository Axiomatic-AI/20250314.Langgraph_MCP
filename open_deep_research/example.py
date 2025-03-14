"""Example script demonstrating the Open Deep Research assistant using Anthropic's Claude.

This script showcases how to use the research assistant to generate a report on a given topic
while running entirely within a Docker container for isolation and security.
"""

import asyncio
import logging
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
    
    # Define research topic
    topic = "Brief overview of Model Control Protocol (MCP) in AI systems"
    
    logger.info(f"Starting research on topic: {topic}")
    logger.info("Generating report plan...")
    
    # Stream initial planning phase
    async for event in graph.astream({"topic": topic}, thread, stream_mode="updates"):
        logger.info(event)
    
    logger.info("Accepting plan and generating report...")
    
    # Accept plan and generate report
    async for event in graph.astream({"resume": True}, thread, stream_mode="updates"):
        logger.info(event)

if __name__ == "__main__":
    asyncio.run(main())
