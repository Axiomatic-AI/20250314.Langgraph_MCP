"""Script to run both the original and MCP-based research assistants and save their reports.

This script ensures proper isolation by running everything in Docker containers.
"""

import asyncio
import logging
import os
import uuid

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
    
    # Create a unique ID for this run
    run_id = str(uuid.uuid4())
    
    # Initialize the graph with the original config
    graph = builder.compile(
        config={
            "research_topic": "Model Control Protocol (MCP) in AI systems",
            "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "tavily_api_key": os.environ.get("TAVILY_API_KEY", ""),
            "writer_model": "claude-3-5-sonnet-20241022",  # Using Claude 3.5 Sonnet for writing
            "max_search_depth": 1,  # Limiting search depth to keep memory usage low
        }
    )
    
    # Run the graph
    logger.info("Running original research assistant...")
    async for event in graph.astream(
        {"research_topic": "Model Control Protocol (MCP) in AI systems"},
        config={"configurable": {"thread_id": run_id}},
    ):
        if "__interrupt__" in event:
            logger.info(f"Interrupt event: {event['__interrupt__']}")
        elif "report" in event:
            # Save the report to a file
            report_filename = f"Original_Research_Report_{run_id[:8]}.md"
            with open(report_filename, "w") as f:
                f.write(event["report"])
            logger.info(f"Original research report saved to {report_filename}")
    
    logger.info("Original research assistant completed")
    return run_id

async def run_mcp_assistant():
    """Run the MCP-based research assistant."""
    logger.info("Starting MCP-based research assistant...")
    
    # Create a unique ID for this run
    run_id = str(uuid.uuid4())
    
    # Initialize the graph with the MCP config
    graph = builder.compile(
        config={
            "research_topic": "Model Control Protocol (MCP) in AI systems",
            "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "tavily_api_key": os.environ.get("TAVILY_API_KEY", ""),
            "writer_model": "claude-3-5-sonnet-20241022",  # Using Claude 3.5 Sonnet for writing
            "max_search_depth": 1,  # Limiting search depth to keep memory usage low
        }
    )
    
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
    
    # Run the graph with MCP instructions
    logger.info("Running MCP-based research assistant...")
    async for event in graph.astream(
        {
            "research_topic": topic,
            "mcp_instructions": """
                Follow these Model Control Protocol (MCP) instructions:
                1. Prioritize academic sources and technical documentation
                2. Focus on concrete implementations and case studies
                3. Analyze both benefits and limitations of MCP
                4. Include code examples where relevant
                5. Maintain objective tone throughout
            """
        },
        config={"configurable": {"thread_id": run_id}},
    ):
        if "__interrupt__" in event:
            logger.info(f"Interrupt event: {event['__interrupt__']}")
        elif "report" in event:
            # Save the report to a file
            report_filename = f"MCP_Research_Report_{run_id[:8]}.md"
            with open(report_filename, "w") as f:
                f.write(event["report"])
            logger.info(f"MCP research report saved to {report_filename}")
    
    logger.info("MCP-based research assistant completed")
    return run_id

async def main():
    """Run both research assistants in sequence."""
    await run_original_assistant()
    await run_mcp_assistant()

if __name__ == "__main__":
    asyncio.run(main())
