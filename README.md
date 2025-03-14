# Model Control Protocol (MCP) with LangGraph

This repository demonstrates the integration of the Model Control Protocol (MCP) with LangGraph for building AI research assistants. It showcases two approaches:

1. **Standard Implementation**: Using Tavily API for search and Anthropic's Claude for LLM capabilities
2. **MCP-based Implementation**: Using a simplified MCP client to wrap the Tavily API, demonstrating how MCP can standardize tool integration

## Project Structure

- `open_deep_research/`: Main project directory containing the research assistant implementation
  - `example.py`: Original implementation using direct API calls
  - `example_with_mcp.py`: MCP-based implementation demonstrating the protocol
  - `example_mcp.py`: Alternative implementation using MCP with existing content
  - `run_comparison.py`: Script to run both implementations side-by-side
  - `20250314_MCP.md`: Example content about Model Control Protocol

## Running the Project

All code runs within Docker containers to ensure proper isolation, consistency, and security:

```bash
# Build the Docker container
docker-compose build

# Run the comparison script
docker-compose run --rm open-deep-research python run_comparison.py

# Run individual examples
docker-compose run --rm open-deep-research python example.py
docker-compose run --rm open-deep-research python example_with_mcp.py
```

## Environment Variables

The project requires the following environment variables:

- `ANTHROPIC_API_KEY`: API key for Anthropic's Claude
- `TAVILY_API_KEY`: API key for Tavily search API

These should be set in a `.env` file in the project root.

## Generated Reports

The research assistants generate reports on the Model Control Protocol (MCP) that are saved to:

- `MCP_Report_Original.md`: Report from the standard implementation
- `MCP_Report_With_MCP.md`: Report from the MCP-based implementation
