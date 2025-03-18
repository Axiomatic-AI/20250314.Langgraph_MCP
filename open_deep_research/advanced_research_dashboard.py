#!/usr/bin/env python
"""Advanced Research Dashboard for Photonics Research.

This dashboard integrates three key advanced capabilities:
1. Mathematical Verification - Parse and verify equations from research papers
2. Scientific Concept Mapping - Extract and visualize concept relationships
3. Advanced Model Integration - Route research questions to specialized models
"""

# Standard library imports
import logging
import os
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
import streamlit as st

# Local application imports
from advanced_research_modules import (
    AdvancedResearchDashboard,
    MathematicalVerification,
    ScientificConceptMapper,
    ModelRouter
)
from photonics_arxiv_agent import PhotonicsArxivAgent

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Advanced Photonics Research Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Run the advanced research dashboard."""
    # Display header
    st.title("Advanced Photonics Research Dashboard")
    st.markdown("""
    This dashboard provides advanced research capabilities for photonics research:
    - **Mathematical Verification**: Verify equations and derivations in research papers
    - **Scientific Concept Mapping**: Build and explore knowledge graphs of research concepts
    - **Advanced Model Integration**: Route research questions to specialized models
    """)
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # API key status
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if anthropic_key:
        st.sidebar.success("✅ Anthropic API Key configured")
    else:
        st.sidebar.error("❌ Anthropic API Key missing")
        
    if tavily_key:
        st.sidebar.success("✅ Tavily API Key configured")
    else:
        st.sidebar.error("❌ Tavily API Key missing")
    
    # Create and render the advanced research dashboard
    dashboard = AdvancedResearchDashboard()
    dashboard.render_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Photonics Research Team | Built with Streamlit and LangGraph")


if __name__ == "__main__":
    main()
