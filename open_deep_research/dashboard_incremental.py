#!/usr/bin/env python
"""Enhanced Dashboard for the Photonics Research Agent with incremental research capabilities.

This script provides an advanced web interface to interact with the Photonics Research Agent,
featuring LangGraph-based incremental research, thread persistence, and branching capabilities.
"""

# Standard library imports
import asyncio
import os
from datetime import datetime

# Third-party imports
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components_v1

# Local application imports
from photonics_arxiv_agent import PhotonicsArxivAgent
from thread_manager import get_thread_manager
from research_branching import ResearchBranchManager, render_branch_ui

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Set page configuration
st.set_page_config(
    page_title="Photonics Research Agent Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.search_topic = ""
    st.session_state.papers = []
    st.session_state.selected_papers = []
    st.session_state.expanded_summaries = set()
    st.session_state.current_thread_id = None
    st.session_state.current_branch_id = None
    st.session_state.research_checkpoints = []
    st.session_state.selected_element = None

# Add title and description
st.title("Photonics Research Agent Dashboard")
st.markdown("""
This dashboard allows you to interact with the Photonics Research Agent, which:
1. Searches for arXiv papers on photonics topics
2. Analyzes photonic circuit designs mentioned in papers
3. Generates comprehensive reports with circuit visualizations
4. Supports incremental research with LangGraph thread persistence
""")

# Check for API keys
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

# Add a test mode option in the sidebar
st.sidebar.header("Configuration")
test_mode = st.sidebar.checkbox("Test Mode (Skip API calls)", value=False, key="test_mode")
max_papers = st.sidebar.slider("Maximum Papers to Analyze", 1, 10, 5)

if not anthropic_api_key or not anthropic_api_key.startswith("sk-ant"):
    st.sidebar.warning("⚠️ Anthropic API key not set or invalid")
else:
    st.sidebar.success("✅ Anthropic API key loaded")

if not tavily_api_key:
    st.sidebar.warning("⚠️ Tavily API key not set")
else:
    st.sidebar.success("✅ Tavily API key loaded")

# Initialize managers
thread_manager = get_thread_manager()
branch_manager = ResearchBranchManager()

# Function to search for papers
async def search_papers(topic, max_papers):
    """Search for papers on the given topic."""
    try:
        agent = PhotonicsArxivAgent(
            anthropic_api_key=anthropic_api_key,
            tavily_api_key=tavily_api_key,
            max_papers=max_papers
        )
        papers = await agent.search_arxiv(topic, max_results=max_papers)
        return papers
    except Exception as e:
        st.error(f"Error searching for papers: {e}")
        raise e

# Function to run the research
async def run_research(topic, papers, thread_id=None, checkpoint_name=None):
    """Run the research process and return the report."""
    if st.session_state.get('test_mode', False):
        # Return mock data for testing
        return (
            "# Photonics Research Report: " + topic + "\n\n"
            "## Test Mode Active\n\n"
            "This is a test report generated without making API calls. "
            "To generate a real report, disable Test Mode in the sidebar and provide valid API keys.\n\n"
        ), [], papers

    try:
        # Create agent
        agent = PhotonicsArxivAgent(
            anthropic_api_key=anthropic_api_key,
            tavily_api_key=tavily_api_key,
            max_papers=len(papers) if papers else max_papers
        )
        
        # Generate report from selected papers
        report = await agent.generate_report_from_papers(topic, papers)
        
        # Extract circuit images if any
        images = []
        
        # Save checkpoint if thread_id is provided
        if thread_id and checkpoint_name:
            thread_data = thread_manager.load_thread(thread_id)
            if thread_data:
                checkpoints = thread_data.get("checkpoints", [])
                checkpoints.append({
                    "name": checkpoint_name,
                    "topic": topic,
                    "papers": papers,
                    "report": report,
                    "timestamp": datetime.now().isoformat()
                })
                thread_manager.save_thread(
                    thread_id=thread_id,
                    metadata=thread_data.get("metadata", {}),
                    topic=thread_data.get("topic", topic),
                    checkpoints=checkpoints
                )
        
        return report, images, papers
    except Exception as e:
        st.error(f"Error generating report: {e}")
        raise e

# Function to display paper cards with selection capability
def display_paper_cards(papers, selected_papers):
    """Display paper cards with selection capability."""
    # CSS for paper cards
    st.markdown("""
    <style>
    .paper-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        transition: all 0.3s;
    }
    .paper-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .paper-card.selected {
        border: 2px solid #4CAF50;
        background-color: #f0f9f0;
    }
    .relevance-badge {
        background-color: #2196F3;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        float: right;
    }
    .paper-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    .paper-summary {
        margin-top: 10px;
        border-left: 3px solid #2196F3;
        padding-left: 10px;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display selection summary
    if selected_papers:
        st.markdown(f"### Selected Papers: {len(selected_papers)}/{len(papers)}")
        
        # Calculate statistics
        years = [int(paper.get('published', '2023').split('-')[0]) for paper in selected_papers]
        avg_year = sum(years) / len(years) if years else 0
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Papers Selected", f"{len(selected_papers)}/{len(papers)}")
        with col2:
            st.metric("Average Year", f"{avg_year:.1f}")
        with col3:
            st.metric("Newest Paper", f"{max(years)}" if years else "N/A")
    
    # Display paper cards
    for i, paper in enumerate(papers):
        # Check if paper is selected
        is_selected = paper in selected_papers
        card_class = "paper-card selected" if is_selected else "paper-card"
        
        # Generate a random relevance score for demo purposes
        relevance = paper.get('relevance', 0.5)
        
        # Paper card HTML
        st.markdown(f"""
        <div class="{card_class}" id="paper-{i}">
            <span class="relevance-badge">Relevance: {relevance:.2f}</span>
            <h3>{paper.get('title', 'Untitled Paper')}</h3>
            <p><strong>Authors:</strong> {', '.join(paper.get('authors', ['Unknown']))}</p>
            <p><strong>Published:</strong> {paper.get('published', 'Unknown')}</p>
            <p><strong>arXiv ID:</strong> <a href="https://arxiv.org/abs/{paper.get('id', '')}" target="_blank">{paper.get('id', 'Unknown')}</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✓ Selected" if is_selected else "+ Select", key=f"select-{i}"):
                if is_selected:
                    st.session_state.selected_papers.remove(paper)
                else:
                    st.session_state.selected_papers.append(paper)
                st.rerun()
        
        with col2:
            if paper.get('id'):
                st.button("Preview PDF", key=f"pdf-{i}")
        
        with col3:
            # Toggle summary expansion
            if i in st.session_state.expanded_summaries:
                if st.button("Hide Summary", key=f"summary-{i}"):
                    st.session_state.expanded_summaries.remove(i)
                    st.rerun()
                st.markdown(f"""
                <div class="paper-summary">
                    {paper.get('summary', 'No summary available.')}
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button("View Full Summary", key=f"summary-{i}"):
                    st.session_state.expanded_summaries.add(i)
                    st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True)

# Function to render DOM interaction elements
def render_dom_interaction():
    """Render DOM interaction elements."""
    # Add JavaScript for DOM interaction
    js_code = """
    <script>
    // Function to handle element selection
    function handleElementSelection(event) {
        const element = event.target;
        
        // Highlight the selected element
        if (window.selectedElement) {
            window.selectedElement.style.outline = '';
        }
        
        element.style.outline = '2px solid red';
        window.selectedElement = element;
        
        // Get element data
        const elementData = {
            tagName: element.tagName,
            id: element.id,
            className: element.className,
            textContent: element.textContent.trim().substring(0, 100),
            attributes: Array.from(element.attributes).map(attr => ({
                name: attr.name,
                value: attr.value
            }))
        };
        
        // Send data to Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: elementData
        }, '*');
    }
    
    // Add click event listeners to all elements
    function addClickListeners() {
        document.querySelectorAll('*').forEach(element => {
            element.addEventListener('click', function(event) {
                event.stopPropagation();
                handleElementSelection(event);
            });
        });
    }
    
    // Initialize
    document.addEventListener('DOMContentLoaded', function() {
        addClickListeners();
        console.log('DOM interaction initialized');
    });
    </script>
    """
    
    # Display the JavaScript
    components_v1.html(js_code, height=0)
    
    # Display selected element data
    if st.session_state.selected_element:
        st.subheader("Selected Element")
        st.json(st.session_state.selected_element)

# Main dashboard layout with tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Research Dashboard", 
    "Research Branching", 
    "MCP Comparison",
    "Advanced Features"
])

# Tab 1: Main Research Dashboard
with tab1:
    st.header("Photonics Research")
    
    # Input for research topic
    topic = st.text_input("Enter a photonics research topic:", 
                         value=st.session_state.search_topic)
    
    # Search button
    search_col, generate_col = st.columns([1, 1])
    with search_col:
        if st.button("Search for Papers"):
            if not topic:
                st.error("Please enter a research topic")
            else:
                st.session_state.search_topic = topic
                with st.spinner("Searching for papers..."):
                    # Run the search asynchronously
                    papers = asyncio.run(search_papers(topic, max_papers))
                    st.session_state.papers = papers
                    st.session_state.selected_papers = []  # Reset selected papers
                    st.success(f"Found {len(papers)} papers")
    
    with generate_col:
        if st.button("Generate Report from Selected Papers"):
            if not st.session_state.selected_papers:
                st.error("Please select at least one paper")
            else:
                with st.spinner("Generating research report..."):
                    # Run the research asynchronously
                    report, images, papers = asyncio.run(
                        run_research(
                            topic, 
                            st.session_state.selected_papers,
                            thread_id=st.session_state.current_thread_id,
                            checkpoint_name=f"Checkpoint-{len(st.session_state.research_checkpoints)}"
                        )
                    )
                    
                    # Add to checkpoints
                    checkpoint = {
                        "topic": topic,
                        "papers": st.session_state.selected_papers,
                        "report": report,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.research_checkpoints.append(checkpoint)
                    
                    # Display report
                    st.markdown(report)
                    
                    # Display images if any
                    if images:
                        st.subheader("Circuit Visualizations")
                        for i, img in enumerate(images):
                            st.image(img, caption=f"Circuit {i+1}")
    
    # Display papers if available
    if st.session_state.papers:
        st.header("Available Papers")
        display_paper_cards(st.session_state.papers, st.session_state.selected_papers)
    
    # Display research checkpoints if available
    if st.session_state.research_checkpoints:
        st.header("Research Checkpoints")
        for i, checkpoint in enumerate(st.session_state.research_checkpoints):
            with st.expander(f"Checkpoint {i+1}: {checkpoint['topic']} ({checkpoint['timestamp']})"):
                st.markdown(checkpoint["report"])
                st.write(f"Papers used: {len(checkpoint['papers'])}")

# Tab 2: Research Branching
with tab2:
    render_branch_ui()

# Tab 3: MCP Comparison
with tab3:
    st.header("MCP vs Standard Research Comparison")
    
    # Embed the MCP comparison dashboard
    st.markdown("""
    The MCP Comparison dashboard allows you to compare the performance of Model Context Protocol (MCP) 
    against standard deep research approaches in photonics research tasks.
    
    Click the button below to open the MCP comparison dashboard.
    """)
    
    if st.button("Open MCP Comparison Dashboard"):
        # This would typically redirect to another page or embed the dashboard
        st.info("Redirecting to MCP Comparison Dashboard...")
        st.markdown("[Click here to open in a new tab](/mcp_comparison)")

# Tab 4: Advanced Features
with tab4:
    st.header("Advanced Features")
    
    # DOM Interaction
    dom_interaction = st.checkbox("Enable DOM Element Interaction", value=False)
    
    if dom_interaction:
        st.subheader("DOM Element Interaction")
        st.markdown("""
        This feature allows you to interact with DOM elements on the page.
        Click on any element to select it and view its properties.
        """)
        
        # Custom component to receive DOM element data
        def handle_dom_element(element_data):
            """Process DOM element data received from JavaScript."""
            if element_data and isinstance(element_data, dict):
                st.session_state.selected_element = element_data
        
        # Render DOM interaction elements
        render_dom_interaction()

# Main function
def main():
    """Initialize and run the dashboard application."""
    pass

if __name__ == "__main__":
    main()
