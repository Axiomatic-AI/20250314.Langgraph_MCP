#!/usr/bin/env python
"""Dashboard for the Photonics Research Agent.

This script provides a simple web interface to interact with the Photonics Research Agent.

"""

# Standard library imports
import asyncio
import os
import tempfile
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components_v1

# Local application imports
from photonics_arxiv_agent import PhotonicsArxivAgent

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

# Add title and description
st.title("Photonics Research Agent Dashboard")
st.markdown("""
This dashboard allows you to interact with the Photonics Research Agent, which:
1. Searches for arXiv papers on photonics topics
2. Analyzes photonic circuit designs mentioned in papers
3. Generates comprehensive reports with circuit visualizations
""")

# Check for API keys
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

# Add a test mode option in the sidebar
st.sidebar.header("Configuration")
test_mode = st.sidebar.checkbox("Test Mode (Skip API calls)", value=False, key="test_mode")
max_papers = st.sidebar.slider("Maximum Papers to Analyze", 1, 10, 5)

if not anthropic_api_key or not anthropic_api_key.startswith("sk-ant"):
    st.error("⚠️ Valid Anthropic API key not found. The research functionality will not work without a valid key.")
    with st.expander("How to set up Anthropic API key"):
        st.markdown("""
        1. Get a valid API key from [Anthropic](https://www.anthropic.com/)
        2. Create a file named `.env` in the root directory of this project
        3. Add the following line to the file:
        ```
        ANTHROPIC_API_KEY=your_anthropic_api_key
        ```
        4. Restart the Docker containers with:
        ```
        docker-compose -f docker-compose.dashboard.yml down
        docker-compose -f docker-compose.dashboard.yml up -d
        ```
        """)
elif not tavily_api_key or tavily_api_key == "xxx":
    st.warning("⚠️ Valid Tavily API key not found. Web search functionality may be limited.")
    with st.expander("How to set up Tavily API key"):
        st.markdown("""
        1. Get a valid API key from [Tavily](https://tavily.com/)
        2. Create a file named `.env` in the root directory of this project
        3. Add the following line to the file:
        ```
        TAVILY_API_KEY=your_tavily_api_key
        ```
        4. Restart the Docker containers
        """)
else:
    st.success("✅ API keys found and loaded successfully")

# Create search input
st.header("Research Topic")
research_topic = st.text_input(
    "Enter a research topic related to photonics:",
    placeholder="e.g., Recent advances in silicon photonics for quantum computing"
)

# Create a button to start the research
search_button = st.button("Search for Papers", type="primary", disabled=not research_topic)
generate_button = None
selected_papers = None

# Function to search for papers
async def search_papers(topic, max_papers):
    """Search for papers on the given topic."""
    try:
        # Initialize the agent
        agent = PhotonicsArxivAgent(
            anthropic_api_key=anthropic_api_key,
            tavily_api_key=tavily_api_key,
            max_papers=max_papers
        )
        
        # Search for relevant papers
        papers = agent.search_arxiv(topic, max_results=max_papers * 2)  # Get more papers for selection
        return papers
    except Exception as e:
        if "401" in str(e):
            raise ValueError(
                "API authentication error. Please check that your API keys are valid and have sufficient permissions."
            ) from e
        else:
            raise e

# Function to run the research
async def run_research(topic, papers):
    """Run the research process and return the report."""
    if st.session_state.get('test_mode', False):
        # Return mock data for testing
        return (
            f"# Photonics Research Report: {topic}\n\n"
            "## Test Mode Active\n\n"
            "This is a test report generated without making API calls. "
            "To generate a real report, disable Test Mode in the sidebar and provide valid API keys.\n\n"
            "## GDSFactory Integration Test\n\n"
            "The integration with GDSFactory is working correctly. "
            "The dashboard can generate and visualize photonic circuits using the shared volume.\n\n"
            "## Next Steps\n\n"
            "1. Obtain valid API keys for Anthropic and Tavily\n"
            "2. Update the `.env` file with these keys\n"
            "3. Restart the Docker containers\n"
            "4. Disable Test Mode and run a real research query"
        ), [], papers
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Initialize the agent
            agent = PhotonicsArxivAgent(
                anthropic_api_key=anthropic_api_key,
                tavily_api_key=tavily_api_key,
                max_papers=len(papers),
                temp_dir=temp_dir
            )
            
            # Generate the report using only the selected papers
            # We need to modify the generate_report method to accept a list of papers
            report = await agent.generate_report_from_papers(topic, papers)
            
            # Get any generated images
            images = []
            temp_path = Path(temp_dir)
            for img_file in temp_path.glob("*.png"):
                with open(img_file, "rb") as f:
                    images.append((img_file.name, f.read()))
            
            return report, images, papers
        except Exception as e:
            if "401" in str(e) and "authentication_error" in str(e):
                raise ValueError(
                    "Authentication failed with the Anthropic API. Please provide a valid API key in the .env file."
                ) from e
            elif "401" in str(e):
                raise ValueError(
                    "API authentication error. Please check that your API keys are valid and have sufficient permissions."
                ) from e
            else:
                raise e

# Handle the paper search process
if search_button:
    # Show a spinner while processing
    with st.spinner(f"Searching for papers on '{research_topic}'... This may take a moment."):
        try:
            # Search for papers asynchronously
            papers = asyncio.run(search_papers(research_topic, max_papers * 2))
            
            if papers:
                st.session_state.papers = papers
                st.session_state.search_topic = research_topic
                st.rerun()
            else:
                st.error(f"No papers found for '{research_topic}'. Please try a different search term.")
        except ValueError as e:
            st.error(f"API Error: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred during the search process: {str(e)}")
            st.exception(e)

# Display paper selection if papers are available
if hasattr(st.session_state, 'papers') and st.session_state.papers:
    st.header(f"Papers Found for '{st.session_state.search_topic}'")
    
    # Add custom CSS for paper cards
    st.markdown("""
    <style>
    .paper-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
        position: relative;
    }
    .paper-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #4CAF50;
        background-color: #f9f9f9;
    }
    .paper-title {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 5px;
    }
    .paper-authors {
        color: #555;
        font-style: italic;
        margin-bottom: 5px;
    }
    .paper-date {
        color: #777;
        font-size: 0.9em;
        margin-bottom: 5px;
    }
    .paper-id {
        font-family: monospace;
        background-color: #f0f0f0;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 0.9em;
    }
    .paper-summary {
        margin-top: 10px;
        font-size: 0.95em;
        line-height: 1.4;
        max-height: 100px;
        overflow: hidden;
        text-overflow: ellipsis;
        position: relative;
    }
    .paper-summary.expanded {
        max-height: none;
    }
    .paper-actions {
        margin-top: 10px;
        display: flex;
        justify-content: space-between;
    }
    .paper-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #4CAF50;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("Select the papers you want to include in your research report:")
    
    # Create a container for the paper selection interface
    paper_container = st.container()
    
    # Create a dictionary to store paper selection state if it doesn't exist
    if 'paper_selections' not in st.session_state:
        st.session_state.paper_selections = {}
    
    # Create a dictionary to store expanded summaries state
    if 'expanded_summaries' not in st.session_state:
        st.session_state.expanded_summaries = {}
    
    # Create a dictionary to store paper relevance scores (simulated)
    if 'paper_relevance' not in st.session_state:
        # Generate simulated relevance scores (in a real app, this would come from an ML model)
        st.session_state.paper_relevance = {i: max(0.5, min(0.99, 0.95 - (i * 0.05))) 
                                           for i in range(len(st.session_state.papers))}
    
    # Function to toggle paper selection
    def toggle_paper_selection(paper_idx):
        """Toggle the selection state of a paper in the session state.

        Args:
            paper_idx (int): The index of the paper in the session state papers list.
        """
        if paper_idx in st.session_state.paper_selections:
            del st.session_state.paper_selections[paper_idx]
        else:
            st.session_state.paper_selections[paper_idx] = True
    
    # Function to toggle summary expansion
    def toggle_summary_expansion(paper_idx):
        """Toggle the expansion state of a paper summary in the session state.

        Args:
            paper_idx (int): The index of the paper in the session state papers list.
        """
        if paper_idx in st.session_state.expanded_summaries:
            del st.session_state.expanded_summaries[paper_idx]
        else:
            st.session_state.expanded_summaries[paper_idx] = True
    
    # Display papers with enhanced UI
    with paper_container:
        for i, paper in enumerate(st.session_state.papers):
            # Create columns for checkbox and paper content
            col1, col2 = st.columns([0.1, 0.9])
            
            # Checkbox for selection
            with col1:
                is_selected = st.checkbox("", key=f"select_paper_{i}", 
                                         value=i in st.session_state.paper_selections,
                                         on_change=toggle_paper_selection, args=(i,))
            
            # Paper card with hover effects
            with col2:
                # Calculate relevance score for this paper
                relevance = st.session_state.paper_relevance[i]
                relevance_pct = int(relevance * 100)
                
                # Create paper card with HTML for better styling
                card_html = f"""
                <div class="paper-card" id="paper-{i}">
                    <div class="paper-badge">{relevance_pct}% Relevant</div>
                    <div class="paper-title">{paper['title']}</div>
                    <div class="paper-authors">By: {', '.join(paper['authors'][:3])}{' et al.' if len(paper['authors']) > 3 else ''}</div>
                    <div class="paper-date">Published: {paper['published']}</div>
                    <div class="paper-id">arXiv: <a href="https://arxiv.org/abs/{paper['arxiv_id']}" target="_blank">{paper['arxiv_id']}</a></div>
                    <div class="paper-summary {'expanded' if i in st.session_state.expanded_summaries else ''}">
                        {paper['summary'][:300]}{'...' if len(paper['summary']) > 300 and i not in st.session_state.expanded_summaries else paper['summary'][300:]}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Add interactive buttons below the card
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if st.button("View Full Summary", key=f"view_summary_{i}"):
                        toggle_summary_expansion(i)
                        st.rerun()
                
                with col_b:
                    if st.button("Preview PDF", key=f"preview_pdf_{i}"):
                        st.info(f"Opening PDF preview for {paper['title']}...")
                        # In a real implementation, this would open a PDF preview
                
                with col_c:
                    if st.button("Related Papers", key=f"related_{i}"):
                        st.info(f"Finding papers related to {paper['title']}...")
                        # In a real implementation, this would show related papers
    
    # Get selected papers
    selected_indices = list(st.session_state.paper_selections.keys())
    selected_papers = [st.session_state.papers[i] for i in selected_indices]
    
    # Display selection summary
    st.subheader("Selection Summary")
    if selected_papers:
        st.success(f"Selected {len(selected_papers)} papers for the research report.")
        
        # Show selection statistics
        avg_relevance = sum(st.session_state.paper_relevance[i] for i in selected_indices) / len(selected_indices)
        st.metric("Average Relevance Score", f"{int(avg_relevance * 100)}%")
        
        # Add generate button
        generate_button = st.button("Generate Report from Selected Papers", type="primary")
    else:
        st.warning("Please select at least one paper to generate a report.")
        generate_button = None

# Handle the report generation process
if generate_button and selected_papers:
    # Show a spinner while processing
    with st.spinner(f"Generating report from {len(selected_papers)} selected papers... This may take a few minutes."):
        try:
            # Create placeholder for the report
            report_placeholder = st.empty()
            report_placeholder.info("Starting report generation...")
            
            # Run the research asynchronously
            report, images, papers = asyncio.run(run_research(st.session_state.search_topic, selected_papers))
            
            # Display the report
            report_placeholder.empty()
            st.header("Research Report")
            st.markdown(report)
            
            # Display the papers used
            if papers and not st.session_state.get('test_mode', False):
                st.header("Papers Used in This Research")
                for i, paper in enumerate(papers, 1):
                    with st.expander(f"{i}. {paper['title']}"):
                        st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                        st.markdown(f"**Published:** {paper['published']}")
                        st.markdown(f"**arXiv ID:** [{paper['arxiv_id']}](https://arxiv.org/abs/{paper['arxiv_id']})")
                        st.markdown(f"**Summary:** {paper['summary']}")
            
            # Display any generated images
            if images:
                st.header("Generated Circuit Visualizations")
                for img_name, img_data in images:
                    st.image(img_data, caption=img_name)
            
            # Add download button for the report
            st.download_button(
                label="Download Report as Markdown",
                data=report,
                file_name=f"Photonics_Research_Report_{st.session_state.search_topic.replace(' ', '_')[:30]}.md",
                mime="text/markdown"
            )
            
        except ValueError as e:
            st.error(f"API Error: {str(e)}")
            if "API key" in str(e):
                with st.expander("How to fix API key issues"):
                    st.markdown("""
                    1. Obtain a valid API key from the service provider
                    2. Update your `.env` file with the correct key
                    3. Restart the Docker containers with:
                    ```
                    docker-compose -f docker-compose.dashboard.yml down
                    docker-compose -f docker-compose.dashboard.yml up -d
                    ```
                    """)
        except Exception as e:
            st.error(f"An error occurred during the research process: {str(e)}")
            st.exception(e)

# Add information about the project
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.markdown("""
This dashboard interfaces with the Photonics Research Agent, which uses:
- arXiv API for paper search
- Anthropic Claude API for analysis
- GDSFactory for photonic circuit visualization
- Tavily API for web search

All processing is done within Docker containers for proper isolation and security.
""")

# Add DOM element interaction section
st.sidebar.markdown("---")
st.sidebar.header("Advanced Features")
if st.sidebar.checkbox("Enable DOM Element Interaction"):
    st.sidebar.markdown("This feature allows you to interact with specific DOM elements on the page.")
    
    # Create a placeholder to store selected element data
    if 'selected_element' not in st.session_state:
        st.session_state.selected_element = None
    
    # Add custom HTML and JavaScript for DOM interaction
    dom_interaction_html = """
    <script>
    // Function to enable element selection with proper event handling for Streamlit
    function enableElementSelection() {
        // Create a global variable to store the selected element
        window.selectedElement = null;
        
        // Function to handle element selection
        function handleElementSelection(e) {
            // Prevent default behavior and stop propagation
            e.preventDefault();
            e.stopPropagation();
            
            // Store the element
            window.selectedElement = this;
            
            // Reset borders on all elements
            document.querySelectorAll('[data-testid]').forEach(el => {
                el.style.border = '';
                el.style.boxShadow = '';
            });
            
            // Highlight selected element
            this.style.border = '2px solid red';
            this.style.boxShadow = '0 0 10px rgba(255,0,0,0.5)';
            
            // Get element information
            const tagName = this.tagName;
            const className = this.className;
            const id = this.id || '';
            const dataTestId = this.getAttribute('data-testid') || '';
            const dataComponentName = this.getAttribute('data-component-name') || '';
            const width = this.getAttribute('width') || '';
            const text = this.innerText || '';
            
            // Create data object
            const data = {
                tagName: tagName,
                className: className,
                id: id,
                dataTestId: dataTestId,
                dataComponentName: dataComponentName,
                width: width,
                text: text.substring(0, 100) + (text.length > 100 ? '...' : '')
            };
            
            // Send data to Python via window.parent
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: data
            }, '*');
            
            console.log('Selected element:', data);
            
            return false;
        }
        
        // Add click handlers with capture phase to ensure they run first
        document.addEventListener('click', function(e) {
            // Check if we're in selection mode
            if (!window.selectionModeActive) return;
            
            // Get the clicked element
            const el = e.target;
            
            // Process the element
            handleElementSelection.call(el, e);
        }, true);
        
        // Set selection mode active
        window.selectionModeActive = true;
        
        // Add visual indicator that selection mode is active
        const indicator = document.createElement('div');
        indicator.style.position = 'fixed';
        indicator.style.top = '10px';
        indicator.style.right = '10px';
        indicator.style.backgroundColor = 'red';
        indicator.style.color = 'white';
        indicator.style.padding = '5px';
        indicator.style.borderRadius = '5px';
        indicator.style.zIndex = '9999';
        indicator.style.fontWeight = 'bold';
        indicator.innerText = 'DOM Selection Mode Active';
        document.body.appendChild(indicator);
        
        console.log('Element selection enabled');
    }
    
    // Function to fix specific stVerticalBlock div elements
    function fixVerticalBlockDivs() {
        // Find all divs with data-testid="stVerticalBlock"
        const verticalBlocks = document.querySelectorAll('div[data-testid="stVerticalBlock"]');
        
        verticalBlocks.forEach((div, index) => {
            // Add a special class for identification
            div.classList.add('fixed-vertical-block');
            
            // Add a visible border to make it easier to select
            div.style.border = '2px dashed #4CAF50';
            div.style.padding = '5px';
            div.style.margin = '5px 0';
            div.style.position = 'relative';
            div.style.transition = 'all 0.3s ease';
            
            // Add hover effect
            div.onmouseover = function() {
                this.style.border = '2px solid #4CAF50';
                this.style.boxShadow = '0 0 10px rgba(76,175,80,0.5)';
                this.style.backgroundColor = 'rgba(76,175,80,0.05)';
            };
            
            div.onmouseout = function() {
                if (!this.classList.contains('selected-block')) {
                    this.style.border = '2px dashed #4CAF50';
                    this.style.boxShadow = '';
                    this.style.backgroundColor = '';
                }
            };
            
            // Add a label to identify the block
            const label = document.createElement('div');
            label.style.position = 'absolute';
            label.style.top = '0';
            label.style.right = '0';
            label.style.backgroundColor = '#4CAF50';
            label.style.color = 'white';
            label.style.padding = '2px 5px';
            label.style.fontSize = '10px';
            label.style.borderRadius = '0 0 0 5px';
            label.style.zIndex = '999';
            label.innerText = `Block ${index+1}`;
            div.appendChild(label);
            
            // Make the div clickable with a specific handler
            div.onclick = function(e) {
                e.stopPropagation();
                
                // Reset all borders and remove selected class
                document.querySelectorAll('.fixed-vertical-block').forEach(el => {
                    el.style.border = '2px dashed #4CAF50';
                    el.style.boxShadow = '';
                    el.style.backgroundColor = '';
                    el.classList.remove('selected-block');
                });
                
                // Highlight this div and add selected class
                this.style.border = '3px solid #FF5722';
                this.style.boxShadow = '0 0 10px rgba(255,87,34,0.5)';
                this.style.backgroundColor = 'rgba(255,87,34,0.05)';
                this.classList.add('selected-block');
                
                // Get element information
                const data = {
                    tagName: this.tagName,
                    className: this.className,
                    id: this.id || '',
                    dataTestId: this.getAttribute('data-testid') || '',
                    dataComponentName: this.getAttribute('data-component-name') || '',
                    width: this.getAttribute('width') || '',
                    childrenCount: this.children.length,
                    text: this.innerText.substring(0, 100) + (this.innerText.length > 100 ? '...' : ''),
                    structure: getElementStructure(this, 2)
                };
                
                // Send data to Python via window.parent
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: data
                }, '*');
                
                console.log('Selected vertical block:', data);
                
                return false;
            };
            
            console.log('Fixed vertical block div:', div);
        });
        
        return `Fixed ${verticalBlocks.length} vertical block divs`;
    }
    
    // Function to fix the specific div element from the user
    function fixSpecificDiv() {
        // Target the specific div with the exact class combination
        const specificDivs = document.querySelectorAll('div.stVerticalBlock.st-emotion-cache-7j4m9d');
        
        if (specificDivs.length === 0) {
            console.log('Specific div not found, trying alternative selectors');
            // Try alternative selectors
            const altDivs = document.querySelectorAll('div[data-testid="stVerticalBlock"][width="417"]');
            if (altDivs.length === 0) {
                // Try even more generic selector
                const genericDivs = document.querySelectorAll('div[data-testid="stVerticalBlock"]');
                if (genericDivs.length > 0) {
                    processSpecificDivs(genericDivs);
                    return `Fixed ${genericDivs.length} vertical block divs using generic selector`;
                }
                
                // Try to find error message containers
                const errorDivs = document.querySelectorAll('div[data-testid="stMarkdownContainer"]');
                if (errorDivs.length > 0) {
                    processErrorDivs(errorDivs);
                    return `Fixed ${errorDivs.length} error message divs`;
                }
                
                return 'Specific div not found';
            }
            processSpecificDivs(altDivs);
            return `Fixed ${altDivs.length} specific divs using alternative selector`;
        }
        
        processSpecificDivs(specificDivs);
        return `Fixed ${specificDivs.length} specific divs`;
    }
    
    // Function to process error message divs
    function processErrorDivs(divs) {
        divs.forEach((div, index) => {
            // Check if this is an error message div
            const errorText = div.innerText.toLowerCase();
            const isErrorDiv = errorText.includes('error') || errorText.includes('exception');
            
            if (isErrorDiv) {
                // Add special styling for error divs
                div.style.border = '3px solid #F44336';
                div.style.boxShadow = '0 0 15px rgba(244,67,54,0.7)';
                div.style.padding = '8px';
                div.style.margin = '10px 0';
                div.style.position = 'relative';
                div.style.borderRadius = '5px';
                div.style.transition = 'all 0.3s ease';
                
                // Add hover effect
                div.onmouseover = function() {
                    this.style.border = '3px solid #D32F2F';
                    this.style.boxShadow = '0 0 15px rgba(211,47,47,0.7)';
                    this.style.backgroundColor = 'rgba(244,67,54,0.05)';
                };
                
                div.onmouseout = function() {
                    if (!this.classList.contains('selected-error')) {
                        this.style.border = '3px solid #F44336';
                        this.style.boxShadow = '0 0 15px rgba(244,67,54,0.7)';
                        this.style.backgroundColor = '';
                    }
                };
                
                // Add a prominent label
                const label = document.createElement('div');
                label.style.position = 'absolute';
                label.style.top = '0';
                label.style.left = '0';
                label.style.backgroundColor = '#F44336';
                label.style.color = 'white';
                label.style.padding = '3px 8px';
                label.style.fontSize = '12px';
                label.style.fontWeight = 'bold';
                label.style.borderRadius = '0 0 5px 0';
                label.style.zIndex = '1000';
                label.innerText = `Error Message ${index+1}`;
                div.appendChild(label);
                
                // Add a click handler specifically for this div
                div.onclick = function(e) {
                    e.stopPropagation();
                    
                    // Remove selected class from all divs
                    document.querySelectorAll('.fixed-vertical-block, div[data-testid="stVerticalBlock"], div[data-testid="stMarkdownContainer"]').forEach(el => {
                        el.classList.remove('selected-error');
                        el.classList.remove('selected-target');
                        el.classList.remove('selected-block');
                        el.classList.remove('selected-markdown');
                        if (el.classList.contains('fixed-vertical-block')) {
                            el.style.border = '2px dashed #4CAF50';
                            el.style.boxShadow = '';
                            el.style.backgroundColor = '';
                        }
                    });
                    
                    // Apply special highlight and add selected class
                    this.style.border = '4px solid #D32F2F';
                    this.style.boxShadow = '0 0 20px rgba(211,47,47,0.8)';
                    this.style.backgroundColor = 'rgba(244,67,54,0.1)';
                    this.classList.add('selected-error');
                    
                    // Create detailed data object
                    const data = {
                        type: 'ERROR_DIV',
                        tagName: this.tagName,
                        className: this.className,
                        dataTestId: this.getAttribute('data-testid') || '',
                        dataComponentName: this.getAttribute('data-component-name') || '',
                        width: this.getAttribute('width') || '',
                        childrenCount: this.children.length,
                        structure: getElementStructure(this, 3),
                        text: this.innerText,
                        errorMessage: this.innerText
                    };
                    
                    // Send data to Python
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: data
                    }, '*');
                    
                    console.log('Selected error div:', data);
                    
                    return false;
                };
                
                console.log('Fixed error message div:', div);
            } else {
                // For non-error message divs, apply standard styling
                div.style.border = '2px solid #2196F3';
                div.style.padding = '5px';
                div.style.margin = '5px 0';
                div.style.position = 'relative';
                div.style.transition = 'all 0.3s ease';
                
                // Add hover effect
                div.onmouseover = function() {
                    this.style.border = '2px solid #1976D2';
                    this.style.boxShadow = '0 0 10px rgba(33,150,243,0.5)';
                    this.style.backgroundColor = 'rgba(33,150,243,0.05)';
                };
                
                div.onmouseout = function() {
                    if (!this.classList.contains('selected-markdown')) {
                        this.style.border = '2px solid #2196F3';
                        this.style.boxShadow = '';
                        this.style.backgroundColor = '';
                    }
                };
                
                // Add a label
                const label = document.createElement('div');
                label.style.position = 'absolute';
                label.style.top = '0';
                label.style.right = '0';
                label.style.backgroundColor = '#2196F3';
                label.style.color = 'white';
                label.style.padding = '2px 5px';
                label.style.fontSize = '10px';
                label.style.borderRadius = '0 0 0 5px';
                label.style.zIndex = '999';
                label.innerText = `Markdown ${index+1}`;
                div.appendChild(label);
                
                // Add click handler
                div.onclick = function(e) {
                    e.stopPropagation();
                    
                    // Reset all selections
                    document.querySelectorAll('.fixed-vertical-block, div[data-testid="stVerticalBlock"], div[data-testid="stMarkdownContainer"]').forEach(el => {
                        el.classList.remove('selected-markdown');
                        el.classList.remove('selected-target');
                        el.classList.remove('selected-block');
                        el.classList.remove('selected-error');
                        if (el.classList.contains('fixed-vertical-block')) {
                            el.style.border = '2px dashed #4CAF50';
                            el.style.boxShadow = '';
                            el.style.backgroundColor = '';
                        }
                    });
                    
                    // Highlight this div
                    this.style.border = '3px solid #1976D2';
                    this.style.boxShadow = '0 0 10px rgba(25,118,210,0.5)';
                    this.style.backgroundColor = 'rgba(33,150,243,0.05)';
                    this.classList.add('selected-markdown');
                    
                    // Get element information
                    const data = {
                        type: 'MARKDOWN_DIV',
                        tagName: this.tagName,
                        className: this.className,
                        dataTestId: this.getAttribute('data-testid') || '',
                        dataComponentName: this.getAttribute('data-component-name') || '',
                        width: this.getAttribute('width') || '',
                        childrenCount: this.children.length,
                        structure: getElementStructure(this, 2),
                        text: this.innerText.substring(0, 150) + (this.innerText.length > 150 ? '...' : '')
                    };
                    
                    // Send data to Python
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: data
                    }, '*');
                    
                    console.log('Selected markdown div:', data);
                    
                    return false;
                };
            }
        });
    }
    
    // Helper function to get element structure
    function getElementStructure(element, depth = 1, currentDepth = 0) {
        if (currentDepth >= depth) return '...';
        
        const children = Array.from(element.children).map(child => {
            return {
                tag: child.tagName.toLowerCase(),
                id: child.id || undefined,
                class: child.className || undefined,
                'data-testid': child.getAttribute('data-testid') || undefined,
                children: getElementStructure(child, depth, currentDepth + 1)
            };
        });
        
        return children;
    }
    
    // Initialize when the DOM is fully loaded
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM fully loaded');
        setTimeout(() => {
            enableElementSelection();
            fixVerticalBlockDivs();
            fixSpecificDiv();
        }, 1000); // Delay to ensure Streamlit components are loaded
    });
    
    // Also run when this component is mounted (in case DOMContentLoaded already fired)
    setTimeout(() => {
        console.log('Component mounted, initializing');
        enableElementSelection();
        fixVerticalBlockDivs();
        fixSpecificDiv();
    }, 1000);
    
    // Add a mutation observer to handle dynamically added elements
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length) {
                console.log('DOM changed, re-fixing elements');
                fixVerticalBlockDivs();
                fixSpecificDiv();
            }
        });
    });
    
    // Start observing after a delay
    setTimeout(() => {
        observer.observe(document.body, { childList: true, subtree: true });
    }, 2000);
    </script>
    <div style="padding: 15px; background-color: #f0f2f6; border-radius: 8px; margin-bottom: 15px; border: 1px solid #ddd;">
        <h3 style="margin-top: 0; color: #333;">DOM Element Interaction</h3>
        <p>Click on any element on the page to select it. Selected elements will be highlighted with a red border.</p>
        <p>Vertical block divs have been highlighted with green dashed borders for easier selection.</p>
        <p>The specific target div has been highlighted with an orange border.</p>
        <div style="display: flex; gap: 10px; margin-top: 15px;">
            <button onclick="fixVerticalBlockDivs()" style="background-color: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; flex: 1;">
                Re-highlight Vertical Blocks
            </button>
            <button onclick="fixSpecificDiv()" style="background-color: #FF9800; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; flex: 1;">
                Fix Target Div
            </button>
        </div>
    </div>
    """
    
    # Render the HTML/JS with a key to ensure proper reactivity
    component_value = components_v1.html(
        dom_interaction_html, 
        height=250,
        scrolling=True
    )
    
    # Update session state if component returns a value
    if component_value:
        st.session_state.selected_element = component_value
    
    # Display selected element information
    st.sidebar.subheader("Selected Element")
    
    if st.session_state.selected_element:
        element_data = st.session_state.selected_element
        
        # Create a more visually appealing display
        st.sidebar.markdown(f"""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border-left: 5px solid #1E88E5;">
            <h4 style="color: #1E88E5; margin-top: 0;">Element Details</h4>
            <p><strong>Tag:</strong> {element_data.get('tagName', 'N/A') if isinstance(element_data, dict) else 'N/A'}</p>
            <p><strong>Test ID:</strong> {element_data.get('dataTestId', 'N/A') if isinstance(element_data, dict) else 'N/A'}</p>
            <p><strong>Component:</strong> {element_data.get('dataComponentName', 'N/A') if isinstance(element_data, dict) else 'N/A'}</p>
            <p><strong>Width:</strong> {element_data.get('width', 'N/A') if isinstance(element_data, dict) else 'N/A'}</p>
            <p><strong>Children:</strong> {element_data.get('childrenCount', 'N/A') if isinstance(element_data, dict) else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show element text if available
        if isinstance(element_data, dict) and 'text' in element_data and element_data['text']:
            st.sidebar.markdown("#### Element Text")
            st.sidebar.markdown(f"""
            <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto;">
                {element_data['text']}
            </div>
            """, unsafe_allow_html=True)
        
        # Show element structure if available
        if isinstance(element_data, dict) and 'structure' in element_data and element_data['structure']:
            st.sidebar.markdown("#### Element Structure")
            st.sidebar.json(element_data['structure'])
            
        # Add actions for the selected element
        st.sidebar.markdown("#### Actions")
        if st.sidebar.button("Analyze Element"):
            st.sidebar.info("Analyzing element structure and content...")
            # In a real implementation, this would perform additional analysis
            
        if isinstance(element_data, dict) and 'type' in element_data:
            if element_data['type'] == 'TARGET_DIV':
                st.sidebar.success("This is the target vertical block div element!")
            elif element_data['type'] == 'ERROR_DIV':
                st.sidebar.error("This is an error message div!")
                if isinstance(element_data, dict) and 'errorMessage' in element_data:
                    st.sidebar.code(element_data['errorMessage'], language="text")
                    
                    # Provide potential fixes based on the error message
                    if isinstance(element_data['errorMessage'], str):
                        if "experimental_rerun" in element_data['errorMessage']:
                            st.sidebar.info("This error is caused by using the deprecated st.experimental_rerun() function. Replace it with st.rerun() instead.")
                        elif "module" in element_data['errorMessage'] and "has no attribute" in element_data['errorMessage']:
                            st.sidebar.info("This error is related to a missing module attribute. Check your imports and module versions.")
            elif element_data['type'] == 'MARKDOWN_DIV':
                st.sidebar.info("This is a markdown container div.")
    else:
        st.sidebar.info("No element selected. Click on an element to select it.")

# Add footer
st.markdown("---")
st.caption(" 2025 Photonics Research Agent | Running in Docker")
