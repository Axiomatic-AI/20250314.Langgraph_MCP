#!/usr/bin/env python
"""LLM-Powered Advanced Research Dashboard for Photonics Research.

This dashboard provides three key advanced capabilities:
1. Mathematical Verification - Parse and verify equations from research papers
2. Scientific Concept Mapping - Extract and visualize concept relationships
3. Advanced Model Integration - Route research questions to specialized models.
"""

# Standard library imports
import logging
import os

# Third-party imports
import networkx as nx
import streamlit as st
from dotenv import load_dotenv

# Local application imports
import llm_advanced_research as lar

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Advanced Research Dashboard", layout="wide")

# Load environment variables
try:
    # First try to load from .env file if it exists
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logger.info("Loaded environment variables from .env file")
    else:
        logger.info("No .env file found, using environment variables")
except Exception as e:
    logger.warning(f"Error loading .env file: {str(e)}")

# Initialize modules with API key from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
logger.info(f"API key {'found' if ANTHROPIC_API_KEY else 'not found'}")

# Check if API key is set, and provide appropriate messages when not available
api_key_error = None
if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_api_key_here":
    api_key_error = "Anthropic API key is missing. Please provide a valid API key."

# Try to initialize components, handle errors
initialization_error = None
try:
    if api_key_error:
        st.error(f"⚠️ {api_key_error}")
        raise ValueError(api_key_error)

    # Check API key validity
    if not lar.check_api_key(ANTHROPIC_API_KEY):
        raise ValueError("Invalid API key")
    
    # Initialize research components
    model_router, math_verifier, concept_mapper, paper_search = lar.init_components(ANTHROPIC_API_KEY)
except Exception as e:
    logger.error(f"Error initializing research modules: {str(e)}")
    initialization_error = str(e)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    .stMarkdown p {
        margin-bottom: 10px;
    }
    .equation-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .concept-node {
        background-color: #4e8df5;
        color: white;
        border-radius: 15px;
        padding: 5px 10px;
        margin: 5px;
        display: inline-block;
    }
    .model-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    /* Fix for plot container sizing */
    .stPlotlyChart {
        width: 100%;
    }
    /* Improve input field styling */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border-radius: 5px;
        border-color: #ddd;
    }
    /* Make buttons more consistent */
    .stButton > button {
        border-radius: 5px;
        background-color: #4e8df5;
        color: white;
    }
    .stButton > button:hover {
        background-color: #3a6fc5;
    }
    /* Fix sidebar spacing */
    .css-1d391kg {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar for global settings
with st.sidebar:
    st.title("Advanced Research Dashboard")
    st.markdown("---")
    st.markdown("### Settings")
    
    # Select API model
    model_name = st.selectbox(
        "Select Model",
        options=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        index=0,
        key="selected_model"
    )
    
    # Apply dark/light mode toggle
    theme_mode = st.radio(
        "Theme",
        options=["Light", "Dark"],
        index=0,
        key="theme_mode"
    )
    
    if theme_mode == "Dark":
        st.markdown("""
        <style>
            .main {
                background-color: #0e1117;
                color: #fff;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #262730;
            }
            .equation-box, .model-card {
                background-color: #262730;
                color: #fff;
            }
            .stTextInput > div > div > input, .stTextArea > div > div > textarea {
                background-color: #262730;
                color: #fff;
            }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(" 2025 Advanced Research Team")

st.title("Advanced Research Dashboard")
st.markdown("Explore research papers, verify mathematical concepts, and map scientific domains.")

# Create tabs for different functionalities
selected_tab = st.radio(
    "Select Function",
    ["Paper Search", "Mathematical Verification", "Scientific Concept Mapping", "Research Question"],
    horizontal=True
)

# Display initialization error if any
if initialization_error:
    st.markdown(f"""
    <div class="error-box">
        <strong>Initialization Error:</strong> {initialization_error}
        <p>Please check your API keys and connections.</p>
    </div>
    """, unsafe_allow_html=True)

# Display the appropriate tab content
if selected_tab == "Paper Search":
    st.header("Research Paper Search")
    st.markdown("""
    Search for research papers on arXiv based on keywords or topics. Select papers for deeper analysis.
    """)
    
    # Initialize session state for paper search if not exists
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_papers' not in st.session_state:
        st.session_state.selected_papers = []
    if 'paper_analysis_question' not in st.session_state:
        st.session_state.paper_analysis_question = ""
    if 'paper_analysis_result' not in st.session_state:
        st.session_state.paper_analysis_result = None
    
    # Paper search form
    with st.form(key="paper_search_form"):
        search_query = st.text_input(
            "Enter search keywords:",
            value=st.session_state.search_query,
            placeholder="e.g., silicon photonics quantum computing",
            key="paper_search_query"
        )
        max_results = st.slider("Maximum results:", 5, 20, 10, key="max_results")
        search_submitted = st.form_submit_button("Search Papers")
    
    # Sample queries
    sample_queries = [
        "silicon photonics quantum computing",
        "thin-film lithium niobate",
        "photonic integrated circuits quantum",
        "CMOS compatible photonics"
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Silicon Photonics", key="sample_query_1"):
            st.session_state.search_query = sample_queries[0]
            st.rerun()
    with col2:
        if st.button("Lithium Niobate", key="sample_query_2"):
            st.session_state.search_query = sample_queries[1]
            st.rerun()
    
    # Process search
    if search_submitted and search_query:
        with st.spinner("Searching papers..."):
            st.session_state.search_query = search_query
            st.session_state.search_results = paper_search.search_papers(search_query, max_results)
    
    # Display search results
    if st.session_state.search_results and not st.session_state.search_results.error:
        st.markdown("### Search Results")
        
        for i, paper in enumerate(st.session_state.search_results.papers):
            paper_id = paper['id']
            is_selected = paper_id in [p['id'] for p in st.session_state.selected_papers]
            
            with st.expander(f"{paper['title']}", expanded=False):
                st.markdown(f"**Authors:** {paper['authors']}")
                st.markdown(f"**Published:** {paper['published']}")
                st.markdown(f"**Abstract:** {paper['abstract']}")
                st.markdown(f"[View PDF]({paper['url']})")
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if is_selected:
                        if st.button("Remove", key=f"remove_{paper_id}"):
                            st.session_state.selected_papers = [p for p in st.session_state.selected_papers if p['id'] != paper_id]
                            st.rerun()
                    else:
                        if st.button("Select", key=f"select_{paper_id}"):
                            st.session_state.selected_papers.append(paper)
                            st.rerun()
    
    # Selected papers section
    if st.session_state.selected_papers:
        st.markdown("### Selected Papers")
        for paper in st.session_state.selected_papers:
            st.markdown(f"- **{paper['title']}** ({paper['published']})")
        
        # Analysis question
        st.markdown("### Ask a Research Question")
        st.markdown("Ask a question about the selected papers for deep analysis.")
        
        question = st.text_area(
            "Your research question:",
            value=st.session_state.paper_analysis_question,
            height=100,
            placeholder="e.g., What are the common fabrication methods discussed in these papers?",
            key="paper_analysis_question_input"
        )
        
        if st.button("Analyze Papers", key="analyze_papers"):
            if question:
                with st.spinner("Analyzing papers..."):
                    st.session_state.paper_analysis_question = question
                    
                    # Prepare context from selected papers
                    context = ""
                    for i, paper in enumerate(st.session_state.selected_papers):
                        context += f"Paper {i+1}: {paper['title']}\n"
                        context += f"Authors: {paper['authors']}\n"
                        context += f"Abstract: {paper['abstract']}\n\n"
                    
                    # Use the model router for analysis
                    analysis = model_router.process_with_models_sync(
                        question, 
                        model_name="claude-3-opus-20240229", 
                        context=context
                    )
                    
                    # Get the synthesized answer if available, otherwise use the first response
                    if isinstance(analysis, dict) and "synthesized_answer" in analysis:
                        st.session_state.paper_analysis_result = analysis["synthesized_answer"]
                    elif isinstance(analysis, dict) and len(analysis) > 0:
                        # Get the first non-routing response
                        for key, value in analysis.items():
                            if key != "_routing_info":
                                st.session_state.paper_analysis_result = value
                                break
                    else:
                        st.session_state.paper_analysis_result = "No analysis results returned."
                    
        # Display analysis result
        if st.session_state.paper_analysis_result:
            st.markdown("### Analysis Result")
            st.markdown(st.session_state.paper_analysis_result)
    
    # Error handling for search
    elif st.session_state.search_results and st.session_state.search_results.error:
        st.error(f"Search error: {st.session_state.search_results.error}")
elif selected_tab == "Mathematical Verification":
    st.header("Mathematical Verification")
    st.markdown("""
    Verify mathematical equations and multi-step derivations using LLMs.
    Enter LaTeX equations or derivation steps to check their validity.
    """)
    
    # Create sub-tabs for equation verification and derivation verification
    eq_tab, deriv_tab = st.tabs(["Single Equation", "Derivation Steps"])
    
    # Single Equation Verification
    with eq_tab:
        st.subheader("Verify a Single Equation")
        
        # Sample equation button
        if st.button("Load Sample Equation", key="load_sample_equation"):
            st.session_state.single_equation = "E = mc^2"
            
        equation = st.text_area(
            "Enter a LaTeX equation (without $ symbols):",
            value=st.session_state.get("single_equation", "E = mc^2"),
            height=100,
            key="single_equation_input"
        )
        
        if st.button("Verify Equation", key="verify_single_equation"):
            with st.spinner("Verifying equation..."):
                try:
                    # Use the synchronous method instead of asyncio.run
                    result = math_verifier.verify_equation_sync(equation)
                    
                    if result.valid:
                        st.success(" Equation is valid!")
                        st.markdown(f"**Parsed expression:** `{result.parsed_expression}`")
                        if result.variables:
                            st.markdown(f"**Variables:** {', '.join(result.variables)}")
                        if result.simplified:
                            st.markdown(f"**Simplified form:** `{result.simplified}`")
                    else:
                        st.error(" Equation is invalid")
                        if result.error:
                            st.markdown(f"**Error:** {result.error}")
                except Exception as e:
                    st.error(f"Error verifying equation: {str(e)}")
                    logger.error(f"Error verifying equation: {str(e)}")
    
    # Derivation Verification
    with deriv_tab:
        st.subheader("Verify a Multi-Step Derivation")
        
        # Initialize derivation steps in session state if needed
        if "derivation_steps" not in st.session_state:
            st.session_state.derivation_steps = [""] * 5
        
        # Sample derivation button
        if st.button("Load Sample Derivation", key="load_sample_derivation"):
            st.session_state.derivation_steps = [
                "F = ma",
                "\\vec{F} = m\\vec{a}",
                "\\vec{F} = m\\frac{d\\vec{v}}{dt}",
                "\\vec{F} = \\frac{d(m\\vec{v})}{dt}",
                "\\vec{F} = \\frac{d\\vec{p}}{dt}"
            ]
        
        # Create layout for steps
        for i in range(5):
            col1, col2 = st.columns([10, 1])
            with col1:
                st.session_state.derivation_steps[i] = st.text_area(
                    f"Step {i+1}",
                    value=st.session_state.derivation_steps[i],
                    height=70,  # Keep minimum height requirement
                    key=f"derivation_step_{i}"
                )
        
        # Verify button
        if st.button("Verify Derivation", key="verify_derivation"):
            # Filter out empty steps
            steps = [step for step in st.session_state.derivation_steps if step.strip()]
            
            if len(steps) < 2:
                st.warning("Please enter at least two derivation steps.")
            else:
                with st.spinner("Verifying derivation steps..."):
                    try:
                        # Use synchronous method instead of asyncio
                        result = math_verifier.verify_derivation_sync(steps)
                        
                        if result.valid_derivation:
                            st.success(" Derivation is valid!")
                            
                            # Show step validity
                            for i, step_result in enumerate(result.steps):
                                if step_result.valid:
                                    st.markdown(f"**Step {step_result.step}**: Valid")
                                else:
                                    st.markdown(f"**Step {step_result.step}**: Invalid - {step_result.error}")
                                    
                            # Display reasoning if available
                            if hasattr(result, 'reasoning') and result.reasoning:
                                st.markdown("### Reasoning")
                                st.markdown(result.reasoning)
                        else:
                            st.error(" Derivation has errors")
                            if result.error:
                                st.markdown(f"**Error:** {result.error}")
                    except Exception as e:
                        st.error(f"Error verifying derivation: {str(e)}")
                        logger.error(f"Error verifying derivation: {str(e)}")
elif selected_tab == "Scientific Concept Mapping":
    st.header("Scientific Concept Mapping")
    st.markdown("""
    Extract key scientific concepts from research papers and visualize their relationships.
    Enter text from a research paper or upload a paper to extract concepts.
    """)
    
    # Create sub-tabs for concept extraction and graph visualization
    extract_tab, graph_tab = st.tabs(["Extract Concepts", "Concept Graph"])
    
    # Concept Extraction
    with extract_tab:
        st.subheader("Extract Scientific Concepts")
        
        SAMPLE_SILICON_PHOTONICS_TEXT = """
        Silicon photonics is an emerging technology that uses silicon integrated circuits to manipulate light for data transmission and processing. It enables the integration of optical components with electronic circuits, creating photonic integrated circuits (PICs). These PICs can incorporate quantum emitters like quantum dots, color centers in diamond, or defects in 2D materials to generate single photons for quantum information processing.
        
        Recent advances in silicon photonics have demonstrated several key benefits: high bandwidth, low power consumption, CMOS compatibility, and scalability for mass production. Waveguides, modulators, photodetectors, and resonant structures can all be fabricated using standard semiconductor manufacturing processes.
        
        When integrating quantum emitters in silicon, researchers have explored several approaches. Direct integration of quantum dots in silicon has challenges due to the indirect bandgap, but progress has been made with germanium quantum dots and defect centers. Hybrid approaches using III-V materials bonded to silicon platforms show promise for efficient light emission and detection.
        
        Photonic crystal cavities in silicon can enhance light-matter interactions, critical for quantum applications. These structures allow for precise control of the electromagnetic environment around quantum emitters, enhancing emission rates and collection efficiencies.
        """
        
        if 'sample_text_clicked' not in st.session_state:
            st.session_state.sample_text_clicked = False
            
        paper_text = st.text_area(
            "Enter text from a research paper:",
            value=SAMPLE_SILICON_PHOTONICS_TEXT if st.session_state.sample_text_clicked else "",
            height=200,
            placeholder="Paste text from a research paper here...",
            key="concept_paper_text"
        )
        
        # Sample paper button
        if st.button("Use Sample Text on Silicon Photonics", key="use_sample_text"):
            st.session_state.sample_text_clicked = True
            st.rerun()
        
        # File uploader for PDF papers (placeholder for now)
        uploaded_file = st.file_uploader("Or upload a research paper (PDF):", type=["pdf"], key="concept_pdf_upload")
        
        if uploaded_file is not None:
            st.info("PDF processing is not yet implemented. Please paste text directly.")
        
        # Extract button
        if st.button("Extract Concepts", key="extract_concepts"):
            if paper_text:
                with st.spinner("Extracting concepts..."):
                    try:
                        # Use synchronous method instead of asyncio
                        result = concept_mapper.extract_concepts_sync(paper_text)
                        
                        # Store the result in session state
                        st.session_state.extracted_concepts = result
                        st.session_state.paper_text = paper_text
                        
                        # Display the extracted concepts
                        st.markdown("### Extracted Concepts")
                        for concept in result.concepts:
                            st.markdown(f"<div class='concept-node'>{concept}</div>", unsafe_allow_html=True)
                        
                        # Display the relationships
                        if result.relationships:
                            st.markdown("### Concept Relationships")
                            for rel in result.relationships:
                                st.markdown(f"- **{rel.source}** → *{rel.relation}* → **{rel.target}**")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.markdown(f"""
                        <div class="error-box">
                            <strong>Debug Information:</strong>
                            <pre>{str(e)}</pre>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please enter some text or upload a paper.")
    
    # Graph Visualization
    with graph_tab:
        st.subheader("Visualize Concept Graph")
        
        # Check if we have extracted concepts
        if "extracted_concepts" not in st.session_state or not st.session_state.extracted_concepts:
            st.info("Extract concepts first in the 'Extract Concepts' tab to visualize the concept graph.")
        else:
            try:
                # Manually create a graph for visualization
                G = nx.DiGraph()
                
                # Add nodes for each concept
                for concept in st.session_state.extracted_concepts.concepts:
                    G.add_node(concept)
                
                # Add edges for each relationship
                for rel in st.session_state.extracted_concepts.relationships:
                    G.add_edge(rel.source, rel.target, label=rel.relation)
                
                # Visualize the graph if it has nodes
                if G.nodes:
                    try:
                        # Generate an interactive Plotly graph
                        plotly_fig = concept_mapper.visualize_concept_graph_plotly(G)
                        st.plotly_chart(plotly_fig, use_container_width=True)
                    
                        st.markdown("### Interactive Graph Features")
                        st.markdown("""
                        - **Zoom**: Scroll to zoom in/out
                        - **Pan**: Click and drag to move around
                        - **Hover**: Hover over nodes for more information
                        - **Select**: Click nodes to see connections
                        """)
                    
                        # Optional static visualization toggle
                        if st.checkbox("Show static visualization (matplotlib)"):
                            static_fig = concept_mapper.visualize_concept_graph(G)
                            st.pyplot(static_fig)
                    except Exception as e:
                        # Fallback to matplotlib if Plotly visualization fails
                        st.warning(f"Interactive visualization failed: {str(e)}. Falling back to static visualization.")
                        static_fig = concept_mapper.visualize_concept_graph(G)
                        st.pyplot(static_fig)
                else:
                    st.info("No concepts or relationships to visualize.")
            except Exception as e:
                st.error(f"An error occurred during graph visualization: {str(e)}")
                st.markdown(f"""
                <div class="error-box">
                    <strong>Debug Information:</strong>
                    <pre>{str(e)}</pre>
                </div>
                """, unsafe_allow_html=True)
else:
    st.header("Advanced Research Question Analysis")
    
    # Create a container for research question input
    research_question = st.text_area("Enter your research question:", 
                                    height=100, 
                                    help="Enter a specific, focused research question for analysis")
    
    # Optional context for the research question
    with st.expander("Additional Context (Optional)"):
        context = st.text_area("Enter any additional context:", height=150)
    
    # Domain selection
    st.subheader("Domain Expertise")
    st.markdown("Our multi-agent system will automatically identify relevant domains, but you can also specify which you think are most important:")
    
    col1, col2 = st.columns(2)
    with col1:
        physics = st.checkbox("Physics", value=False)
        biology = st.checkbox("Biology", value=False)
        chemistry = st.checkbox("Chemistry", value=False)
    
    with col2:
        computer_science = st.checkbox("Computer Science", value=False)
        mathematics = st.checkbox("Mathematics", value=False)
        interdisciplinary = st.checkbox("Interdisciplinary", value=True)
    
    # Get selected domains
    selected_domains = []
    if physics:
        selected_domains.append("physics")
    if biology:
        selected_domains.append("biology")
    if chemistry:
        selected_domains.append("chemistry")
    if computer_science:
        selected_domains.append("computer_science")
    if mathematics:
        selected_domains.append("mathematics")
    if interdisciplinary:
        selected_domains.append("interdisciplinary")
    
    # Process the research question
    if st.button("Analyze Research Question"):
        if not research_question:
            st.warning("Please enter a research question.")
        else:
            try:
                # Get the model from the sidebar if it exists
                selected_model = st.session_state.get("selected_model", "claude-3-opus-20240229")
                
                # Show a spinner while processing
                with st.spinner("Processing research question with multiple specialized agents..."):
                    # Get responses from the model router
                    responses = model_router.process_with_models_sync(research_question, selected_model)
                    
                    # Check for errors
                    if "error" in responses:
                        st.error(f"An error occurred: {responses['error']}")
                    else:
                        # Display routing information
                        if "_routing_info" in responses:
                            routing_info = responses.pop("_routing_info")
                            
                            with st.expander("Domain Analysis", expanded=True):
                                st.subheader("Relevant Domains:")
                                if "relevant_domains" in routing_info:
                                    domains = routing_info["relevant_domains"]
                                    for domain in domains:
                                        if domain in model_router.domain_agents:
                                            agent = model_router.domain_agents[domain]
                                            st.markdown(f"- **{agent['name']}**: {agent['description']}")
                                else:
                                    st.markdown("No specific domains identified.")
                        
                        # Display the synthesized answer if available
                        if "synthesized_answer" in responses:
                            st.subheader("Collaborative Research Answer")
                            st.markdown(responses["synthesized_answer"])
                            
                            # Show individual agent responses in expandable sections
                            with st.expander("View Individual Expert Analyses", expanded=False):
                                for agent_name, response in responses.items():
                                    if agent_name not in ["synthesized_answer", "_routing_info"]:
                                        domain = agent_name.split("_")[0]
                                        if domain in model_router.domain_agents:
                                            agent = model_router.domain_agents[domain]
                                            st.markdown(f"### {agent['name']}")
                                            st.markdown(response)
                                            st.markdown("---")
                        else:
                            # If there's no synthesized answer, just show the regular response
                            for model_name, response in responses.items():
                                if model_name not in ["_routing_info"]:
                                    st.subheader("Analysis Results")
                                    st.markdown(response)
                                    
            except Exception as e:
                st.error(f"Error processing research question: {str(e)}")
                st.markdown(f"""
                <div class="error-box">
                    <strong>Debug Information:</strong>
                    <pre>{str(e)}</pre>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
**Photonics Research Advanced Dashboard** | Powered by LLMs and LangChain
""")
