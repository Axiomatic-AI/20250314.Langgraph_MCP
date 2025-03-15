"""Research Branching module for the Photonics Research Dashboard.

This module provides functionality to create, manage, and visualize branching
research paths using LangGraph's thread management capabilities.
"""

# Standard library imports
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Local imports
from thread_manager import ThreadManager, get_thread_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearchBranch:
    """Represents a branch in the research process."""
    
    def __init__(
        self,
        branch_id: str,
        parent_id: Optional[str],
        thread_id: str,
        topic: str,
        description: str,
        checkpoint_idx: int,
        created_at: Optional[str] = None
    ):
        """Initialize a research branch.
        
        Args:
            branch_id: Unique identifier for this branch
            parent_id: ID of the parent branch (None if root)
            thread_id: LangGraph thread ID associated with this branch
            topic: Research topic for this branch
            description: Description of this research branch
            checkpoint_idx: Index of the checkpoint in the parent thread
            created_at: Creation timestamp
        """
        self.branch_id = branch_id
        self.parent_id = parent_id
        self.thread_id = thread_id
        self.topic = topic
        self.description = description
        self.checkpoint_idx = checkpoint_idx
        self.created_at = created_at or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "thread_id": self.thread_id,
            "topic": self.topic,
            "description": self.description,
            "checkpoint_idx": self.checkpoint_idx,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchBranch':
        """Create a branch from dictionary data."""
        return cls(
            branch_id=data["branch_id"],
            parent_id=data["parent_id"],
            thread_id=data["thread_id"],
            topic=data["topic"],
            description=data["description"],
            checkpoint_idx=data["checkpoint_idx"],
            created_at=data["created_at"]
        )


class ResearchBranchManager:
    """Manages branching research paths."""
    
    def __init__(self):
        """Initialize the research branch manager."""
        self.thread_manager = get_thread_manager()
        self.branches_file = os.path.join(
            os.path.dirname(__file__), 
            "research_threads", 
            "branches.json"
        )
        self._ensure_branches_file()
    
    def _ensure_branches_file(self) -> None:
        """Ensure the branches file exists."""
        if not os.path.exists(self.branches_file):
            with open(self.branches_file, "w") as f:
                json.dump([], f)
    
    def create_root_branch(self, topic: str, description: str) -> ResearchBranch:
        """Create a new root branch (starting point for research).
        
        Args:
            topic: Research topic
            description: Description of the research
            
        Returns:
            The created branch
        """
        branch_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())
        
        branch = ResearchBranch(
            branch_id=branch_id,
            parent_id=None,  # Root branch has no parent
            thread_id=thread_id,
            topic=topic,
            description=description,
            checkpoint_idx=0
        )
        
        # Save branch
        self._save_branch(branch)
        
        # Initialize empty thread data
        self.thread_manager.save_thread(
            thread_id=thread_id,
            metadata={"branch_id": branch_id, "is_root": True},
            topic=topic,
            checkpoints=[{"state": "initialized", "timestamp": datetime.now().isoformat()}]
        )
        
        return branch
    
    def create_branch(
        self, 
        parent_id: str, 
        topic: str, 
        description: str,
        checkpoint_idx: int
    ) -> Optional[ResearchBranch]:
        """Create a new branch from an existing one.
        
        Args:
            parent_id: ID of the parent branch
            topic: Research topic for the new branch
            description: Description of the new branch
            checkpoint_idx: Index of the checkpoint to branch from
            
        Returns:
            The created branch or None if parent not found
        """
        # Find parent branch
        parent = self.get_branch(parent_id)
        if not parent:
            logger.error(f"Parent branch {parent_id} not found")
            return None
        
        # Load parent thread
        parent_thread = self.thread_manager.load_thread(parent.thread_id)
        if not parent_thread:
            logger.error(f"Parent thread {parent.thread_id} not found")
            return None
        
        # Ensure checkpoint exists
        checkpoints = parent_thread.get("checkpoints", [])
        if checkpoint_idx >= len(checkpoints):
            logger.error(f"Checkpoint {checkpoint_idx} not found in thread {parent.thread_id}")
            return None
        
        # Create new branch
        branch_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())
        
        branch = ResearchBranch(
            branch_id=branch_id,
            parent_id=parent_id,
            thread_id=thread_id,
            topic=topic,
            description=description,
            checkpoint_idx=checkpoint_idx
        )
        
        # Save branch
        self._save_branch(branch)
        
        # Create new thread with checkpoint data
        checkpoint_data = checkpoints[checkpoint_idx]
        self.thread_manager.save_thread(
            thread_id=thread_id,
            metadata={
                "branch_id": branch_id,
                "parent_branch_id": parent_id,
                "parent_thread_id": parent.thread_id,
                "parent_checkpoint_idx": checkpoint_idx
            },
            topic=topic,
            checkpoints=[
                {
                    "state": "branched",
                    "parent_checkpoint": checkpoint_data,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        )
        
        return branch
    
    def get_branch(self, branch_id: str) -> Optional[ResearchBranch]:
        """Get a branch by ID.
        
        Args:
            branch_id: ID of the branch to get
            
        Returns:
            The branch or None if not found
        """
        branches = self._load_branches()
        for branch_data in branches:
            if branch_data.get("branch_id") == branch_id:
                return ResearchBranch.from_dict(branch_data)
        return None
    
    def list_branches(self) -> List[ResearchBranch]:
        """List all branches.
        
        Returns:
            List of all branches
        """
        branches = self._load_branches()
        return [ResearchBranch.from_dict(b) for b in branches]
    
    def get_branch_tree(self) -> nx.DiGraph:
        """Get the branch tree as a NetworkX graph.
        
        Returns:
            NetworkX DiGraph representing the branch tree
        """
        branches = self.list_branches()
        
        G = nx.DiGraph()
        
        # Add nodes
        for branch in branches:
            G.add_node(
                branch.branch_id,
                topic=branch.topic,
                description=branch.description,
                thread_id=branch.thread_id,
                created_at=branch.created_at
            )
        
        # Add edges
        for branch in branches:
            if branch.parent_id:
                G.add_edge(branch.parent_id, branch.branch_id)
        
        return G
    
    def visualize_branch_tree(self) -> plt.Figure:
        """Create a visualization of the branch tree.
        
        Returns:
            Matplotlib figure with the branch tree visualization
        """
        G = self.get_branch_tree()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=2000, 
            node_color="lightblue",
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            width=2, 
            edge_color="gray",
            arrowsize=20,
            ax=ax
        )
        
        # Draw labels
        labels = {node: G.nodes[node]["topic"][:20] + "..." for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, 
            labels=labels,
            font_size=10,
            font_weight="bold",
            ax=ax
        )
        
        # Set title
        plt.title("Research Branch Tree", fontsize=16)
        plt.axis("off")
        
        return fig
    
    def delete_branch(self, branch_id: str) -> bool:
        """Delete a branch and its thread.
        
        Args:
            branch_id: ID of the branch to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        branch = self.get_branch(branch_id)
        if not branch:
            logger.error(f"Branch {branch_id} not found")
            return False
        
        # Delete thread
        self.thread_manager.delete_thread(branch.thread_id)
        
        # Delete branch
        branches = self._load_branches()
        branches = [b for b in branches if b.get("branch_id") != branch_id]
        self._save_branches(branches)
        
        return True
    
    def _load_branches(self) -> List[Dict[str, Any]]:
        """Load branches from file."""
        try:
            with open(self.branches_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading branches: {e}")
            return []
    
    def _save_branches(self, branches: List[Dict[str, Any]]) -> None:
        """Save branches to file."""
        try:
            with open(self.branches_file, "w") as f:
                json.dump(branches, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving branches: {e}")
    
    def _save_branch(self, branch: ResearchBranch) -> None:
        """Save a branch to the branches file."""
        branches = self._load_branches()
        
        # Check if branch already exists
        for i, b in enumerate(branches):
            if b.get("branch_id") == branch.branch_id:
                branches[i] = branch.to_dict()
                self._save_branches(branches)
                return
        
        # Add new branch
        branches.append(branch.to_dict())
        self._save_branches(branches)


# Streamlit UI components for research branching
def render_branch_ui():
    """Render the research branching UI in Streamlit."""
    st.header("Research Branching")
    
    branch_manager = ResearchBranchManager()
    
    # Tabs for different branch operations
    tab1, tab2, tab3 = st.tabs(["View Branches", "Create Branch", "Branch Details"])
    
    # Tab 1: View Branches
    with tab1:
        st.subheader("Research Branch Tree")
        
        branches = branch_manager.list_branches()
        if not branches:
            st.info("No research branches found. Create a new root branch to get started.")
        else:
            # Visualize branch tree
            fig = branch_manager.visualize_branch_tree()
            st.pyplot(fig)
            
            # List branches in a table
            branch_data = []
            for branch in branches:
                branch_data.append({
                    "ID": branch.branch_id[:8] + "...",
                    "Topic": branch.topic,
                    "Description": branch.description[:30] + "..." if len(branch.description) > 30 else branch.description,
                    "Created": branch.created_at.split("T")[0]
                })
            
            st.dataframe(branch_data)
    
    # Tab 2: Create Branch
    with tab2:
        st.subheader("Create New Research Branch")
        
        # Option to create root or child branch
        branch_type = st.radio(
            "Branch Type",
            ["Root Branch (New Research)", "Child Branch (From Existing)"]
        )
        
        if branch_type == "Root Branch (New Research)":
            # Create root branch
            topic = st.text_input("Research Topic")
            description = st.text_area("Description")
            
            if st.button("Create Root Branch"):
                if topic and description:
                    branch = branch_manager.create_root_branch(topic, description)
                    st.success(f"Created root branch: {branch.branch_id}")
                else:
                    st.error("Please provide both topic and description")
        else:
            # Create child branch
            branches = branch_manager.list_branches()
            if not branches:
                st.warning("No existing branches to branch from. Create a root branch first.")
            else:
                # Select parent branch
                parent_options = {
                    f"{b.topic} ({b.branch_id[:8]}...)": b.branch_id 
                    for b in branches
                }
                parent_selection = st.selectbox(
                    "Parent Branch",
                    list(parent_options.keys())
                )
                parent_id = parent_options[parent_selection]
                
                # Get parent thread checkpoints
                parent = branch_manager.get_branch(parent_id)
                if parent:
                    parent_thread = branch_manager.thread_manager.load_thread(parent.thread_id)
                    if parent_thread:
                        checkpoints = parent_thread.get("checkpoints", [])
                        
                        # Select checkpoint
                        checkpoint_options = []
                        for i, cp in enumerate(checkpoints):
                            timestamp = cp.get("timestamp", "Unknown")
                            state = cp.get("state", "Unknown")
                            checkpoint_options.append(f"Checkpoint {i}: {state} ({timestamp})")
                        
                        checkpoint_idx = 0
                        if checkpoint_options:
                            checkpoint_selection = st.selectbox(
                                "Branch from Checkpoint",
                                checkpoint_options
                            )
                            checkpoint_idx = checkpoint_options.index(checkpoint_selection)
                        
                        # Branch details
                        topic = st.text_input("New Branch Topic", value=parent.topic)
                        description = st.text_area(
                            "New Branch Description",
                            value=f"Branched from {parent.topic}"
                        )
                        
                        if st.button("Create Branch"):
                            if topic and description:
                                branch = branch_manager.create_branch(
                                    parent_id=parent_id,
                                    topic=topic,
                                    description=description,
                                    checkpoint_idx=checkpoint_idx
                                )
                                if branch:
                                    st.success(f"Created branch: {branch.branch_id}")
                                else:
                                    st.error("Failed to create branch")
                            else:
                                st.error("Please provide both topic and description")
    
    # Tab 3: Branch Details
    with tab3:
        st.subheader("Branch Details")
        
        branches = branch_manager.list_branches()
        if not branches:
            st.info("No research branches found.")
        else:
            # Select branch
            branch_options = {
                f"{b.topic} ({b.branch_id[:8]}...)": b.branch_id 
                for b in branches
            }
            branch_selection = st.selectbox(
                "Select Branch",
                list(branch_options.keys())
            )
            branch_id = branch_options[branch_selection]
            
            # Get branch details
            branch = branch_manager.get_branch(branch_id)
            if branch:
                # Display branch info
                st.json({
                    "branch_id": branch.branch_id,
                    "parent_id": branch.parent_id,
                    "thread_id": branch.thread_id,
                    "topic": branch.topic,
                    "description": branch.description,
                    "created_at": branch.created_at
                })
                
                # Thread details
                thread = branch_manager.thread_manager.load_thread(branch.thread_id)
                if thread:
                    st.subheader("Thread Checkpoints")
                    
                    checkpoints = thread.get("checkpoints", [])
                    for i, cp in enumerate(checkpoints):
                        with st.expander(f"Checkpoint {i}"):
                            st.json(cp)
                
                # Delete branch
                if st.button("Delete Branch", type="primary"):
                    if branch_manager.delete_branch(branch_id):
                        st.success(f"Deleted branch: {branch_id}")
                        st.rerun()
                    else:
                        st.error("Failed to delete branch")


# Main function for testing
def main():
    """Main function for testing."""
    st.set_page_config(
        page_title="Research Branching",
        page_icon="🌿",
        layout="wide"
    )
    
    st.title("Research Branching")
    render_branch_ui()


if __name__ == "__main__":
    main()
