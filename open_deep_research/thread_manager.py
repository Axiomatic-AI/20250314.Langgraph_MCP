"""Thread Manager for LangGraph persistence in the Photonics Research Dashboard.

This module provides functionality to save, load, and manage LangGraph threads,
enabling incremental research capabilities with proper checkpointing.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
from langchain_core.runnables import RunnableConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
THREADS_DIR = os.path.join(os.path.dirname(__file__), "research_threads")
os.makedirs(THREADS_DIR, exist_ok=True)


class ThreadManager:
    """Manages LangGraph threads for the Photonics Research Dashboard."""
    
    def __init__(self, max_threads: int = 50):
        """Initialize the thread manager.
        
        Args:
            max_threads: Maximum number of threads to store
        """
        self.max_threads = max_threads
        self._ensure_threads_dir()
    
    def _ensure_threads_dir(self) -> None:
        """Ensure the threads directory exists."""
        if not os.path.exists(THREADS_DIR):
            os.makedirs(THREADS_DIR)
            logger.info(f"Created threads directory at {THREADS_DIR}")
    
    def save_thread(self, thread_id: str, metadata: Dict[str, Any], 
                   topic: str, checkpoints: List[Dict[str, Any]]) -> str:
        """Save a thread to disk with metadata.
        
        Args:
            thread_id: The ID of the thread
            metadata: Metadata about the thread (creation time, last modified, etc.)
            topic: The research topic
            checkpoints: List of checkpoint data for the thread
            
        Returns:
            Path to the saved thread file
        """
        # Create thread data structure
        thread_data = {
            "thread_id": thread_id,
            "metadata": {
                **metadata,
                "last_modified": datetime.now().isoformat(),
                "memory_usage_mb": self._get_current_memory_usage()
            },
            "topic": topic,
            "checkpoints": checkpoints
        }
        
        # Save to file
        thread_path = os.path.join(THREADS_DIR, f"{thread_id}.json")
        with open(thread_path, "w") as f:
            json.dump(thread_data, f, indent=2)
        
        logger.info(f"Saved thread {thread_id} to {thread_path}")
        
        # Clean up old threads if needed
        self._cleanup_old_threads()
        
        return thread_path
    
    def load_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Load a thread from disk.
        
        Args:
            thread_id: The ID of the thread to load
            
        Returns:
            Thread data if found, None otherwise
        """
        thread_path = os.path.join(THREADS_DIR, f"{thread_id}.json")
        if not os.path.exists(thread_path):
            logger.warning(f"Thread {thread_id} not found at {thread_path}")
            return None
        
        try:
            with open(thread_path, "r") as f:
                thread_data = json.load(f)
            
            logger.info(f"Loaded thread {thread_id} from {thread_path}")
            return thread_data
        except Exception as e:
            logger.error(f"Error loading thread {thread_id}: {e}")
            return None
    
    def list_threads(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List available threads with their metadata.
        
        Args:
            limit: Maximum number of threads to return
            
        Returns:
            List of thread metadata
        """
        thread_files = [f for f in os.listdir(THREADS_DIR) if f.endswith(".json")]
        threads = []
        
        for thread_file in thread_files[:limit]:
            thread_path = os.path.join(THREADS_DIR, thread_file)
            try:
                with open(thread_path, "r") as f:
                    thread_data = json.load(f)
                
                # Extract summary data
                threads.append({
                    "thread_id": thread_data.get("thread_id"),
                    "topic": thread_data.get("topic"),
                    "last_modified": thread_data.get("metadata", {}).get("last_modified"),
                    "checkpoint_count": len(thread_data.get("checkpoints", [])),
                    "memory_usage_mb": thread_data.get("metadata", {}).get("memory_usage_mb")
                })
            except Exception as e:
                logger.error(f"Error reading thread file {thread_file}: {e}")
        
        # Sort by last modified time (newest first)
        threads.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
        
        return threads[:limit]
    
    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread.
        
        Args:
            thread_id: The ID of the thread to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        thread_path = os.path.join(THREADS_DIR, f"{thread_id}.json")
        if not os.path.exists(thread_path):
            logger.warning(f"Thread {thread_id} not found at {thread_path}")
            return False
        
        try:
            os.remove(thread_path)
            logger.info(f"Deleted thread {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting thread {thread_id}: {e}")
            return False
    
    def create_runnable_config(self, thread_id: str, 
                              checkpoint_interval: int = 5) -> RunnableConfig:
        """Create a LangGraph runnable config with the thread ID.
        
        Args:
            thread_id: The ID of the thread to use
            checkpoint_interval: How often to create checkpoints (in steps)
            
        Returns:
            RunnableConfig for LangGraph
        """
        return {"configurable": {"thread_id": thread_id}}
    
    def _cleanup_old_threads(self) -> None:
        """Clean up old threads if we exceed the maximum."""
        thread_files = [f for f in os.listdir(THREADS_DIR) if f.endswith(".json")]
        
        if len(thread_files) <= self.max_threads:
            return
        
        # Get thread modification times
        thread_times = []
        for thread_file in thread_files:
            thread_path = os.path.join(THREADS_DIR, thread_file)
            try:
                mtime = os.path.getmtime(thread_path)
                thread_times.append((thread_path, mtime))
            except Exception:
                continue
        
        # Sort by modification time (oldest first)
        thread_times.sort(key=lambda x: x[1])
        
        # Delete oldest threads
        threads_to_delete = len(thread_files) - self.max_threads
        for i in range(threads_to_delete):
            if i < len(thread_times):
                try:
                    os.remove(thread_times[i][0])
                    logger.info(f"Cleaned up old thread: {thread_times[i][0]}")
                except Exception as e:
                    logger.error(f"Error cleaning up thread {thread_times[i][0]}: {e}")
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB


# Convenience function to get a thread manager instance
def get_thread_manager() -> ThreadManager:
    """Get a thread manager instance."""
    return ThreadManager()
