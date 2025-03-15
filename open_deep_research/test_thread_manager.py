#!/usr/bin/env python
"""Unit tests for the Thread Manager module.

This script tests the functionality of the ThreadManager class to ensure
proper thread persistence and management for the LangGraph integration.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime

import thread_manager
from thread_manager import ThreadManager


class TestThreadManager(unittest.TestCase):
    """Test cases for the ThreadManager class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test threads
        self.test_dir = tempfile.mkdtemp()
        
        # Save original threads directory
        self.original_threads_dir = thread_manager.THREADS_DIR
        
        # Override the module-level THREADS_DIR constant for testing
        thread_manager.THREADS_DIR = self.test_dir
        
        # Create a test thread manager
        self.thread_manager = ThreadManager(max_threads=5)
        
        # Ensure the test directory exists
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        
        # Restore the original threads directory
        thread_manager.THREADS_DIR = self.original_threads_dir
    
    def test_save_and_load_thread(self):
        """Test saving and loading a thread."""
        # Create test data
        thread_id = "test-thread-1"
        metadata = {"test_key": "test_value"}
        topic = "Test Topic"
        checkpoints = [{"state": "test_state", "timestamp": datetime.now().isoformat()}]
        
        # Save thread
        thread_path = self.thread_manager.save_thread(
            thread_id=thread_id,
            metadata=metadata,
            topic=topic,
            checkpoints=checkpoints
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(thread_path))
        
        # Load thread
        loaded_thread = self.thread_manager.load_thread(thread_id)
        
        # Verify data
        self.assertEqual(loaded_thread["thread_id"], thread_id)
        self.assertEqual(loaded_thread["topic"], topic)
        self.assertEqual(len(loaded_thread["checkpoints"]), len(checkpoints))
        self.assertEqual(loaded_thread["checkpoints"][0]["state"], checkpoints[0]["state"])
        
        # Verify metadata was updated with memory usage
        self.assertIn("memory_usage_mb", loaded_thread["metadata"])
        self.assertIn("last_modified", loaded_thread["metadata"])
        self.assertEqual(loaded_thread["metadata"]["test_key"], "test_value")
    
    def test_list_threads(self):
        """Test listing threads."""
        # Clear any existing threads in the test directory
        for file in os.listdir(thread_manager.THREADS_DIR):
            if file.endswith(".json"):
                os.remove(os.path.join(thread_manager.THREADS_DIR, file))
                
        # Create multiple test threads
        for i in range(3):
            thread_id = f"test-thread-{i}"
            self.thread_manager.save_thread(
                thread_id=thread_id,
                metadata={"index": i},
                topic=f"Test Topic {i}",
                checkpoints=[{"state": f"test_state_{i}", "timestamp": datetime.now().isoformat()}]
            )
        
        # List threads
        threads = self.thread_manager.list_threads()
        
        # Verify thread count
        self.assertEqual(len(threads), 3)
        
        # Verify thread data
        thread_ids = [t["thread_id"] for t in threads]
        self.assertIn("test-thread-0", thread_ids)
        self.assertIn("test-thread-1", thread_ids)
        self.assertIn("test-thread-2", thread_ids)
    
    def test_delete_thread(self):
        """Test deleting a thread."""
        # Create a test thread
        thread_id = "test-thread-delete"
        self.thread_manager.save_thread(
            thread_id=thread_id,
            metadata={},
            topic="Test Topic",
            checkpoints=[]
        )
        
        # Verify thread exists
        self.assertTrue(self.thread_manager.load_thread(thread_id) is not None)
        
        # Delete thread
        result = self.thread_manager.delete_thread(thread_id)
        
        # Verify deletion was successful
        self.assertTrue(result)
        
        # Verify thread no longer exists
        self.assertTrue(self.thread_manager.load_thread(thread_id) is None)
    
    def test_cleanup_old_threads(self):
        """Test automatic cleanup of old threads."""
        # Clear any existing threads in the test directory
        for file in os.listdir(thread_manager.THREADS_DIR):
            if file.endswith(".json"):
                os.remove(os.path.join(thread_manager.THREADS_DIR, file))
                
        # Create more threads than the maximum
        for i in range(10):  # Max is 5
            thread_id = f"test-thread-{i}"
            self.thread_manager.save_thread(
                thread_id=thread_id,
                metadata={},
                topic=f"Test Topic {i}",
                checkpoints=[]
            )
            
            # Add a small delay to ensure different modification times
            import time
            time.sleep(0.1)
        
        # List threads
        threads = self.thread_manager.list_threads(limit=10)
        
        # Verify only max_threads threads remain
        self.assertEqual(len(threads), 5)
        
        # Verify the oldest threads were deleted (thread-0 through thread-4)
        thread_ids = [t["thread_id"] for t in threads]
        for i in range(5):
            self.assertNotIn(f"test-thread-{i}", thread_ids)
        
        # Verify the newest threads were kept (thread-5 through thread-9)
        for i in range(5, 10):
            self.assertIn(f"test-thread-{i}", thread_ids)


if __name__ == "__main__":
    unittest.main()
